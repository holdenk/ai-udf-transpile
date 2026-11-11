# SPDX-License-Identifier: Apache-2.0
"""Static UDF extraction: find PySpark UDFs in a Python file without running it.

The runtime hook keys a UDF on what Spark hands it (the parsed ``def``, the
declared return type, the live function's captures). This module rebuilds
the same key from the file alone, so a one-off pass over a directory can
pre-fill the catalog before any job runs. It only ever *parses* the target
file; the only thing it imports is the standard library, to resolve a
captured stdlib module or constant exactly the way the live function would.

Recognized shapes (each needs annotated params and an atomic return type,
the same gate the runtime hook applies):

* ``@udf`` / ``@udf(LongType())`` / ``@F.udf(returnType="bigint")`` on a def
* ``udf(plus_one, LongType())`` / ``F.udf(f=plus_one, returnType=...)``
* ``spark.udf.register("name", plus_one, LongType())``

Everything the static view cannot prove matches the runtime key is skipped
with a reason rather than guessed at: lambdas (no annotations), functions
imported from elsewhere, captures of non-literal or non-stdlib values, locals
of an enclosing function, names bound more than once, non-``udf`` decorators,
Arrow UDFs, ``asNondeterministic()``.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ai_udf_transpile.keys import (
    BUILTIN_NAMES,
    _is_allowed_literal,
    canonical_source,
    closure_fingerprint,
    decorator_callee,
    dotted_parts,
    free_names,
    is_pyspark_udf_qualname,
    strip_udf_decorators,
    udf_key,
)
from ai_udf_transpile.types import input_categories, input_spark_types, return_spark_type

# Bump when extraction changes what it finds, so the prefill file tracker
# rescans files an older extractor already marked done.
EXTRACTOR_VERSION = 1

# Builtins that ``from pyspark.sql.functions import *`` replaces with Column
# functions (pyspark.sql.functions.__all__ & dir(builtins), Spark master).
# A UDF in such a file that calls ``round`` gets the Spark one at runtime --
# which the runtime hook refuses to capture -- so statically it is a decline.
PYSPARK_FUNCTIONS_SHADOWED_BUILTINS = frozenset(
    {"abs", "ascii", "bin", "filter", "hash", "hex", "max", "min", "pow", "round", "slice", "sum"}
)

# Stdlib modules with import-time side effects; never import them to resolve.
_NO_IMPORT = frozenset({"antigravity", "this", "idlelib", "tkinter", "turtle", "turtledemo"})

# DDL type strings -> Spark simpleString, for _parse_datatype_string parity.
_DDL_SIMPLE = {
    "tinyint": "tinyint",
    "byte": "tinyint",
    "smallint": "smallint",
    "short": "smallint",
    "int": "int",
    "integer": "int",
    "bigint": "bigint",
    "long": "bigint",
    "float": "float",
    "real": "float",
    "double": "double",
    "string": "string",
    "boolean": "boolean",
    "binary": "binary",
}
_TYPE_CLASS_SIMPLE = {
    "ByteType": "tinyint",
    "ShortType": "smallint",
    "IntegerType": "int",
    "LongType": "bigint",
    "FloatType": "float",
    "DoubleType": "double",
    "StringType": "string",
    "BooleanType": "boolean",
    "BinaryType": "binary",
}
_DECIMAL_RE = re.compile(r"^(?:decimal|dec|numeric)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?$")
# Spark's _transpile_func only engages for NumericType/StringType/BooleanType/BinaryType.
_ATOMIC_SIMPLE = frozenset(_DDL_SIMPLE.values())


@dataclass
class ExtractedUDF:
    path: str
    lineno: int
    name: str
    source_text: str
    param_names: list[str]
    input_types: list[str]
    input_categories: list[str]
    return_type: str
    captures: dict[str, Any]
    closure_fingerprint: str

    def key(self, spark_version: str) -> str:
        return udf_key(
            self.source_text,
            self.param_names,
            self.input_types,
            self.return_type,
            spark_version,
            self.closure_fingerprint,
        )


@dataclass
class SkippedUDF:
    path: str
    lineno: int
    name: str
    reason: str


@dataclass
class FileExtraction:
    path: str
    sha256: str
    udfs: list[ExtractedUDF] = field(default_factory=list)
    skipped: list[SkippedUDF] = field(default_factory=list)
    error: Optional[str] = None


class _Skip(Exception):
    pass


# -- scopes -------------------------------------------------------------------


@dataclass
class _Binding:
    kind: str  # import | from | assign | def | class | param | other
    lineno: int
    target: Optional[str] = None  # imports: the qualified name bound
    node: Optional[ast.AST] = None  # assign: the value; def: the FunctionDef


@dataclass
class _Scope:
    kind: str  # module | function | class | lambda
    node: ast.AST
    bindings: dict[str, list[_Binding]] = field(default_factory=dict)
    star_imports: list[Optional[str]] = field(default_factory=list)

    def add(self, name: str, binding: _Binding) -> None:
        self.bindings.setdefault(name, []).append(binding)

    def only(self, name: str) -> Optional[_Binding]:
        found = self.bindings.get(name) or []
        return found[0] if len(found) == 1 else None


class _BindingCollector(ast.NodeVisitor):
    """Names a scope binds. Does not descend into nested scopes' bodies."""

    def __init__(self, scope: _Scope):
        self.scope = scope

    def _store(self, target: ast.AST, kind: str = "other", value: Optional[ast.AST] = None) -> None:
        if isinstance(target, ast.Name):
            self.scope.add(target.id, _Binding(kind, target.lineno, node=value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._store(elt)
        elif isinstance(target, ast.Starred):
            self._store(target.value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.scope.add(alias.asname, _Binding("import", node.lineno, target=alias.name))
            else:
                top = alias.name.split(".")[0]
                self.scope.add(top, _Binding("import", node.lineno, target=top))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module if node.level == 0 else None
        for alias in node.names:
            if alias.name == "*":
                self.scope.star_imports.append(module)
                continue
            target = f"{module}.{alias.name}" if module else None
            self.scope.add(alias.asname or alias.name, _Binding("from", node.lineno, target=target))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._store(target, "assign" if isinstance(target, ast.Name) else "other", node.value)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._store(node.target, "assign", node.value)
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._store(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._store(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._store(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_withitem(self, node: ast.withitem) -> None:
        if node.optional_vars is not None:
            self._store(node.optional_vars)
        self.visit(node.context_expr)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.scope.add(node.name, _Binding("other", node.lineno))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._store(target)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.add(node.name, _Binding("def", node.lineno, node=node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.add(node.name, _Binding("other", node.lineno))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.add(node.name, _Binding("class", node.lineno))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def _comprehension(self, node: ast.AST) -> None:
        return  # comprehension targets are local to the comprehension

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.scope.add(node.name, _Binding("other", node.lineno))
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.scope.add(node.name, _Binding("other", node.lineno))

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.scope.add(node.rest, _Binding("other", node.lineno))
        self.generic_visit(node)


def _build_scope(kind: str, node: ast.AST) -> _Scope:
    scope = _Scope(kind, node)
    collector = _BindingCollector(scope)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            scope.add(arg.arg, _Binding("param", getattr(arg, "lineno", 0)))
        for arg in (args.vararg, args.kwarg):
            if arg is not None:
                scope.add(arg.arg, _Binding("param", getattr(arg, "lineno", 0)))
        body = [node.body] if isinstance(node, ast.Lambda) else node.body
    else:
        body = getattr(node, "body", [])
    for stmt in body:
        collector.visit(stmt)
    return scope


def _globals_declared(tree: ast.AST) -> set[str]:
    """Names some function rebinds with ``global``: their module value is not static."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


# -- resolution -----------------------------------------------------------------


def _visible(stack: list[_Scope]) -> list[_Scope]:
    """Scopes a name in ``stack[-1]`` can see, innermost first (class bodies are skipped)."""
    out = [stack[-1]]
    for scope in reversed(stack[:-1]):
        if scope.kind != "class":
            out.append(scope)
    return out


def _qualname(parts: list[str], stack: list[_Scope]) -> Optional[str]:
    """Statically resolve ``F.udf`` to ``pyspark.sql.functions.udf`` through imports."""
    head, rest = parts[0], parts[1:]
    for scope in _visible(stack):
        found = scope.bindings.get(head)
        if not found:
            continue
        targets = {b.target for b in found if b.kind in {"import", "from"}}
        if len(targets) != 1 or any(b.kind not in {"import", "from"} for b in found):
            return None  # rebound, or bound to something that is not an import
        target = targets.pop()
        return ".".join([target, *rest]) if target else None
    if not rest:
        # `from pyspark.sql.functions import *` then a bare `udf`.
        for scope in _visible(stack):
            for module in scope.star_imports:
                if module and module.startswith("pyspark."):
                    return f"{module}.{head}"
    return None


def _is_udf_callee(node: ast.AST, stack: list[_Scope]) -> bool:
    parts = dotted_parts(node)
    return parts is not None and is_pyspark_udf_qualname(_qualname(parts, stack))


def _safe_import(name: str) -> Any:
    top = name.split(".")[0]
    if top not in getattr(sys, "stdlib_module_names", frozenset()) or top in _NO_IMPORT:
        raise _Skip(f"captures non-stdlib module {name!r}")
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise _Skip(f"could not import stdlib module {name!r}: {exc}") from exc


def _capture_value(value: Any, name: str) -> tuple[bool, Any]:
    """(include, capture) for a resolved value -- extract_captures' rules."""
    from types import ModuleType

    if name in BUILTIN_NAMES and value is getattr(builtins, name, object()):
        return False, None
    if _is_allowed_literal(value):
        return True, value
    if isinstance(value, ModuleType):
        if (value.__name__ or "").split(".")[0] in getattr(sys, "stdlib_module_names", frozenset()):
            return True, {"__module__": value.__name__}
        raise _Skip(f"captures non-stdlib module {value.__name__!r} as {name!r}")
    raise _Skip(f"captures {name!r}, a {type(value).__name__} (only literals and stdlib modules key)")


def _resolve_capture(name: str, module: _Scope, rebound: set[str]) -> tuple[bool, Any]:
    found = module.bindings.get(name)
    if not found:
        if module.star_imports:
            for star in module.star_imports:
                if star in {"pyspark.sql.functions", "pyspark.sql.functions.builtin"}:
                    if name in PYSPARK_FUNCTIONS_SHADOWED_BUILTINS:
                        raise _Skip(f"{name!r} is pyspark.sql.functions.{name} here (star import)")
                elif star == "pyspark.sql.types":
                    continue
                elif star and star.split(".")[0] in getattr(sys, "stdlib_module_names", frozenset()):
                    mod = _safe_import(star)
                    if hasattr(mod, name) and (getattr(mod, "__all__", None) is None or name in mod.__all__):
                        return _capture_value(getattr(mod, name), name)
                else:
                    raise _Skip(f"{name!r} may come from star import of {star or 'a relative module'}")
        if name in BUILTIN_NAMES:
            return False, None
        raise _Skip(f"free name {name!r} is not bound in this file")
    if name in rebound:
        raise _Skip(f"{name!r} is rebound with `global`; its value is not static")
    if len(found) != 1:
        raise _Skip(f"{name!r} is bound {len(found)} times at module level; its value is not static")
    binding = found[0]
    if binding.kind == "import":
        mod = _safe_import(binding.target or name)
        return _capture_value(mod, name)
    if binding.kind == "from":
        if not binding.target:
            raise _Skip(f"{name!r} comes from a relative import")
        module_name, _, attr = binding.target.rpartition(".")
        if module_name.split(".")[0] not in getattr(sys, "stdlib_module_names", frozenset()):
            raise _Skip(f"captures {name!r} from non-stdlib module {module_name!r}")
        try:
            mod = _safe_import(binding.target)  # `from os import path` is a module
        except _Skip:
            mod = _safe_import(module_name)
            if not hasattr(mod, attr):
                raise _Skip(f"stdlib {module_name!r} has no {attr!r}") from None
            return _capture_value(getattr(mod, attr), name)
        return _capture_value(mod, name)
    if binding.kind == "assign":
        try:
            value = ast.literal_eval(binding.node)
        except Exception:
            raise _Skip(f"captures {name!r}, which is not a literal") from None
        return _capture_value(value, name)
    if binding.kind == "def":
        raise _Skip(f"calls module-level function {name!r} (functions do not key)")
    raise _Skip(f"captures {name!r} (a {binding.kind}); only literals and stdlib modules key")


def _static_captures(fn: ast.FunctionDef, params: list[str], stack: list[_Scope]) -> dict[str, Any]:
    module = stack[0]
    rebound = _globals_declared(module.node)
    enclosing = [s for s in stack[1:] if s.kind in {"function", "lambda"}]
    captures: dict[str, Any] = {}
    for name in sorted(free_names(fn, params)):
        for scope in enclosing:
            if name in scope.bindings:
                raise _Skip(f"captures {name!r} from an enclosing function (closure values are not static)")
        include, value = _resolve_capture(name, module, rebound)
        if include:
            captures[name] = value
    return captures


# -- return types -----------------------------------------------------------------


def _literal_int(node: Optional[ast.AST], default: int) -> int:
    if node is None:
        return default
    value = ast.literal_eval(node)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError
    return value


def _simple_from_ddl(text: str) -> Optional[str]:
    t = " ".join(text.strip().lower().split())
    if t in _DDL_SIMPLE:
        return _DDL_SIMPLE[t]
    m = _DECIMAL_RE.match(t)
    if m:
        precision = int(m.group(1)) if m.group(1) else 10
        scale = int(m.group(2)) if m.group(2) else 0
        return f"decimal({precision},{scale})"
    return None


def _return_simple(node: Optional[ast.AST], stack: list[_Scope], depth: int = 0) -> Optional[str]:
    """Spark simpleString for a returnType expression, or None when not static/atomic."""
    if node is None:
        return "string"  # udf()'s default returnType is StringType()
    if depth > 4:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _simple_from_ddl(node.value)
    if isinstance(node, ast.Call):
        parts = dotted_parts(node.func)
        if not parts:
            return None
        cls = parts[-1]
        if cls in _TYPE_CLASS_SIMPLE and not node.args and not node.keywords:
            return _TYPE_CLASS_SIMPLE[cls]
        if cls == "DecimalType":
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            args = list(node.args) + [None, None]
            try:
                precision = _literal_int(args[0] or kwargs.get("precision"), 10)
                scale = _literal_int(args[1] or kwargs.get("scale"), 0)
            except Exception:
                return None
            return f"decimal({precision},{scale})"
        return None
    if isinstance(node, ast.Name):
        for scope in _visible(stack):
            if node.id in scope.bindings:
                binding = scope.only(node.id)
                if binding is not None and binding.kind == "assign":
                    return _return_simple(binding.node, stack, depth + 1)
                return None
    return None


def _is_type_expr(node: ast.AST) -> bool:
    """``udf(LongType())`` / ``udf("bigint")``: the first positional is a returnType."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Call):
        parts = dotted_parts(node.func)
        return bool(parts) and parts[-1].endswith("Type")
    return False


# -- the walk -------------------------------------------------------------------


def _udf_args(call: ast.Call) -> tuple[Optional[ast.AST], Optional[ast.AST], dict[str, ast.AST]]:
    """(f, returnType, kwargs) of a ``udf(...)`` call, either calling convention."""
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    positional = list(call.args)
    f = kwargs.get("f")
    if f is None and positional and not _is_type_expr(positional[0]):
        f = positional.pop(0)
    rt = kwargs.get("returnType")
    if rt is None and positional:
        rt = positional[0]
    return f, rt, kwargs


def _uses_arrow(kwargs: dict[str, ast.AST]) -> bool:
    node = kwargs.get("useArrow")
    return isinstance(node, ast.Constant) and node.value is True


class _Finder(ast.NodeVisitor):
    def __init__(self, path: str, module: ast.Module):
        self.path = path
        self.stack: list[_Scope] = [_build_scope("module", module)]
        self.parents: dict[int, ast.AST] = {}
        for parent in ast.walk(module):
            for child in ast.iter_child_nodes(parent):
                self.parents[id(child)] = parent
        self.udfs: list[ExtractedUDF] = []
        self.skipped: list[SkippedUDF] = []
        self._seen: set[tuple[int, str]] = set()
        # def node -> stack it was defined in, for udf(f) calls that name it.
        # Filled by a first pass, so `register("x", f)` inside main() can name
        # a `def f` that appears further down the file.
        self._def_stacks: dict[int, list[_Scope]] = {}
        self._detect = False
        # The `udf(...)` of a `@udf(...)` decorator: the def handles it.
        self._decorator_calls: set[int] = set()

    def run(self, module: ast.Module) -> None:
        self.visit(module)
        self.stack = [_build_scope("module", module)]
        self._detect = True
        self.visit(module)

    # scope plumbing

    def _push(self, kind: str, node: ast.AST) -> None:
        self.stack.append(_build_scope(kind, node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._detect:
            self._def_stacks[id(node)] = list(self.stack)
        udf_decorators = [d for d in node.decorator_list if _is_udf_callee(decorator_callee(d), self.stack)]
        for decorator in udf_decorators:
            self._decorator_calls.add(id(decorator))
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if udf_decorators and self._detect:
            self._from_decorator(node, udf_decorators[0])
        self._push("function", node)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._push("function", node)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        self._push("class", node)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._push("lambda", node)
        self.visit(node.body)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if not self._detect or id(node) in self._decorator_calls:
            pass
        elif _is_udf_callee(node.func, self.stack):
            self._from_udf_call(node)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            receiver = node.func.value
            if isinstance(receiver, ast.Attribute) and receiver.attr == "udf":
                self._from_register(node)
        self.generic_visit(node)

    # the three shapes

    def _from_decorator(self, fn: ast.FunctionDef, decorator: ast.AST) -> None:
        rt: Optional[ast.AST] = None
        if isinstance(decorator, ast.Call):
            # As a decorator the function is never an argument: the first
            # positional, whatever it looks like, is the returnType.
            kwargs = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
            rt = kwargs.get("returnType") or (decorator.args[0] if decorator.args else None)
            if _uses_arrow(kwargs):
                self._skip(fn.lineno, fn.name, "Arrow UDF (useArrow=True) is never transpiled")
                return
        if self.stack[-1].kind == "class":
            self._skip(fn.lineno, fn.name, "method UDFs are not extracted")
            return
        self._add(fn, rt, list(self.stack), fn.lineno)

    def _from_udf_call(self, call: ast.Call) -> None:
        f, rt, kwargs = _udf_args(call)
        if f is None:
            return  # udf(LongType()) used as a decorator factory; the def is handled there
        label = ast.unparse(f) if not isinstance(f, ast.Lambda) else "<lambda>"
        if _uses_arrow(kwargs):
            self._skip(call.lineno, label, "Arrow UDF (useArrow=True) is never transpiled")
            return
        parent = self.parents.get(id(call))
        if isinstance(parent, ast.Attribute) and parent.attr == "asNondeterministic":
            self._skip(call.lineno, label, "nondeterministic UDFs are never transpiled")
            return
        self._from_function_ref(f, rt, call.lineno, label)

    def _from_register(self, call: ast.Call) -> None:
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        args = list(call.args)
        f = args[1] if len(args) > 1 else kwargs.get("f")
        rt = args[2] if len(args) > 2 else kwargs.get("returnType")
        if f is None:
            return
        label = ast.unparse(f) if not isinstance(f, ast.Lambda) else "<lambda>"
        self._from_function_ref(f, rt, call.lineno, label)

    def _from_function_ref(self, f: ast.AST, rt: Optional[ast.AST], lineno: int, label: str) -> None:
        if isinstance(f, ast.Lambda):
            self._skip(lineno, label, "lambda UDFs carry no type annotations")
            return
        if not isinstance(f, ast.Name):
            self._skip(lineno, label, "function is not a plain name in this file")
            return
        fn, why = self._lookup_def(f.id)
        if fn is None:
            if why:
                self._skip(lineno, label, why)
            return
        if any(_is_udf_callee(decorator_callee(d), self._def_stacks[id(fn)]) for d in fn.decorator_list):
            return  # already a UDF object; its decorator was extracted where it is defined
        self._add(fn, rt, self._def_stacks[id(fn)], lineno)

    def _lookup_def(self, name: str) -> tuple[Optional[ast.FunctionDef], str]:
        for scope in _visible(self.stack):
            found = scope.bindings.get(name)
            if not found:
                continue
            if len(found) != 1:
                return None, f"{name!r} is bound {len(found)} times in its scope"
            binding = found[0]
            if binding.kind == "def" and id(binding.node) in self._def_stacks:
                return binding.node, ""
            if binding.kind in {"import", "from"}:
                return None, f"defined in another module ({binding.target or 'relative import'})"
            if (
                binding.kind == "assign"
                and isinstance(binding.node, ast.Call)
                and _is_udf_callee(binding.node.func, self.stack)
            ):
                return None, ""  # registering a udf(...) object; that call was extracted
            return None, f"{name!r} is a {binding.kind}, not a def in this file"
        return None, f"{name!r} is not defined in this file"

    # building one UDF

    def _skip(self, lineno: int, name: str, reason: str) -> None:
        self.skipped.append(SkippedUDF(self.path, lineno, name, reason))

    def _add(self, fn: ast.FunctionDef, rt: Optional[ast.AST], stack: list[_Scope], lineno: int) -> None:
        try:
            udf = self._build(fn, rt, stack)
        except _Skip as exc:
            self._skip(lineno, fn.name, str(exc))
            return
        marker = (id(fn), udf.return_type)
        if marker in self._seen:
            return
        self._seen.add(marker)
        self.udfs.append(udf)

    def _build(self, fn: ast.FunctionDef, rt: Optional[ast.AST], stack: list[_Scope]) -> ExtractedUDF:
        stripped = strip_udf_decorators(fn, lambda parts: is_pyspark_udf_qualname(_qualname(parts, stack)))
        if stripped.decorator_list:
            raise _Skip("decorated with something other than pyspark's udf")
        args = stripped.args
        if args.posonlyargs:
            raise _Skip("positional-only parameters are not keyed")
        if args.defaults or args.kwonlyargs or args.vararg or args.kwarg:
            raise _Skip("default, variadic, or keyword-only parameters are never transpiled")
        params = [a.arg for a in args.args]
        if stack[-1].kind == "class":
            raise _Skip("method UDFs are not extracted")
        simple = _return_simple(rt, stack)
        if simple is None:
            shown = ast.unparse(rt) if rt is not None else "?"
            raise _Skip(f"return type {shown} is not a static atomic type")
        if simple not in _ATOMIC_SIMPLE and not simple.startswith("decimal("):
            raise _Skip(f"return type {simple} is never transpiled (atomic types only)")
        out_type = return_spark_type(simple)
        in_types = input_spark_types(stripped, params)
        in_cats = input_categories(stripped, params)
        if out_type is None or in_types is None or in_cats is None or len(in_types) != len(params):
            raise _Skip("every parameter needs a supported type annotation")
        if not params:
            raise _Skip("zero-argument UDFs are not keyed")
        captures = _static_captures(stripped, params, stack)
        return ExtractedUDF(
            path=self.path,
            lineno=fn.lineno,
            name=fn.name,
            source_text=canonical_source(stripped),
            param_names=params,
            input_types=in_types,
            input_categories=in_cats,
            return_type=out_type,
            captures=captures,
            closure_fingerprint=closure_fingerprint(captures),
        )


def extract_source(source: Union[str, bytes], path: str = "<string>") -> FileExtraction:
    data = source.encode("utf-8") if isinstance(source, str) else source
    out = FileExtraction(path=path, sha256=hashlib.sha256(data).hexdigest())
    try:
        tree = ast.parse(data, filename=path)
    except (SyntaxError, ValueError) as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        return out
    finder = _Finder(path, tree)
    finder.run(tree)
    out.udfs = finder.udfs
    out.skipped = finder.skipped
    return out


def extract_file(path: Union[str, Path]) -> FileExtraction:
    p = Path(path)
    return extract_source(p.read_bytes(), str(p))
