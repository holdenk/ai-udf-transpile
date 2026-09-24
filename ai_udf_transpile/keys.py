# SPDX-License-Identifier: Apache-2.0
"""Canonical source, literal/stdlib captures, and the cache key."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import sys
import textwrap
from types import FunctionType, ModuleType
from typing import Any, Optional

ALLOWED_LITERAL_TYPES = (int, float, bool, str, bytes, type(None))
BUILTIN_NAMES = set(dir(builtins))


def canonical_source(function_ast: ast.AST) -> str:
    return ast.unparse(function_ast)


def canonical_source_text(src: str) -> str:
    tree = ast.parse(textwrap.dedent(src).strip())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return ast.unparse(fn)


def canonical_source_from_func(func: Any) -> str:
    src = textwrap.dedent(inspect.getsource(func))
    return canonical_source_text(src)


def _is_allowed_literal(value: Any) -> bool:
    if isinstance(value, ALLOWED_LITERAL_TYPES):
        return True
    if isinstance(value, tuple):
        return all(_is_allowed_literal(v) for v in value)
    return False


def _is_stdlib_module(mod: ModuleType) -> bool:
    name = (getattr(mod, "__name__", "") or "").split(".")[0]
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    return bool(name) and name in stdlib


def _stable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_stable(v) for v in value]
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, ModuleType):
        return {"__module__": value.__name__}
    return value


def _bound_names(function_ast: ast.FunctionDef, public_params: list[str]) -> set[str]:
    bound = set(public_params)
    bound.add(function_ast.name)
    for arg in function_ast.args.args:
        bound.add(arg.arg)
    for arg in function_ast.args.posonlyargs:
        bound.add(arg.arg)
    for arg in function_ast.args.kwonlyargs:
        bound.add(arg.arg)
    if function_ast.args.vararg:
        bound.add(function_ast.args.vararg.arg)
    if function_ast.args.kwarg:
        bound.add(function_ast.args.kwarg.arg)
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.FunctionDef) and node is not function_ast:
            bound.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def free_names(function_ast: ast.FunctionDef, public_params: list[str]) -> set[str]:
    bound = _bound_names(function_ast, public_params)
    names: set[str] = set()
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound:
                names.add(node.id)
    return names


def _resolve_name(func: Any, name: str) -> Any:
    if func is None:
        raise KeyError(name)
    if isinstance(func, FunctionType) and func.__closure__ and func.__code__.co_freevars:
        for var, cell in zip(func.__code__.co_freevars, func.__closure__):
            if var == name:
                return cell.cell_contents
    globals_dict = getattr(func, "__globals__", None) or {}
    if name in globals_dict:
        return globals_dict[name]
    raise KeyError(name)


def extract_captures(
    func: Any,
    function_ast: ast.FunctionDef,
    public_params: list[str],
) -> Optional[dict[str, Any]]:
    """Return a JSON-serialisable capture dict, or None when the UDF cannot be keyed.

    Allowed: literals (and tuples thereof), stdlib modules, and names that resolve
    to the matching builtin. Anything else declines.
    """
    names = free_names(function_ast, public_params)
    if not names:
        return {}
    if func is None:
        # Free names with no live function: builtins-only is OK.
        if names <= BUILTIN_NAMES:
            return {}
        return None
    captures: dict[str, Any] = {}
    for name in sorted(names):
        try:
            value = _resolve_name(func, name)
        except (KeyError, ValueError):
            if name in BUILTIN_NAMES:
                continue
            return None
        if name in BUILTIN_NAMES and value is getattr(builtins, name, object()):
            continue
        if _is_allowed_literal(value):
            captures[name] = value
            continue
        if isinstance(value, ModuleType) and _is_stdlib_module(value):
            captures[name] = {"__module__": value.__name__}
            continue
        return None
    return captures


def closure_fingerprint(captures: dict[str, Any]) -> str:
    if not captures:
        return ""
    payload = json.dumps(
        {k: _stable(v) for k, v in sorted(captures.items())},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def udf_key(
    source: str,
    params: list[str],
    input_types: list[str],
    return_type: str,
    spark_version: str,
    fingerprint: str,
) -> str:
    material = "\0".join(
        [
            source,
            ",".join(params),
            json.dumps(list(input_types), separators=(",", ":")),
            return_type,
            spark_version,
            fingerprint,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def func_from_spark_stack() -> Optional[Any]:
    """Best-effort: Spark's ``_transpile_func`` holds ``func`` in a parent frame."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "_transpile_func" and "func" in frame.f_locals:
            return frame.f_locals["func"]
        frame = frame.f_back
    return None
