# SPDX-License-Identifier: Apache-2.0
"""Hypothesis differential check plus an import-aware exec sandbox."""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import linecache
import logging
import re
import textwrap
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STDLIB_ALLOWLIST = frozenset(
    {
        "math",
        "cmath",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "operator",
        "statistics",
        "string",
        "re",
        "json",
        "hashlib",
        "collections",
        "typing",
        "dataclasses",
        "numbers",
        "os",
        "sys",
        "copy",
        "random",
    }
)

_SENTINEL_RAISED = object()

# reflect/java_method run arbitrary JVM methods from a SQL expression.
# Runtime/ProcessBuilder in a Java UDF runs at class-load, before any row
# check; reflection (`Class.forName`/`getMethod`/`loadClass`) reaches the
# same places without naming Runtime, and io/net/script are process and
# network surface a row function never needs. Denylist, not allowlist:
# extend it when a rewrite legitimately needs one of these and re-review.
# \s* around dots: valid Java allows whitespace/comments between tokens of
# a qualified name (`Runtime . getRuntime()` compiles fine).
# try_reflect is reflect with the exception swallowed: the same arbitrary
# static call (`try_reflect('java.lang.System', 'getenv', 'X')` reads the
# driver's environment into a column), and `\breflect` does not see it.
# The Java list is still a denylist and still leaks at the edges; the
# entries past io/net are the ones a rewrite reached without naming either
# (files and sockets through java.nio, TLS through javax.net, the process
# environment, and the getDeclared* spellings of reflection).
_FORBIDDEN_SQL = re.compile(r"\b(?:try_)?(?:reflect|java_method)\s*\(", re.IGNORECASE)
_FORBIDDEN_JAVA = re.compile(
    r"\b(?:"
    r"Runtime\s*\.\s*getRuntime"
    r"|System\s*\.\s*(?:load|loadLibrary|exit|getenv|getProperty|getProperties|setProperty)"
    r"|getenv\s*\("
    r"|ProcessBuilder"
    r"|Class\s*\.\s*forName"
    r"|loadClass\s*\("
    r"|(?:get\w*)?ClassLoader"
    r"|get(?:Declared)?(?:Method|Constructor|Field)s?\s*\("
    r"|MethodHandles"
    r"|exec\s*\("
    r"|java\s*\.\s*(?:io|net)\s*\."
    r"|java\s*\.\s*nio\s*\.\s*(?:file|channels)"
    r"|javax\s*\.\s*(?:script|net)"
    r")"
)
# A Rust row function is pure computation over its arguments. `unsafe` is
# also a compile error (the generated glue forbids it for the module); the
# rest is what safe Rust can still reach: the filesystem, processes, sockets
# and the environment through std, files read at COMPILE time through
# include!/env!/#[path], and FFI through extern. arrow and the SDK are the
# glue's business -- a row function never needs them. A module is named
# either as a path segment (`fs::read`, `std::fs`) or in a `use`. A `use`
# can also rename the crate root or glob it (`use std as s; use s::fs as q`,
# `use std::*; use fs as q`), after which the module is never followed by
# `::`: so a forbidden module anywhere in a `use`, a renamed crate root, and
# a crate-root glob are refused outright. Denylist, not a sandbox: the same
# caveat as the Java scan.
_RUST_MODULES = r"(?:fs|process|net|env|os|ffi|ptr|thread|io|path|arch|intrinsics)"
_FORBIDDEN_RUST = re.compile(
    r"\b(?:unsafe|extern)\b"
    r"|\b" + _RUST_MODULES + r"\s*::"
    r"|\b(?:std|core|alloc)\s*::\s*" + _RUST_MODULES + r"\b"
    r"|\buse\b[^;]*\b" + _RUST_MODULES + r"\b"
    r"|\b(?:std|core|alloc)\s+as\b"
    r"|\buse\b[^;]*\b(?:std|core|alloc)\s*::\s*(?:\*|\{[^;]*\*)"
    r"|\b(?:arrow\w*|comet_udf_sdk|libc)\s*::"
    r"|\b(?:SystemTime|Instant)\b"  # the clock: Comet plans the call as immutable
    # A macro can paste a module name into a path (`std::$m::read`), which
    # no token scan sees; a row function has no use for defining one.
    r"|\bmacro_rules\b"
    r"|\b(?:include|include_str|include_bytes|env|option_env|asm|global_asm|concat_idents)\s*!(?!=)"
    r"|#\s*!?\s*\[\s*(?:path|link\w*|no_mangle|export_name|used|global_allocator|cfg_attr|proc_macro\w*)\b"
)
_PLACEHOLDER = re.compile(r"_udf_param_\d+")


def _quote_closes(text: str, i: int) -> bool:
    """Quote at ``i`` closes its literal: an even run of backslashes before it.

    ``'\\\\'`` (escaped backslash, the normal spelling of a backslash regex)
    ends in a quote that DOES close; treating any backslash-preceded quote as
    escaped leaves the literal open and hides whatever follows it.
    """
    run = 0
    j = i - 1
    while j >= 0 and text[j] == "\\":
        run += 1
        j -= 1
    return run % 2 == 0


def lambda_captured_placeholder(sql: str) -> Optional[str]:
    """The `_udf_param_N` referenced inside a `->` lambda body, if any.

    TranspiledPythonUDF substitutes placeholders in the expression tree but
    does not descend into higher-order-function lambdas, so a reference there
    fails analysis with UNRESOLVED_COLUMN and takes the query down. Spark
    also refuses to attach a rewrite whose return type is an array, which
    means the live reconstruction probe never runs for those UDFs — this
    scan is what still rejects the shape.
    """
    depth = 0
    i = 0
    quote: Optional[str] = None
    while i < len(sql):
        char = sql[i]
        if quote is not None:
            if char == quote and _quote_closes(sql, i):
                quote = None
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            continue
        if char == "(":
            depth += 1
            i += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if sql.startswith("->", i):
            body_depth = depth
            j = i + 2
            body_quote: Optional[str] = None
            while j < len(sql):
                nxt = sql[j]
                if body_quote is not None:
                    if nxt == body_quote and _quote_closes(sql, j):
                        body_quote = None
                    j += 1
                    continue
                if nxt in {"'", '"'}:
                    body_quote = nxt
                    j += 1
                    continue
                if nxt == "(":
                    depth += 1
                elif nxt == ")":
                    depth -= 1
                    if depth < body_depth:
                        break
                j += 1
            token = _placeholder_outside_strings(sql[i + 2 : j])
            if token:
                return token
            i = j
            continue
        i += 1
    return None


def _placeholder_outside_strings(fragment: str) -> Optional[str]:
    """First `_udf_param_N` that is not inside a SQL string, if any."""
    quote: Optional[str] = None
    kept: list[str] = []
    i = 0
    while i < len(fragment):
        char = fragment[i]
        if quote is not None:
            if char == quote and _quote_closes(fragment, i):
                quote = None
            i += 1
            continue
        if char in {"'", '"'}:
            quote = char
            i += 1
            continue
        kept.append(char)
        i += 1
    match = _PLACEHOLDER.search("".join(kept))
    return match.group(0) if match else None


def lambda_captures_placeholder(sql: str) -> bool:
    return lambda_captured_placeholder(sql) is not None


def sql_reconstruction_error(result: Any) -> Optional[str]:
    """Analysis failure we can see without asking Spark to attach the rewrite."""
    token = lambda_captured_placeholder(getattr(result, "sql", None) or "")
    if token:
        return (
            "verified rewrite fails plan analysis when reconstructed: "
            f"{token} is referenced inside a lambda body"
        )
    return None


def _sql_for_forbidden_scan(sql: str) -> str:
    """Blank literals and comments so `reflect` and reflect/* */( still match.

    Leftmost-wins alternation (mirrors ``_JAVA_SKIP``): a literal starting
    before a ``--`` inside it consumes it, so comment syntax inside a string
    never deletes a real call from the scan, and a mention inside a literal
    is blanked with it.
    """
    return _SQL_SKIP.sub(_blank_skippable, sql).replace("`", "")


_SQL_SKIP = re.compile(
    r"'(?:[^']|'')*'"  # string literal ('' doubling)
    r"|/\*.*?\*/"  # block comment
    r"|--[^\n\r]*",  # line comment
    re.DOTALL,
)


# Comments and string/char/text-block contents, blanked before the forbidden
# scan. Leftmost match wins, so a string starting before a `//` or `/*` inside
# it consumes it -- comment syntax inside a literal is never treated as a
# comment (a real call cannot hide behind `"http://..."` on the same line),
# and a mention inside a literal is blanked with it (no false positive).
# Line comments end at CR too: javac treats \r as a line terminator.
_JAVA_SKIP = re.compile(
    r"//[^\n\r]*"  # line comment
    r"|/\*.*?\*/"  # block comment
    r'|"""(?:[^"\\]|\\.|"(?!""))*"""'  # text block
    r'|"(?:[^"\\\n]|\\.)*"'  # string literal
    r"|'(?:[^'\\\n]|\\.)*'",  # char literal
    re.DOTALL,
)

_JAVA_UNICODE_ESCAPE = re.compile(r"\\u+[0-9a-fA-F]{4}")


def _translate_java_unicode_escapes(java: str) -> str:
    """JLS 3.3: `\\uXXXX` translates before lexing, even inside comments and
    literals, so the scan must see the translated text -- `Runtime\\u002egetRuntime`
    compiles, and `// x \\u000d Runtime.getRuntime()` is a comment that ends
    mid-line. A backslash is eligible only when the run of backslashes right
    before it is even (`\\\\u0041` is an escaped backslash plus text). Single
    pass: translated output is not rescanned.
    """
    out: list[str] = []
    i = 0
    run = 0
    while i < len(java):
        if java[i] == "\\" and run % 2 == 0:
            match = _JAVA_UNICODE_ESCAPE.match(java, i)
            if match:
                out.append(chr(int(match.group(0)[-4:], 16)))
                i = match.end()
                run = 0
                continue
        out.append(java[i])
        run = run + 1 if java[i] == "\\" else 0
        i += 1
    return "".join(out)


def _blank_skippable(match: re.Match) -> str:
    """Blank a comment or literal, keeping newlines (line structure survives)."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _java_for_forbidden_scan(java: str) -> str:
    """Scan what javac would lex: unicode escapes translated, comments and
    literals blanked (the SQL side mirrors the comment half of this).

    The evasions that matter are whitespace/comments around the dot of a
    qualified name (`Runtime/* */./* */getRuntime()` compiles) and unicode
    escapes for the tokens themselves; a comment cannot split an identifier
    (`Run/* */time` does not compile).
    """
    return _JAVA_SKIP.sub(_blank_skippable, _translate_java_unicode_escapes(java))


# Comments and literals, blanked before the Rust scan. A lexer, not a regex:
# Rust block comments NEST, and a regex that stops at the first `*/` treats
# the rest of an outer comment as code -- `/* /* */ " */ exit(0); // "`
# then reads as a string covering the call. Raw strings (`r#"..."#`) have no
# escapes, and `'a` is a lifetime unless a closing quote follows one char.
_RUST_CHAR = re.compile(r"b?'(?:[^'\\\n]|\\(?:u\{[^}\n]*\}|x[0-9a-fA-F]{2}|.))'")
# Raw strings: r"", byte br"", and C cr"" (stable since Rust 1.77). A
# backslash in any of them is a plain character, so a raw string read as an
# escaped one would run on past its closing quote and hide the code after it.
_RUST_RAW_START = re.compile(r'[bc]?r(#*)"')


def _rust_blank_literals(rust: str) -> str:
    """Blank comments and literals (keeping newlines) the way rustc lexes them.

    An unterminated comment or literal blanks to the end: rustc rejects that
    source anyway, so treating the tail as inert can only reject less of
    something that will never compile.
    """
    out = list(rust)
    n = len(rust)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        ch = rust[i]
        if rust.startswith("//", i):
            j = rust.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if rust.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if rust.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif rust.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        ident_before = i > 0 and (rust[i - 1].isalnum() or rust[i - 1] == "_")
        raw = None if ident_before else _RUST_RAW_START.match(rust, i)
        if raw:
            close = '"' + raw.group(1)
            j = rust.find(close, raw.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
            continue
        if ch == '"' or (ch == "b" and not ident_before and rust.startswith('b"', i)):
            j = i + (2 if ch == "b" else 1)
            while j < n and rust[j] != '"':
                j += 2 if rust[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
            continue
        if ch == "'" or (ch == "b" and not ident_before and rust.startswith("b'", i)):
            lit = _RUST_CHAR.match(rust, i)
            if lit:
                blank(i, lit.end())
                i = lit.end()
                continue
        i += 1
    return "".join(out)


# `r#fs` is the identifier `fs`: drop the raw prefix so `std::r#fs` reads as
# what it names. (Raw strings also start `r#`, but those are already blank.)
_RUST_RAW_IDENT = re.compile(r"\br#(?=[A-Za-z_])")


def _rust_for_forbidden_scan(rust: str) -> str:
    return _RUST_RAW_IDENT.sub("", _rust_blank_literals(rust))


_RUST_FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
_RUST_IMPL = re.compile(r"\bimpl\b")
_RUST_LOCAL_PATH_ROOT = re.compile(r"\b(?:struct|enum|union|mod|type)\s+([A-Za-z_]\w*)")
# What carries a call without naming a function: trait dispatch (an operator,
# `{}` formatting, `==`, `.next()` in a for loop all reach an impl's method
# without its name), an erased closure (`dyn Fn`), `Any` (a fn pointer behind
# a downcast), and a type that holds a function (a closure handed a struct
# holding itself).
_RUST_HIDDEN_CALL = re.compile(
    r"\bimpl\b[^{;]*\bfor\b(?!\s*<)"
    r"|\btrait\s+[A-Za-z_]"
    r"|\bdyn\b"
    r"|\bAny\b"
)
_RUST_FN_TYPE = re.compile(r"\bfn\s*\(|\bFn(?:Mut|Once)?\b")


def _rust_block_end(scanned: str, start: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``start`` (or the text's end)."""
    depth = 0
    for i in range(start, len(scanned)):
        if scanned[i] == "{":
            depth += 1
        elif scanned[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(scanned)


def _rust_item_open(scanned: str, pos: int) -> Optional[int]:
    """The ``{`` opening the item whose header starts at ``pos``; None at a ``;`` first.

    ``;`` inside ``(..)``/``[..]`` is an array length (``[u8; 4]``), not the end.
    """
    i = _rust_item_stop(scanned, pos)
    return i if i < len(scanned) and scanned[i] == "{" else None


def _rust_item_stop(scanned: str, pos: int) -> int:
    """The first ``{`` or ``;`` outside ``(..)``/``[..]`` from ``pos`` (or the end)."""
    depth = 0
    for i in range(pos, len(scanned)):
        ch = scanned[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth <= 0 and ch in "{;":
            return i
    return len(scanned)


def _rust_item_end(scanned: str, pos: int) -> int:
    """Where the item whose header starts at ``pos`` ends: its ``;`` or closing ``}``."""
    i = _rust_item_stop(scanned, pos)
    return _rust_block_end(scanned, i) if i < len(scanned) and scanned[i] == "{" else i


def _rust_fn_bodies(scanned: str) -> list[tuple[str, int, int, int]]:
    """(name, fn_start, name_end, end) for every ``fn`` with a body, in blanked source.

    ``fn_start..end`` runs from ``fn`` to the closing brace; ``name_end..end``
    is the body proper, parameters included.
    """
    out = []
    for match in _RUST_FN.finditer(scanned):
        open_ = _rust_item_open(scanned, match.end())
        if open_ is not None:
            out.append((match.group(1), match.start(), match.end(), _rust_block_end(scanned, open_)))
    return out


def _rust_recursion(scanned: str) -> Optional[str]:
    """Why the blanked source may recurse, if it may.

    A Rust stack overflow is not a panic: it aborts the process, and in
    local mode that process is the driver. Deep enough input reaches any
    recursion depth, so a rewrite that can reach a function from itself is
    refused -- directly, through a function value (``let f = udf; f(x)``,
    a fn-pointer table), or through a cycle of helpers or methods.

    The check is by name, not by scope: any mention of a function's name in
    a body is an edge, a local of the same name included (a scope-aware
    exception let ``let f = udf; { let udf = 0; } f(x)`` through), and a
    name is every function with it (dynamic dispatch). A path reaches a free
    function only through ``crate``/``self``/``super``/a local module; a
    method also through ``.name`` and any ``Type::name``. Calls that name no
    function -- trait dispatch, erased closures, ``Any`` -- are refused
    outright, as is a function value outside every body (a ``static``).
    Runaway loops are the prompt's to prevent.
    """
    hidden = _RUST_HIDDEN_CALL.search(scanned)
    if hidden:
        return (
            f"rust rewrite uses `{hidden.group(0).split()[0]}`: trait impls, `dyn`, and `Any` "
            "can call a function without naming it, so recursion through them cannot be "
            "ruled out; use plain functions and inherent methods"
        )
    for match in _RUST_LOCAL_PATH_ROOT.finditer(scanned):
        kind = match.group(0).split()[0]
        if kind == "mod":
            continue
        if _RUST_FN_TYPE.search(scanned[match.end() : _rust_item_end(scanned, match.end())]):
            return (
                f"rust rewrite defines a type holding a function ({match.group(1)}): a "
                "closure handed such a value can call itself; pass functions as arguments"
            )
    impls = []
    for match in _RUST_IMPL.finditer(scanned):
        open_ = _rust_item_open(scanned, match.end())
        if open_ is not None:
            impls.append((open_, _rust_block_end(scanned, open_)))
    bodies = _rust_fn_bodies(scanned)
    names = {name for name, *_ in bodies}
    if not names:
        return None
    methods = {name for name, start, *_ in bodies if any(a < start < b for a, b in impls)}
    roots = {"crate", "self", "super", "Self"} | {m.group(1) for m in _RUST_LOCAL_PATH_ROOT.finditer(scanned)}
    root = "|".join(re.escape(r) for r in sorted(roots))

    def mention(name: str) -> re.Pattern:
        word = re.escape(name)
        alts = [
            rf"(?<![.\w:']){word}\b(?!\s*!)",
            rf"\b(?:{root})(?:\s*::\s*\w+)*\s*::\s*{word}\b",
        ]
        if name in methods:
            alts.append(rf"(?:\.|::)\s*{word}\b(?!\s*!)")
        return re.compile("|".join(alts))

    patterns = {name: mention(name) for name in names}
    # Outside every body; a `fn name` (with a body or without) declares, not mentions.
    outside = list(scanned)
    for _, start, _, end in bodies:
        outside[start : end + 1] = " " * (end + 1 - start)
    for m in _RUST_FN.finditer(scanned):
        outside[m.start() : m.end()] = " " * (m.end() - m.start())
    outside_text = "".join(outside)
    for name, pattern in patterns.items():
        if pattern.search(outside_text):
            return (
                f"rust rewrite uses the function {name} as a value outside a function "
                "body (a static, const, or use): it can then be called from anywhere, "
                "itself included"
            )
    edges: dict[str, set[str]] = {name: set() for name in names}
    for name, _, name_end, end in bodies:
        # From after `fn name`: the declaration itself is no self-edge
        # (a nested `fn`'s text is part of the body and counts).
        text = scanned[name_end:end]
        for other, pattern in patterns.items():
            if pattern.search(text):
                edges[name].add(other)
    for node in sorted(names):
        seen: set[str] = set()
        stack = list(edges[node])
        while stack:
            cur = stack.pop()
            if cur == node:
                return (
                    f"rust rewrite is recursive ({node}): a stack overflow aborts the "
                    "process instead of failing the query (any mention of a function's "
                    "name counts as a call, a local or parameter of the same name "
                    "included: rename it)"
                )
            if cur not in seen:
                seen.add(cur)
                stack.extend(edges.get(cur, ()))
    return None


@functools.lru_cache(maxsize=256)
def _rust_rejected(rust: str) -> Optional[str]:
    """Scan verdict for one Rust source; memoized, since every lookup re-checks a served row."""
    scanned = _rust_for_forbidden_scan(rust)
    if _FORBIDDEN_RUST.search(scanned):
        return (
            "rust rewrite uses a forbidden construct "
            "(unsafe/extern/fs/process/net/env/io/clock/include!/env!/macro_rules!/#[path]/arrow)"
        )
    return _rust_recursion(scanned)


def rejected_rewrite(result: Any) -> Optional[str]:
    """Return an error if the rewrite must not be compiled or executed.

    A binary-only java result (register_impl impl_binary) skips the Java
    scan: the binary is trusted exactly like human-supplied catalyst_sql,
    which this gate never re-parses either.
    """
    sql = getattr(result, "sql", None) or ""
    java = getattr(result, "java_source", None) or ""
    rust = getattr(result, "rust_source", None) or ""
    if _FORBIDDEN_SQL.search(_sql_for_forbidden_scan(sql)):
        return "rewrite invokes reflect or java_method"
    if _FORBIDDEN_JAVA.search(_java_for_forbidden_scan(java)):
        return "java rewrite uses a forbidden API (Runtime/ProcessBuilder/reflection/exec/io/net/script)"
    if rust:
        return _rust_rejected(rust)
    return None


class VerifyFailed(Exception):
    """Hypothesis or sandbox determined the rewrite is not equivalent."""


def collect_imported_module_names(source_text: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _copy_defining_module_imports(func: Any, ns: dict[str, Any]) -> None:
    globals_dict = getattr(func, "__globals__", None) or {}
    for key, value in globals_dict.items():
        if inspect.ismodule(value) and key not in ns:
            ns[key] = value
    try:
        srcfile = inspect.getsourcefile(func)
    except TypeError:
        srcfile = None
    if not srcfile:
        return
    try:
        with open(srcfile, encoding="utf-8") as fh:
            file_src = fh.read()
        tree = ast.parse(file_src)
    except Exception:
        return
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in globals_dict:
                    ns.setdefault(bound, globals_dict[bound])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in globals_dict:
                    ns.setdefault(bound, globals_dict[bound])


def _hydrate_captures(captures: Optional[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in (captures or {}).items():
        if isinstance(value, dict) and "__module__" in value:
            out[name] = importlib.import_module(value["__module__"])
        elif isinstance(value, dict) and "__bytes__" in value:
            out[name] = bytes.fromhex(value["__bytes__"])
        else:
            out[name] = value
    return out


class _OnDemandModules(dict):
    def __missing__(self, key: str) -> Any:
        if key in STDLIB_ALLOWLIST:
            mod = importlib.import_module(key)
            self[key] = mod
            return mod
        raise KeyError(key)


def exec_udf_source(
    source_text: str,
    captures: Optional[dict[str, Any]] = None,
    func: Any = None,
) -> dict[str, Any]:
    """Exec ``source_text`` in an import-aware namespace. Raises NameError/ImportError."""
    ns: _OnDemandModules = _OnDemandModules()
    ns.update(_hydrate_captures(captures))
    if func is not None:
        _copy_defining_module_imports(func, ns)
    for name in collect_imported_module_names(source_text):
        if name in STDLIB_ALLOWLIST and name not in ns:
            ns[name] = importlib.import_module(name)
    dedented = textwrap.dedent(source_text)
    # Compile under a pseudo-filename registered with linecache so
    # inspect.getsource works on the exec'd function -- Spark's transpiler
    # hook reads source that way, and the reconstruction smoke test rebuilds
    # the UDF from exactly this text.
    filename = f"<ai_udf_verify_{abs(hash(dedented))}>"
    linecache.cache[filename] = (len(dedented), None, dedented.splitlines(True), filename)
    # Not a hard import boundary: allowlisted stdlib loads on demand and the
    # defining module's imports come along (the UDF's own module is trusted).
    exec(compile(dedented, filename, "exec"), ns, ns)  # noqa: S102
    return ns


def load_python_udf(
    source_text: str,
    captures: Optional[dict[str, Any]] = None,
    func: Any = None,
) -> Callable[..., Any]:
    ns = exec_udf_source(source_text, captures, func)
    callables = [
        v for v in ns.values() if inspect.isfunction(v) and getattr(v, "__name__", None) != "<lambda>"
    ]
    if not callables:
        raise VerifyFailed("no function defined in source_text")
    return callables[-1]


def _strategy_for(spark_type: str, *, party_on_strings: bool = False):
    from hypothesis import strategies as st

    t = spark_type.strip().lower()
    if t in {"bigint", "long"}:
        # int64 range: the (2^40, 2^53] region is where int/float math
        # diverges (the prompt's own floor(x / y) trick); int32-only draws
        # never differentially test it.
        return st.one_of(st.none(), st.integers(min_value=-(2**63), max_value=2**63 - 1))
    if t in {"int", "integer", "smallint", "tinyint", "byte", "short"}:
        return st.one_of(st.none(), st.integers(min_value=-(2**31), max_value=2**31 - 1))
    if t.startswith("decimal"):
        return st.one_of(st.none(), st.decimals(allow_nan=False, allow_infinity=False, places=4))
    if t in {"double", "float"}:
        return st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False, width=32),
        )
    if t == "string" and not party_on_strings:
        return st.none()
    if t == "string":
        # Exclude surrogates (invalid in Spark strings) and codepoints
        # unassigned in this Python's Unicode: the JVM may run a newer
        # Unicode and case-map them (e.g. U+A7D5 -> U+A7D4 on JDKs with
        # Unicode 16 vs Python 3.13's 15.1), a version skew no rewrite can
        # control. Assigned-char mappings are stable per Unicode policy.
        return st.one_of(
            st.none(),
            st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cn")), max_size=16),
        )
    if t in {"boolean", "bool"}:
        return st.one_of(st.none(), st.booleans())
    if t == "binary":
        return st.one_of(st.none(), st.binary(max_size=16))
    if t in {"timestamp", "timestamp_ntz"}:
        import datetime as _dt

        # Naive datetimes, 1900-2100: covers pre-1970 (negative epochs) and
        # the 2038 boundary without platform strftime corner cases.
        return st.one_of(
            st.none(),
            st.datetimes(min_value=_dt.datetime(1900, 1, 1), max_value=_dt.datetime(2100, 12, 31)),
        )
    if t.startswith("map<") and t.endswith(">"):
        key_t, _, val_t = t[4:-1].partition(",")
        if key_t.strip() == "string" and val_t.strip() == "string":

            def _text(max_size: int):
                return st.text(
                    alphabet=st.characters(blacklist_categories=("Cs", "Cn")),
                    max_size=max_size,
                )

            return st.one_of(
                st.none(),
                st.dictionaries(_text(8), _text(16), max_size=4),
            )
    if t.startswith("array<") and t.endswith(">"):
        inner = t[6:-1].strip()
        if inner == "string":
            # Membership-test UDFs (x in lst) care about short lists of short
            # strings. Elements are never None: python's `None in [None]` is
            # True while SQL array_contains(arr, NULL) is NULL, and null
            # elements in the constant membership lists these UDFs close over
            # / receive do not occur -- a real sample containing one fails
            # verification closed, which is the honest outcome.
            return st.one_of(
                st.none(),
                st.lists(
                    st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cn")), max_size=8),
                    max_size=5,
                ),
            )
        if inner == "array<string>":
            # Flatten-style UDFs iterate the inner lists, so inner lists are
            # never None (python raises TypeError on `for x in None`, and a
            # python raise allows any sql result -- no signal either way).
            # But ELEMENTS may be None: python passes them through and
            # Spark's flatten preserves them, while explode/collect_list and
            # filter-style rewrites drop them -- null elements are where
            # those rewrites die.
            return st.one_of(
                st.none(),
                st.lists(
                    st.lists(
                        st.one_of(
                            st.none(),
                            st.text(
                                alphabet=st.characters(blacklist_categories=("Cs", "Cn")),
                                max_size=8,
                            ),
                        ),
                        max_size=4,
                    ),
                    max_size=4,
                ),
            )
    raise VerifyFailed(f"unsupported input type for verify: {spark_type}")


def _spark_type(simple: str):
    from pyspark.sql.types import (
        BinaryType,
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
    )

    t = simple.strip().lower()
    if t.startswith("map<") and t.endswith(">"):
        key_t, _, val_t = t[4:-1].partition(",")
        return MapType(_spark_type(key_t), _spark_type(val_t))
    if t.startswith("array<") and t.endswith(">"):
        from pyspark.sql.types import ArrayType

        return ArrayType(_spark_type(t[6:-1]))
    if t == "timestamp":
        from pyspark.sql.types import TimestampType

        return TimestampType()
    if t == "timestamp_ntz":
        from pyspark.sql.types import TimestampNTZType

        return TimestampNTZType()
    return {
        "tinyint": ByteType(),
        "byte": ByteType(),
        "smallint": ShortType(),
        "short": ShortType(),
        "int": IntegerType(),
        "integer": IntegerType(),
        "bigint": LongType(),
        "long": LongType(),
        "float": FloatType(),
        "double": DoubleType(),
        "string": StringType(),
        "boolean": BooleanType(),
        "bool": BooleanType(),
        "binary": BinaryType(),
    }.get(t, LongType() if t.startswith("decimal") else StringType())


# One Spark job per example dominates verify time. A hundred rows in one
# DataFrame is the cheap path; a batch that throws (ANSI divide-by-zero on a
# single row fails the whole job) falls back to per-row eval.
SQL_BATCH_ROWS = 100


def _param_fields(input_types: list[str]):
    from pyspark.sql.types import StructField

    return [
        StructField(f"_udf_param_{i}", _spark_type(input_types[i]), True) for i in range(len(input_types))
    ]


def _row_values(args: tuple) -> dict[str, Any]:
    return {f"_udf_param_{i}": args[i] for i in range(len(args))}


_COMET_RDD_CONVERT = "spark.comet.convert.rdd.enabled"
_NATIVE_ROWS_LOCK = threading.Lock()
# Per SparkSession (conf is per session): id -> {"depth": n, "prior": value}.
_NATIVE_ROWS: dict[int, dict[str, Any]] = {}


@contextmanager
def _comet_native_rows(spark: Any, native: bool):
    """Let Comet take the operator above an in-memory DataFrame.

    A Rust UDF runs only inside Comet's native execution, and the verifier's
    rows come from ``createDataFrame`` (an RDD scan Comet leaves to Spark
    unless asked to convert it). Session-wide for the duration, like the
    untyped-UDF switch in javac: harmless to a concurrent query, which at
    worst gets its own RDD scans converted to Arrow.
    """
    if not native:
        yield
        return
    # Refcounted: overlapping verifications (the inline worker and a
    # register_impl call) must restore the value from before the FIRST one,
    # not whatever the other had set.
    # Keyed by session: the spark object is held for the whole scope, so its
    # id cannot be reused by another session while the entry exists.
    session = id(spark)
    with _NATIVE_ROWS_LOCK:
        state = _NATIVE_ROWS.get(session)
        if state is None:
            state = _NATIVE_ROWS[session] = {"depth": 0, "prior": spark.conf.get(_COMET_RDD_CONVERT, None)}
            spark.conf.set(_COMET_RDD_CONVERT, "true")
        state["depth"] += 1
    try:
        yield
    finally:
        with _NATIVE_ROWS_LOCK:
            state["depth"] -= 1
            if state["depth"] == 0:
                del _NATIVE_ROWS[session]
                if state["prior"] is None:
                    spark.conf.unset(_COMET_RDD_CONVERT)
                else:
                    spark.conf.set(_COMET_RDD_CONVERT, state["prior"])


def _eval_sql(
    spark: Any, sql: str, args: tuple, input_types: list[str], return_type: str, native: bool = False
):
    from pyspark.sql import Row
    from pyspark.sql.types import StructType

    df = spark.createDataFrame([Row(**_row_values(args))], StructType(_param_fields(input_types)))
    with _comet_native_rows(spark, native):
        value = df.selectExpr(f"({sql}) AS result").collect()[0][0]
    if not _representable(value, return_type):
        # Production always casts the result; verify did not. A value that
        # fits the tolerance but not the return type passed verification and
        # then died on the first real query's ANSI overflow.
        raise ArithmeticError(f"sql result {value!r} does not fit {return_type}")
    return value


def _eval_sql_batch(
    spark: Any, sql: str, rows: list[tuple], input_types: list[str], native: bool = False
) -> list:
    """Evaluate ``sql`` on many argument tuples in one job. Order matches ``rows``.

    ``native`` keeps the projection directly above the scan: a global sort
    puts a Spark exchange under it, and a Rust UDF above a Spark operator is
    a query error. Rows are matched back by id either way.
    """
    from pyspark.sql import Row
    from pyspark.sql.types import IntegerType, StructField, StructType

    fields = [StructField("_batch_id", IntegerType(), False)] + _param_fields(input_types)
    data = []
    for idx, args in enumerate(rows):
        values = {"_batch_id": idx}
        values.update(_row_values(args))
        data.append(Row(**values))
    df = spark.createDataFrame(data, StructType(fields))
    if not native:
        df = df.orderBy("_batch_id")
    with _comet_native_rows(spark, native):
        collected = df.selectExpr("_batch_id", f"({sql}) AS result").collect()
    by_id = {int(record[0]): record[1] for record in collected}
    return [by_id[i] for i in range(len(rows))]


def smoke_test_reconstruction(
    spark: Any,
    *,
    source_text: str,
    captures: Optional[dict[str, Any]],
    input_types: list[str],
    return_type: str,
    func: Any = None,
) -> Optional[str]:
    """Force the cached rewrite through the real TranspiledPythonUDF path on an
    empty DataFrame; return an error string if plan analysis fails, else None.

    hypothesis_check evaluates candidate SQL with the params bound as ordinary
    columns, where outer references inside higher-order function lambdas
    resolve fine -- but the TranspiledPythonUDF placeholder substitution does
    not descend into lambda bodies, so a verified rewrite can still fail
    analysis (UNRESOLVED_COLUMN _udf_param_N) when it reaches a real query,
    taking the whole query down instead of falling back. The catalog row must
    already be written. A provisional row is visible only on this thread, so
    other queries keep the Python UDF until the caller promotes it.

    Only a genuine hook decline (nothing reconstructed, e.g. Spark's own
    return-type gate skipping the UDF before the hook) returns None -- no
    opinion. Harness problems that would leave the probe silently disarmed
    (confs that cannot be set, exceptions while constructing the UDF) return
    an error: the row must not be promoted reconstruction-unchecked, or the
    first real query pays for it.
    """
    from ai_udf_transpile.catalog import expose_provisional

    with expose_provisional():
        return _smoke_test_reconstruction(
            spark,
            source_text=source_text,
            captures=captures,
            input_types=input_types,
            return_type=return_type,
            func=func,
        )


def _smoke_test_reconstruction(
    spark: Any,
    *,
    source_text: str,
    captures: Optional[dict[str, Any]],
    input_types: list[str],
    return_type: str,
    func: Any = None,
) -> Optional[str]:
    if spark is None:
        return None
    try:
        try:
            spark.conf.set("spark.sql.experimental.optimizer.transpilePyUDFs", "true")
            current = spark.conf.get("spark.sql.experimental.optimizer.pyTranspilers", "") or ""
            if "ai" not in current.split(","):
                spark.conf.set(
                    "spark.sql.experimental.optimizer.pyTranspilers",
                    ",".join([x for x in current.split(",") if x] + ["ai"]),
                )
        except Exception:
            logger.debug("could not enable transpile confs for smoke test", exc_info=True)
            return "smoke test could not arm the transpile confs"
        from pyspark.sql.types import StructField, StructType
        from pyspark.sql.udf import UserDefinedFunction

        from ai_udf_transpile.transpiler import register_transpiler

        register_transpiler()
        f = func
        if f is None:
            f = load_python_udf(source_text, captures)
        fields = [StructField(f"arg{i}", _spark_type(t), True) for i, t in enumerate(input_types)]
        df = spark.createDataFrame([], StructType(fields))
        udf = UserDefinedFunction(f, _spark_type(return_type))
        transpiled = list(getattr(udf, "transpiled", None) or [])
        if not transpiled:
            return None  # hook declined / nothing reconstructed: no opinion
    except Exception as exc:
        logger.debug("smoke test harness could not construct the UDF", exc_info=True)
        return f"smoke test harness failed: {exc}"
    try:
        df.select(udf(*[f"arg{i}" for i in range(len(input_types))])).schema
    except Exception as exc:
        return f"verified rewrite fails plan analysis when reconstructed: {exc}"
    return None


def _java_udf_expr(
    spark: Any,
    result: Any,
    input_types: list[str],
    return_type: str,
) -> Optional[str]:
    """Compile (if needed) and register the Java UDF under a content-addressed name."""
    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    binary = getattr(result, "binary", None)
    class_name = getattr(result, "class_name", None)
    java_source = getattr(result, "java_source", None)
    janino: Optional[bool] = None
    if not binary and java_source:
        compiled = compile_java(spark, java_source, class_name)
        binary, class_name, janino = compiled.jar_bytes, compiled.class_name, compiled.janino
        result.binary = binary
        result.class_name = class_name
    if not (binary and class_name):
        return None
    fname = verify_function_name("ai_udf_verify", binary)
    register_java_udf(spark, fname, class_name, binary, _spark_type(return_type), janino=janino)
    args = ", ".join(f"_udf_param_{i}" for i in range(len(input_types)))
    return f"{fname}({args})"


def _rust_udf_expr(
    spark: Any,
    result: Any,
    input_types: list[str],
    return_type: str,
) -> str:
    """Build (if needed) and register the Rust UDF; the SQL that calls it.

    Blocks on cargo: this is the worker, not UDF definition. The build lands
    in the same content-addressed cache a later lookup serves from.
    """
    from ai_udf_transpile import rustc

    reason = rustc.comet_unavailable(spark)
    if reason:
        raise VerifyFailed(f"a rust rewrite can only be verified under Comet: {reason}")
    compiled = rustc.compile_rust(spark, result.rust_source or "", input_types, return_type)
    rustc.register_rust_udf(spark, compiled, input_types, return_type)
    result.entry = compiled.name
    return rustc.call_expr(compiled.name, input_types)


_INT_RETURN_RANGES = {
    "int": (-(2**31), 2**31 - 1),
    "bigint": (-(2**63), 2**63 - 1),
}


def _representable(value: Any, return_type: str) -> bool:
    """Whether a python result fits the declared Spark return type.

    A python UDF whose result does not fit its declared return type fails at
    execution too (e.g. ``round(1e37)`` returns a 38-digit int that no bigint
    column can hold), so the value is out of contract: the sql side may do
    anything there, including raise ANSI overflow on the final cast.
    """
    rng = _INT_RETURN_RANGES.get(return_type)
    if rng is not None and isinstance(value, int) and not isinstance(value, bool):
        return rng[0] <= value <= rng[1]
    return True


def _within_tolerance(py_value: Any, sql_value: Any, tolerance: float) -> bool:
    """Exact match, or absolute numeric distance within ``tolerance``.

    Booleans stay exact (``True == 1`` is already Python's ``==``). Strings,
    lists, and timestamps are exact: a tolerance does not fuzzy-match them.
    """
    if py_value == sql_value:
        return True
    if tolerance <= 0:
        return False
    if isinstance(py_value, bool) or isinstance(sql_value, bool):
        return False
    if not isinstance(py_value, (int, float)) or not isinstance(sql_value, (int, float)):
        return False
    try:
        return abs(float(py_value) - float(sql_value)) <= tolerance
    except (TypeError, ValueError, OverflowError):
        return False


def hypothesis_check(
    *,
    source_text: str,
    captures: dict[str, Any],
    result: Any,
    input_types: list[str],
    return_type: str,
    spark: Any,
    max_examples: int = 20,
    func: Any = None,
    samples: Optional[list] = None,
    tolerance: float = 0.0,
    party_on_strings: bool = False,
) -> tuple[bool, Optional[str]]:
    """Return (ok, error). Analysis failure / mismatch → (False, msg). Never raises to the worker.

    ``party_on_strings`` invents string values. Off by default: a declared
    string may be a date, JSON, or a number, and invented text is not
    evidence. Real sampled rows still run either way.
    """
    rejected = rejected_rewrite(result)
    if rejected:
        return False, rejected
    try:
        python_fn = load_python_udf(source_text, captures, func)
    except (NameError, ImportError, SyntaxError, VerifyFailed) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"sandbox exec failed: {exc}"

    kind = getattr(result, "kind", "catalyst")
    sql = getattr(result, "sql", None)
    if kind == "catalyst" and sql:
        eval_expr = sql
    elif kind == "java_udf":
        if spark is None:
            return False, "java_udf verify requires Spark"
        try:
            eval_expr = _java_udf_expr(spark, result, input_types, return_type)
        except Exception as exc:
            return False, f"java_udf verify setup failed: {exc}"
        if eval_expr is None:
            return False, "java_udf result has no class/binary to verify"
    elif kind == "rust":
        if spark is None:
            return False, "rust verify requires Spark"
        try:
            eval_expr = _rust_udf_expr(spark, result, input_types, return_type)
        except Exception as exc:
            return False, f"rust verify setup failed: {exc}"
    else:
        return False, f"verify of target_kind={kind} is not implemented"

    if spark is None:
        return False, "hypothesis_check requires a SparkSession"
    # A Rust UDF is evaluated by Comet or not at all.
    native = kind == "rust"

    import time

    from hypothesis import HealthCheck, example, given, settings
    from hypothesis import strategies as st

    try:
        strategies = [_strategy_for(t, party_on_strings=party_on_strings) for t in input_types]
    except VerifyFailed as exc:
        # e.g. register_impl input_types with an unknown simple string;
        # the docstring says never raises.
        return False, f"{type(exc).__name__}: {exc}"
    mismatch: list[str] = []
    py_successes = [0]  # a rewrite is only verified if python succeeds somewhere
    # (args, python value, python exception). SQL runs later, ~100 rows per job.
    queued: list[tuple[tuple, Any, Optional[BaseException]]] = []
    example_count = 0
    sql_batches = 0
    started = time.perf_counter()

    def _judge(
        args: tuple,
        py_value: Any,
        py_exc: Optional[BaseException],
        sql_value: Any,
        sql_exc: Optional[BaseException],
    ) -> None:
        if sql_exc is not None:
            if py_exc is None:
                mismatch.append(f"sql raised {sql_exc!r} on {args!r} but python returned {py_value!r}")
            return
        if py_exc is not None:
            # Spark's own hypothesis policy: python raise + sql value is allowed.
            return
        if not _within_tolerance(py_value, sql_value, tolerance):
            mismatch.append(f"mismatch on {args!r}: python={py_value!r} sql={sql_value!r}")

    def _flush(batch: list[tuple[tuple, Any, Optional[BaseException]]]) -> None:
        nonlocal sql_batches
        if not batch:
            return
        sql_batches += 1
        args_list = [item[0] for item in batch]
        try:
            sql_values = _eval_sql_batch(spark, eval_expr, args_list, input_types, native)
        except Exception:
            logger.debug("sql batch of %d failed; retrying per row", len(batch), exc_info=True)
            for args, py_value, py_exc in batch:
                try:
                    sql_value = _eval_sql(spark, eval_expr, args, input_types, return_type, native)
                    _judge(args, py_value, py_exc, sql_value, None)
                except Exception as exc:
                    _judge(args, py_value, py_exc, _SENTINEL_RAISED, exc)
            return
        for (args, py_value, py_exc), sql_value in zip(batch, sql_values):
            _judge(args, py_value, py_exc, sql_value, None)

    def _run(args: tuple) -> None:
        nonlocal example_count
        example_count += 1
        py_exc: Optional[BaseException] = None
        try:
            py_value = python_fn(*args)
        except (NameError, ImportError) as exc:
            # Sandbox couldn't resolve a name/import — not a per-row Python error.
            mismatch.append(f"{type(exc).__name__}: {exc} on {args!r}")
            return
        except Exception as exc:  # Python UDF raised
            py_exc = exc
            py_value = _SENTINEL_RAISED
        else:
            py_successes[0] += 1
            if not _representable(py_value, return_type):
                # Out of contract: executing the python UDF with the declared
                # return type would fail as well, so the sql side may do
                # anything (ANSI overflow on cast(bround(1e37) as bigint)).
                return
        queued.append((args, py_value, py_exc))

    def _explicit_examples() -> list[tuple]:
        """Real sampled rows plus built-in interesting strings.

        Random text rarely exercises parsers (a JSON UDF raises on random
        strings and so does the SQL side -- a vacuous match), so real sampled
        values always run as @example. Structured string examples run only
        when party_on_strings is set.
        """
        from ai_udf_transpile.sampling import (
            BUILTIN_ARRAY_EXAMPLES,
            BUILTIN_BOOL_EXAMPLES,
            BUILTIN_DOUBLE_EXAMPLES,
            BUILTIN_INT_EXAMPLES,
            BUILTIN_MAP_EXAMPLES,
            BUILTIN_NESTED_ARRAY_EXAMPLES,
            BUILTIN_STRING_EXAMPLES,
            BUILTIN_TIMESTAMP_EXAMPLES,
            CROSS_STRING_EXAMPLES,
        )

        examples: list[tuple] = []
        arity = len(input_types)
        for sample in samples or []:
            try:
                values = tuple(sample)
            except TypeError:
                continue
            if len(values) == arity:
                examples.append(values)
        for i, spark_type in enumerate(input_types):
            stype = spark_type.strip().lower()
            if stype == "string" and party_on_strings:
                for text in BUILTIN_STRING_EXAMPLES:
                    examples.append(tuple(text if j == i else None for j in range(arity)))
            elif stype in {"bigint", "long", "int", "integer", "smallint", "tinyint"}:
                for n in BUILTIN_INT_EXAMPLES:
                    examples.append(tuple(n if j == i else None for j in range(arity)))
            elif stype in {"double", "float"}:
                for n in BUILTIN_DOUBLE_EXAMPLES:
                    examples.append(tuple(n if j == i else None for j in range(arity)))
            elif stype.startswith("map<"):
                for m in BUILTIN_MAP_EXAMPLES:
                    examples.append(tuple(dict(m) if j == i else None for j in range(arity)))
            elif stype.startswith("array<array<"):
                for a in BUILTIN_NESTED_ARRAY_EXAMPLES:
                    examples.append(tuple([list(x) for x in a] if j == i else None for j in range(arity)))
            elif stype.startswith("array<"):
                for a in BUILTIN_ARRAY_EXAMPLES:
                    examples.append(tuple(list(a) if j == i else None for j in range(arity)))
            elif stype in {"boolean", "bool"}:
                for b in BUILTIN_BOOL_EXAMPLES:
                    examples.append(tuple(b if j == i else None for j in range(arity)))
            elif stype in {"timestamp", "timestamp_ntz"}:
                for ts in BUILTIN_TIMESTAMP_EXAMPLES:
                    examples.append(tuple(ts if j == i else None for j in range(arity)))
        if arity >= 2:
            # One-param-at-a-time built-ins never combine interesting values
            # across params (e.g. a valid timestamp start AND a 'nan'
            # duration), which is where coercion guards get exercised -- a
            # rewrite missing an isnan guard passes vacuously without them.
            # The product explodes combinatorially (10 strings x 10 strings x
            # 5 arrays x 5 arrays = 2500 for a 4-param UDF), so enumerate
            # diagonally (by index sum): every param's values appear within
            # the first few combos instead of being truncated lexicographically.
            import itertools

            per_param: list[list] = []
            for spark_type in input_types:
                stype = spark_type.strip().lower()
                if stype == "string":
                    per_param.append(list(CROSS_STRING_EXAMPLES) if party_on_strings else [None])
                elif stype in {"bigint", "long", "int", "integer", "smallint", "tinyint"}:
                    per_param.append(list(BUILTIN_INT_EXAMPLES))
                elif stype in {"double", "float"}:
                    per_param.append(list(BUILTIN_DOUBLE_EXAMPLES))
                elif stype.startswith("map<"):
                    per_param.append([dict(m) for m in BUILTIN_MAP_EXAMPLES])
                elif stype.startswith("array<array<"):
                    per_param.append([[list(x) for x in a] for a in BUILTIN_NESTED_ARRAY_EXAMPLES])
                elif stype.startswith("array<"):
                    per_param.append([list(a) for a in BUILTIN_ARRAY_EXAMPLES])
                elif stype in {"boolean", "bool"}:
                    per_param.append(list(BUILTIN_BOOL_EXAMPLES))
                elif stype in {"timestamp", "timestamp_ntz"}:
                    per_param.append(list(BUILTIN_TIMESTAMP_EXAMPLES))
                else:
                    per_param.append([None])
            index_combos = sorted(
                itertools.product(*(range(len(p)) for p in per_param)),
                key=lambda c: (sum(c), c),
            )
            for combo in index_combos:
                examples.append(tuple(per_param[i][idx] for i, idx in enumerate(combo)))
        return examples[:160]

    check = _run
    for ex in _explicit_examples():
        check = example(ex)(check)  # inside @given: documented ordering
    check = given(st.tuples(*strategies) if strategies else st.just(()))(check)
    check = settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        database=None,
    )(check)

    def _log_timing() -> None:
        elapsed = time.perf_counter() - started
        logger.info(
            "hypothesis_check examples=%d sql_rows=%d batches=%d seconds=%.2f",
            example_count,
            len(queued),
            sql_batches,
            elapsed,
        )

    try:
        check()
    except Exception as exc:
        _log_timing()
        return False, f"hypothesis failed: {exc}"
    for offset in range(0, len(queued), SQL_BATCH_ROWS):
        if mismatch:
            break
        _flush(queued[offset : offset + SQL_BATCH_ROWS])
    _log_timing()
    if mismatch:
        return False, mismatch[0]
    if py_successes[0] == 0:
        # python raised on EVERY example (e.g. a boto3/KMS UDF in an
        # environment without credentials): under "python raise + sql value
        # is allowed" any rewrite would pass vacuously. No signal -> fail
        # closed.
        return False, "python raised on every example; no signal to verify against"
    return True, None
