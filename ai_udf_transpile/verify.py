# SPDX-License-Identifier: Apache-2.0
"""Hypothesis differential check plus an import-aware exec sandbox."""

from __future__ import annotations

import ast
import contextlib
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
_RUST_TYPE_ITEM = re.compile(r"\b(struct|enum|union|mod|type)\s+([A-Za-z_]\w*)")
# `static`/`const` items -- not `const fn`, not `'static`, not `*const T`,
# not a `<const N: usize>` generic parameter (checked by the caller).
_RUST_STATIC = re.compile(r"(?<![\w'*])(?:static|const)\b(?!\s+(?:fn|unsafe|async|extern)\b)")
_RUST_USE = re.compile(r"\buse\b[^;]*;")
_RUST_AS = re.compile(r"\bas\s+([A-Za-z_]\w*)")
_RUST_TRAIT_IMPL_HEADER = re.compile(r"\bfor\b(?!\s*<)")
# What carries a call without naming a function: `trait` items, an erased
# closure or value (`dyn`), and `std::any` (a fn pointer behind a downcast).
# Trait impls (`impl X for Y`) are found by `_rust_recursion` from each
# `impl` header, since a header can hold `;` (`impl From<[i64; 2]> for D`).
_RUST_HIDDEN_CALL = re.compile(
    r"\btrait\s+[A-Za-z_]"
    r"|\bdyn\b"
    r"|\b(?:std|core|alloc)\s*::\s*(?:any\b|\{[^;]*\bany\b)"
)
_RUST_FN_TYPE = re.compile(r"\bfn\s*\(|\bFn(?:Mut|Once)?\b")
# A field whose type is built only from these is a std value: a `.name(..)`
# call on `self.field` reaches std's method, never one of the rewrite's own.
# A local type, a type parameter, or anything else leaves the call an edge.
_RUST_STD_FIELD_WORDS = frozenset(
    "i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str "
    "String Vec VecDeque HashMap HashSet BTreeMap BTreeSet BinaryHeap Option Box Rc "
    "Cell RefCell Reverse Ordering Peekable Chars CharIndices Bytes Rev Range "
    "std core alloc collections iter string vec cell rc cmp mut pub".split()
)
# Real row functions are a few KB; the scan below is quadratic in the worst
# case, so a bound keeps a pathological reply from stalling the worker.
_RUST_MAX_SOURCE = 20_000


_RUST_BRACES = re.compile(r"[{}]")
_RUST_HEADER_PUNCT = re.compile(r"[()\[\]<>{;]")


def _rust_block_end(scanned: str, start: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``start`` (or the text's end)."""
    depth = 0
    for m in _RUST_BRACES.finditer(scanned, start):
        depth += 1 if m.group(0) == "{" else -1
        if depth == 0:
            return m.start()
    return len(scanned)


def _rust_item_stop(scanned: str, pos: int) -> int:
    """The ``{`` opening, or ``;`` ending, the item whose header starts at ``pos``.

    Generic arguments are skipped: ``;`` inside ``(..)``/``[..]``/``<..>`` is an
    array length (``impl From<[i64; 2]>``), and a ``{..}`` inside ``<..>`` is a
    const-generic argument (``impl W<{ 1 + 1 }>``), not the body. ``->`` and
    ``=>`` close nothing, and ``<``/``>`` inside ``(..)``/``[..]`` are operators
    (``[u8; 1 << 2]``), not generics.
    """
    depth = angle = 0
    n = len(scanned)
    hit = _RUST_HEADER_PUNCT.search(scanned, pos)
    while hit:
        i, ch = hit.start(), hit.group(0)
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "<" and depth <= 0:  # inside (..)/[..] it may be `1 << 2`
            angle += 1
        elif ch == ">" and depth <= 0 and (i == 0 or scanned[i - 1] not in "-="):
            angle -= 1
        elif ch == "{":
            if depth > 0 or angle > 0:
                hit = _RUST_HEADER_PUNCT.search(scanned, _rust_block_end(scanned, i) + 1)
                continue
            return i
        elif ch == ";" and depth <= 0 and angle <= 0:
            return i
        hit = _RUST_HEADER_PUNCT.search(scanned, i + 1)
    return n


def _rust_item_open(scanned: str, pos: int) -> Optional[int]:
    """The ``{`` opening the item whose header starts at ``pos``; None at a ``;`` first."""
    i = _rust_item_stop(scanned, pos)
    return i if i < len(scanned) and scanned[i] == "{" else None


def _rust_item_end(scanned: str, pos: int) -> int:
    """Where the item whose header starts at ``pos`` ends: its ``;`` or closing ``}``."""
    i = _rust_item_stop(scanned, pos)
    return _rust_block_end(scanned, i) if i < len(scanned) and scanned[i] == "{" else i


def _rust_stmt_end(scanned: str, pos: int) -> int:
    """The ``;`` ending a ``static``/``const`` item (initializer expressions included)."""
    depth = 0
    for i in range(pos, len(scanned)):
        ch = scanned[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return i
        elif ch == ";" and depth == 0:
            return i
    return len(scanned)


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


def _rust_split_top(text: str) -> list[str]:
    """``text`` split at commas outside any bracket."""
    parts, depth, last = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>" and not (ch == ">" and i and text[i - 1] in "-="):
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


def _rust_std_fields(scanned: str, type_name: str) -> set[str]:
    """Fields of the one struct ``type_name`` whose types are std values only.

    A word the rewrite defines itself (``struct Range``, ``use S as Vec``)
    is never std, whatever its name.
    """
    local = {m.group(2) for m in _RUST_TYPE_ITEM.finditer(scanned)}
    for use in _RUST_USE.finditer(scanned):
        local |= set(_RUST_AS.findall(use.group(0)))
    std_words = _RUST_STD_FIELD_WORDS - local
    defs = [m for m in _RUST_TYPE_ITEM.finditer(scanned) if m.group(2) == type_name]
    if len(defs) != 1 or defs[0].group(1) != "struct":
        return set()
    stop = _rust_item_stop(scanned, defs[0].end())
    head = scanned[defs[0].end() : stop]
    params = head.split("(")[0]
    if "<" in params and re.sub(r"'\w+|[\s<>,:+]", "", params):
        return set()  # a type parameter: a field may be any type (lifetimes are fine)
    if stop < len(scanned) and scanned[stop] == "{":
        fields = []
        for part in _rust_split_top(scanned[stop + 1 : _rust_block_end(scanned, stop)]):
            m = re.match(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?([A-Za-z_]\w*)\s*:(?!:)(.*)", part, re.S)
            if not m:
                return set()
            fields.append((m.group(1), m.group(2)))
    else:
        open_ = head.find("(")
        if open_ < 0:
            return set()
        close = _rust_stmt_end(scanned, defs[0].end() + open_ + 1)
        inner = scanned[defs[0].end() + open_ + 1 : close]
        fields = [(str(i), part) for i, part in enumerate(_rust_split_top(inner))]
    return {
        name
        for name, typ in fields
        if all(w in std_words for w in re.findall(r"[A-Za-z_]\w*", re.sub(r"'\w+", "", typ)))
    }


_RUST_CHAIN_CALL = re.compile(r"\s*\.\s*[A-Za-z_]\w*\s*(?:::\s*<([^;{}()|]*)>\s*)?\(")


def _rust_std_receiver(before: str, fields: set[str]) -> bool:
    """``before`` ends in ``self.<std field>`` plus std calls that keep it std.

    Each call in the chain must take no identifiers (``.iter()``, ``.get(0)``,
    ``.split(',')`` with the literal blanked): a closure, a variable or a fn
    item could hand a user type in (``.and(Some(s)).unwrap().go()``), and a
    turbofish may only name std types.
    """
    found = list(re.finditer(r"(?<![\w.])self\s*\.\s*(\w+)", before))
    if not found or found[-1].group(1) not in fields:
        return False
    tail, i = before[found[-1].end() :], 0
    while True:
        call = _RUST_CHAIN_CALL.match(tail, i)
        if not call:
            break
        if call.group(1) and not all(
            w in _RUST_STD_FIELD_WORDS for w in re.findall(r"[A-Za-z_]\w*", call.group(1))
        ):
            return False
        depth, k = 1, call.end()
        while k < len(tail) and depth:
            depth += {"(": 1, ")": -1}.get(tail[k], 0)
            k += 1
        if depth or re.search(r"[A-Za-z_|]", tail[call.end() : k - 1]):
            return False
        i = k
    return not tail[i:].strip()


def _rust_alias_resolver(scanned: str):
    """A function mapping a ``use .. as`` alias (chains included) to the name it renames."""
    aliases = {}
    for use in _RUST_USE.finditer(scanned):
        for target, alias in re.findall(r"([A-Za-z_]\w*)\s+as\s+([A-Za-z_]\w*)", use.group(0)):
            aliases[alias] = target

    def resolve(name: str) -> str:
        for _ in range(len(aliases) + 1):
            if name not in aliases or aliases[name] == name:
                return name
            name = aliases[name]
        return name

    return resolve


_RUST_PRIM_SIZE = {
    "i8": 1, "u8": 1, "bool": 1, "i16": 2, "u16": 2, "i32": 4, "u32": 4, "f32": 4, "char": 4,
    "i64": 8, "u64": 8, "f64": 8, "isize": 8, "usize": 8, "i128": 16, "u128": 16,
}  # fmt: skip
# Anything else (String, Vec, Box, Option<..> of those, references) is a few
# words on the stack; its contents live on the heap.
_RUST_OTHER_SIZE = 24
# What one rewrite may put on the stack, all arrays and local values together.
# Executor task threads often get 1 MB; a lookup table is a few KB.
_RUST_STACK_BUDGET = 256 * 1024
_RUST_INFINITE = float("inf")


def _rust_const_value(expr: str, consts: dict[str, str], depth: int = 0) -> float:
    """An array length as a number: literals, + - * / % << >> ( ), and consts."""
    if depth > 8:
        return _RUST_INFINITE
    text = re.sub(r"(?<=\d)_(?=\d)", "", expr)
    text = re.sub(r"\b(\d+)(?:usize|u\d+|i\d+|isize)\b", r"\1", text)

    def const(m: re.Match) -> str:
        name = m.group(0)
        if name in consts:
            value = _rust_const_value(consts[name], consts, depth + 1)
            return "1e30" if value == _RUST_INFINITE else str(int(value))
        return "@"  # unknown name: not sizable

    text = re.sub(r"\b[A-Za-z_]\w*\b", const, text)
    if "@" in text:
        return _RUST_INFINITE
    return _rust_int_expr(text)


_RUST_INT_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
}
_RUST_INT_LIMIT = 1 << 64


def _rust_int_expr(text: str) -> float:
    """An integer expression of literals and + - * / % << >> ( ), or infinity.

    No ``eval``: ``9**9**9`` or ``1 << 10**9`` would stall the worker. Any
    value past 2**64 (or a shift past 64 bits) is infinite, i.e. not sizable.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return _RUST_INFINITE

    def value(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            inner = value(node.operand)
            result = -inner if isinstance(node.op, ast.USub) else inner
        elif isinstance(node, ast.BinOp) and type(node.op) in _RUST_INT_OPS:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= right < 64:
                raise OverflowError
            if isinstance(node.op, (ast.FloorDiv, ast.Div, ast.Mod)) and right == 0:
                raise ZeroDivisionError
            result = _RUST_INT_OPS[type(node.op)](left, right)
        else:
            raise ValueError
        if abs(result) >= _RUST_INT_LIMIT:
            raise OverflowError
        return result

    try:
        result = value(tree.body)
    except (ValueError, OverflowError, ZeroDivisionError, RecursionError):
        return _RUST_INFINITE
    return float(result) if result >= 0 else _RUST_INFINITE


def _rust_stack_estimate(scanned: str, types: dict[str, str], resolve) -> Optional[str]:
    """Why the source may need more stack than ``_RUST_STACK_BUDGET``, if it may.

    Counts every array type or repeat expression (``[T; N]``, ``[x; N]``,
    nested ones multiplied) and every mention of a local type in a function
    body at its size. Over-counts on purpose: a type definition and its uses
    both count.
    """
    consts = {
        m.group(1): m.group(2)
        for m in re.finditer(r"\bconst\s+([A-Za-z_]\w*)\s*:[^=;]*=\s*([^;]*);", scanned)
    }
    sizes: dict[str, float] = {}

    def type_size(text: str, seen: frozenset = frozenset()) -> float:
        text = text.strip()
        if not text:
            return 0
        if text.startswith(("&", "*", "fn")):
            return 8
        if text.startswith("["):
            return bracket_size(text[1 : _rust_close(text, 0)], seen)
        if text.startswith("("):
            return sum(type_size(p, seen) for p in _rust_split_top(text[1 : _rust_close(text, 0)]))
        m = re.match(r"(?:[A-Za-z_]\w*\s*::\s*)*([A-Za-z_]\w*)", text)
        name = resolve(m.group(1)) if m else ""
        if name in _RUST_PRIM_SIZE:
            return _RUST_PRIM_SIZE[name]
        if name in types and name not in seen:
            return local_size(name, seen | {name})
        return _RUST_OTHER_SIZE

    def local_size(name: str, seen: frozenset) -> float:
        if name not in sizes:
            body = types[name]
            fields = [
                re.sub(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?[A-Za-z_]\w*\s*:(?!:)", "", f, count=1)
                for f in _rust_split_top(_rust_inner(body))
            ]
            # A struct is the sum of its fields; an enum is at least its largest
            # variant, which the sum bounds from above.
            sizes[name] = sum(type_size(f, seen) for f in fields) or _RUST_OTHER_SIZE
        return sizes[name]

    def bracket_size(inner: str, seen: frozenset = frozenset()) -> float:
        parts = _rust_split_semicolon(inner)
        if len(parts) != 2:
            return 0  # a slice type, an index, or an array literal: no repeat
        elem, length = parts
        n = _rust_const_value(length, consts)
        if n == _RUST_INFINITE:
            return _RUST_INFINITE
        nested = [bracket_size(elem[i + 1 : _rust_close(elem, i)], seen) for i in _rust_top_brackets(elem)]
        if nested:
            each = max(nested)
        else:
            words = [resolve(w) for w in re.findall(r"[A-Za-z_]\w*", elem)]
            local = [w for w in words if w in types]
            prim = [_RUST_PRIM_SIZE[w] for w in words if w in _RUST_PRIM_SIZE]
            each = max([local_size(w, seen | {w}) for w in local] + prim or [8])
        return n * each

    def occurrences(text: str):
        """Every array to count: ``vec![x; n]`` is heap, but an array inside it is not."""
        for i in _rust_top_brackets(text):
            inner = text[i + 1 : _rust_close(text, i)]
            if re.search(r"\bvec\s*!\s*$", text[:i]):
                yield from occurrences(inner)
            else:
                yield inner

    total = 0.0
    for inner in occurrences(scanned):
        total += bracket_size(inner)
        if total == _RUST_INFINITE:
            return (
                "rust rewrite has an array whose length this check cannot size (use "
                "integer literals or consts): a large array on the stack overflows it"
            )
    for name, _, name_end, end in _rust_fn_bodies(scanned):
        for word in re.findall(r"[A-Za-z_]\w*", scanned[name_end:end]):
            if resolve(word) in types:
                total += local_size(resolve(word), frozenset({resolve(word)}))
    if total > _RUST_STACK_BUDGET:
        return (
            f"rust rewrite may put about {int(total) // 1024} KB on the stack (arrays and "
            f"local values; at most {_RUST_STACK_BUDGET // 1024} KB): a stack overflow aborts "
            "the process; put large buffers in a Vec"
        )
    return None


def _rust_close(text: str, i: int) -> int:
    """Index of the bracket closing the ``[``/``(`` at ``i`` (or the end)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] in "([{":
            depth += 1
        elif text[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j
    return len(text)


def _rust_top_brackets(text: str) -> list[int]:
    """Start of every outermost ``[..]`` in ``text`` (nested ones are their contents)."""
    out, i = [], 0
    while True:
        i = text.find("[", i)
        if i < 0:
            return out
        out.append(i)
        i = _rust_close(text, i) + 1


def _rust_split_semicolon(text: str) -> list[str]:
    """``text`` split at ``;`` outside any bracket."""
    parts, depth, last = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return parts


def _rust_inner(body: str) -> str:
    """The text inside the first ``{..}`` or ``(..)`` of a type definition."""
    for i, ch in enumerate(body):
        if ch in "{(":
            return body[i + 1 : _rust_close(body, i)]
        if ch == ";":
            break
    return body


def _rust_impl_target(header: str) -> Optional[str]:
    """``S`` from an inherent ``impl<..> S<..>`` header (after the keyword)."""
    text = header.lstrip()
    if text.startswith("<"):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "<"
            depth -= ch == ">" and (i == 0 or text[i - 1] not in "-=")
            if depth == 0:
                text = text[i + 1 :]
                break
    m = re.match(r"\s*([A-Za-z_]\w*)", text)
    return m.group(1) if m else None


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
    function only through ``crate``/``self``/``super``/``Self``/a local type,
    module or ``use .. as`` alias; a method also through ``.name``, except
    ``self.field.name(..)`` on a field whose type is std values only
    (``self.items.push(c)`` inside ``fn push``).

    Calls that name no function are refused outright: trait impls (an
    operator, ``{}``, ``.into()`` reach an impl's method by trait), ``trait``,
    ``dyn``, ``std::any``, a type, ``static`` or ``const`` holding a function,
    and a function value outside every body. A call in a ``const``/``static``
    initializer is compile-time evaluation and is allowed.
    Runaway loops are the prompt's to prevent.
    """
    hidden = _RUST_HIDDEN_CALL.search(scanned)
    if hidden:
        return (
            f"rust rewrite uses `{hidden.group(0).split()[0]}`: `trait`, `dyn` and `std::any` "
            "can call a function without naming it, so recursion through them cannot be "
            "ruled out; use plain functions and inherent methods"
        )
    impls = []  # (target, open, end) of every `impl` with a body
    for match in _RUST_IMPL.finditer(scanned):
        stop = _rust_item_stop(scanned, match.end())
        header = scanned[match.end() : stop]
        if _RUST_TRAIT_IMPL_HEADER.search(header):
            return (
                "rust rewrite implements a trait (`impl .. for ..`): an operator, `{}`, "
                "`.into()` or a for loop reaches the impl's method without naming it, so "
                "recursion through it cannot be ruled out; use plain functions and "
                "inherent methods (`impl Type { .. }`)"
            )
        if stop < len(scanned) and scanned[stop] == "{":
            impls.append((_rust_impl_target(header), stop, _rust_block_end(scanned, stop)))
    for match in _RUST_TYPE_ITEM.finditer(scanned):
        if match.group(1) == "mod":
            continue
        if _RUST_FN_TYPE.search(scanned[match.end() : _rust_item_end(scanned, match.end())]):
            return (
                f"rust rewrite defines a type holding a function ({match.group(2)}): a "
                "closure handed such a value can call itself; pass functions as arguments"
            )
    # A recursive type (`struct Node { next: Option<Box<Node>> }`) recurses
    # with no function at all: dropping, cloning or comparing a long chain
    # overflows inside std's glue. `Self` and `use .. as` aliases name it too.
    types = {
        m.group(2): scanned[m.end() : _rust_item_end(scanned, m.end())]
        for m in _RUST_TYPE_ITEM.finditer(scanned)
        if m.group(1) != "mod"
    }
    resolve = _rust_alias_resolver(scanned)
    type_edges = {
        name: {resolve(w) for w in re.findall(r"[A-Za-z_]\w*", text) if resolve(w) in types}
        | ({name} if re.search(r"\bSelf\b", text) else set())
        for name, text in types.items()
    }
    for node in sorted(types):
        seen: set[str] = set()
        stack = list(type_edges[node])
        while stack:
            cur = stack.pop()
            if cur == node:
                return (
                    f"rust rewrite defines a recursive type ({node}): dropping, cloning or "
                    "comparing a deep value recurses and can overflow the stack; use a Vec "
                    "with indices instead"
                )
            if cur not in seen:
                seen.add(cur)
                stack.extend(type_edges.get(cur, ()))
    # No recursion is needed to overflow: one big array on the stack does it
    # (`let mut counts = [0u32; 1 << 20];` is 4 MB, the stack 1-8 MB).
    big = _rust_stack_estimate(scanned, types, resolve)
    if big:
        return big
    statics = []  # (start, end) of every static/const item
    for match in _RUST_STATIC.finditer(scanned):
        before = scanned[: match.start()].rstrip()
        if before.endswith(("<", ",")):  # `<const N: usize>`: a generic parameter
            continue
        end = _rust_stmt_end(scanned, match.end())
        if _RUST_FN_TYPE.search(scanned[match.end() : end]):
            return (
                "rust rewrite has a static or const holding a function: whatever "
                "stores a function there can be called back from anywhere, itself included"
            )
        statics.append((match.start(), end))
    bodies = _rust_fn_bodies(scanned)
    names = {name for name, *_ in bodies}
    if not names:
        return None
    methods = {name for name, start, *_ in bodies if any(a < start < b for _, a, b in impls)}
    roots = {"crate", "self", "super", "Self"} | {m.group(2) for m in _RUST_TYPE_ITEM.finditer(scanned)}
    for use in _RUST_USE.finditer(scanned):
        roots |= set(_RUST_AS.findall(use.group(0)))
    root = "|".join(re.escape(r) for r in sorted(roots))
    std_fields = {t: _rust_std_fields(scanned, t) for t, _, _ in impls if t}

    def patterns(name: str) -> tuple[re.Pattern, Optional[re.Pattern]]:
        """(any-mention pattern, method-call pattern) for ``name``."""
        word = re.escape(name)
        # Bare: not after one `.` (a field or method) or `::` (a path), but
        # after `..` (a range) or one `:` (a field init) it is the value.
        alts = [
            rf"(?<![\w'])(?<![^.]\.)(?<!::){word}\b(?!\s*!)",
            rf"\b(?:{root})(?:\s*::\s*\w+)*\s*::\s*{word}\b",
        ]
        if name not in methods:
            return re.compile("|".join(alts)), None
        alts.append(rf">\s*::\s*{word}\b")  # `<S>::go`, `<S as X>::go`, `S::<T>::go`
        # `.name` reaches a method only as a call (`x.go(..)`, `x.go::<T>(..)`);
        # without one it is a field (`self.consonants` in `fn consonants`).
        return re.compile("|".join(alts)), re.compile(rf"\.\s*{word}\s*(?:\(|::)")

    compiled = {name: patterns(name) for name in names}

    def mentions(text: str, name: str, fields: set[str]) -> bool:
        plain, method = compiled[name]
        if plain.search(text):
            return True
        if method is None:
            return False
        for hit in method.finditer(text):
            if not _rust_std_receiver(text[: hit.start()], fields):
                return True
        return False

    # Outside every body only a static/const initializer or a `use` can hold a
    # function value (type definitions with function types are refused above).
    # A call in an initializer is compile-time evaluation: blank its callee.
    outside = [" "] * len(scanned)
    for start, end in statics:
        text = re.sub(r"\b[A-Za-z_]\w*(?=\s*\()", lambda m: " " * len(m.group(0)), scanned[start:end])
        outside[start:end] = text
    for use in _RUST_USE.finditer(scanned):
        outside[use.start() : use.end()] = use.group(0)
    for _, start, _, end in bodies:
        outside[start : end + 1] = " " * (end + 1 - start)
    outside_text = "".join(outside)
    for name in sorted(names):
        if mentions(outside_text, name, set()):
            return (
                f"rust rewrite uses the function {name} as a value outside a function "
                "body (a static, const, or use): it can then be called from anywhere, "
                "itself included"
            )
    edges: dict[str, set[str]] = {name: set() for name in names}
    for name, fn_start, name_end, end in bodies:
        # From after `fn name`: the declaration itself is no self-edge
        # (a nested `fn`'s text is part of the body and counts).
        text = scanned[name_end:end]
        owner = [t for t, a, b in impls if a < fn_start < b]
        fields = std_fields.get(owner[-1], set()) if owner and owner[-1] else set()
        for other in names:
            if mentions(text, other, fields):
                edges[name].add(other)
    for node in sorted(names):
        seen: set[str] = set()
        stack = list(edges[node])
        while stack:
            cur = stack.pop()
            if cur == node:
                return (
                    f"rust rewrite is recursive ({node}): a stack overflow aborts the "
                    "process instead of failing the query (the check is by name: a local, "
                    "a parameter, or a method named like a std method it calls on anything "
                    "but a std-typed field of self counts too: rename it)"
                )
            if cur not in seen:
                seen.add(cur)
                stack.extend(edges.get(cur, ()))
    return None


@functools.lru_cache(maxsize=256)
def _rust_rejected(rust: str) -> Optional[str]:
    """Scan verdict for one Rust source; memoized, since every lookup re-checks a served row."""
    if len(rust) > _RUST_MAX_SOURCE:
        return f"rust rewrite is too long ({len(rust)} chars, at most {_RUST_MAX_SOURCE}): keep it small"
    scanned = _rust_for_forbidden_scan(rust)
    if not scanned.isascii():
        # Comments and literals are blank by now: this is an identifier, which
        # the ASCII-only name patterns below would not see.
        return "rust rewrite uses a non-ASCII identifier: use ASCII names"
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


class VerifyQueryTimeout(RuntimeError):
    """A verification query ran past ``verifyQueryTimeoutSeconds`` and was cancelled."""


@contextlib.contextmanager
def _query_deadline(spark: Any):
    """Cancel the Spark jobs started inside after ``verifyQueryTimeoutSeconds``.

    A candidate whose loop never ends (Java or Rust; SQL cannot loop) would
    otherwise hold verification, and the worker, forever. Cancelling the job
    group fails the driver's ``collect()`` at once. The task thread itself
    cannot be stopped mid-row -- a JVM loop ignores the interrupt until it
    returns, native code never sees it -- so in local mode it keeps one core
    busy until the process exits. That costs a core; a hang costs the worker.
    """
    import threading
    import uuid

    from ai_udf_transpile import conf

    seconds = conf.get_float(conf.VERIFY_QUERY_TIMEOUT, spark, 300.0)
    sc = getattr(spark, "sparkContext", None)
    if seconds <= 0 or sc is None:
        yield
        return
    keys = ("spark.jobGroup.id", "spark.job.description", "spark.job.interruptOnCancel")
    prior = {k: sc.getLocalProperty(k) for k in keys}
    group = f"ai-udf-verify-{uuid.uuid4().hex}"
    fired = threading.Event()

    def cancel() -> None:
        fired.set()
        sc.cancelJobGroup(group)

    sc.setJobGroup(group, "ai_udf_transpile verification", interruptOnCancel=True)
    timer = threading.Timer(seconds, cancel)
    timer.daemon = True
    timer.start()
    try:
        yield
    except Exception as exc:
        if fired.is_set():
            raise VerifyQueryTimeout(
                f"verification query ran longer than {seconds:g}s and was cancelled "
                "(a loop that does not terminate on this input?)"
            ) from exc
        raise
    finally:
        timer.cancel()
        for key, value in prior.items():
            sc.setLocalProperty(key, value)


def _eval_sql(
    spark: Any, sql: str, args: tuple, input_types: list[str], return_type: str, native: bool = False
):
    from pyspark.sql import Row
    from pyspark.sql.types import StructType

    df = spark.createDataFrame([Row(**_row_values(args))], StructType(_param_fields(input_types)))
    with _comet_native_rows(spark, native), _query_deadline(spark):
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
    if native:
        # One partition: a candidate that never terminates spins one task
        # thread per partition it runs in, and a cancelled native task cannot
        # be stopped -- one core lost per runaway candidate, not all of them.
        df = spark.createDataFrame(spark.sparkContext.parallelize(data, 1), StructType(fields))
    else:
        df = spark.createDataFrame(data, StructType(fields)).orderBy("_batch_id")
    with _comet_native_rows(spark, native), _query_deadline(spark):
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
        except VerifyQueryTimeout as exc:
            # Not a per-row error to retry: each retry would leave one more
            # task thread spinning and wait out the deadline again.
            mismatch.append(str(exc))
            return
        except Exception:
            logger.debug("sql batch of %d failed; retrying per row", len(batch), exc_info=True)
            for args, py_value, py_exc in batch:
                try:
                    sql_value = _eval_sql(spark, eval_expr, args, input_types, return_type, native)
                    _judge(args, py_value, py_exc, sql_value, None)
                except VerifyQueryTimeout as exc:
                    mismatch.append(f"{exc} on {args!r}")
                    return
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
