# SPDX-License-Identifier: Apache-2.0
"""Static guess at whether a verified rewrite will be slower than the Python.

A rewrite removes one thing for certain: the per-row trip to the Python
worker. Everything the body does still has to be done, by a different
engine. So the question is how the body's work fares after the move:

  speedup = (overhead + work) / (jvm_overhead + k * work)

where k is what the work costs in its new engine relative to CPython. A
Python-level loop becoming one native builtin has a tiny k and wins big; a
regex stays a regex (k near 1, and above it once the rewrite has to spell
out Python's unicode classes or emulate ``\\b`` with lookarounds); and a C
string method (``strip``, ``splitlines``, ``startswith``) re-expressed as a
regex has a k in the tens, because one ``regexp_*`` evaluation costs about
as much as the whole Python round trip it replaced.

This module reads the two texts and says which of those cases it sees. It
is report-only: nothing here gates serving, and the verdict is about sign,
not size. The thresholds were set on the seven FHI pipeline UDFs (see
examples/bench_fhi_udfs.py, which prints this next to the measured ratio)
and checked against the documented examples for false alarms; that is one
machine and one corpus, so treat `at-risk` as "measure before serving".
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

OK = "ok"
AT_RISK = "at-risk"
LIKELY_REGRESSION = "likely-regression"
NOT_SERVED = "not-served"

# A body making this many regex calls per row is regex-dominated: the
# round trip is a small share of its cost, so a rewrite that only moves the
# same regexes has a ceiling near 1x.
LOW_HEADROOM_REGEX_CALLS = 5

_RE_FUNCS = frozenset({"search", "match", "fullmatch", "sub", "subn", "findall", "finditer", "split"})
_SQL_REGEX_CALL = re.compile(
    r"\b(regexp_replace|regexp_extract_all|regexp_extract|regexp_like|regexp_instr|regexp_substr"
    r"|regexp_count|regexp|split)\s*\(",
    re.IGNORECASE,
)
_SQL_REGEX_OP = re.compile(r"\b(RLIKE|REGEXP)\b(?!\s*\()", re.IGNORECASE)
_LEADING_NULL_GUARD = re.compile(
    r"^\s*CASE\s+WHEN\s+_udf_param_\d+\s+IS\s+NULL\s+THEN\s+(?:CAST\s*\(\s*)?NULL\b.*?\bWHEN\b",
    re.IGNORECASE | re.DOTALL,
)
_ASCII_GUARD = re.compile(
    r"\blength\s*\(\s*(_udf_param_\d+)\s*\)\s*=\s*octet_length\s*\(\s*\1\s*\)", re.IGNORECASE
)
# Python's unicode whitespace / line breaks / digits / word chars written
# out for java.util.regex, and \b rebuilt from a lookbehind+lookahead pair.
_UNICODE_CLASS = re.compile(r"\\+(?:u2028|u3000|x85|xa0|p\{)", re.IGNORECASE)
_LOOKBEHIND = re.compile(r"\(\?<[=!]")
_JAVA_REGEX = re.compile(r"\bPattern\s*\.\s*compile\s*\(|\.\s*(?:replaceAll|replaceFirst|matches|split)\s*\(")


@dataclass(frozen=True)
class RegressionRisk:
    verdict: str
    flags: tuple[str, ...]
    python_regex_calls: int
    rewrite_regex_evals: int

    def __str__(self) -> str:
        return self.verdict + (f" ({', '.join(self.flags)})" if self.flags else "")


def _literal_len(node: Optional[ast.AST], bindings: dict[str, int]) -> int:
    """How many times a ``for`` over this iterable runs, when that is static."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return max(1, len(node.elts))
    if isinstance(node, ast.Name):
        return bindings.get(node.id, 1)
    return 1


def python_regex_calls(source_text: str) -> int:
    """Regex calls on the longest path through the function body.

    A call inside ``for p in <tuple literal>`` counts once per element; a
    loop over data (``for line in text.splitlines()``) counts once, since
    its length is not in the source.
    """
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return 0
    func = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if func is None:
        return 0
    re_names = {"re"}
    bindings: dict[str, int] = {}
    compiled: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Import):
            re_names.update(a.asname or a.name for a in node.names if a.name == "re")
        elif (
            isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if isinstance(node.value, (ast.Tuple, ast.List)):
                bindings[name] = max(1, len(node.value.elts))
            elif _is_re_call(node.value, re_names, {"compile"}):
                compiled.add(name)

    def count(node: ast.AST, multiplier: int) -> int:
        total = 0
        if isinstance(node, ast.Call):
            f = node.func
            if _is_re_call(node, re_names, _RE_FUNCS) or (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in compiled
                and f.attr in _RE_FUNCS
            ):
                total += multiplier
        if isinstance(node, (ast.For, ast.AsyncFor)):
            inner = multiplier * _literal_len(node.iter, bindings)
            total += count(node.iter, multiplier)
            for child in node.body + node.orelse:
                total += count(child, inner)
            return total
        for child in ast.iter_child_nodes(node):
            total += count(child, multiplier)
        return total

    return sum(count(stmt, 1) for stmt in func.body)


def _is_re_call(node: Optional[ast.AST], re_names: set[str], attrs: frozenset[str] | set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in re_names
        and node.func.attr in attrs
    )


def _split_sql(sql: str) -> tuple[str, list[str]]:
    """(sql with string literals blanked, the literal bodies)."""
    out: list[str] = []
    literals: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c not in "'\"":
            out.append(c)
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < n:
            if sql[j] == "\\" and j + 1 < n:
                body.append(sql[j : j + 2])
                j += 2
            elif sql[j] == c and j + 1 < n and sql[j + 1] == c:
                body.append(c)
                j += 2
            elif sql[j] == c:
                break
            else:
                body.append(sql[j])
                j += 1
        literals.append("".join(body))
        out.append(c + " " * (j - i - 1) + c)
        i = j + 1
    return "".join(out), literals


def _call_text(sql: str, start: int) -> str:
    """The balanced-paren call beginning at ``start`` (literals respected)."""
    blanked, _ = _split_sql(sql)
    depth = 0
    for k in range(blanked.index("(", start), len(blanked)):
        if blanked[k] == "(":
            depth += 1
        elif blanked[k] == ")":
            depth -= 1
            if depth == 0:
                return sql[start : k + 1]
    return sql[start:]


def _sql_features(sql: str) -> dict[str, Any]:
    blanked, literals = _split_sql(sql)
    calls = [_call_text(sql, m.start()) for m in _SQL_REGEX_CALL.finditer(blanked)]
    text = "\n".join(literals)
    return {
        "evals": len(calls) + len(_SQL_REGEX_OP.findall(blanked)),
        "lambda": "->" in blanked,
        "null_guard_first": bool(_LEADING_NULL_GUARD.match(blanked)),
        "ascii_guard": bool(_ASCII_GUARD.search(blanked)),
        "repeated": len(calls) != len(set(calls)),
        "unicode_class": bool(_UNICODE_CLASS.search(text)),
        "lookbehind": bool(_LOOKBEHIND.search(text)),
    }


def _java_features(java: str) -> dict[str, Any]:
    return {
        "evals": len(_JAVA_REGEX.findall(java)),
        "lambda": False,
        "null_guard_first": False,
        "ascii_guard": False,
        "repeated": False,
        "unicode_class": bool(_UNICODE_CLASS.search(java)),
        "lookbehind": bool(_LOOKBEHIND.search(java)),
    }


def regression_risk(source_text: str, result: Any) -> RegressionRisk:
    """Verdict for one (python source, rewrite) pair. ``result`` is a
    TranspileResult (or anything with ``sql`` / ``java_source``)."""
    sql = getattr(result, "sql", None) or ""
    java = getattr(result, "java_source", None) or ""
    f = _sql_features(sql) if sql else _java_features(java)
    e_py = python_regex_calls(source_text)
    e_rw = int(f["evals"])
    flags: list[str] = []
    if f["lambda"]:
        # Spark 4.3 keeps the python UDF for any option containing a
        # LambdaFunction, silently: there is nothing to time.
        return RegressionRisk(NOT_SERVED, ("lambda",), e_py, e_rw)

    # Behind an ASCII guard only one arm runs, and it is the cheap one for
    # ASCII rows: the emulation in the other arm is not paid there.
    emulation = (f["unicode_class"] or f["lookbehind"]) and not f["ascii_guard"]
    if f["unicode_class"]:
        flags.append("unicode-classes")
    if f["lookbehind"]:
        flags.append("lookaround")
    if f["ascii_guard"]:
        flags.append("ascii-guard")
    if f["repeated"]:
        flags.append("repeated-regex")
    if f["null_guard_first"] and f["repeated"]:
        # The guard is the only always-evaluated WHEN condition, so Spark's
        # subexpression elimination shares nothing after it.
        flags.append("null-guard-defeats-cse")
    if java and e_rw:
        flags.append("java-regex")

    verdict = OK
    if e_rw and not e_py:
        # The python used C string methods; the rewrite brought regexes.
        flags.append("regex-introduced")
        verdict = LIKELY_REGRESSION if e_rw >= 2 else AT_RISK
    elif e_rw and e_py:
        low_headroom = e_py >= LOW_HEADROOM_REGEX_CALLS
        if low_headroom:
            flags.append("regex-dominated")
        if low_headroom and emulation:
            verdict = LIKELY_REGRESSION
        elif low_headroom or f["lookbehind"] or java:
            verdict = AT_RISK
    return RegressionRisk(verdict, tuple(flags), e_py, e_rw)
