# SPDX-License-Identifier: Apache-2.0
"""Deterministic backend: walk CPython bytecode and emit Catalyst SQL.

No AI in the loop. The bytecode is the source of truth, so the rewrite is
exact for the ops we support. Anything we do not recognise (CALL, IMPORT,
LOAD_ATTR, BUILD_*, ...) is a BackendDecline at Python time -- before any
SQL is emitted or verified.

Supports the common single-expression and if/else shapes:
  - arithmetic (+, -, *, /, %, unary -)
  - comparisons (<, <=, ==, !=, >, >=, is, is not)
  - boolean and/or via short-circuit jumps
  - if/else via POP_JUMP_IF_FALSE / POP_JUMP_IF_NOT_NONE

Python truthiness is not SQL truthiness (None and 0 and '' are falsy;
CASE WHEN NULL falls through but NOT NULL is NULL), so every jump
condition and `not` operand goes through _truth(), built from the kind
the symbolic stack tracks for each fragment.

% is floored in Python but truncated in Spark, so the emitted SQL carries
the sign-correction CASE (same form the fake backend uses).  == and != use
<=> (null-safe) because Python returns False for None == x but Spark's ==
returns NULL.  // and ** are declined: Spark has no infix // or **.
"""

from __future__ import annotations

import dis
import sys
import types
from typing import Any, Optional

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.keys import canonical_source_text
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob, TranspileResult

# BINARY_OP argrepr -> sql_infix.  Only ops that (a) parse in Spark and
# (b) have clean semantics.  // and ** have no Spark infix; <<, >>, &, |
# have wrapping/bool mismatches.
_BINARY_OPS: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",  # handled specially via _floored
}

# COMPARE_OP argval -> sql.  == and != use <=> for null-safety.
_COMPARE_OPS: dict[str, str] = {
    "<": "<",
    "<=": "<=",
    "==": "<=>",
    "!=": "<=>",  # wrapped in NOT by the handler
    ">": ">",
    ">=": ">=",
}

# input_types -> truthiness class. "other" params are declined if a jump
# or `not` consumes them (no faithful generic SQL).
_TRUTH_KINDS: dict[str, str] = {
    "boolean": "bool",
    "bool": "bool",
    "string": "string",
    "bigint": "numeric",
    "long": "numeric",
    "int": "numeric",
    "integer": "numeric",
    "smallint": "numeric",
    "tinyint": "numeric",
    "byte": "numeric",
    "short": "numeric",
    "double": "numeric",
    "float": "numeric",
}


def _truth_kind(spark_type: str) -> str:
    """Kind for a (possibly parameterized) type spelling: decimal(10,2),
    varchar(20) and friends must not fall to "other" and decline."""
    name = spark_type.strip().lower()
    if name.startswith("decimal"):
        return "numeric"
    if name.startswith(("varchar", "char")):
        return "string"
    return _TRUTH_KINDS.get(name, "other")


class BytecodeBackend:
    name = "bytecode"

    def run(self, job: TranspileJob, sandbox: Any) -> TranspileResult:
        # The walker is written against 3.13 instruction shapes (fused
        # TO_BOOL/RETURN_CONST/LOAD_FAST_LOAD_FAST, jump-argval rules).
        # Other versions decline rather than emit SQL from mismatched
        # bytecode -- fail closed.
        if sys.version_info[:2] != (3, 13):
            raise BackendDecline(f"CPython 3.13 bytecode required (running {sys.version.split()[0]})")
        try:
            key = canonical_source_text(job.source_text)
        except Exception:
            raise BackendDecline("could not parse source text")
        code = compile(key, "<bytecode-backend>", "exec")
        func_code = next((c for c in code.co_consts if isinstance(c, types.CodeType)), None)
        if func_code is None:
            raise BackendDecline("no function code object found")
        if func_code.co_exceptiontable:
            raise BackendDecline("try/except not supported")
        param_kinds = [_truth_kind(t) for t in job.input_types]
        evaluator = _Evaluator(job.param_names, func_code, param_kinds)
        result = evaluator.evaluate(list(dis.get_instructions(func_code)))
        if result is None:
            raise BackendDecline("function has no return path")
        (sandbox / "OUT.sql").write_text(result[0] + "\n", encoding="utf-8")
        return TranspileResult(kind=KIND_CATALYST, sql=result[0])


# One entry per value on the symbolic stack: (sql, truthiness kind).
# kind is one of "bool", "numeric", "string", "null", "mixed", "other".
_Entry = tuple[str, str]


class _Evaluator:
    def __init__(self, param_names: list[str], code: types.CodeType, param_kinds: list[str]):
        self.param_names = param_names
        self.co_consts = code.co_consts
        self.param_kinds = param_kinds

    def _is_string(self, _sql: str, kind: str) -> bool:
        return kind == "string"

    def evaluate(self, instrs: list[dis.Instruction]) -> Optional[_Entry]:
        self._idx = {ins.offset: n for n, ins in enumerate(instrs)}
        return self._eval_range(instrs, 0, len(instrs), [])

    def _eval_range(
        self,
        instrs: list[dis.Instruction],
        start: int,
        end: int,
        stack: list[_Entry],
    ) -> Optional[_Entry]:
        i = start
        while i < end:
            ins = instrs[i]
            op = ins.opname
            if op in ("RESUME", "NOP", "CACHE", "TO_BOOL"):
                i += 1
                continue
            if op in ("LOAD_FAST", "LOAD_FAST_LOAD_FAST"):
                names = ins.argval if op == "LOAD_FAST_LOAD_FAST" else (ins.argval,)
                if not isinstance(names, tuple):
                    raise BackendDecline(f"{op} without name")
                for nm in names:
                    if nm not in self.param_names:
                        raise BackendDecline(f"{op} unknown var {nm}")
                    idx = self.param_names.index(nm)
                    stack.append((f"_udf_param_{idx}", self.param_kinds[idx]))
                i += 1
                continue
            if op == "LOAD_CONST":
                sql, kind = self._const_sql(ins.argval)
                stack.append((sql, kind))
                i += 1
                continue
            if op == "RETURN_VALUE":
                return stack[-1] if stack else None
            if op == "RETURN_CONST":
                return self._const_sql(ins.argval)
            if op == "POP_TOP":
                stack.pop()
                i += 1
                continue
            if op == "COPY":
                stack.append(stack[-(ins.arg or 1)])
                i += 1
                continue
            if op == "SWAP":
                n = ins.arg or 1
                stack[-1], stack[-n] = stack[-n], stack[-1]
                i += 1
                continue
            if op == "BINARY_OP":
                (b_sql, b_kind), (a_sql, a_kind) = stack.pop(), stack.pop()
                r = ins.argrepr
                if r == "%":
                    self._numeric_only(r, a_kind, b_kind)
                    stack.append((_floored(a_sql, b_sql), "numeric"))
                elif r == "+":
                    if self._is_string(a_sql, a_kind) or self._is_string(b_sql, b_kind):
                        stack.append((f"concat({a_sql}, {b_sql})", "string"))
                    else:
                        self._numeric_only(r, a_kind, b_kind)
                        stack.append((f"({a_sql} + {b_sql})", "numeric"))
                elif r in _BINARY_OPS:
                    self._numeric_only(r, a_kind, b_kind)
                    stack.append((f"({a_sql} {_BINARY_OPS[r]} {b_sql})", "numeric"))
                else:
                    raise BackendDecline(f"unsupported BINARY_OP {r!r}")
                i += 1
                continue
            if op == "UNARY_NEGATIVE":
                a_sql, a_kind = stack.pop()
                self._numeric_only("unary -", a_kind, None)
                stack.append((f"(-{a_sql})", "numeric"))
                i += 1
                continue
            if op == "UNARY_NOT":
                sql = self._not_sql(stack.pop())
                stack.append((sql, "bool"))
                i += 1
                continue
            if op == "COMPARE_OP":
                (b_sql, b_kind), (a_sql, a_kind) = stack.pop(), stack.pop()
                cmp = _COMPARE_OPS.get(ins.argval)
                if cmp is None:
                    raise BackendDecline(f"unsupported COMPARE_OP {ins.argval!r}")
                if not self._comparable(a_kind, b_kind):
                    # Spark would implicitly coerce ('5' == 5 is true in SQL,
                    # False in python) or fail analysis: not a rewrite, a lie.
                    raise BackendDecline(f"comparison of {a_kind} and {b_kind} operands not supported")
                if ins.argval == "!=":
                    stack.append((f"(NOT ({a_sql} <=> {b_sql}))", "bool"))
                else:
                    stack.append((f"({a_sql} {cmp} {b_sql})", "bool"))
                i += 1
                continue
            if op == "IS_OP":
                (b_sql, b_kind), (a_sql, a_kind) = stack.pop(), stack.pop()
                kw = "IS NOT" if ins.arg else "IS"
                if b_sql == "NULL":
                    stack.append((f"({a_sql} {kw} NULL)", "bool"))
                elif a_sql == "NULL":
                    stack.append((f"({b_sql} {kw} NULL)", "bool"))
                else:
                    raise BackendDecline("IS_OP on non-None not supported")
                i += 1
                continue
            if op in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
                cond = self._truth(stack.pop())
                return self._branch(instrs, i, end, stack, cond, op == "POP_JUMP_IF_TRUE")
            if op in ("POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE"):
                cond = stack.pop()[0]
                return self._branch(instrs, i, end, stack, f"{cond} IS NULL", op == "POP_JUMP_IF_NONE")
            raise BackendDecline(f"unsupported bytecode: {op}")
        # Fall-through with no RETURN: a branch's top-of-stack is the result
        # (short-circuit and/or fall through to a shared RETURN_VALUE).
        return stack[-1] if stack else None

    def _branch(
        self,
        instrs: list[dis.Instruction],
        i: int,
        end: int,
        stack: list[_Entry],
        cond: str,
        jump_if_cond: bool,
    ) -> _Entry:
        """Split at a conditional jump.  Both sub-ranges run to `end` so shared
        code after the join point (e.g. the NOT in `not (a and b)`) is seen by
        both branches.  jump_if_cond=True means the jump target is the THEN
        case; False means the fall-through is."""
        target = self._idx.get(instrs[i].argval)
        if target is None or not (i < target <= end):
            raise BackendDecline(f"jump target {instrs[i].argval!r} out of range")
        fall = self._eval_range(instrs, i + 1, end, list(stack))
        jump = self._eval_range(instrs, target, end, list(stack))
        if fall is None or jump is None:
            raise BackendDecline("branch with no return")
        then, other = (jump, fall) if jump_if_cond else (fall, jump)
        kind = then[1] if then[1] == other[1] else "mixed"
        return (f"CASE WHEN {cond} THEN {then[0]} ELSE {other[0]} END", kind)

    def _truth(self, entry: _Entry) -> str:
        """SQL matching Python truthiness: None, 0, and '' are falsy."""
        sql, kind = entry
        if kind == "bool":
            return sql
        if kind == "numeric":
            return f"({sql} IS NOT NULL AND {sql} <> 0)"
        if kind == "string":
            return f"({sql} IS NOT NULL AND length({sql}) > 0)"
        raise BackendDecline(f"no truthiness SQL for a {kind or 'unknown'}-typed value")

    def _not_sql(self, entry: _Entry) -> str:
        """`not v`. NOT NULL is NULL in SQL, but `not None` is True."""
        cond = self._truth(entry)
        return f"CASE WHEN {cond} THEN false ELSE true END"

    def _numeric_only(self, op: str, a_kind: Optional[str], b_kind: Optional[str]) -> None:
        # numeric only, NULL folds in. "other" params (timestamp/binary/array)
        # are declined: python raises on ts + 5 and Spark would happily
        # date-shift -- inventing semantics the function never had.
        for kind in (a_kind, b_kind):
            if kind is not None and kind not in ("numeric", "null"):
                raise BackendDecline(f"{op!r} on a {kind} operand not supported")

    @staticmethod
    def _comparable(a_kind: str, b_kind: str) -> bool:
        """True when Spark compares these without changing python semantics.

        NULL folds with anything (python returns False for None == x, and
        NULL <=> x is false). Anything else must be the same class: Spark's
        implicit string->number coercion makes '5' == 5 true in SQL where
        python says False.
        """
        if a_kind == "null" or b_kind == "null":
            return True
        return a_kind == b_kind and a_kind in ("numeric", "string", "bool")

    def _const_sql(self, val: Any) -> _Entry:
        if val is None:
            return "NULL", "null"
        if val is True:
            return "true", "bool"
        if val is False:
            return "false", "bool"
        if isinstance(val, (int, float)):
            if isinstance(val, float) and val != val:
                return "cast('NaN' as double)", "numeric"
            if val == float("inf"):
                return "cast('Infinity' as double)", "numeric"
            if val == float("-inf"):
                return "cast('-Infinity' as double)", "numeric"
            return str(val), "numeric"
        if isinstance(val, str):
            if "\\" in val:
                raise BackendDecline("backslash in string const not supported")
            return f"'{val.replace(chr(39), chr(39) + chr(39))}'", "string"
        raise BackendDecline(f"unsupported const type {type(val).__name__}")


def _floored(a: str, b: str) -> str:
    """SQL matching Python's floor semantics for %.

    Spark's % truncates toward zero; Python floors.  The fix is the
    truncated result plus a sign-correction when signs differ and remainder
    is non-zero.
    """
    base = f"({a} % {b})"
    return f"({base} + CASE WHEN {base} <> 0 AND (({a} < 0) <> ({b} < 0)) THEN {b} ELSE 0 END)"
