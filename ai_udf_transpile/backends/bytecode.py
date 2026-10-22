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

% is floored in Python but truncated in Spark, so the emitted SQL carries
the sign-correction CASE (same form the fake backend uses).  == and != use
<=> (null-safe) because Python returns False for None == x but Spark's ==
returns NULL.  // and ** are declined: Spark has no infix // or **.
"""

from __future__ import annotations

import dis
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


class BytecodeBackend:
    name = "bytecode"

    def run(self, job: TranspileJob, sandbox: Any) -> TranspileResult:
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
        string_params = {i for i, t in enumerate(job.input_types) if t == "string"}
        evaluator = _Evaluator(job.param_names, func_code, string_params)
        result = evaluator.evaluate(list(dis.get_instructions(func_code)))
        if result is None:
            raise BackendDecline("function has no return path")
        (sandbox / "OUT.sql").write_text(result + "\n", encoding="utf-8")
        return TranspileResult(kind=KIND_CATALYST, sql=result)


class _Evaluator:
    def __init__(self, param_names: list[str], code: types.CodeType, string_params: set[int]):
        self.param_names = param_names
        self.co_consts = code.co_consts
        self.string_params = string_params

    def _is_string(self, expr: str) -> bool:
        """True if expr is a string literal, concat(...), or string-typed param."""
        if expr.startswith("'") or expr.startswith("concat("):
            return True
        return any(expr == f"_udf_param_{i}" for i in self.string_params)

    def evaluate(self, instrs: list[dis.Instruction]) -> Optional[str]:
        self._idx = {ins.offset: n for n, ins in enumerate(instrs)}
        return self._eval_range(instrs, 0, len(instrs), [])

    def _eval_range(
        self,
        instrs: list[dis.Instruction],
        start: int,
        end: int,
        stack: list[str],
    ) -> Optional[str]:
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
                    stack.append(f"_udf_param_{self.param_names.index(nm)}")
                i += 1
                continue
            if op == "LOAD_CONST":
                stack.append(self._const_sql(ins.argval))
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
                b, a = stack.pop(), stack.pop()
                r = ins.argrepr
                if r == "%":
                    stack.append(_floored(a, b))
                elif r == "+" and (self._is_string(a) or self._is_string(b)):
                    stack.append(f"concat({a}, {b})")
                elif r in _BINARY_OPS:
                    stack.append(f"({a} {_BINARY_OPS[r]} {b})")
                else:
                    raise BackendDecline(f"unsupported BINARY_OP {r!r}")
                i += 1
                continue
            if op == "UNARY_NEGATIVE":
                stack.append(f"(-{stack.pop()})")
                i += 1
                continue
            if op == "UNARY_NOT":
                stack.append(f"(NOT {stack.pop()})")
                i += 1
                continue
            if op == "COMPARE_OP":
                b, a = stack.pop(), stack.pop()
                cmp = _COMPARE_OPS.get(ins.argval)
                if cmp is None:
                    raise BackendDecline(f"unsupported COMPARE_OP {ins.argval!r}")
                if ins.argval == "!=":
                    stack.append(f"(NOT ({a} <=> {b}))")
                else:
                    stack.append(f"({a} {cmp} {b})")
                i += 1
                continue
            if op == "IS_OP":
                b, a = stack.pop(), stack.pop()
                kw = "IS NOT" if ins.arg else "IS"
                if b == "NULL":
                    stack.append(f"({a} {kw} NULL)")
                elif a == "NULL":
                    stack.append(f"({b} {kw} NULL)")
                else:
                    raise BackendDecline("IS_OP on non-None not supported")
                i += 1
                continue
            if op in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
                return self._branch(instrs, i, end, stack, stack.pop(), op == "POP_JUMP_IF_TRUE")
            if op in ("POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE"):
                return self._branch(instrs, i, end, stack, f"{stack.pop()} IS NULL", op == "POP_JUMP_IF_NONE")
            raise BackendDecline(f"unsupported bytecode: {op}")
        # Fall-through with no RETURN: a branch's top-of-stack is the result
        # (short-circuit and/or fall through to a shared RETURN_VALUE).
        return stack[-1] if stack else None

    def _branch(
        self,
        instrs: list[dis.Instruction],
        i: int,
        end: int,
        stack: list[str],
        cond: str,
        jump_if_cond: bool,
    ) -> str:
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
        return f"CASE WHEN {cond} THEN {then} ELSE {other} END"

    def _const_sql(self, val: Any) -> str:
        if val is None:
            return "NULL"
        if val is True:
            return "true"
        if val is False:
            return "false"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            if "\\" in val:
                raise BackendDecline("backslash in string const not supported")
            return f"'{val.replace(chr(39), chr(39) + chr(39))}'"
        raise BackendDecline(f"unsupported const type {type(val).__name__}")


def _floored(a: str, b: str) -> str:
    """SQL matching Python's floor semantics for %.

    Spark's % truncates toward zero; Python floors.  The fix is the
    truncated result plus a sign-correction when signs differ and remainder
    is non-zero.
    """
    base = f"({a} % {b})"
    return f"({base} + CASE WHEN {base} <> 0 AND (({a} < 0) <> ({b} < 0)) THEN {b} ELSE 0 END)"
