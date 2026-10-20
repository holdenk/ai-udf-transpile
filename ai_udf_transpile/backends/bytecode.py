# SPDX-License-Identifier: Apache-2.0
"""Deterministic backend: walk CPython bytecode and emit Catalyst SQL.

No AI in the loop. The bytecode is the source of truth, so the rewrite is
exact for the ops we support. Anything we do not recognise (CALL, IMPORT,
LOAD_ATTR, BUILD_*, ...) is a BackendDecline at Python time -- before any
SQL is emitted or verified.

Supports the common single-expression and if/else shapes:
  - arithmetic (+, -, *, /, //, %, **, unary -)
  - comparisons (<, <=, ==, !=, >, >=, is, is not)
  - boolean and/or via short-circuit jumps
  - if/else via POP_JUMP_IF_FALSE / POP_JUMP_IF_NOT_NONE

% and // are floored in Python but truncated in Spark, so the emitted SQL
carries the sign-correction CASE (same form the fake backend uses).
"""

from __future__ import annotations

import ast
import dis
import types
from typing import Any, Optional

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.keys import canonical_source_text
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob, TranspileResult

# BINARY_OP argrepr -> (sql_infix, needs_floor_fix).  The floor fix is for
# ops where Python floors but Spark truncates toward zero.
_BINARY_OPS: dict[str, tuple[str, bool]] = {
    "+": ("+", False),
    "-": ("-", False),
    "*": ("*", False),
    "/": ("/", False),
    "//": ("//", True),  # floored in python, truncated in spark
    "%": ("%", True),  # floored in python, truncated in spark
    "**": ("**", False),
    "<<": ("<<", False),
    ">>": (">>", False),
    "&": ("&", False),
    "|": ("|", False),
}

_COMPARE_OPS: dict[str, str] = {
    "<": "<",
    "<=": "<=",
    "==": "==",
    "!=": "!=",
    ">": ">",
    ">=": ">=",
}

# ops we hard-decline on (no SQL equivalent, or too much surface)
_DECLINE_OPS = frozenset(
    {
        "CALL",
        "CALL_FUNCTION",
        "CALL_FUNCTION_KW",
        "CALL_FUNCTION_EX",
        "CALL_KW",
        "LOAD_ATTR",
        "LOAD_METHOD",
        "MAKE_FUNCTION",
        "MAKE_CELL",
        "IMPORT_NAME",
        "IMPORT_FROM",
        "BUILD_LIST",
        "BUILD_MAP",
        "BUILD_TUPLE",
        "BUILD_SET",
        "BUILD_CONST_KEY_MAP",
        "FORMAT_VALUE",
        "FORMAT_SIMPLE",
        "FORMAT_WITH_KWARGS",
        "BINARY_SUBSCR",
        "STORE_SUBSCR",
        "DELETE_SUBSCR",
        "UNPACK_SEQUENCE",
        "UNPACK_EX",
        "STORE_FAST",
        "STORE_FAST_CHECK",
        "STORE_FAST_MAYBE_NULL",
        "STORE_SUBSCR",
        "STORE_ATTR",
        "LOAD_GLOBAL",  # builtins via LOAD_GLOBAL + CALL are declined at CALL
        "LOAD_DEREF",
        "LOAD_CLOSURE",
        "LOAD_CLASSDEREF",  # closures
        "LIST_APPEND",
        "LIST_EXTEND",
        "DICT_MERGE",
        "SET_ADD",
        "SET_UPDATE",
        "MAP_ADD",
        "GEN_START",
        "SEND",
        "GET_ITER",
        "FOR_ITER",
        "YIELD_VALUE",
        "YIELD_FROM",
        "RETURN_GENERATOR",
        "SETUP_FINALLY",
        "SETUP_EXCEPT",
        "SETUP_WITH",
        "SETUP_ASYNC_WITH",
        "POP_BLOCK",
        "PUSH_EXC_INFO",
        "CHECK_EXC_MATCH",
        "CHECK_EG_MATCH",
        "RAISE_VARARGS",
        "RERAISE",
        "PUSH_NULL",
        "LOAD_ASSERTION_ERROR",
        "LOAD_NAME",
        "DELETE_FAST",
        "DELETE_GLOBAL",
        "DELETE_NAME",
        "DELETE_ATTR",
        "BINARY_SLICE",
        "STORE_SLICE",
    }
)


class BytecodeBackend:
    name = "bytecode"

    def run(self, job: TranspileJob, sandbox: Any) -> TranspileResult:
        try:
            key = canonical_source_text(job.source_text)
        except Exception:
            key = (job.source_text or "").strip()
        # Compile the canonical source so the bytecode matches what Spark
        # would actually run (dedented, normalised).
        tree = ast.parse(key)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        sql = _compile_function(tree, fn, job.param_names, job.input_types)
        (sandbox / "OUT.sql").write_text(sql + "\n", encoding="utf-8")
        return TranspileResult(kind=KIND_CATALYST, sql=sql)


def _compile_function(
    tree: ast.Module, fn: ast.FunctionDef, param_names: list[str], input_types: list[str]
) -> str:
    code = compile(tree, "<bytecode-backend>", "exec")
    # Find the code object for our function.
    func_code = _find_code(code, fn.name)
    if func_code is None:
        raise BackendDecline(f"could not locate code object for {fn.name}")
    instrs = list(dis.get_instructions(func_code))
    string_params = {i for i, t in enumerate(input_types) if t == "string"}
    evaluator = _Evaluator(param_names, func_code, string_params)
    result = evaluator.evaluate(instrs)
    if result is None:
        raise BackendDecline("function has no return path")
    return result


def _find_code(code: types.CodeType, name: str) -> Optional[types.CodeType]:
    for const in code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == name:
            return const
    return None


class _Evaluator:
    def __init__(self, param_names: list[str], code: types.CodeType, string_params: set[int]):
        self.param_names = param_names
        self.code = code
        self.co_consts = code.co_consts
        self.string_params = string_params

    def _is_string(self, expr: str) -> bool:
        """True if expr is a string literal or a string-typed param ref."""
        if expr.startswith("'"):
            return True
        for i in self.string_params:
            if expr == f"_udf_param_{i}":
                return True
        return False

    def evaluate(self, instrs: list[dis.Instruction]) -> Optional[str]:
        """Walk instructions, return the SQL for the return value, or None."""
        # Build a map of jump targets -> index for branch handling.
        targets = {}
        for i, ins in enumerate(instrs):
            if ins.is_jump_target:
                targets[ins.offset] = i
        return self._eval_range(instrs, 0, len(instrs), targets, [])

    def _eval_range(
        self,
        instrs: list[dis.Instruction],
        start: int,
        end: int,
        targets: dict[int, int],
        stack: list[str],
    ) -> Optional[str]:
        i = start
        while i < end:
            ins = instrs[i]
            op = ins.opname
            if op in ("RESUME", "NOP", "CACHE"):
                i += 1
                continue
            if op == "LOAD_FAST" or op == "LOAD_FAST_CHECK" or op == "LOAD_FAST_MAYBE_NULL":
                idx = ins.arg
                if idx is None or idx >= len(self.param_names):
                    raise BackendDecline(f"LOAD_FAST index {idx} out of range")
                stack.append(f"_udf_param_{idx}")
                i += 1
                continue
            if op == "LOAD_FAST_LOAD_FAST":
                # Python 3.13+: packs two fast vars in one op. argval is (name, name).
                names = ins.argval
                if not isinstance(names, tuple) or len(names) != 2:
                    raise BackendDecline("LOAD_FAST_LOAD_FAST without name pair")
                for nm in names:
                    if nm not in self.param_names:
                        raise BackendDecline(f"LOAD_FAST_LOAD_FAST unknown var {nm}")
                    idx = self.param_names.index(nm)
                    stack.append(f"_udf_param_{idx}")
                i += 1
                continue
            if op == "LOAD_CONST":
                stack.append(self._const_sql(ins.arg))
                i += 1
                continue
            if op == "RETURN_VALUE":
                if not stack:
                    raise BackendDecline("RETURN_VALUE with empty stack")
                return stack[-1]
            if op == "RETURN_CONST":
                return self._const_sql(ins.arg)
            if op == "POP_TOP":
                if stack:
                    stack.pop()
                i += 1
                continue
            if op == "COPY":
                n = ins.arg or 1
                if not stack or n > len(stack):
                    raise BackendDecline("COPY underflow")
                stack.append(stack[-n])
                i += 1
                continue
            if op == "SWAP":
                n = ins.arg or 1
                if n > len(stack):
                    raise BackendDecline("SWAP underflow")
                stack[-1], stack[-n] = stack[-n], stack[-1]
                i += 1
                continue
            if op == "BINARY_OP":
                if len(stack) < 2:
                    raise BackendDecline("BINARY_OP underflow")
                b = stack.pop()
                a = stack.pop()
                spec = _BINARY_OPS.get(ins.argrepr)
                if spec is None:
                    raise BackendDecline(f"unsupported BINARY_OP {ins.argrepr!r}")
                infix, floor = spec
                if infix == "+" and (self._is_string(a) or self._is_string(b)):
                    expr = f"concat({a}, {b})"
                elif floor:
                    expr = _floored(a, b, infix)
                else:
                    expr = f"{a} {infix} {b}"
                stack.append(expr)
                i += 1
                continue
            if op == "UNARY_NEGATIVE" or op == "UNARY_NOT" or op == "UNARY_POSITIVE" or op == "UNARY_INVERT":
                if not stack:
                    raise BackendDecline("unary underflow")
                a = stack.pop()
                if op == "UNARY_NEGATIVE":
                    stack.append(f"-{a}")
                elif op == "UNARY_NOT":
                    stack.append(f"NOT {a}")
                elif op == "UNARY_POSITIVE":
                    stack.append(a)
                else:
                    raise BackendDecline("UNARY_INVERT has no SQL equivalent")
                i += 1
                continue
            if op == "COMPARE_OP":
                if len(stack) < 2:
                    raise BackendDecline("COMPARE_OP underflow")
                b = stack.pop()
                a = stack.pop()
                cmp = _COMPARE_OPS.get(ins.argval)
                if cmp is None:
                    raise BackendDecline(f"unsupported COMPARE_OP {ins.argval!r}")
                stack.append(f"{a} {cmp} {b}")
                i += 1
                continue
            if op == "IS_OP":
                if len(stack) < 2:
                    raise BackendDecline("IS_OP underflow")
                b = stack.pop()
                a = stack.pop()
                negate = bool(ins.arg)
                if b == "NULL":
                    stack.append(f"{a} IS{' NOT' if negate else ''} NULL")
                elif a == "NULL":
                    stack.append(f"{b} IS{' NOT' if negate else ''} NULL")
                else:
                    raise BackendDecline("IS_OP on non-None not supported")
                i += 1
                continue
            if op == "TO_BOOL":
                # SQL booleans are already booleans; no-op.
                i += 1
                continue
            if op == "POP_JUMP_IF_FALSE" or op == "POP_JUMP_IF_TRUE":
                return self._conditional_jump(instrs, i, end, targets, stack, invert=op == "POP_JUMP_IF_TRUE")
            if op == "POP_JUMP_IF_NOT_NONE" or op == "POP_JUMP_IF_NONE":
                return self._none_jump(instrs, i, end, targets, stack, negate=op == "POP_JUMP_IF_NONE")
            if op in _DECLINE_OPS:
                raise BackendDecline(f"unsupported bytecode: {op}")
            raise BackendDecline(f"unsupported bytecode: {op}")
        # Fall-through: no RETURN hit.  The top of stack is the branch result
        # (short-circuit and/or patterns fall through to a shared RETURN_VALUE).
        return stack[-1] if stack else None

    def _conditional_jump(
        self,
        instrs: list[dis.Instruction],
        i: int,
        end: int,
        targets: dict[int, int],
        stack: list[str],
        *,
        invert: bool,
    ) -> str:
        ins = instrs[i]
        if not stack:
            raise BackendDecline("conditional jump with empty stack")
        cond = stack.pop()
        # POP_JUMP_IF_FALSE: jump when false.  Fall-through = true branch.
        # POP_JUMP_IF_TRUE: jump when true.  Fall-through = false branch.
        if invert:
            cond = f"(NOT {cond})"
        target_off = ins.argval if isinstance(ins.argval, int) else ins.arg
        target_idx = targets.get(target_off, None)
        if target_idx is None:
            raise BackendDecline(f"jump target {target_off} not found")
        true_result = self._eval_range(instrs, i + 1, target_idx, targets, list(stack))
        false_result = self._eval_range(instrs, target_idx, end, targets, list(stack))
        if true_result is None or false_result is None:
            raise BackendDecline("branch with no return")
        return f"CASE WHEN {cond} THEN {true_result} ELSE {false_result} END"

    def _none_jump(
        self,
        instrs: list[dis.Instruction],
        i: int,
        end: int,
        targets: dict[int, int],
        stack: list[str],
        *,
        negate: bool,
    ) -> str:
        ins = instrs[i]
        if not stack:
            raise BackendDecline("none-jump with empty stack")
        val = stack.pop()
        # POP_JUMP_IF_NOT_NONE: jump when not None.  Fall-through = val IS None.
        # POP_JUMP_IF_NONE: jump when None.  Fall-through = val IS NOT None.
        if negate:
            cond = f"{val} IS NOT NULL"
        else:
            cond = f"{val} IS NULL"
        target_off = ins.argval if isinstance(ins.argval, int) else ins.arg
        target_idx = targets.get(target_off, None)
        if target_idx is None:
            raise BackendDecline(f"jump target {target_off} not found")
        fallthrough_result = self._eval_range(instrs, i + 1, target_idx, targets, list(stack))
        target_result = self._eval_range(instrs, target_idx, end, targets, list(stack))
        if fallthrough_result is None or target_result is None:
            raise BackendDecline("branch with no return")
        return f"CASE WHEN {cond} THEN {fallthrough_result} ELSE {target_result} END"

    def _const_sql(self, arg: Optional[int]) -> str:
        if arg is None or arg >= len(self.co_consts):
            raise BackendDecline("LOAD_CONST out of range")
        val = self.co_consts[arg]
        if val is None:
            return "NULL"
        if val is True:
            return "true"
        if val is False:
            return "false"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            escaped = val.replace("'", "''")
            return f"'{escaped}'"
        raise BackendDecline(f"unsupported const type {type(val).__name__}")


def _floored(a: str, b: str, op: str) -> str:
    """Emit SQL matching Python's floor semantics for % and //.

    Spark's % and // truncate toward zero; Python floors.  The fix is the
    truncated result plus a sign-correction when the signs differ and the
    remainder is non-zero.
    """
    base = f"({a} {op} {b})"
    if op == "%":
        return f"{base} + CASE WHEN {base} <> 0 AND (({a} < 0) <> ({b} < 0)) THEN {b} ELSE 0 END"
    if op == "//":
        rem = f"({a} % {b})"
        return f"{base} - CASE WHEN {rem} <> 0 AND (({a} < 0) <> ({b} < 0)) THEN 1 ELSE 0 END"
    return base
