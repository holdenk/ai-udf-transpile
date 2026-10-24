# SPDX-License-Identifier: Apache-2.0
"""Compile the supported bytecode subset into a Java UDF that interprets it.

The interpreter is ``java/BytecodeEval.java``, pasted into each UDF class so
the jar is self-contained. Same opcode allowlist as the SQL bytecode backend
(IS_OP is None-only here too). Known limits: decimal(0) is falsy but
decimal *arithmetic* throws (java cannot be exact); ``is`` on non-None is
declined; string order is UTF-16 code units; empty arrays/maps read truthy
and mixed decimal/long equality can diverge -- all caught by the
hypothesis gate, so a UDF that hits one is never served.
"""

from __future__ import annotations

import dis
import hashlib
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.keys import canonical_source_from_func, canonical_source_text

_JAVA_PATH = Path(__file__).resolve().parents[1] / "java" / "BytecodeEval.java"
_SKIP = {"RESUME", "NOP", "CACHE", "TO_BOOL", "EXTENDED_ARG"}
_BIN = {"+": "OP_BIN_ADD", "-": "OP_BIN_SUB", "*": "OP_BIN_MUL", "/": "OP_BIN_TRUEDIV", "%": "OP_BIN_MOD"}
_CMP = {
    "<": "OP_CMP_LT",
    "<=": "OP_CMP_LE",
    "==": "OP_CMP_EQ",
    "!=": "OP_CMP_NE",
    ">": "OP_CMP_GT",
    ">=": "OP_CMP_GE",
}
_JUMP = {
    "POP_JUMP_IF_FALSE": "OP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE": "OP_JUMP_IF_TRUE",
    "POP_JUMP_IF_NONE": "OP_JUMP_IF_NONE",
    "POP_JUMP_IF_NOT_NONE": "OP_JUMP_IF_NOT_NONE",
}

_OP_RE = re.compile(r"public static final int (OP_[A-Z0-9_]+) = (\d+);")


def opcode_numbers(java_source: Optional[str] = None) -> dict[str, int]:
    text = java_source if java_source is not None else _JAVA_PATH.read_text(encoding="utf-8")
    found = {name: int(value) for name, value in _OP_RE.findall(text)}
    if len(found) < 20:
        raise BackendDecline("BytecodeEval.java is missing opcode constants")
    return found


@dataclass
class JavaEval:
    class_name: str
    java_source: str
    code: list[int]
    consts: list[Any]


def emit_java_eval(func: Any) -> JavaEval:
    """Java UDF source for ``func``. Raises BackendDecline on an unsupported op."""
    source = canonical_source_from_func(func)
    return emit_java_eval_source(source)


def emit_java_eval_source(source_text: str) -> JavaEval:
    # 3.13 instruction shapes only (see backends/bytecode.py); anything
    # else declines fail-closed instead of encoding mismatched bytecode.
    if sys.version_info[:2] != (3, 13):
        raise BackendDecline(f"CPython 3.13 bytecode required (running {sys.version.split()[0]})")
    try:
        key = canonical_source_text(source_text)
    except Exception as exc:
        raise BackendDecline("could not parse source text") from exc
    module = compile(key, "<bytecode-eval>", "exec")
    func_code = next((c for c in module.co_consts if isinstance(c, types.CodeType)), None)
    if func_code is None:
        raise BackendDecline("no function code object found")
    if func_code.co_exceptiontable:
        raise BackendDecline("try/except not supported")
    ops = opcode_numbers()
    code, consts = _encode(func_code, ops)
    arity = func_code.co_argcount
    if not 1 <= arity <= 10:
        raise BackendDecline(f"arity {arity} outside 1..10")
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    class_name = f"ai_udf.Bc{digest}"
    simple = class_name.rsplit(".", 1)[1]
    java = _render(simple, arity, code, consts, _interpreter_body())
    return JavaEval(class_name=class_name, java_source=java, code=code, consts=consts)


def _encode(func_code: types.CodeType, ops: dict[str, int]) -> tuple[list[int], list[Any]]:
    kept: list[dis.Instruction] = []
    for ins in dis.get_instructions(func_code):
        if ins.opname in _SKIP:
            continue
        kept.append(ins)
    # RETURN_CONST and LOAD_FAST_LOAD_FAST expand to more than one op, so the
    # jump map is in pre-expansion indexes. A jump may land on a skipped op
    # (TO_BOOL); that resolves to the next kept instruction.
    expanded_pc: dict[int, int] = {}
    program: list[tuple[str, int]] = []
    names = list(func_code.co_varnames[: func_code.co_argcount])
    for origin, ins in enumerate(kept):
        expanded_pc[origin] = len(program)
        program.extend(_lower_one(ins, names, kept, kept[origin - 1] if origin else None))
    flat: list[int] = []
    for name, arg in program:
        # Jumps were recorded as pre-expansion indexes. Remap now.
        if name.startswith("OP_JUMP_"):
            arg = expanded_pc[arg]
        flat.append(ops[name])
        flat.append(arg)
    loaded: set[int] = set()
    for ins in kept:
        if ins.opname in {"LOAD_CONST", "RETURN_CONST"}:
            loaded.add(ins.arg)
    consts: list[Any] = []
    for i, val in enumerate(func_code.co_consts):
        consts.append(val if i in loaded else None)
        if i in loaded:
            _java_const(val)  # decline here, not at render time
    return flat, consts


def _lower_one(
    ins: dis.Instruction,
    names: list[str],
    kept: list[dis.Instruction],
    prev: Optional[dis.Instruction] = None,
) -> list[tuple[str, int]]:
    op = ins.opname
    if op == "LOAD_FAST":
        return [("OP_LOAD_FAST", _local(ins.argval, names))]
    if op == "LOAD_FAST_LOAD_FAST":
        if not isinstance(ins.argval, tuple):
            raise BackendDecline("LOAD_FAST_LOAD_FAST without name")
        return [("OP_LOAD_FAST", _local(nm, names)) for nm in ins.argval]
    if op == "LOAD_CONST":
        return [("OP_LOAD_CONST", ins.arg)]
    if op == "RETURN_CONST":
        return [("OP_LOAD_CONST", ins.arg), ("OP_RETURN", 0)]
    if op == "RETURN_VALUE":
        return [("OP_RETURN", 0)]
    if op == "POP_TOP":
        return [("OP_POP", 0)]
    if op == "COPY":
        return [("OP_COPY", ins.arg or 1)]
    if op == "SWAP":
        return [("OP_SWAP", ins.arg or 2)]
    if op == "BINARY_OP":
        name = _BIN.get(ins.argrepr)
        if name is None:
            raise BackendDecline(f"unsupported BINARY_OP {ins.argrepr!r}")
        return [(name, 0)]
    if op == "UNARY_NEGATIVE":
        return [("OP_UNARY_NEG", 0)]
    if op == "UNARY_NOT":
        return [("OP_UNARY_NOT", 0)]
    if op == "COMPARE_OP":
        raw = ins.argval if isinstance(ins.argval, str) else ins.argrepr
        name = _CMP.get(raw)
        if name is None:
            raise BackendDecline(f"unsupported COMPARE_OP {ins.argval!r}")
        return [(name, 0)]
    if op == "IS_OP":
        # Reference equality is only `x is None` semantics. The SQL backend
        # declines IS on non-None; match it unless TOS is provably the None
        # const.
        if prev is None or prev.opname != "LOAD_CONST" or prev.argval is not None:
            raise BackendDecline("IS_OP on non-None not supported")
        return [("OP_IS", 1 if ins.arg else 0)]
    if op in _JUMP:
        return [(_JUMP[op], _jump_origin(ins.argval, kept))]
    raise BackendDecline(f"unsupported bytecode: {op}")


def _jump_origin(target_offset: int, kept: list[dis.Instruction]) -> int:
    for i, ins in enumerate(kept):
        if ins.offset >= target_offset:
            return i
    raise BackendDecline(f"jump target {target_offset!r} out of range")


def _local(name: Any, names: list[str]) -> int:
    if not isinstance(name, str) or name not in names:
        raise BackendDecline(f"LOAD_FAST unknown var {name}")
    return names.index(name)


def _java_const(val: Any) -> str:
    if val is None:
        return "null"
    if val is True:
        return "Boolean.TRUE"
    if val is False:
        return "Boolean.FALSE"
    if isinstance(val, int):
        if val == -(2**63):
            return "Long.MIN_VALUE"
        if not -(2**63) <= val <= 2**63 - 1:
            raise BackendDecline("int const does not fit long")
        return f"Long.valueOf({val}L)"
    if isinstance(val, float):
        if val != val:
            return "Double.NaN"
        if val == float("inf"):
            return "Double.POSITIVE_INFINITY"
        if val == float("-inf"):
            return "Double.NEGATIVE_INFINITY"
        return f"Double.valueOf({val!r}d)"
    if isinstance(val, str):
        return _java_string(val)
    raise BackendDecline(f"unsupported const type {type(val).__name__}")


def _java_string(val: str) -> str:
    out = ['"']
    for ch in val:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) > 0xFFFF:
            cp = ord(ch) - 0x10000
            out.append(f"\\u{0xD800 + (cp >> 10):04x}\\u{0xDC00 + (cp & 0x3FF):04x}")
        elif ord(ch) < 32 or ord(ch) > 126:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _interpreter_body() -> str:
    text = _JAVA_PATH.read_text(encoding="utf-8")
    marker = "public final class BytecodeEval"
    idx = text.find(marker)
    if idx < 0:
        raise BackendDecline("BytecodeEval.java has no class")
    body = "static final class Eval" + text[idx + len(marker) :]
    return body.replace("private BytecodeEval()", "private Eval()")


def _render(simple: str, arity: int, code: list[int], consts: list[Any], interpreter: str) -> str:
    iface = f"UDF{arity}"
    params = ", ".join(f"Object a{i}" for i in range(arity))
    if arity == 1:
        invoke = "Eval.eval1(CODE, CONSTS, a0)"
    elif arity == 2:
        invoke = "Eval.eval2(CODE, CONSTS, a0, a1)"
    else:
        args = ", ".join(f"a{i}" for i in range(arity))
        invoke = f"Eval.evalN(CODE, CONSTS, new Object[] {{{args}}})"
    const_literals = ", ".join(_java_const(c) for c in consts)
    code_literals = ", ".join(str(n) for n in code)
    return (
        "package ai_udf;\n\n"
        f"import org.apache.spark.sql.api.java.{iface};\n\n"
        f"public class {simple} implements {iface}<{', '.join(['Object'] * (arity + 1))}> {{\n"
        f"{interpreter}\n"
        f"    private static final int[] CODE = {{{code_literals}}};\n"
        f"    private static final Object[] CONSTS = {{{const_literals}}};\n\n"
        "    @Override\n"
        f"    public Object call({params}) {{\n"
        f"        return {invoke};\n"
        "    }\n"
        "}\n"
    )
