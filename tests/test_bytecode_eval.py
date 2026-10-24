# SPDX-License-Identifier: Apache-2.0
"""Java interpreter for the bytecode subset. No Spark required for the encoder."""

from __future__ import annotations

import pytest

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode_eval import emit_java_eval, emit_java_eval_source, opcode_numbers
from ai_udf_transpile.backends.fake import (
    both_positive,
    greet,
    is_none_branch,
    mod_bucket,
    plus_one,
    upper_useragent,
)


def test_off_313_declines_fail_closed(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "version_info", (3, 11, 0))
    with pytest.raises(BackendDecline, match="3.13"):
        emit_java_eval(plus_one)


def test_opcode_numbers_match_the_java_source():
    ops = opcode_numbers()
    assert ops["OP_LOAD_FAST"] == 1
    assert ops["OP_RETURN"] == 21
    assert ops["OP_SWAP"] == 24


def test_plus_one_program_is_load_add_return():
    ops = opcode_numbers()
    emitted = emit_java_eval(plus_one)
    code = emitted.code
    assert code[:2] == [ops["OP_LOAD_FAST"], 0]
    assert code[2:4] == [ops["OP_LOAD_CONST"], 1]
    assert code[4:6] == [ops["OP_BIN_ADD"], 0]
    assert code[6:8] == [ops["OP_RETURN"], 0]
    assert "Eval.eval1" in emitted.java_source
    assert "static final class Eval" in emitted.java_source
    assert emitted.class_name.startswith("ai_udf.Bc")


def test_is_none_branch_jumps():
    ops = opcode_numbers()
    code = emit_java_eval(is_none_branch).code
    assert ops["OP_JUMP_IF_NOT_NONE"] in code


def test_load_attr_declined():
    with pytest.raises(BackendDecline, match="LOAD_ATTR"):
        emit_java_eval(upper_useragent)


def test_is_op_on_non_none_declined():
    src = "def f(x: int, y: int) -> bool:\n    return x is y\n"
    with pytest.raises(BackendDecline, match="IS_OP"):
        emit_java_eval_source(src)


def test_is_none_still_lowers():
    src = "def f(x: int) -> bool:\n    return x is None\n"
    code = emit_java_eval_source(src).code
    assert opcode_numbers()["OP_IS"] in code


def test_is_op_polarity():
    ops = opcode_numbers()
    is_code = emit_java_eval_source("def f(x: int) -> bool:\n    return x is None\n").code
    is_not_code = emit_java_eval_source("def f(x: int) -> bool:\n    return x is not None\n").code
    # Java: arg == 0 ? same : !same. Flipping this flips every `is not None`.
    assert is_code[is_code.index(ops["OP_IS"]) + 1] == 0
    assert is_not_code[is_not_code.index(ops["OP_IS"]) + 1] == 1


def test_jump_landing_on_to_bool_skips_forward():
    src = "def not_and(a: bool, b: bool) -> bool:\n    return not (a and b)\n"
    ops = opcode_numbers()
    code = emit_java_eval_source(src).code
    # COPY, then a jump that in CPython targets TO_BOOL. That op is skipped,
    # so the jump lands on UNARY_NOT.
    jump_at = code.index(ops["OP_JUMP_IF_FALSE"])
    target = code[jump_at + 1]
    assert code[target * 2] == ops["OP_UNARY_NOT"]


def test_non_bmp_const_is_a_surrogate_pair():
    src = 'def f(s: str) -> str:\n    return s + "\U0001f600"\n'
    java = emit_java_eval_source(src).java_source
    assert "\\ud83d\\ude00" in java


def test_floor_div_declined():
    def f(x: int, y: int) -> int:
        return x // y

    with pytest.raises(BackendDecline, match="BINARY_OP"):
        emit_java_eval(f)


@pytest.mark.spark
def test_java_eval_matches_python(spark):
    from pyspark.sql.types import BooleanType, LongType, StringType

    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    cases = [
        (plus_one, LongType(), "x long", [(1,), (41,), (0,)], lambda r: r[0] + 1),
        (
            is_none_branch,
            LongType(),
            "x long",
            [(5,), (None,)],
            lambda r: -1 if r[0] is None else r[0],
        ),
        (
            both_positive,
            BooleanType(),
            "a long, b long",
            [(1, 2), (-1, 2), (1, -2), (0, 1)],
            lambda r: r[0] is not None and r[1] is not None and r[0] > 0 and r[1] > 0,
        ),
        (greet, StringType(), "s string", [("holden",), ("",)], lambda r: "hi " + r[0]),
        (
            mod_bucket,
            LongType(),
            "value long, modulus long",
            [(10, 3), (7, 100), (-7, 3), (7, -3), (-7, -3)],
            lambda r: r[0] % r[1],
        ),
    ]
    for func, rtype, schema, rows, expect in cases:
        emitted = emit_java_eval(func)
        compiled = compile_java(spark, emitted.java_source, emitted.class_name)
        name = verify_function_name("bce", compiled.jar_bytes)
        register_java_udf(spark, name, compiled.class_name, compiled.jar_bytes, rtype, janino=compiled.janino)
        df = spark.createDataFrame(rows, schema)
        cols = ", ".join(schema.split(", ")[i].split()[0] for i in range(len(rows[0])))
        got = [r[0] for r in df.selectExpr(f"{name}({cols}) AS r").collect()]
        assert got == [expect(r) for r in rows], func.__name__


@pytest.mark.spark
def test_java_eval_branches(spark):
    from pyspark.sql.types import BooleanType, LongType

    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    def nested(x: int, y: int) -> int:
        if x > 0:
            if y > 0:
                return 1
            return 2
        return 3

    def chained(x: int) -> bool:
        return 0 < x < 10

    def not_and(a: bool, b: bool) -> bool:
        return not (a and b)

    specs = [
        (nested, LongType(), "x long, y long", [(1, 1), (1, -1), (-1, 1)], lambda r: nested(*r)),
        (chained, BooleanType(), "x long", [(5,), (0,), (10,), (-1,)], lambda r: chained(*r)),
        (
            not_and,
            BooleanType(),
            "a boolean, b boolean",
            [(True, True), (True, False), (False, True)],
            lambda r: not_and(*r),
        ),
    ]
    for func, rtype, schema, rows, expect in specs:
        emitted = emit_java_eval(func)
        compiled = compile_java(spark, emitted.java_source, emitted.class_name)
        name = verify_function_name("bce", compiled.jar_bytes)
        register_java_udf(spark, name, compiled.class_name, compiled.jar_bytes, rtype, janino=compiled.janino)
        df = spark.createDataFrame(rows, schema)
        cols = ", ".join(part.split()[0] for part in schema.split(", "))
        got = [r[0] for r in df.selectExpr(f"{name}({cols}) AS r").collect()]
        assert got == [expect(r) for r in rows], func.__name__


@pytest.mark.spark
def test_java_eval_long_double_past_mantissa(spark):
    from pyspark.sql.types import BooleanType

    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    def eq(a: int, b: float) -> bool:
        return a == b

    def gt(a: int, b: float) -> bool:
        return a > b

    big = 1 << 53
    rows = [(big + 1, float(big)), (big, float(big)), (1, 1.5)]
    for func, expect in (
        (eq, [False, True, False]),
        (gt, [True, False, False]),
    ):
        emitted = emit_java_eval(func)
        compiled = compile_java(spark, emitted.java_source, emitted.class_name)
        name = verify_function_name("bce", compiled.jar_bytes)
        register_java_udf(
            spark, name, compiled.class_name, compiled.jar_bytes, BooleanType(), janino=compiled.janino
        )
        df = spark.createDataFrame(rows, "a long, b double")
        got = [r[0] for r in df.selectExpr(f"{name}(a, b) AS r").collect()]
        assert got == expect, func.__name__
