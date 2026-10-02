# SPDX-License-Identifier: Apache-2.0
"""Prove the Hypothesis differential gate actually catches wrong rewrites.

Every test here would be a silent correctness bug if the gate were vacuous,
so each negative test asserts a specific error string, not just ``not ok``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from ai_udf_transpile.backends.fake import BACKWARDS_JAVA
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileResult
from ai_udf_transpile.verify import _strategy_for, hypothesis_check

pytestmark = pytest.mark.spark

PLUS_ONE = "def plus_one(x: int) -> int:\n    return x + 1\n"
IS_NONE_BRANCH = "def is_none_branch(x: int) -> int:\n    if x is None:\n        return -1\n    return x\n"
SAFE_TEN_DIV = "def safe_ten_div(x: int) -> int:\n    if x == 0:\n        return -1\n    return int(10 / x)\n"
BACKWARDS_PY = (
    "def backwards(name: str) -> str:\n    if name is None:\n        return None\n    return name[::-1]\n"
)
# Same shape as the real fixture but returns the input unchanged.
BACKWARDS_BAD_JAVA = """package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class BackwardsBad implements UDF1<Object, Object> {
    @Override
    public Object call(Object s) {
        return s;
    }
}
"""


def _check(source, result, input_types, return_type, spark, max_examples=50):
    return hypothesis_check(
        source_text=source,
        captures={},
        result=result,
        input_types=input_types,
        return_type=return_type,
        spark=spark,
        max_examples=max_examples,
    )


def _sql(sql: str) -> TranspileResult:
    return TranspileResult(kind=KIND_CATALYST, sql=sql)


def test_correct_sql_passes(spark):
    ok, err = _check(PLUS_ONE, _sql("_udf_param_0 + 1"), ["bigint"], "bigint", spark)
    assert ok, err


def test_python_raise_sql_null_is_allowed(spark):
    # plus_one raises TypeError on None; SQL returns NULL. Policy: allowed.
    ok, err = _check(PLUS_ONE, _sql("_udf_param_0 + 1"), ["bigint"], "bigint", spark)
    assert ok, err


def test_tolerance_accepts_a_small_numeric_gap(spark):
    src = "def ident(x: float) -> float:\n    return x\n"
    near = _sql("_udf_param_0 + 1e-8")
    ok, err = _check(src, near, ["double"], "double", spark, max_examples=3)
    assert not ok
    assert "mismatch" in (err or "")
    ok, err = hypothesis_check(
        source_text=src,
        captures={},
        result=near,
        input_types=["double"],
        return_type="double",
        spark=spark,
        max_examples=3,
        tolerance=1e-6,
    )
    assert ok, err


def test_wrong_sql_rejected(spark):
    ok, err = _check(PLUS_ONE, _sql("_udf_param_0 + 2"), ["bigint"], "bigint", spark)
    assert not ok
    assert "mismatch" in (err or "")


def test_null_semantics_mismatch_rejected(spark):
    # SQL drops the None branch: python returns -1, SQL returns NULL.
    ok, err = _check(IS_NONE_BRANCH, _sql("_udf_param_0"), ["bigint"], "bigint", spark)
    assert not ok
    assert "mismatch" in (err or "")
    assert "None" in (err or "")


def test_sql_raise_python_return_rejected(spark):
    # ANSI divide-by-zero raises where the Python UDF returns -1.
    ok, err = _check(SAFE_TEN_DIV, _sql("10 DIV _udf_param_0"), ["bigint"], "bigint", spark, 100)
    assert not ok
    assert "sql raised" in (err or "")


def test_matching_null_and_divide_passes(spark):
    ok, err = _check(
        SAFE_TEN_DIV,
        _sql(
            "CASE WHEN _udf_param_0 IS NULL THEN NULL "
            "WHEN _udf_param_0 = 0 THEN -1 ELSE 10 DIV _udf_param_0 END"
        ),
        ["bigint"],
        "bigint",
        spark,
        100,
    )
    assert ok, err


def test_java_udf_correct_passes(spark):
    ok, err = _check(
        BACKWARDS_PY,
        TranspileResult(kind=KIND_JAVA_UDF, java_source=BACKWARDS_JAVA, class_name="ai_udf.Backwards"),
        ["string"],
        "string",
        spark,
    )
    assert ok, err


def test_java_udf_wrong_impl_rejected(spark):
    ok, err = _check(
        BACKWARDS_PY,
        TranspileResult(
            kind=KIND_JAVA_UDF,
            java_source=BACKWARDS_BAD_JAVA,
            class_name="ai_udf.BackwardsBad",
        ),
        ["string"],
        "string",
        spark,
    )
    assert not ok
    assert "mismatch" in (err or "")


def test_java_udf_verify_requires_spark():
    ok, err = hypothesis_check(
        source_text=BACKWARDS_PY,
        captures={},
        result=TranspileResult(kind=KIND_JAVA_UDF, java_source=BACKWARDS_JAVA, class_name="ai_udf.Backwards"),
        input_types=["string"],
        return_type="string",
        spark=None,
    )
    assert not ok
    assert "requires Spark" in (err or "")


def test_strategies_generate_null_zero_and_text():
    from hypothesis import find

    # find() raises NoSuchExample if the value cannot be generated.
    assert find(_strategy_for("string"), lambda v: v is None) is None
    assert find(_strategy_for("string"), lambda v: isinstance(v, str) and len(v) > 3)
    assert find(_strategy_for("bigint"), lambda v: v == 0) == 0
    assert find(_strategy_for("boolean"), lambda v: v is False) is False
