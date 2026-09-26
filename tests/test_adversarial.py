# SPDX-License-Identifier: Apache-2.0
"""Adversarial pack: python/Spark lookalikes that bite, empirically grounded.

Every claim below was battery-tested against a real Spark master build:

- python `%` floors (sign of divisor); Spark `%` truncates toward zero, and
  pmod only matches python for POSITIVE divisors (pmod(7, -3) == 1 but
  7 % -3 == -2). Faithful: truncated remainder plus a sign correction.
- python `//` floors; Spark `div` truncates. floor(x / y) matches within the
  int32-ranged verify strategy (double division is exact below 2**53).
- python round is banker's; Spark round is half-up. bround is the banker's
  one, and the cast must be bigint: cast(bround(1e16) as int) raises under
  ANSI where python returns 10000000000000000. Past 2**63 the python result
  fits no bigint column at all -- out of contract for `-> int`, so
  verification skips the comparison there (a real python UDF execution
  would fail to fit the value into LongType too).
- Spark split takes a REGEX: split(s, '.') turns 'a.b' into ['', '', '', ''].
- python zfill never truncates and pads after a +/- sign; lpad does both
  wrong ('abcd' -> 'abc', '-5' -> '0-5').
- python weekday() is Monday=0; dayofweek is Sunday=1.
- python casefold is not lower ('ß'.casefold() == 'ss').
- str(float) differs from cast: '1e+16' vs '1.0E16', and str(None) is the
  string 'None', not NULL.
- python int() accepts underscores ('1_000' == 1000); casts reject them.
- python strip removes all whitespace; trim removes spaces only.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable, register_impl
from ai_udf_transpile.backends.fake import mod_bucket, round_half, split_dot, weekday_of, zfill3
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

MOD_FAITHFUL = (
    "(_udf_param_0 % _udf_param_1) + CASE WHEN (_udf_param_0 % _udf_param_1) <> 0 "
    "AND ((_udf_param_0 < 0) <> (_udf_param_1 < 0)) THEN _udf_param_1 ELSE 0 END"
)
ROUND_FAITHFUL = "cast(bround(_udf_param_0) as bigint)"
WEEKDAY_FAITHFUL = "pmod(dayofweek(_udf_param_0) + 5, 7)"
SPLIT_FAITHFUL = "split(_udf_param_0, '\\\\.')"
ZFILL_FAITHFUL = (
    "CASE WHEN length(_udf_param_0) >= 3 THEN _udf_param_0 "
    "WHEN substr(_udf_param_0, 1, 1) IN ('-', '+') "
    "THEN concat(substr(_udf_param_0, 1, 1), lpad(substr(_udf_param_0, 2), 2, '0')) "
    "ELSE lpad(_udf_param_0, 3, '0') END"
)


def casefold_lower(s: str) -> str:
    return s.casefold()


def float_str(x: float) -> str:
    return str(x)


def parse_int(s: str) -> int:
    return int(s)


def _check(func, sql, spark, input_types, return_type):
    return hypothesis_check(
        source_text=canonical_source_from_func(func),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=input_types,
        return_type=return_type,
        spark=spark,
        max_examples=20,
    )


# --- faithful rewrites pass -------------------------------------------------


def test_floored_modulo_passes(spark):
    ok, err = _check(mod_bucket, MOD_FAITHFUL, spark, ["bigint", "bigint"], "bigint")
    assert ok, err


def test_bankers_round_passes(spark):
    ok, err = _check(round_half, ROUND_FAITHFUL, spark, ["double"], "bigint")
    assert ok, err


def test_out_of_contract_round_result_not_a_mismatch(spark):
    # round(1e37) returns a 38-digit int no bigint column can hold: executing
    # the python UDF with the declared `-> int` would fail too, so the ANSI
    # overflow on the final cast is out of contract, not a mismatch. 1e37 is
    # in BUILTIN_DOUBLE_EXAMPLES, so test_bankers_round_passes exercises the
    # skip on every run; this pins the rule itself.
    from ai_udf_transpile.verify import _representable

    assert not _representable(10**37, "bigint")
    assert not _representable(-(2**63) - 1, "bigint")
    assert _representable(2**63 - 1, "bigint")
    assert not _representable(2**31, "int")
    assert _representable(10**37, "double")
    assert _representable(10**37, "string")
    assert _representable(True, "bigint")


def test_weekday_passes(spark):
    ok, err = _check(weekday_of, WEEKDAY_FAITHFUL, spark, ["timestamp"], "bigint")
    assert ok, err


def test_split_escaped_dot_passes(spark):
    ok, err = _check(split_dot, SPLIT_FAITHFUL, spark, ["string"], "array<string>")
    assert ok, err


def test_zfill_sign_aware_passes(spark):
    ok, err = _check(zfill3, ZFILL_FAITHFUL, spark, ["string"], "string")
    assert ok, err


# --- naive lookalikes rejected (all deterministic via built-in examples) ----


def test_naive_modulo_rejected(spark):
    # (-7, 3): python 2, truncated -1.
    ok, err = _check(mod_bucket, "_udf_param_0 % _udf_param_1", spark, ["bigint", "bigint"], "bigint")
    assert not ok
    assert "mismatch" in (err or "")


def test_pmod_rejected_for_negative_divisor(spark):
    # (7, -3): python -2 (sign of divisor), pmod 1 (always non-negative).
    ok, err = _check(mod_bucket, "pmod(_udf_param_0, _udf_param_1)", spark, ["bigint", "bigint"], "bigint")
    assert not ok
    assert "mismatch" in (err or "")


def test_naive_round_rejected(spark):
    # 2.5: python banker's 2, Spark half-up 3.
    ok, err = _check(round_half, "round(_udf_param_0)", spark, ["double"], "bigint")
    assert not ok
    assert "mismatch" in (err or "")


def test_bround_to_int_rejected_overflow(spark):
    # 1e16: python returns 10000000000000000; int32 cast raises under ANSI.
    ok, err = _check(round_half, "cast(bround(_udf_param_0) as int)", spark, ["double"], "bigint")
    assert not ok


def test_naive_dayofweek_rejected(spark):
    ok, err = _check(weekday_of, "dayofweek(_udf_param_0)", spark, ["timestamp"], "bigint")
    assert not ok
    assert "mismatch" in (err or "")


def test_naive_split_regex_dot_rejected(spark):
    # 'a.b' -> ['', '', '', ''] when the dot is treated as regex any-char.
    ok, err = _check(split_dot, "split(_udf_param_0, '.')", spark, ["string"], "array<string>")
    assert not ok
    assert "mismatch" in (err or "")


def test_naive_lpad_for_zfill_rejected(spark):
    # '-5': python '-05', lpad '0-5'.
    ok, err = _check(zfill3, "lpad(_udf_param_0, 3, '0')", spark, ["string"], "string")
    assert not ok
    assert "mismatch" in (err or "")


def test_lower_for_casefold_rejected(spark):
    # 'ß'.casefold() == 'ss' but lower('ß') == 'ß'.
    ok, err = _check(casefold_lower, "lower(_udf_param_0)", spark, ["string"], "string")
    assert not ok
    assert "mismatch" in (err or "")


def test_cast_for_float_str_rejected(spark):
    # None: python str(None) == 'None'; cast yields NULL. Also '1e+16' vs '1.0E16'.
    ok, err = _check(float_str, "cast(_udf_param_0 as string)", spark, ["double"], "string")
    assert not ok
    assert "mismatch" in (err or "")


def test_try_cast_for_int_rejected_underscore(spark):
    # '1_000': python 1000; try_cast NULL.
    ok, err = _check(parse_int, "try_cast(_udf_param_0 as int)", spark, ["string"], "bigint")
    assert not ok
    assert "mismatch" in (err or "")


def test_underscore_tolerant_parse_int_passes(spark):
    # python raises on malformed input (sql may do anything there), and
    # stripping underscores first matches python on '1_000'.
    ok, err = _check(
        parse_int, "try_cast(regexp_replace(_udf_param_0, '_', '') as int)", spark, ["string"], "bigint"
    )
    assert ok, err


# --- end to end through the fake backend ------------------------------------


def test_mod_bucket_end_to_end(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    rows = [(7, 3), (-7, 3), (7, -3), (-7, -3), (None, 3)]
    df = spark.createDataFrame(rows, "x bigint, y bigint")
    expected = [mod_bucket(a, b) if a is not None and b is not None else None for a, b in rows]
    # python: None % 3 raises; the UDF would error -- compare only non-null.
    expected = [a % b for a, b in rows if a is not None]
    df = df.filter("x is not null")

    first = UserDefinedFunction(mod_bucket, LongType())
    assert not first.transpiled
    got = [r[0] for r in df.select(first("x", "y")).collect()]
    assert got == expected

    deadline = time.time() + 60
    second = UserDefinedFunction(mod_bucket, LongType())
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(mod_bucket, LongType())
    assert second.transpiled, "fake backend never landed the floored-modulo rewrite"
    got = [r[0] for r in df.select(second("x", "y")).collect()]
    assert got == expected


def test_register_impl_round_half(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark, round_half, kind="catalyst", catalyst_sql="round(_udf_param_0)", return_type=LongType()
        )
    register_impl(spark, round_half, kind="catalyst", catalyst_sql=ROUND_FAITHFUL, return_type=LongType())
    row = get_catalog()._conn.execute("SELECT status, origin FROM cache").fetchone()
    assert tuple(row) == ("success", "human")
