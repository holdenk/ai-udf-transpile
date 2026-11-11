# SPDX-License-Identifier: Apache-2.0
"""Rust target against a real Comet: build, register, verify, serve.

Runs only where the session has Comet live (CI's comet child, or locally
with AI_UDF_TRANSPILE_TEST_COMET_JAR), cargo is on PATH, and a Comet checkout
($COMET_HOME) matching the jar provides comet-udf-sdk. Skips cleanly anywhere
else; the JVM-free half of the target is tests/test_rust_target.py.
"""

from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, register_impl, rustc
from ai_udf_transpile.backends.fake import (
    BACKWARDS_RS,
    COLLATZ_STEPS_RS,
    PLUS_ONE_RS,
    FakeBackend,
    backwards_rust,
    collatz_steps,
    plus_one_rust,
)
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_RUST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check
from ai_udf_transpile.worker import poll_once

pytestmark = pytest.mark.spark

_TARGETS = "catalyst,java_udf,rust"


_COMET_SHUFFLE = "spark.comet.shuffle.enabled"


@pytest.fixture
def comet(spark):
    """The shared session with Comet's extension live, for the test's duration.

    conftest's Comet session keeps Spark's shuffle manager, and under it
    Comet's extension turns itself off unless Comet shuffle is explicitly
    disabled. That is a runtime conf, so it is flipped here and put back:
    the rest of the suite keeps running exactly the session it always has.
    """
    if "CometPlugin" not in (spark.conf.get("spark.plugins", "") or ""):
        pytest.skip("needs a Comet session (AI_UDF_TRANSPILE_TEST_COMET_JAR)")
    if not shutil.which("cargo"):
        pytest.skip("needs cargo")
    prior = spark.conf.get(_COMET_SHUFFLE, None)
    spark.conf.set(_COMET_SHUFFLE, "false")
    try:
        reason = rustc.comet_unavailable(spark)
        if reason:
            pytest.skip(f"needs a live Comet session: {reason}")
        try:
            rustc.resolve_sdk(spark)
        except rustc.RustUnavailable as exc:
            pytest.skip(f"needs comet-udf-sdk: {exc}")
        yield spark
    finally:
        if prior is None:
            spark.conf.unset(_COMET_SHUFFLE)
        else:
            spark.conf.set(_COMET_SHUFFLE, prior)


@pytest.fixture
def rust_session(comet, sqlite_path):
    """enable() with the rust target opted in and the worker driven by hand."""
    enable(comet, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    comet.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.TARGETS, _TARGETS)
    conf.set_value(conf.PARTY_ON_STRINGS, "true")
    override = os.environ.get("AI_UDF_TRANSPILE_TEST_RUST_BUILD_DIR")
    if override:
        conf.set_value(conf.RUST_BUILD_DIR, override)
    yield comet
    comet.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "catalyst,ai")


def _parquet(spark, tmp_path, rows, schema):
    """Comet scans Parquet natively, so the operator above it is Comet's too."""
    path = str(tmp_path / "data")
    spark.createDataFrame(rows, schema).write.mode("overwrite").parquet(path)
    return spark.read.parquet(path)


def _plan(df) -> str:
    return df._jdf.queryExecution().executedPlan().toString()


def _status(catalog) -> tuple:
    return tuple(catalog._conn.execute("SELECT status, target_kind, impl_entry, error FROM cache").fetchone())


def test_register_and_call_natively(rust_session, tmp_path):
    spark = rust_session
    compiled = rustc.compile_rust(spark, BACKWARDS_RS, ["string"], "string")
    name = rustc.register_rust_udf(spark, compiled, ["string"], "string")
    df = _parquet(spark, tmp_path, [("héllo 🎉",), (None,), ("",)], "s string")
    out = df.selectExpr(f"{rustc.call_expr(name, ['string']).replace('_udf_param_0', 's')} AS r")
    assert "CometProject" in _plan(out)
    assert sorted(out.collect(), key=lambda r: (r[0] is None, r[0])) == [("",), ("🎉 olléh",), (None,)]


def test_int_column_into_bigint_parameter(rust_session, tmp_path):
    # Comet's signature check inserts no casts; call_expr's CAST does.
    spark = rust_session
    compiled = rustc.compile_rust(spark, PLUS_ONE_RS, ["bigint"], "bigint")
    name = rustc.register_rust_udf(spark, compiled, ["bigint"], "bigint")
    df = _parquet(spark, tmp_path, [(1,), (41,)], "i int")
    call = rustc.call_expr(name, ["bigint"]).replace("_udf_param_0", "i")
    assert sorted(r[0] for r in df.selectExpr(f"{call} AS r").collect()) == [2, 42]


def test_overflow_is_a_query_error_not_a_wrapped_value(rust_session, tmp_path):
    spark = rust_session
    # A bare `+` instead of the fixture's checked_add: the per-package
    # overflow checks turn the wrap into a panic, which the SDK turns into
    # a query error.
    source = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(|v| v + 1)) }\n"
    compiled = rustc.compile_rust(spark, source, ["bigint"], "bigint")
    name = rustc.register_rust_udf(spark, compiled, ["bigint"], "bigint")
    df = _parquet(spark, tmp_path, [(2**63 - 1,)], "x bigint")
    with pytest.raises(Exception, match="overflow"):
        df.selectExpr(f"{name}(x) AS r").collect()


def test_hypothesis_accepts_a_faithful_rewrite(rust_session):
    result = TranspileResult(kind=KIND_RUST, rust_source=COLLATZ_STEPS_RS)
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(collatz_steps),
        captures={},
        result=result,
        input_types=["bigint"],
        return_type="bigint",
        spark=rust_session,
        max_examples=8,
        func=collatz_steps,
    )
    assert (ok, err) == (True, None)
    assert result.entry and result.entry.startswith("ai_udf_rs_")
    # The verifier's temporary Comet conversion switch is put back.
    assert rust_session.conf.get("spark.comet.convert.rdd.enabled", "unset") in ("unset", "false")


def test_hypothesis_rejects_a_wrong_rewrite(rust_session):
    wrong = PLUS_ONE_RS.replace("checked_add(1)", "checked_add(2)")
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(plus_one_rust),
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source=wrong),
        input_types=["bigint"],
        return_type="bigint",
        spark=rust_session,
        max_examples=8,
        func=plus_one_rust,
    )
    assert ok is False and "mismatch" in err


def test_hypothesis_rejects_code_points_counted_as_bytes(rust_session):
    # The classic: reversing bytes instead of chars. Only non-ASCII input
    # tells them apart, which party_on_strings generates.
    wrong = (
        "pub fn udf(name: Option<&str>) -> Result<Option<String>, String> {\n"
        "    Ok(name.map(|s| String::from_utf8_lossy(&s.bytes().rev().collect::<Vec<u8>>()).into_owned()))\n"
        "}\n"
    )
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(backwards_rust),
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source=wrong),
        input_types=["string"],
        return_type="string",
        spark=rust_session,
        max_examples=8,
        func=backwards_rust,
        party_on_strings=True,
    )
    assert ok is False and "mismatch" in err


def test_compile_failure_fails_verification_with_rustc_output(rust_session):
    broken = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(|v| v + nope)) }\n"
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(plus_one_rust),
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source=broken),
        input_types=["bigint"],
        return_type="bigint",
        spark=rust_session,
        func=plus_one_rust,
    )
    assert ok is False
    assert "rust verify setup failed" in err and "cannot find value `nope`" in err


def test_end_to_end_miss_then_native_hit(rust_session, tmp_path):
    spark = rust_session
    first = UserDefinedFunction(collatz_steps, LongType())
    assert not first.transpiled  # a miss: pending row, Python UDF
    catalog = get_catalog()
    assert poll_once(catalog, FakeBackend(), spark)
    status, kind, entry, error = _status(catalog)
    assert (status, kind, error) == ("success", "rust", None)
    assert entry.startswith("ai_udf_rs_")

    second = UserDefinedFunction(collatz_steps, LongType())
    assert second.transpiled and entry in str(second.transpiled[0])
    df = _parquet(spark, tmp_path, [(n,) for n in [1, 2, 3, 6, 7, 27, 97, 871]], "n bigint")
    out = df.select("n", second("n").alias("steps"))
    plan = _plan(out)
    assert "CometProject" in plan and entry in plan and "PythonUDF" not in plan
    assert dict(out.collect()) == {n: collatz_steps(n) for n in [1, 2, 3, 6, 7, 27, 97, 871]}


def test_end_to_end_string_udf(rust_session, tmp_path):
    spark = rust_session
    assert not UserDefinedFunction(backwards_rust, StringType()).transpiled
    assert poll_once(get_catalog(), FakeBackend(), spark)
    assert _status(get_catalog())[:2] == ("success", "rust")
    served = UserDefinedFunction(backwards_rust, StringType())
    df = _parquet(spark, tmp_path, [("abc",), (None,), ("añb",)], "s string")
    assert {r[0] for r in df.select(served("s")).collect()} == {"cba", None, "bña"}


def test_target_off_means_no_rust_is_asked_for_or_served(comet, sqlite_path, tmp_path):
    # Same UDF, same Comet session, rust NOT in targets: the fake's only
    # fixture is Rust, so it declines, and the UDF stays interpreted Python.
    enable(comet, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    comet.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    try:
        assert not UserDefinedFunction(collatz_steps, LongType()).transpiled
        assert poll_once(get_catalog(), FakeBackend(), comet)
        status, kind, _entry, error = _status(get_catalog())
        assert status == "failed" and kind is None and "declined" in error
        assert not UserDefinedFunction(collatz_steps, LongType()).transpiled
    finally:
        comet.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "catalyst,ai")


def test_served_row_goes_back_to_python_when_the_target_is_switched_off(rust_session):
    spark = rust_session
    UserDefinedFunction(plus_one_rust, LongType())
    assert poll_once(get_catalog(), FakeBackend(), spark)
    assert UserDefinedFunction(plus_one_rust, LongType()).transpiled
    conf.set_value(conf.TARGETS, "catalyst,java_udf")
    assert not UserDefinedFunction(plus_one_rust, LongType()).transpiled


def test_register_impl_rust(rust_session, tmp_path):
    spark = rust_session

    def triple(x: int) -> int:
        return x * 3

    source = (
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {\n"
        "    match x {\n"
        '        Some(v) => v.checked_mul(3).map(Some).ok_or_else(|| "overflow".to_string()),\n'
        "        None => Ok(None),\n"
        "    }\n"
        "}\n"
    )
    register_impl(spark, triple, kind="rust", impl_source=source, return_type=LongType())
    row = get_catalog().get(_only_key(get_catalog()))
    assert (row.status, row.target_kind, row.origin) == ("success", "rust", "human")
    served = UserDefinedFunction(triple, LongType())
    assert served.transpiled
    df = _parquet(spark, tmp_path, [(2,), (None,)], "x bigint")
    assert {r[0] for r in df.select(served("x")).collect()} == {6, None}


def test_register_impl_rust_rejects_a_wrong_rewrite(rust_session):
    def double(x: int) -> int:
        return x * 2

    source = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(|v| v / 2)) }\n"
    with pytest.raises(ValueError, match="Hypothesis failed"):
        register_impl(rust_session, double, kind="rust", impl_source=source, return_type=LongType())


def test_register_impl_rust_needs_the_target(comet, sqlite_path):
    enable(comet, sqlite_path=sqlite_path, backend="fake", inline_worker=False)

    def same(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="cannot run in this session: rust is not in"):
        register_impl(comet, same, kind="rust", impl_source=PLUS_ONE_RS, return_type=LongType())


def _only_key(catalog) -> str:
    return catalog._conn.execute("SELECT udf_key FROM cache").fetchone()[0]
