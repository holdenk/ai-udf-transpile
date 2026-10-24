# SPDX-License-Identifier: Apache-2.0
"""End-to-end Spark tests for the bytecode backend."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import BooleanType, LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, shutdown
from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.backends.fake import both_positive, greet, is_none_branch, mod_bucket, plus_one
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check
from ai_udf_transpile.worker import poll_once

pytestmark = pytest.mark.spark


def _transpile(spark, sqlite_path, func, rtype, input_types=None):
    """Run the bytecode backend on one UDF and return the catalog row."""
    enable(spark, sqlite_path=sqlite_path, backend="bytecode", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    first = UserDefinedFunction(func, rtype)
    assert not first.transpiled
    catalog = get_catalog()
    assert catalog.count() == 1
    assert poll_once(catalog, BytecodeBackend(), spark)
    cur = catalog._conn.execute("SELECT udf_key, status, catalyst_sql FROM cache")
    key, status, sql = cur.fetchone()
    assert status == "success", f"expected success, got {status}"
    return sql


def test_plus_one_end_to_end(spark, sqlite_path):
    sql = _transpile(spark, sqlite_path, plus_one, LongType())
    assert sql == "(_udf_param_0 + 1)"
    second = UserDefinedFunction(plus_one, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(1,), (41,)], ["x"])
    assert [r[0] for r in df.select(second("x")).collect()] == [2, 42]
    shutdown()


def test_is_none_branch_end_to_end(spark, sqlite_path):
    sql = _transpile(spark, sqlite_path, is_none_branch, LongType())
    assert "CASE WHEN" in sql and "IS NULL" in sql
    second = UserDefinedFunction(is_none_branch, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(5,), (None,)], ["x"])
    assert [r[0] for r in df.select(second("x")).collect()] == [5, -1]
    shutdown()


def test_greet_hypothesis_verified(spark, sqlite_path):
    # greet raises on None, and the strings here really are text.
    conf.set_value(conf.PARTY_ON_STRINGS, "true")
    # Bytecode emits 'hi ' + name; Hypothesis must confirm it matches python.
    sql = _transpile(spark, sqlite_path, greet, StringType(), input_types=["string"])
    assert sql == "concat('hi ', _udf_param_0)"
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(greet),
        captures={},
        result=TranspileResult(kind="catalyst", sql=sql),
        input_types=["string"],
        return_type="string",
        spark=spark,
        max_examples=20,
        func=greet,
        party_on_strings=True,
    )
    assert ok, f"bytecode greet SQL failed Hypothesis: {err}"
    shutdown()


def test_both_positive_hypothesis_verified(spark, sqlite_path):
    sql = _transpile(spark, sqlite_path, both_positive, BooleanType(), input_types=["bigint", "bigint"])
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(both_positive),
        captures={},
        result=TranspileResult(kind="catalyst", sql=sql),
        input_types=["bigint", "bigint"],
        return_type="boolean",
        spark=spark,
        max_examples=50,
        func=both_positive,
    )
    assert ok, f"bytecode both_positive SQL failed Hypothesis: {err}"
    shutdown()


def test_mod_bucket_hypothesis_verified(spark, sqlite_path):
    sql = _transpile(spark, sqlite_path, mod_bucket, LongType(), input_types=["bigint", "bigint"])
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(mod_bucket),
        captures={},
        result=TranspileResult(kind="catalyst", sql=sql),
        input_types=["bigint", "bigint"],
        return_type="bigint",
        spark=spark,
        max_examples=50,
        func=mod_bucket,
    )
    assert ok, f"bytecode mod_bucket SQL failed Hypothesis: {err}"
    shutdown()
