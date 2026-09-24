# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable
from ai_udf_transpile.backends.fake import FakeBackend, always_decline, plus_one
from ai_udf_transpile.catalog import WAIT
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check
from ai_udf_transpile.worker import poll_once

pytestmark = pytest.mark.spark


def test_fake_plus_one_end_to_end(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    first = UserDefinedFunction(plus_one, LongType())
    assert not first.transpiled
    catalog = get_catalog()
    assert catalog.count() == 1
    assert poll_once(catalog, FakeBackend(), spark)
    row = catalog.oldest_pending()
    assert row is None
    # fetch the success row
    cur = catalog._conn.execute("SELECT udf_key, status FROM cache")
    key, status = cur.fetchone()
    assert status == "success"
    second = UserDefinedFunction(plus_one, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(1,), (None,)], ["x"])
    # None + 1: python raises, sql returns null — collect the non-null row
    assert df.filter("x is not null").select(second("x")).collect()[0][0] == 2


def test_always_decline_cooldown_does_not_reinsert(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(always_decline, LongType())
    catalog = get_catalog()
    assert catalog.count() == 1
    assert poll_once(catalog, FakeBackend(), spark)
    row_key = catalog._conn.execute("SELECT udf_key FROM cache").fetchone()[0]
    row = catalog.get(row_key)
    assert row.status == "failed"
    UserDefinedFunction(always_decline, LongType())
    assert catalog.count() == 1
    kind, again = catalog.lookup(row_key)
    assert kind == WAIT
    assert again.status == "failed"


def test_wrong_sql_fails_hypothesis(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    from ai_udf_transpile.keys import canonical_source_from_func

    src = canonical_source_from_func(plus_one)
    ok, err = hypothesis_check(
        source_text=src,
        captures={},
        result=TranspileResult(kind="catalyst", sql="_udf_param_0 * 99"),
        input_types=["bigint"],
        return_type="bigint",
        spark=spark,
        max_examples=8,
        func=plus_one,
    )
    assert ok is False
    assert err
