# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable
from ai_udf_transpile.backends.fake import plus_one
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.worker import inline_thread_alive

pytestmark = pytest.mark.spark


def test_inline_worker_fills_cache(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    assert inline_thread_alive()
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    first = UserDefinedFunction(plus_one, LongType())
    assert not first.transpiled
    catalog = get_catalog()
    deadline = time.time() + 90
    row = None
    while time.time() < deadline:
        rows = catalog._conn.execute("SELECT udf_key, status FROM cache").fetchall()
        if rows and rows[0][1] == "success":
            row = rows[0]
            break
        time.sleep(0.5)
    assert row is not None, "inline worker never produced a success row"
    second = UserDefinedFunction(plus_one, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(5,)], ["x"])
    assert df.select(second("x")).collect()[0][0] == 6
