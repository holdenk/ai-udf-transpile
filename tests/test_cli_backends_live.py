# SPDX-License-Identifier: Apache-2.0
"""Live end-to-end tests against the real AI CLI backends (coco/cursor/claude).

These never run in CI: they require AI_UDF_LIVE=1 and the backend binary on
PATH with working auth. Run locally with:

    AI_UDF_LIVE=1 SPARK_HOME=~/spark .venv/bin/pytest tests/test_cli_backends_live.py -q
"""

from __future__ import annotations

import os
import shutil
import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable
from ai_udf_transpile.backends.fake import upper_useragent  # importable by executors
from ai_udf_transpile.transpiler import get_catalog

LIVE = os.environ.get("AI_UDF_LIVE") == "1"

pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(not LIVE, reason="set AI_UDF_LIVE=1 to run real CLI backends"),
]

BACKENDS = {
    "coco": conf.DEFAULTS[conf.BINARY_COCO],
    "cursor": conf.DEFAULTS[conf.BINARY_CURSOR],
    "claude": conf.DEFAULTS[conf.BINARY_CLAUDE],
}


def plus_one(x: int) -> int:
    return x + 1


def greet(name: str) -> str:
    return "hi " + name


def _wait_success(catalog, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = catalog._conn.execute("SELECT status, target_kind, error FROM cache").fetchall()
        if rows and rows[0][0] == "success":
            return rows[0]
        if rows and rows[0][0] == "failed":
            pytest.fail(f"backend row failed: {rows[0][2]}")
        time.sleep(2)
    pytest.fail("backend never produced a success row")


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_live_backend_plus_one(spark, sqlite_path, backend):
    if shutil.which(BACKENDS[backend]) is None:
        pytest.skip(f"{BACKENDS[backend]} not on PATH")
    conf.set_value(conf.CLI_TIMEOUT, "600")
    enable(spark, sqlite_path=sqlite_path, backend=backend, inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    first = UserDefinedFunction(plus_one, LongType())
    assert not first.transpiled
    row = _wait_success(get_catalog())
    assert row[1] in {"catalyst", "java_udf"}

    second = UserDefinedFunction(plus_one, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(5,), (None,)], ["x"])
    assert [r[0] for r in df.select(second("x")).collect()] == [6, None]


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_live_backend_greet(spark, sqlite_path, backend):
    if shutil.which(BACKENDS[backend]) is None:
        pytest.skip(f"{BACKENDS[backend]} not on PATH")
    conf.set_value(conf.CLI_TIMEOUT, "600")
    enable(spark, sqlite_path=sqlite_path, backend=backend, inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    first = UserDefinedFunction(greet, StringType())
    assert not first.transpiled
    _wait_success(get_catalog())

    second = UserDefinedFunction(greet, StringType())
    assert second.transpiled
    df = spark.createDataFrame([("bo",), (None,)], ["name"])
    assert [r[0] for r in df.select(second("name")).collect()] == ["hi bo", None]


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_live_backend_upper_useragent(spark, sqlite_path, backend):
    """SPARK-21935's UDF against a real backend: upper(col) must verify."""
    if shutil.which(BACKENDS[backend]) is None:
        pytest.skip(f"{BACKENDS[backend]} not on PATH")
    conf.set_value(conf.CLI_TIMEOUT, "600")
    enable(spark, sqlite_path=sqlite_path, backend=backend, inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "curl/7.68.0"]
    df = spark.createDataFrame([(a,) for a in agents], ["ua"])
    first = UserDefinedFunction(upper_useragent, StringType())
    assert not first.transpiled
    assert [r[0] for r in df.select(first("ua")).collect()] == [a.upper() for a in agents]

    _wait_success(get_catalog())
    second = UserDefinedFunction(upper_useragent, StringType())
    assert second.transpiled
    assert [r[0] for r in df.select(second("ua")).collect()] == [a.upper() for a in agents]
