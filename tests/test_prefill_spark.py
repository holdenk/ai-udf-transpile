# SPDX-License-Identifier: Apache-2.0
"""Prefill against a real classic SparkSession.

The point of a pre-fill is that the job's own UDF construction later finds
the row: these tests pre-fill a script, then import that same script and
check its ``@udf`` is served the cached rewrite without a new pending row.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import uuid

import pytest

from ai_udf_transpile import conf, enable
from ai_udf_transpile.backends.fake import FakeBackend
from ai_udf_transpile.catalog import HIT
from ai_udf_transpile.catalog.writeback import WritebackCatalog
from ai_udf_transpile.prefill import FileTracker, main, prefill_paths

# No module-level importorskip: nothing below imports pyspark, and the
# `spark` fixture owns the rule (fail when SPARK_HOME is set but PySpark is
# unusable, skip when Spark is not configured at all).

pytestmark = pytest.mark.spark

JOB = textwrap.dedent(
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import LongType


    @udf(returnType=LongType())
    def plus_one(x: int) -> int:
        return x + 1
    """
)


def _job_dir(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "prefill_job.py").write_text(JOB)
    return root


def _import_job(path, monkeypatch):
    name = f"prefill_job_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _ai_only(spark):
    # Keep Spark's built-in transpiler out, so a hit is the cache's doing.
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")


def _prefill(spark, catalog, root, sqlite_path):
    tracker = FileTracker(sqlite_path)
    try:
        return prefill_paths(
            [str(root)],
            catalog=catalog,
            backend=FakeBackend(),
            spark=spark,
            spark_version=str(spark.version),
            tracker=tracker,
            log=lambda _line: None,
        )
    finally:
        tracker.close()


def test_decorated_udf_engages_the_hook(spark, sqlite_path, tmp_path, monkeypatch):
    """Before the decorator strip, `udf` was a capture and every @udf declined."""
    catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    _ai_only(spark)
    _import_job(_job_dir(tmp_path) / "prefill_job.py", monkeypatch)
    assert catalog.count() == 1
    row = catalog.oldest_pending()
    assert row.source_text.startswith("def plus_one(x: int) -> int:")


def test_prefilled_row_serves_the_jobs_own_udf(spark, sqlite_path, tmp_path, monkeypatch):
    catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    _ai_only(spark)
    root = _job_dir(tmp_path)
    report = _prefill(spark, catalog, root, sqlite_path)
    assert [o.outcome for o in report.outcomes] == ["transpiled"]
    key = report.outcomes[0].udf_key
    assert catalog.count() == 1

    module = _import_job(root / "prefill_job.py", monkeypatch)
    assert module.plus_one._unwrapped.transpiled, "the job's @udf must hit the pre-filled row"
    assert catalog.count() == 1, "the runtime computed a different key than the pre-fill"
    assert catalog.get(key).status == "success"
    df = spark.createDataFrame([(1,), (41,)], ["x"])
    assert [r[0] for r in df.select(module.plus_one("x")).collect()] == [2, 42]


def test_prefill_flushes_writeback_below_threshold(spark, sqlite_path, tmp_path):
    table = f"default.prefill_wb_{uuid.uuid4().hex[:12]}"
    try:
        conf.set_value(conf.WRITEBACK_TABLE, table, spark)
        conf.set_value(conf.WRITEBACK_FORMAT, "parquet", spark)
        conf.set_value(conf.WRITEBACK_THRESHOLD, "32", spark)
        catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
        assert isinstance(catalog, WritebackCatalog)
        _ai_only(spark)
        report = _prefill(spark, catalog, _job_dir(tmp_path), sqlite_path)
        key = report.outcomes[0].udf_key
        assert spark.sql(f"SELECT COUNT(*) FROM {table}").collect()[0][0] == 0
        assert catalog.flush() == 1
        assert spark.sql(f"SELECT COUNT(*) FROM {table}").collect()[0][0] == 1
        assert catalog.flush() == 0

        # A fresh process (new SQLite file) is served from the shared table.
        fresh = enable(spark, sqlite_path=str(tmp_path / "fresh.sqlite"), backend="fake", inline_worker=False)
        kind, row = fresh.lookup(key, spark=spark)
        assert kind == HIT and row.catalyst_sql == "_udf_param_0 + 1"
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table}")


def test_cli_round_trip(spark, sqlite_path, tmp_path, capsys):
    root = _job_dir(tmp_path)
    argv = [
        str(root),
        "--sqlite-path",
        sqlite_path,
        "--backend",
        "fake",
        "--conf",
        "spark.sql.experimental.optimizer.pyTranspilers=ai",
    ]
    assert main(argv) == 0
    assert "1 transpiled" in capsys.readouterr().out
    assert main(argv) == 0
    assert "1 unchanged" in capsys.readouterr().out
