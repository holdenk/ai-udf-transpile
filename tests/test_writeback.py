# SPDX-License-Identifier: Apache-2.0
"""Write-back hybrid catalog: SQLite staging + shared parquet/iceberg table.

The shared table is append-only (parquet has no row-level UPDATE), carries
verified successes only, and every interaction with it is fail-open.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("pyspark")

from ai_udf_transpile import conf, enable
from ai_udf_transpile.catalog import HIT, MISS, open_catalog
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.catalog.writeback import WritebackCatalog
from ai_udf_transpile.keys import canonical_source_text, udf_key
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def _table() -> str:
    return f"default.wb_{uuid.uuid4().hex[:12]}"


def _success(catalog, name: str, sql: str = "_udf_param_0 + 1") -> str:
    src = canonical_source_text(f"def {name}(x: int) -> int:\n    return x + 1")
    key = udf_key(src, ["x"], ["bigint"], "bigint", "4.1.0", "fp")
    catalog.upsert_success(
        udf_key=key,
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="4.1.0",
        closure_fingerprint="fp",
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        origin="human",
        hypothesis_passed=True,
    )
    return key


def _enable_wb(spark, sqlite_path, table, threshold="2", fmt="parquet"):
    conf.set_value(conf.WRITEBACK_TABLE, table, spark)
    conf.set_value(conf.WRITEBACK_FORMAT, fmt, spark)
    conf.set_value(conf.WRITEBACK_THRESHOLD, threshold, spark)
    catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    assert isinstance(catalog, WritebackCatalog)
    return catalog


def _remote_count(spark, table) -> int:
    return int(spark.sql(f"SELECT COUNT(*) FROM {table}").collect()[0][0])


@pytest.fixture
def table(spark):
    name = _table()
    yield name
    spark.sql(f"DROP TABLE IF EXISTS {name}")


def test_disabled_by_default(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    assert isinstance(get_catalog(), SqliteCatalog)


def test_flush_only_at_threshold(spark, sqlite_path, table):
    catalog = _enable_wb(spark, sqlite_path, table, threshold="3")
    _success(catalog, "wb_a")
    _success(catalog, "wb_b")
    assert _remote_count(spark, table) == 0, "flushed below threshold"
    _success(catalog, "wb_c")
    assert _remote_count(spark, table) == 3
    assert catalog._staging.staged_successes() == []
    row = catalog.get(
        udf_key(
            canonical_source_text("def wb_a(x: int) -> int:\n    return x + 1"),
            ["x"],
            ["bigint"],
            "bigint",
            "4.1.0",
            "fp",
        )
    )
    assert row.written_back_at is not None


def test_lookup_falls_through_and_backfills(spark, sqlite_path, table, tmp_path):
    writer = _enable_wb(spark, sqlite_path, table, threshold="1")
    key = _success(writer, "wb_shared", sql="_udf_param_0 * 10")
    assert _remote_count(spark, table) == 1

    # A fresh process: new SQLite file, same shared table.
    reader = _enable_wb(spark, str(tmp_path / "reader.sqlite"), table, threshold="100")
    assert reader._staging.get(key) is None
    kind, row = reader.lookup(key, spark=spark)
    assert kind == HIT
    assert row.catalyst_sql == "_udf_param_0 * 10"
    # Backfilled locally, marked written-back so it never echoes to the table.
    local = reader._staging.get(key)
    assert local is not None and local.status == "success"
    assert local.written_back_at is not None
    assert reader._staging.staged_successes() == []
    kind, _ = reader.lookup(key, spark=spark)
    assert kind == HIT
    assert _remote_count(spark, table) == 1


def test_failed_rows_stay_local(spark, sqlite_path, table):
    catalog = _enable_wb(spark, sqlite_path, table, threshold="1")
    src = canonical_source_text("def wb_bad(x: int) -> int:\n    return x + 1")
    key = udf_key(src, ["x"], ["bigint"], "bigint", "4.1.0", "fp")
    catalog.insert_pending(
        udf_key=key,
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="4.1.0",
        closure_fingerprint="fp",
        captures={},
    )
    catalog.mark_failed(key, "hypothesis failed", origin="fake")
    _success(catalog, "wb_good")  # triggers the flush
    assert _remote_count(spark, table) == 1
    remote = spark.sql(f"SELECT udf_key FROM {table}").collect()
    assert remote[0][0] != key


def test_local_pending_not_clobbered_by_remote(spark, sqlite_path, table, tmp_path):
    writer = _enable_wb(spark, sqlite_path, table, threshold="1")
    key = _success(writer, "wb_race", sql="_udf_param_0 + 2")

    reader = _enable_wb(spark, str(tmp_path / "reader.sqlite"), table, threshold="100")
    src = canonical_source_text("def wb_race(x: int) -> int:\n    return x + 1")
    reader.insert_pending(
        udf_key=key,
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="4.1.0",
        closure_fingerprint="fp",
        captures={},
    )
    kind, row = reader.lookup(key, spark=spark)
    assert kind == HIT, "remote success should beat local pending"
    assert row.catalyst_sql == "_udf_param_0 + 2"
    # The in-flight local row is left alone (the worker's success upserts later).
    assert reader._staging.get(key).status == "pending"


def test_writeback_failure_is_fail_open(spark, sqlite_path, table, monkeypatch):
    catalog = _enable_wb(spark, sqlite_path, table, threshold="1")

    def boom(rows):
        raise RuntimeError("simulated parquet write failure")

    monkeypatch.setattr(catalog, "_flush", boom)
    key = _success(catalog, "wb_resilient")  # must not raise
    kind, row = catalog.lookup(key, spark=spark)
    assert kind == HIT  # local SQLite still serves the success
    assert catalog._staging.staged_successes() != [], "row stays staged for a later flush"


def test_iceberg_without_runtime_is_fail_open(spark, sqlite_path, table):
    try:
        spark._jvm.Class.forName("org.apache.iceberg.spark.SparkCatalog")
        pytest.skip("iceberg runtime present; fail-open path not exercised")
    except Exception:
        pass
    catalog = _enable_wb(spark, sqlite_path, table, threshold="1", fmt="iceberg")
    assert not catalog._available
    key = _success(catalog, "wb_ice")
    kind, _ = catalog.lookup(key, spark=spark)
    assert kind == HIT  # local-only, still works


def test_open_catalog_without_spark_warns_but_works(sqlite_path, caplog):
    conf.set_value(conf.CATALOG, "sqlite")
    conf.set_value(conf.SQLITE_PATH, sqlite_path)
    conf.set_value(conf.WRITEBACK_TABLE, "default.wb_no_spark")
    catalog = open_catalog(None, sqlite_path=sqlite_path)
    assert isinstance(catalog, SqliteCatalog)
    assert "write-back disabled" in caplog.text


def test_miss_stays_miss(spark, sqlite_path, table):
    catalog = _enable_wb(spark, sqlite_path, table, threshold="1")
    kind, row = catalog.lookup("nonexistent-key", spark=spark)
    assert kind == MISS
    assert row is None
