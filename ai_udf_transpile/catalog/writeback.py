# SPDX-License-Identifier: Apache-2.0
"""SQLite staging + shared parquet/iceberg write-back hybrid catalog.

Local SQLite stays the hot path (claims, cooldowns, samples). Verified
success rows accumulate as *staged* rows and are appended to a shared
parquet (or iceberg) table once ``writebackThreshold`` of them have
collected; lookups fall through to that table, so other processes and
machines see verified rewrites without waiting on their own backends.

Parquet has no row-level UPDATE, so write-back is append-only and reads
dedupe by latest ``updated_at``. Failed rows stay local: the shared table
carries verified successes only. Every table interaction is fail-open --
a misconfigured or unreachable table never breaks UDF definition.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ai_udf_transpile.catalog import HIT, CacheRow, dumps, tolerance_serves
from ai_udf_transpile.catalog.delta import _ident, _row_from_spark, _sql_str
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult

logger = logging.getLogger(__name__)

_FORMATS = {"parquet", "iceberg"}

# Same columns as the delta catalog's table so rows map 1:1 across catalogs.
_COLUMNS = [
    "udf_key",
    "source_text",
    "param_names",
    "input_types",
    "input_categories",
    "return_type",
    "spark_version",
    "closure_fingerprint",
    "captures_json",
    "status",
    "target_kind",
    "catalyst_sql",
    "impl_source",
    "impl_class",
    "impl_entry",
    "impl_binary",
    "origin",
    "backend",
    "model",
    "error",
    "hypothesis_passed",
    "tolerance",
    "attempt_count",
    "failed_at",
    "claimed_at",
    "created_at",
    "updated_at",
]

_DDL = """
    udf_key STRING, source_text STRING, param_names STRING, input_types STRING,
    input_categories STRING, return_type STRING, spark_version STRING,
    closure_fingerprint STRING, captures_json STRING, status STRING,
    target_kind STRING, catalyst_sql STRING, impl_source STRING, impl_class STRING,
    impl_entry STRING, impl_binary BINARY, origin STRING, backend STRING,
    model STRING, error STRING, hypothesis_passed BOOLEAN, tolerance DOUBLE,
    attempt_count INT,
    failed_at STRING, claimed_at STRING, created_at STRING, updated_at STRING
"""


def _table_tuple(row: CacheRow) -> tuple:
    return (
        row.udf_key,
        row.source_text,
        dumps(row.param_names),
        dumps(row.input_types),
        dumps(row.input_categories),
        row.return_type,
        row.spark_version,
        row.closure_fingerprint,
        dumps(row.captures),
        row.status,
        row.target_kind,
        row.catalyst_sql,
        row.impl_source,
        row.impl_class,
        row.impl_entry,
        row.impl_binary,
        row.origin,
        row.backend,
        row.model,
        row.error,
        row.hypothesis_passed,
        row.tolerance,
        int(row.attempt_count or 0),
        row.failed_at,
        row.claimed_at,
        row.created_at,
        row.updated_at,
    )


class WritebackCatalog:
    """Wraps a SqliteCatalog: staged successes flush to a shared table; reads fall through."""

    def __init__(self, staging: SqliteCatalog, spark: Any, table: str, fmt: str, threshold: int):
        if fmt not in _FORMATS:
            raise ValueError(f"writeback format must be one of {sorted(_FORMATS)}, not {fmt!r}")
        self._staging = staging
        self._spark = spark
        self._table = _ident(table)
        self._format = fmt
        self._threshold = max(1, int(threshold))
        self._flush_lock = threading.Lock()
        self._available = False
        try:
            self._spark.sql(f"CREATE TABLE IF NOT EXISTS {self._table} ({_DDL}) USING {self._format}")
            try:
                self._spark.sql(f"ALTER TABLE {self._table} ADD COLUMN tolerance DOUBLE")
            except Exception:
                logger.debug("writeback tolerance column already present", exc_info=True)
            self._available = True
        except Exception:
            # e.g. USING iceberg without the iceberg runtime/catalog configured.
            logger.warning(
                "writeback table %s (format %s) could not be created; running local-only",
                self._table,
                self._format,
                exc_info=True,
            )

    # -- write-back ------------------------------------------------------

    def _maybe_flush(self) -> None:
        if not self._available:
            return
        try:
            if len(self._staging.staged_successes()) < self._threshold:
                return
            with self._flush_lock:
                staged = self._staging.staged_successes()
                if len(staged) < self._threshold:
                    return
                self._flush(staged)
        except Exception:
            # Write-back is best-effort: the local row is already a success.
            logger.warning("writeback to %s failed; rows stay staged", self._table, exc_info=True)

    def _flush(self, staged: list[CacheRow]) -> None:
        from pyspark.sql.types import (
            BinaryType,
            BooleanType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )

        def col_type(name: str) -> Any:
            if name == "impl_binary":
                return BinaryType()
            if name == "hypothesis_passed":
                return BooleanType()
            if name == "tolerance":
                from pyspark.sql.types import DoubleType

                return DoubleType()
            if name == "attempt_count":
                return IntegerType()
            return StringType()

        schema = StructType([StructField(name, col_type(name), True) for name in _COLUMNS])
        df = self._spark.createDataFrame([_table_tuple(row) for row in staged], schema)
        df.write.mode("append").saveAsTable(self._table)
        self._staging.mark_written_back([row.udf_key for row in staged])
        logger.info("wrote back %d verified UDF(s) to %s", len(staged), self._table)

    def _remote_success(self, udf_key: str) -> Optional[CacheRow]:
        if not self._available:
            return None
        try:
            rows = self._spark.sql(
                f"SELECT * FROM {self._table} "
                f"WHERE udf_key = {_sql_str(udf_key)} AND status = 'success' "
                "ORDER BY updated_at DESC LIMIT 1"
            ).collect()
        except Exception:
            logger.debug("writeback lookup against %s failed", self._table, exc_info=True)
            return None
        return _row_from_spark(rows[0]) if rows else None

    def _backfill(self, remote: CacheRow) -> None:
        """Warm the local cache with a remote success; marked written-back so it never echoes."""
        result = TranspileResult(
            kind=remote.target_kind or KIND_CATALYST,
            sql=remote.catalyst_sql,
            java_source=remote.impl_source,
            class_name=remote.impl_class,
            binary=remote.impl_binary,
            entry=remote.impl_entry,
            model=remote.model,
        )
        self._staging.upsert_success(
            udf_key=remote.udf_key,
            source_text=remote.source_text,
            param_names=list(remote.param_names),
            input_types=list(remote.input_types),
            input_categories=list(remote.input_categories),
            return_type=remote.return_type,
            spark_version=remote.spark_version,
            closure_fingerprint=remote.closure_fingerprint,
            captures=dict(remote.captures),
            result=result,
            origin=remote.origin or "writeback",
            hypothesis_passed=remote.hypothesis_passed,
            tolerance=remote.tolerance,
        )
        self._staging.mark_written_back([remote.udf_key])

    # -- hybrid reads ------------------------------------------------------

    def get(self, udf_key: str) -> Optional[CacheRow]:
        row = self._staging.get(udf_key)
        return row if row is not None else self._remote_success(udf_key)

    def lookup(self, udf_key: str, *, spark: Any = None) -> tuple[str, Optional[CacheRow]]:
        kind, row = self._staging.lookup(udf_key, spark=spark)
        if kind == HIT:
            return kind, row
        remote = self._remote_success(udf_key)
        if remote is not None and remote.reconstructable() and tolerance_serves(remote.tolerance, spark):
            if row is None:
                # Pure miss locally: warm the cache. A pending/running local
                # row is left alone -- the in-flight worker's success upserts
                # over it harmlessly.
                try:
                    self._backfill(remote)
                except Exception:
                    logger.debug("writeback backfill failed for %s", udf_key[:12], exc_info=True)
            return HIT, remote
        return kind, row

    # -- writes: delegate, then maybe flush --------------------------------

    def mark_success(
        self,
        udf_key: str,
        result: TranspileResult,
        origin: str,
        *,
        hypothesis_passed: bool = True,
        visible: bool = True,
        tolerance: Optional[float] = None,
    ) -> None:
        # Staging only. note_verified flushes after reconstruction has passed,
        # so a rewrite that fails the smoke test never reaches the shared table.
        self._staging.mark_success(
            udf_key,
            result,
            origin,
            hypothesis_passed=hypothesis_passed,
            visible=visible,
            tolerance=tolerance,
        )

    def upsert_success(self, **kwargs: Any) -> None:
        self._staging.upsert_success(**kwargs)

    def promote_success(self, udf_key: str) -> None:
        self._staging.promote_success(udf_key)

    def note_verified(self, udf_key: str) -> None:
        self._staging.mark_publish_ready(udf_key)
        self._maybe_flush()

    # -- local-only concepts: straight delegation --------------------------

    def insert_pending(self, **kwargs: Any) -> None:
        self._staging.insert_pending(**kwargs)

    def claim(self, udf_key: str, backend: str) -> bool:
        return self._staging.claim(udf_key, backend)

    def mark_failed(self, udf_key: str, error: str, **kwargs: Any) -> None:
        # Failures stay local: the shared table carries verified successes only.
        self._staging.mark_failed(udf_key, error, **kwargs)

    def reclaim_stale(self, timeout_seconds: int, max_retries: int) -> int:
        return self._staging.reclaim_stale(timeout_seconds, max_retries)

    def oldest_pending(self) -> Optional[CacheRow]:
        return self._staging.oldest_pending()

    def count(self) -> int:
        return self._staging.count()

    def record_samples(self, udf_key: str, samples: list[list]) -> None:
        self._staging.record_samples(udf_key, samples)

    def samples_for(self, udf_key: str, limit: int = 32) -> list[list]:
        return self._staging.samples_for(udf_key, limit)

    @property
    def path(self) -> str:
        return self._staging.path

    def close(self) -> None:
        self._staging.close()
