# SPDX-License-Identifier: Apache-2.0
"""Catalog protocol, cache row, and factory."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

from ai_udf_transpile import conf
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileJob, TranspileResult

logger = logging.getLogger(__name__)

# lookup() results
HIT = "hit"
WAIT = "wait"
MISS = "miss"

# A row is 'provisional' while reconstruction is still being checked. Lookup
# serves it only on the thread that asked to see it, so a bad rewrite cannot
# take down a query running on another thread during that check.
_provisional = threading.local()


def provisional_visible() -> bool:
    return bool(getattr(_provisional, "on", False))


@contextmanager
def expose_provisional():
    previous = provisional_visible()
    _provisional.on = True
    try:
        yield
    finally:
        _provisional.on = previous


def publish_verified(catalog: Any, udf_key: str) -> None:
    """Append a fully checked success to the write-back table, if this catalog has one."""
    note = getattr(catalog, "note_verified", None)
    if callable(note):
        note(udf_key)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%S")


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def loads_list(text: Optional[str]) -> list:
    if not text:
        return []
    data = json.loads(text)
    return list(data) if isinstance(data, list) else []


def loads_dict(text: Optional[str]) -> dict:
    if not text:
        return {}
    data = json.loads(text)
    return dict(data) if isinstance(data, dict) else {}


@dataclass
class CacheRow:
    udf_key: str
    source_text: str = ""
    param_names: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    input_categories: list[str] = field(default_factory=list)
    return_type: str = ""
    spark_version: str = ""
    closure_fingerprint: str = ""
    captures: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    target_kind: Optional[str] = None
    catalyst_sql: Optional[str] = None
    impl_source: Optional[str] = None
    impl_class: Optional[str] = None
    impl_entry: Optional[str] = None
    impl_binary: Optional[bytes] = None
    origin: Optional[str] = None
    backend: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    hypothesis_passed: Optional[bool] = None
    # Absolute numeric tolerance the rewrite was checked at. None means the
    # row predates the column and was checked exactly (tolerance 0).
    tolerance: Optional[float] = None
    attempt_count: int = 0
    failed_at: Optional[str] = None
    claimed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Local SQLite staging flag: set once this success row has been appended
    # to the configured write-back table. Not a shared-table column.
    written_back_at: Optional[str] = None

    def reconstructable(self) -> bool:
        if self.status != "success":
            return False
        if self.target_kind == KIND_CATALYST and self.catalyst_sql:
            return True
        if self.target_kind == KIND_JAVA_UDF and (self.impl_class or self.impl_source or self.impl_binary):
            return True
        return False

    def as_job(self) -> TranspileJob:
        return TranspileJob(
            udf_key=self.udf_key,
            source_text=self.source_text,
            param_names=list(self.param_names),
            input_types=list(self.input_types),
            input_categories=list(self.input_categories),
            return_type=self.return_type,
            captures=dict(self.captures),
            spark_version=self.spark_version,
            closure_fingerprint=self.closure_fingerprint,
        )


class Catalog(Protocol):
    def get(self, udf_key: str) -> Optional[CacheRow]: ...

    def lookup(self, udf_key: str, *, spark: Any = None) -> tuple[str, Optional[CacheRow]]: ...

    def insert_pending(
        self,
        *,
        udf_key: str,
        source_text: str,
        param_names: list[str],
        input_types: list[str],
        input_categories: list[str],
        return_type: str,
        spark_version: str,
        closure_fingerprint: str,
        captures: dict[str, Any],
    ) -> None: ...

    def claim(self, udf_key: str, backend: str) -> bool: ...

    def mark_success(
        self,
        udf_key: str,
        result: TranspileResult,
        origin: str,
        *,
        hypothesis_passed: bool = True,
        visible: bool = True,
        tolerance: Optional[float] = None,
    ) -> None: ...

    def promote_success(self, udf_key: str) -> None: ...

    def mark_failed(
        self,
        udf_key: str,
        error: str,
        *,
        origin: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None: ...

    def upsert_success(
        self,
        *,
        udf_key: str,
        source_text: str,
        param_names: list[str],
        input_types: list[str],
        input_categories: list[str],
        return_type: str,
        spark_version: str,
        closure_fingerprint: str,
        captures: dict[str, Any],
        result: TranspileResult,
        origin: str,
        hypothesis_passed: Optional[bool],
        visible: bool = True,
        tolerance: Optional[float] = None,
    ) -> None: ...

    def reclaim_stale(self, timeout_seconds: int, max_retries: int) -> int: ...

    def oldest_pending(self) -> Optional[CacheRow]: ...

    def count(self) -> int: ...

    def record_samples(self, udf_key: str, samples: list[list]) -> None: ...

    def samples_for(self, udf_key: str, limit: int = 32) -> list[list]: ...


def configured_tolerance(spark: Any = None) -> float:
    """Session absolute tolerance. Negative or non-numeric values count as 0."""
    try:
        value = conf.get_float(conf.TOLERANCE, spark, float(conf.DEFAULTS[conf.TOLERANCE]))
    except (TypeError, ValueError):
        return 0.0
    if value < 0 or value != value:
        return 0.0
    return value


def tolerance_serves(tested: Optional[float], spark: Any = None) -> bool:
    """Whether a rewrite checked at ``tested`` is tight enough for this session.

    Smaller is stricter. A cached rewrite is served only when it was tested at
    a tolerance less than or equal to the configured one; a looser check is
    not reused. Missing ``tested`` is 0, the exact checks from before the column.
    """
    allowed = configured_tolerance(spark)
    if tested is None:
        tested = 0.0
    return tested <= allowed


def cooldown_active(row: CacheRow, spark: Any = None) -> bool:
    seconds = conf.get_int(conf.FAIL_COOLDOWN, spark, int(conf.DEFAULTS[conf.FAIL_COOLDOWN]))
    failed_at = parse_iso(row.failed_at)
    if failed_at is None:
        return False
    return utc_now() < failed_at + timedelta(seconds=seconds)


def retries_exhausted(row: CacheRow, spark: Any = None) -> bool:
    max_retries = conf.get_int(conf.MAX_RETRIES, spark, int(conf.DEFAULTS[conf.MAX_RETRIES]))
    return row.attempt_count >= max_retries


def open_catalog(spark: Any = None, *, sqlite_path: Optional[str] = None) -> Catalog:
    kind = conf.get_value(conf.CATALOG, spark, conf.DEFAULTS[conf.CATALOG]).strip().lower()
    if kind == "sqlite":
        from ai_udf_transpile.catalog.sqlite import SqliteCatalog

        path = sqlite_path or conf.get_value(conf.SQLITE_PATH, spark, "")
        if not path:
            raise ValueError("sqlite catalog requires spark.sql.experimental.aiUdfTranspile.sqlitePath")
        catalog: Catalog = SqliteCatalog(path)
        table = conf.get_value(conf.WRITEBACK_TABLE, spark, conf.DEFAULTS[conf.WRITEBACK_TABLE]).strip()
        if table:
            if spark is None:
                logger.warning(
                    "writeback table %s configured but no SparkSession; write-back disabled", table
                )
            else:
                from ai_udf_transpile.catalog.writeback import WritebackCatalog

                fmt = conf.get_value(conf.WRITEBACK_FORMAT, spark, conf.DEFAULTS[conf.WRITEBACK_FORMAT])
                threshold = conf.get_int(
                    conf.WRITEBACK_THRESHOLD, spark, int(conf.DEFAULTS[conf.WRITEBACK_THRESHOLD])
                )
                catalog = WritebackCatalog(catalog, spark, table, fmt.strip().lower(), threshold)
        return catalog
    if kind == "delta":
        from ai_udf_transpile.catalog.delta import DeltaCatalog

        if spark is None:
            raise ValueError("delta catalog requires a SparkSession")
        table = conf.get_value(conf.TABLE, spark, conf.DEFAULTS[conf.TABLE])
        return DeltaCatalog(spark, table)
    raise ValueError(f"unknown catalog {kind!r}; use sqlite or delta")
