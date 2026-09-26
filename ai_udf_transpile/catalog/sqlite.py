# SPDX-License-Identifier: Apache-2.0
"""SQLite catalog with real row-level CAS (UPDATE ... WHERE status='pending')."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from ai_udf_transpile.catalog import (
    HIT,
    MISS,
    WAIT,
    CacheRow,
    cooldown_active,
    dumps,
    iso_now,
    loads_dict,
    loads_list,
    retries_exhausted,
)
from ai_udf_transpile.targets import TranspileResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    udf_key TEXT PRIMARY KEY,
    source_text TEXT,
    param_names TEXT,
    input_types TEXT,
    input_categories TEXT,
    return_type TEXT NOT NULL,
    spark_version TEXT,
    closure_fingerprint TEXT,
    captures_json TEXT,
    status TEXT NOT NULL,
    target_kind TEXT,
    catalyst_sql TEXT,
    impl_source TEXT,
    impl_class TEXT,
    impl_entry TEXT,
    impl_binary BLOB,
    origin TEXT,
    backend TEXT,
    model TEXT,
    error TEXT,
    hypothesis_passed INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    failed_at TEXT,
    claimed_at TEXT,
    created_at TEXT,
    updated_at TEXT
)
"""

_SAMPLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    udf_key TEXT NOT NULL,
    args_json TEXT NOT NULL,
    created_at TEXT
)
"""


def _row_from_sql(raw: sqlite3.Row) -> CacheRow:
    hyp = raw["hypothesis_passed"]
    keys = set(raw.keys())
    return CacheRow(
        udf_key=raw["udf_key"],
        source_text=raw["source_text"] or "",
        param_names=loads_list(raw["param_names"]),
        input_types=loads_list(raw["input_types"]),
        input_categories=loads_list(raw["input_categories"]),
        return_type=raw["return_type"] or "",
        spark_version=raw["spark_version"] or "",
        closure_fingerprint=raw["closure_fingerprint"] or "",
        captures=loads_dict(raw["captures_json"]),
        status=raw["status"],
        target_kind=raw["target_kind"],
        catalyst_sql=raw["catalyst_sql"],
        impl_source=raw["impl_source"],
        impl_class=raw["impl_class"],
        impl_entry=raw["impl_entry"],
        impl_binary=raw["impl_binary"],
        origin=raw["origin"],
        backend=raw["backend"],
        model=raw["model"] if "model" in keys else None,
        error=raw["error"],
        hypothesis_passed=None if hyp is None else bool(hyp),
        attempt_count=int(raw["attempt_count"] or 0),
        failed_at=raw["failed_at"],
        claimed_at=raw["claimed_at"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
        written_back_at=raw["written_back_at"] if "written_back_at" in keys else None,
    )


class SqliteCatalog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(_SCHEMA)
        self._conn.execute(_SAMPLES_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the first schema version."""
        with self._lock:
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(cache)")}
            if "model" not in cols:
                self._conn.execute("ALTER TABLE cache ADD COLUMN model TEXT")
            if "written_back_at" not in cols:
                self._conn.execute("ALTER TABLE cache ADD COLUMN written_back_at TEXT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _fetch(self, udf_key: str) -> Optional[CacheRow]:
        cur = self._conn.execute("SELECT * FROM cache WHERE udf_key = ?", (udf_key,))
        raw = cur.fetchone()
        return _row_from_sql(raw) if raw else None

    def get(self, udf_key: str) -> Optional[CacheRow]:
        with self._lock:
            return self._fetch(udf_key)

    def lookup(self, udf_key: str, *, spark: Any = None) -> tuple[str, Optional[CacheRow]]:
        with self._lock:
            row = self._fetch(udf_key)
            if row is None:
                return MISS, None
            if row.reconstructable():
                return HIT, row
            if row.status in {"pending", "running"}:
                return WAIT, row
            if row.status == "failed":
                if cooldown_active(row, spark) or retries_exhausted(row, spark):
                    return WAIT, row
                now = iso_now()
                self._conn.execute(
                    "UPDATE cache SET status = 'pending', error = NULL, updated_at = ? "
                    "WHERE udf_key = ? AND status = 'failed'",
                    (now, udf_key),
                )
                return WAIT, self._fetch(udf_key)
            return WAIT, row

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
    ) -> None:
        if not return_type or not input_types or len(input_types) != len(param_names):
            raise ValueError("insert_pending requires known input_types and return_type")
        now = iso_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO cache (
                    udf_key, source_text, param_names, input_types, input_categories,
                    return_type, spark_version, closure_fingerprint, captures_json,
                    status, attempt_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (
                    udf_key,
                    source_text,
                    dumps(param_names),
                    dumps(input_types),
                    dumps(input_categories),
                    return_type,
                    spark_version,
                    closure_fingerprint,
                    dumps(captures),
                    now,
                    now,
                ),
            )

    def claim(self, udf_key: str, backend: str) -> bool:
        now = iso_now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    """
                    UPDATE cache
                    SET status = 'running',
                        claimed_at = ?,
                        backend = ?,
                        attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE udf_key = ? AND status = 'pending'
                    """,
                    (now, backend, now, udf_key),
                )
                self._conn.execute("COMMIT")
                return cur.rowcount == 1
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def mark_success(
        self,
        udf_key: str,
        result: TranspileResult,
        origin: str,
        *,
        hypothesis_passed: bool = True,
    ) -> None:
        now = iso_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE cache SET
                    status = 'success',
                    target_kind = ?,
                    catalyst_sql = ?,
                    impl_source = ?,
                    impl_class = ?,
                    impl_entry = ?,
                    impl_binary = ?,
                    origin = ?,
                    model = ?,
                    hypothesis_passed = ?,
                    error = NULL,
                    failed_at = NULL,
                    claimed_at = NULL,
                    updated_at = ?
                WHERE udf_key = ?
                """,
                (
                    result.kind,
                    result.sql,
                    result.java_source,
                    result.class_name,
                    result.entry,
                    result.binary,
                    origin,
                    result.model,
                    1 if hypothesis_passed else 0,
                    now,
                    udf_key,
                ),
            )

    def mark_failed(
        self,
        udf_key: str,
        error: str,
        *,
        origin: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        now = iso_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE cache SET
                    status = 'failed',
                    error = ?,
                    failed_at = ?,
                    claimed_at = NULL,
                    hypothesis_passed = 0,
                    origin = COALESCE(?, origin),
                    model = COALESCE(?, model),
                    updated_at = ?
                WHERE udf_key = ?
                """,
                (error, now, origin, model, now, udf_key),
            )

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
    ) -> None:
        if not return_type or not input_types:
            raise ValueError("upsert_success requires known types")
        now = iso_now()
        hyp = None if hypothesis_passed is None else (1 if hypothesis_passed else 0)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cache (
                    udf_key, source_text, param_names, input_types, input_categories,
                    return_type, spark_version, closure_fingerprint, captures_json,
                    status, target_kind, catalyst_sql, impl_source, impl_class,
                    impl_entry, impl_binary, origin, model, hypothesis_passed,
                    attempt_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'success', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(udf_key) DO UPDATE SET
                    source_text = excluded.source_text,
                    param_names = excluded.param_names,
                    input_types = excluded.input_types,
                    input_categories = excluded.input_categories,
                    return_type = excluded.return_type,
                    spark_version = excluded.spark_version,
                    closure_fingerprint = excluded.closure_fingerprint,
                    captures_json = excluded.captures_json,
                    status = 'success',
                    target_kind = excluded.target_kind,
                    catalyst_sql = excluded.catalyst_sql,
                    impl_source = excluded.impl_source,
                    impl_class = excluded.impl_class,
                    impl_entry = excluded.impl_entry,
                    impl_binary = excluded.impl_binary,
                    origin = excluded.origin,
                    model = excluded.model,
                    hypothesis_passed = excluded.hypothesis_passed,
                    error = NULL,
                    failed_at = NULL,
                    claimed_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    udf_key,
                    source_text,
                    dumps(param_names),
                    dumps(input_types),
                    dumps(input_categories),
                    return_type,
                    spark_version,
                    closure_fingerprint,
                    dumps(captures),
                    result.kind,
                    result.sql,
                    result.java_source,
                    result.class_name,
                    result.entry,
                    result.binary,
                    origin,
                    result.model,
                    hyp,
                    now,
                    now,
                ),
            )

    def reclaim_stale(self, timeout_seconds: int, max_retries: int) -> int:
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        now = iso_now()
        with self._lock:
            pending = self._conn.execute(
                """
                UPDATE cache
                SET status = 'pending', claimed_at = NULL, updated_at = ?
                WHERE status = 'running'
                  AND (claimed_at IS NULL OR claimed_at < ?)
                  AND attempt_count < ?
                """,
                (now, cutoff, max_retries),
            )
            failed = self._conn.execute(
                """
                UPDATE cache
                SET status = 'failed', failed_at = ?, claimed_at = NULL, updated_at = ?,
                    error = COALESCE(error, 'stale running claim exhausted retries')
                WHERE status = 'running'
                  AND (claimed_at IS NULL OR claimed_at < ?)
                  AND attempt_count >= ?
                """,
                (now, now, cutoff, max_retries),
            )
            return int(pending.rowcount + failed.rowcount)

    def oldest_pending(self) -> Optional[CacheRow]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM cache WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
            )
            raw = cur.fetchone()
            return _row_from_sql(raw) if raw else None

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM cache")
            return int(cur.fetchone()[0])

    def staged_successes(self) -> list[CacheRow]:
        """Verified rows not yet appended to the write-back table (if any)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM cache WHERE status = 'success' AND written_back_at IS NULL"
            )
            return [_row_from_sql(raw) for raw in cur.fetchall()]

    def mark_written_back(self, udf_keys: list[str]) -> None:
        if not udf_keys:
            return
        now = iso_now()
        with self._lock:
            self._conn.executemany(
                "UPDATE cache SET written_back_at = ? WHERE udf_key = ?",
                [(now, key) for key in udf_keys],
            )

    def record_samples(self, udf_key: str, samples: list[list]) -> None:
        from ai_udf_transpile.sampling import record_samples

        record_samples(self.path, udf_key, samples)

    def samples_for(self, udf_key: str, limit: int = 32) -> list[list]:
        from ai_udf_transpile.sampling import decode_args

        with self._lock:
            cur = self._conn.execute(
                "SELECT args_json FROM samples WHERE udf_key = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (udf_key, int(limit)),
            )
            out = []
            for (text,) in cur.fetchall():
                decoded = decode_args(text)
                if decoded is not None:
                    out.append(decoded)
            return out
