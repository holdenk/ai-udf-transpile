# SPDX-License-Identifier: Apache-2.0
"""Optional Delta Lake catalog. Requires Delta on the Spark classpath."""

from __future__ import annotations

import re
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

_TABLE_NAME = re.compile(r"^[A-Za-z_][\w.]*$")


def require_delta(spark: Any) -> None:
    try:
        spark._jvm.Class.forName("io.delta.tables.DeltaTable")
    except Exception as exc:
        raise RuntimeError(
            "catalog=delta requires Delta Lake on the classpath; "
            "use sqlite or add Delta (org.apache.spark:delta-spark)"
        ) from exc


def _ident(table: str) -> str:
    if not _TABLE_NAME.match(table):
        raise ValueError(f"unsafe delta table name: {table!r}")
    return table


def _sql_str(value: Optional[str]) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _sql_blob_hex(data: Optional[bytes]) -> str:
    if data is None:
        return "NULL"
    return "X'" + data.hex() + "'"


def _row_from_spark(rec: Any) -> CacheRow:
    as_dict = rec.asDict() if hasattr(rec, "asDict") else dict(rec)
    hyp = as_dict.get("hypothesis_passed")
    binary = as_dict.get("impl_binary")
    if binary is not None and not isinstance(binary, (bytes, bytearray, type(None))):
        binary = bytes(binary)
    return CacheRow(
        udf_key=as_dict["udf_key"],
        source_text=as_dict.get("source_text") or "",
        param_names=loads_list(as_dict.get("param_names")),
        input_types=loads_list(as_dict.get("input_types")),
        input_categories=loads_list(as_dict.get("input_categories")),
        return_type=as_dict.get("return_type") or "",
        spark_version=as_dict.get("spark_version") or "",
        closure_fingerprint=as_dict.get("closure_fingerprint") or "",
        captures=loads_dict(as_dict.get("captures_json")),
        status=as_dict.get("status") or "pending",
        target_kind=as_dict.get("target_kind"),
        catalyst_sql=as_dict.get("catalyst_sql"),
        impl_source=as_dict.get("impl_source"),
        impl_class=as_dict.get("impl_class"),
        impl_entry=as_dict.get("impl_entry"),
        impl_binary=binary,
        origin=as_dict.get("origin"),
        backend=as_dict.get("backend"),
        error=as_dict.get("error"),
        hypothesis_passed=None if hyp is None else bool(hyp),
        attempt_count=int(as_dict.get("attempt_count") or 0),
        failed_at=as_dict.get("failed_at"),
        claimed_at=as_dict.get("claimed_at"),
        created_at=as_dict.get("created_at"),
        updated_at=as_dict.get("updated_at"),
    )


class DeltaCatalog:
    def __init__(self, spark: Any, table: str):
        require_delta(spark)
        self.spark = spark
        self.table = _ident(table)
        self._ensure()

    def _ensure(self) -> None:
        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                udf_key STRING,
                source_text STRING,
                param_names STRING,
                input_types STRING,
                input_categories STRING,
                return_type STRING,
                spark_version STRING,
                closure_fingerprint STRING,
                captures_json STRING,
                status STRING,
                target_kind STRING,
                catalyst_sql STRING,
                impl_source STRING,
                impl_class STRING,
                impl_entry STRING,
                impl_binary BINARY,
                origin STRING,
                backend STRING,
                error STRING,
                hypothesis_passed BOOLEAN,
                attempt_count INT,
                failed_at STRING,
                claimed_at STRING,
                created_at STRING,
                updated_at STRING
            ) USING delta
            """
        )

    def _select(self, udf_key: str) -> Optional[CacheRow]:
        rows = self.spark.sql(
            f"SELECT * FROM {self.table} WHERE udf_key = {_sql_str(udf_key)} LIMIT 1"
        ).collect()
        if not rows:
            return None
        return _row_from_spark(rows[0])

    def get(self, udf_key: str) -> Optional[CacheRow]:
        return self._select(udf_key)

    def lookup(self, udf_key: str, *, spark: Any = None) -> tuple[str, Optional[CacheRow]]:
        row = self._select(udf_key)
        if row is None:
            return MISS, None
        if row.reconstructable():
            return HIT, row
        if row.status in {"pending", "running"}:
            return WAIT, row
        if row.status == "failed":
            session = spark if spark is not None else self.spark
            if cooldown_active(row, session) or retries_exhausted(row, session):
                return WAIT, row
            now = iso_now()
            self.spark.sql(
                f"""
                UPDATE {self.table}
                SET status = 'pending', error = NULL, updated_at = {_sql_str(now)}
                WHERE udf_key = {_sql_str(udf_key)} AND status = 'failed'
                """
            )
            return WAIT, self._select(udf_key)
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
        if self._select(udf_key) is not None:
            return
        now = iso_now()
        self.spark.sql(
            f"""
            INSERT INTO {self.table} (
                udf_key, source_text, param_names, input_types, input_categories,
                return_type, spark_version, closure_fingerprint, captures_json,
                status, attempt_count, created_at, updated_at
            ) VALUES (
                {_sql_str(udf_key)}, {_sql_str(source_text)}, {_sql_str(dumps(param_names))},
                {_sql_str(dumps(input_types))}, {_sql_str(dumps(input_categories))},
                {_sql_str(return_type)}, {_sql_str(spark_version)},
                {_sql_str(closure_fingerprint)}, {_sql_str(dumps(captures))},
                'pending', 0, {_sql_str(now)}, {_sql_str(now)}
            )
            """
        )

    def claim(self, udf_key: str, backend: str) -> bool:
        now = iso_now()
        before = self._select(udf_key)
        if before is None or before.status != "pending":
            return False
        self.spark.sql(
            f"""
            MERGE INTO {self.table} t
            USING (SELECT {_sql_str(udf_key)} AS udf_key) s
            ON t.udf_key = s.udf_key AND t.status = 'pending'
            WHEN MATCHED THEN UPDATE SET
                status = 'running',
                claimed_at = {_sql_str(now)},
                backend = {_sql_str(backend)},
                attempt_count = t.attempt_count + 1,
                updated_at = {_sql_str(now)}
            """
        )
        after = self._select(udf_key)
        return (
            after is not None
            and after.status == "running"
            and after.attempt_count == ((before.attempt_count or 0) + 1)
        )

    def mark_success(
        self,
        udf_key: str,
        result: TranspileResult,
        origin: str,
        *,
        hypothesis_passed: bool = True,
    ) -> None:
        now = iso_now()
        self.spark.sql(
            f"""
            UPDATE {self.table} SET
                status = 'success',
                target_kind = {_sql_str(result.kind)},
                catalyst_sql = {_sql_str(result.sql)},
                impl_source = {_sql_str(result.java_source)},
                impl_class = {_sql_str(result.class_name)},
                impl_entry = {_sql_str(result.entry)},
                impl_binary = {_sql_blob_hex(result.binary)},
                origin = {_sql_str(origin)},
                hypothesis_passed = {str(bool(hypothesis_passed)).upper()},
                error = NULL,
                failed_at = NULL,
                claimed_at = NULL,
                updated_at = {_sql_str(now)}
            WHERE udf_key = {_sql_str(udf_key)}
            """
        )

    def mark_failed(self, udf_key: str, error: str) -> None:
        now = iso_now()
        self.spark.sql(
            f"""
            UPDATE {self.table} SET
                status = 'failed',
                error = {_sql_str(error)},
                failed_at = {_sql_str(now)},
                claimed_at = NULL,
                hypothesis_passed = FALSE,
                updated_at = {_sql_str(now)}
            WHERE udf_key = {_sql_str(udf_key)}
            """
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
        hyp = "NULL" if hypothesis_passed is None else str(bool(hypothesis_passed)).upper()
        self.spark.sql(
            f"""
            MERGE INTO {self.table} t
            USING (SELECT {_sql_str(udf_key)} AS udf_key) s
            ON t.udf_key = s.udf_key
            WHEN MATCHED THEN UPDATE SET
                source_text = {_sql_str(source_text)},
                param_names = {_sql_str(dumps(param_names))},
                input_types = {_sql_str(dumps(input_types))},
                input_categories = {_sql_str(dumps(input_categories))},
                return_type = {_sql_str(return_type)},
                spark_version = {_sql_str(spark_version)},
                closure_fingerprint = {_sql_str(closure_fingerprint)},
                captures_json = {_sql_str(dumps(captures))},
                status = 'success',
                target_kind = {_sql_str(result.kind)},
                catalyst_sql = {_sql_str(result.sql)},
                impl_source = {_sql_str(result.java_source)},
                impl_class = {_sql_str(result.class_name)},
                impl_entry = {_sql_str(result.entry)},
                impl_binary = {_sql_blob_hex(result.binary)},
                origin = {_sql_str(origin)},
                hypothesis_passed = {hyp},
                error = NULL,
                failed_at = NULL,
                claimed_at = NULL,
                updated_at = {_sql_str(now)}
            WHEN NOT MATCHED THEN INSERT (
                udf_key, source_text, param_names, input_types, input_categories,
                return_type, spark_version, closure_fingerprint, captures_json,
                status, target_kind, catalyst_sql, impl_source, impl_class,
                impl_entry, impl_binary, origin, hypothesis_passed,
                attempt_count, created_at, updated_at
            ) VALUES (
                {_sql_str(udf_key)}, {_sql_str(source_text)}, {_sql_str(dumps(param_names))},
                {_sql_str(dumps(input_types))}, {_sql_str(dumps(input_categories))},
                {_sql_str(return_type)}, {_sql_str(spark_version)},
                {_sql_str(closure_fingerprint)}, {_sql_str(dumps(captures))},
                'success', {_sql_str(result.kind)}, {_sql_str(result.sql)},
                {_sql_str(result.java_source)}, {_sql_str(result.class_name)},
                {_sql_str(result.entry)}, {_sql_blob_hex(result.binary)},
                {_sql_str(origin)}, {hyp}, 0, {_sql_str(now)}, {_sql_str(now)}
            )
            """
        )

    def reclaim_stale(self, timeout_seconds: int, max_retries: int) -> int:
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        now = iso_now()
        before = self.spark.sql(f"SELECT COUNT(*) AS n FROM {self.table} WHERE status = 'running'").collect()[
            0
        ][0]
        self.spark.sql(
            f"""
            UPDATE {self.table}
            SET status = 'pending', claimed_at = NULL, updated_at = {_sql_str(now)}
            WHERE status = 'running'
              AND (claimed_at IS NULL OR claimed_at < {_sql_str(cutoff)})
              AND attempt_count < {int(max_retries)}
            """
        )
        self.spark.sql(
            f"""
            UPDATE {self.table}
            SET status = 'failed', failed_at = {_sql_str(now)}, claimed_at = NULL,
                updated_at = {_sql_str(now)},
                error = COALESCE(error, 'stale running claim exhausted retries')
            WHERE status = 'running'
              AND (claimed_at IS NULL OR claimed_at < {_sql_str(cutoff)})
              AND attempt_count >= {int(max_retries)}
            """
        )
        after = self.spark.sql(f"SELECT COUNT(*) AS n FROM {self.table} WHERE status = 'running'").collect()[
            0
        ][0]
        return int(before) - int(after)

    def oldest_pending(self) -> Optional[CacheRow]:
        rows = self.spark.sql(
            f"SELECT * FROM {self.table} WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).collect()
        if not rows:
            return None
        return _row_from_spark(rows[0])

    def count(self) -> int:
        return int(self.spark.sql(f"SELECT COUNT(*) AS n FROM {self.table}").collect()[0][0])
