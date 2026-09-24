# SPDX-License-Identifier: Apache-2.0
"""Worker: claim pending rows, run a backend, Hypothesis-gate, write success/failed.

``process_row`` is the unit of work. vNext could feed it keys from Kafka; v1 polls
SQLite/Delta. Do not deploy a broker.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.backends import BackendDecline, BackendError, get_backend
from ai_udf_transpile.catalog import Catalog, open_catalog
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.verify import hypothesis_check

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()
_worker_spark: Any = None


def stop_inline() -> None:
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
        thread.join(timeout=5)


def inline_thread_alive() -> bool:
    return _thread is not None and _thread.is_alive()


def process_row(
    catalog: Catalog,
    row: Any,
    backend: Any,
    spark: Any = None,
    *,
    verify_fn: Optional[Callable[..., tuple[bool, Optional[str]]]] = None,
) -> None:
    """Run one claimed row to success or failed. Never leaves status=running."""
    key = row.udf_key
    verify_fn = verify_fn or hypothesis_check
    try:
        if (row.origin == "human") and (row.catalyst_sql or row.impl_class or row.impl_source):
            result = TranspileResult(
                kind=row.target_kind or "catalyst",
                sql=row.catalyst_sql,
                java_source=row.impl_source,
                class_name=row.impl_class,
                binary=row.impl_binary,
                entry=row.impl_entry,
            )
            origin = "human"
        else:
            job = row.as_job()
            with make_sandbox(job) as sandbox:
                result = backend.run(job, sandbox)
            origin = getattr(backend, "name", "unknown")
        max_examples = conf.get_int(conf.MAX_EXAMPLES, spark, int(conf.default_max_examples()))
        ok, err = verify_fn(
            source_text=row.source_text,
            captures=row.captures,
            result=result,
            input_types=row.input_types,
            return_type=row.return_type,
            spark=spark,
            max_examples=max_examples,
        )
        if ok:
            catalog.mark_success(key, result, origin, hypothesis_passed=True)
            logger.info("transpile success key=%s origin=%s kind=%s", key[:12], origin, result.kind)
            return
        catalog.mark_failed(key, err or "hypothesis failed")
        logger.info("transpile failed key=%s: %s", key[:12], err)
    except BackendDecline as exc:
        try:
            catalog.mark_failed(key, f"declined: {exc}")
        except Exception:
            logger.exception("failed to mark declined row %s", key)
        logger.info("backend declined key=%s: %s", key[:12], exc)
    except (BackendError, Exception) as exc:
        try:
            catalog.mark_failed(key, f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.exception("failed to mark error row %s", key)
        logger.exception("process_row failed key=%s", key)


def poll_once(
    catalog: Catalog,
    backend: Any,
    spark: Any = None,
    *,
    verify_fn: Optional[Callable[..., tuple[bool, Optional[str]]]] = None,
) -> bool:
    """Reclaim stale claims, CAS-claim the oldest pending row, process it.

    Returns True if a row was claimed (whether it ended success or failed).
    """
    timeout = conf.get_int(conf.CLAIM_TIMEOUT, spark, int(conf.DEFAULTS[conf.CLAIM_TIMEOUT]))
    max_retries = conf.get_int(conf.MAX_RETRIES, spark, int(conf.DEFAULTS[conf.MAX_RETRIES]))
    catalog.reclaim_stale(timeout, max_retries)
    row = catalog.oldest_pending()
    if row is None:
        return False
    if not catalog.claim(row.udf_key, getattr(backend, "name", "unknown")):
        return False
    claimed = catalog.get(row.udf_key) or row
    process_row(catalog, claimed, backend, spark, verify_fn=verify_fn)
    return True


def _loop(catalog: Catalog, spark: Any) -> None:
    interval = conf.get_float(conf.POLL_INTERVAL, spark, float(conf.DEFAULTS[conf.POLL_INTERVAL]))
    while not _stop.is_set():
        try:
            backend = get_backend(spark=spark)
            poll_once(catalog, backend, spark)
        except Exception:
            logger.exception("inline worker poll failed")
        _stop.wait(interval)


def maybe_start_inline(spark: Any, catalog: Catalog) -> None:
    global _thread, _worker_spark
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _worker_spark = spark
        _thread = threading.Thread(
            target=_loop,
            name="ai-udf-transpile-worker",
            args=(catalog, spark),
            daemon=True,
        )
        _thread.start()
        logger.info("started inline ai-udf-transpile worker")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Poll the AI UDF transpile catalog")
    parser.add_argument("--sqlite-path", default=os.environ.get("AI_UDF_TRANSPILE_SQLITE"))
    parser.add_argument("--backend", default=os.environ.get("AI_UDF_TRANSPILE_BACKEND", "fake"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="process at most one row and exit")
    parser.add_argument("--no-spark", action="store_true", help="skip SparkSession (verify will fail)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    if not args.sqlite_path:
        parser.error("--sqlite-path is required for the standalone worker")

    conf.set_value(conf.CATALOG, "sqlite")
    conf.set_value(conf.SQLITE_PATH, args.sqlite_path)
    conf.set_value(conf.BACKEND, args.backend)
    conf.set_value(conf.POLL_INTERVAL, args.poll_interval)

    spark = None
    if not args.no_spark:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession() or SparkSession._instantiatedSession
            if spark is None:
                spark = (
                    SparkSession.builder.master("local[1]")
                    .appName("ai-udf-transpile-worker")
                    .config("spark.ui.enabled", "false")
                    .config("spark.sql.ansi.enabled", "true")
                    .getOrCreate()
                )
        except Exception:
            logger.exception("could not start SparkSession; verify will fail closed")
            spark = None

    catalog = open_catalog(spark, sqlite_path=args.sqlite_path)
    backend = get_backend(args.backend, spark)
    if args.once:
        poll_once(catalog, backend, spark)
        return 0
    _stop.clear()
    try:
        while not _stop.is_set():
            poll_once(catalog, backend, spark)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
