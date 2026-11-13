# SPDX-License-Identifier: Apache-2.0
"""One-off catalog pre-fill: extract UDFs from Python files and cross-compile them.

Point it at scripts (or a directory of them) before the jobs run::

    python -m ai_udf_transpile.prefill jobs/ --sqlite-path /tmp/ai_udf_transpile.sqlite \\
        --backend coco \\
        --writeback-table polaris.db.ai_udf_writeback --writeback-format iceberg \\
        --conf spark.sql.catalog.polaris=org.apache.iceberg.spark.SparkCatalog ...

Each file is parsed (never imported) by :mod:`ai_udf_transpile.extract`,
which rebuilds the exact cache key the runtime hook will compute. Every UDF
then goes through the same path a runtime miss does: ``lookup`` (a verified
row, local or in the write-back table, is a hit and costs nothing),
``insert_pending``, claim, :func:`ai_udf_transpile.worker.process_row`
(backend, Hypothesis gate, reconstruction smoke test), and publication to
the write-back table. Staged rows are flushed at the end regardless of
``writebackThreshold`` so a one-off run leaves nothing stranded locally.

Files are tracked by path + SHA-256 in a ``prefill_files`` table in the same
SQLite file: a file whose bytes, Spark version and extractor version match a
completed scan is not re-read, and identical content under another path is
not re-processed. A file is complete once every UDF in it has settled
(verified, or failed for a reason other than a transient backend error), so
an interrupted or rate-limited run picks up where it stopped.

Keys include the Spark version and the canonical (``ast.unparse``) source,
so pre-fill with the same Spark build and Python minor version the jobs use.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.catalog import HIT, MISS, CacheRow, dumps, infra_failure, iso_now, loads_list
from ai_udf_transpile.extract import EXTRACTOR_VERSION, ExtractedUDF, extract_file
from ai_udf_transpile.targets import NO_RUST_SUFFIX

logger = logging.getLogger(__name__)

_SKIP_DIRS = frozenset({"__pycache__", "node_modules", "site-packages", "venv"})

_FILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prefill_files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    spark_version TEXT NOT NULL,
    extractor_version INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    udf_keys TEXT,
    skipped TEXT,
    error TEXT,
    scanned_at TEXT
)
"""


class FileTracker:
    """``prefill_files``: which file contents were already fully pre-filled."""

    def __init__(self, sqlite_path: str):
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute(_FILES_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def done(self, path: str, sha256: str, spark_version: str) -> Optional[sqlite3.Row]:
        """The completed scan that covers these bytes: this path's own, else any path's."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM prefill_files WHERE sha256 = ? AND spark_version = ? "
                "AND extractor_version = ? AND complete = 1 "
                "ORDER BY (path = ?) DESC, scanned_at DESC LIMIT 1",
                (sha256, spark_version, EXTRACTOR_VERSION, path),
            )
            return cur.fetchone()

    def record(
        self,
        path: str,
        sha256: str,
        spark_version: str,
        *,
        udf_keys: list[str],
        skipped: list[dict],
        complete: bool,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO prefill_files (
                    path, sha256, spark_version, extractor_version, complete,
                    udf_keys, skipped, error, scanned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    sha256 = excluded.sha256,
                    spark_version = excluded.spark_version,
                    extractor_version = excluded.extractor_version,
                    complete = excluded.complete,
                    udf_keys = excluded.udf_keys,
                    skipped = excluded.skipped,
                    error = excluded.error,
                    scanned_at = excluded.scanned_at
                """,
                (
                    path,
                    sha256,
                    spark_version,
                    EXTRACTOR_VERSION,
                    1 if complete else 0,
                    dumps(udf_keys),
                    dumps(skipped),
                    error,
                    iso_now(),
                ),
            )


@dataclass
class UDFOutcome:
    path: str
    lineno: int
    name: str
    udf_key: str
    outcome: str  # cached | transpiled | failed | queued | in-flight | previously-failed
    detail: Optional[str] = None


@dataclass
class PrefillReport:
    files_scanned: int = 0
    files_unchanged: int = 0
    files_errored: int = 0
    udfs_found: int = 0
    udfs_skipped: int = 0
    udfs_gated: int = 0
    outcomes: list[UDFOutcome] = field(default_factory=list)
    flushed: int = 0

    def count(self, outcome: str) -> int:
        return sum(1 for o in self.outcomes if o.outcome == outcome)

    def summary(self) -> str:
        parts = [
            f"files: {self.files_scanned} scanned, {self.files_unchanged} unchanged, "
            f"{self.files_errored} unparseable",
            f"udfs: {self.udfs_found} found, {self.udfs_skipped} skipped (not keyable), "
            f"{self.udfs_gated} gated by inputCategories",
            "results: "
            + ", ".join(
                f"{self.count(name)} {name}"
                for name in ("cached", "transpiled", "failed", "queued", "in-flight", "previously-failed")
            ),
        ]
        if self.flushed:
            parts.append(f"write-back: {self.flushed} row(s) flushed")
        return "\n".join(parts)


def iter_python_files(paths: Iterable[str], exclude: Iterable[str] = ()) -> Iterator[Path]:
    """Explicit files as given; directories walked for ``*.py``, hidden/venv dirs skipped."""
    patterns = list(exclude)
    seen: set[Path] = set()

    def excluded(p: Path) -> bool:
        text = p.as_posix()
        return any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns)

    for raw in paths:
        root = Path(raw)
        if root.is_file():
            candidates: Iterable[Path] = [root]
        elif root.is_dir():
            candidates = sorted(
                p
                for p in root.rglob("*.py")
                if not any(
                    part.startswith(".") or part in _SKIP_DIRS for part in p.relative_to(root).parts[:-1]
                )
            )
        else:
            logger.warning("prefill: %s does not exist", raw)
            continue
        for p in candidates:
            resolved = p.resolve()
            if resolved in seen or excluded(p):
                continue
            seen.add(resolved)
            yield resolved


def _settled(row: Optional[CacheRow]) -> bool:
    """Nothing more for a pre-fill to do: verified, or failed for a real reason."""
    if row is None:
        return False
    if row.status == "success":
        return True
    return row.status == "failed" and not infra_failure(row)


def _still_settled(catalog: Any, keys: list[str], spark: Any) -> bool:
    """Whether a finished scan still holds, asked through ``lookup``.

    The file's bytes are unchanged, but the catalog may not be: a stricter
    tolerance than the rows were verified at (lookup re-queues them), a
    failed row past its cooldown (lookup makes it pending again), a row
    deleted. Any of those means the file is worth a pass.
    """
    for key in keys:
        kind, row = catalog.lookup(key, spark=spark)
        if kind != HIT and not _settled(row):
            return False
    return True


def _prefill_udf(
    udf: ExtractedUDF,
    key: str,
    *,
    catalog: Any,
    backend: Any,
    spark: Any,
    spark_version: str,
    queue_only: bool,
    verify_fn: Optional[Callable[..., tuple[bool, Optional[str]]]],
) -> UDFOutcome:
    from ai_udf_transpile.transpiler import _rust_elsewhere
    from ai_udf_transpile.worker import process_row

    # Same routing as the hook: a Rust row this session cannot serve sends the
    # UDF to its +no-rust twin key, which gets its own SQL/Java rewrite.
    peek = catalog.get(key)
    if peek is not None and _rust_elsewhere(peek, spark):
        spark_version += NO_RUST_SUFFIX
        key = udf.key(spark_version)

    def outcome(name: str, detail: Optional[str] = None) -> UDFOutcome:
        return UDFOutcome(udf.path, udf.lineno, udf.name, key, name, detail)

    kind, row = catalog.lookup(key, spark=spark)
    if kind == HIT:
        return outcome("cached", (row.origin or "") if row is not None else None)
    if kind == MISS:
        catalog.insert_pending(
            udf_key=key,
            source_text=udf.source_text,
            param_names=list(udf.param_names),
            input_types=list(udf.input_types),
            input_categories=list(udf.input_categories),
            return_type=udf.return_type,
            spark_version=spark_version,
            closure_fingerprint=udf.closure_fingerprint,
            captures=dict(udf.captures),
        )
    row = catalog.get(key)
    if row is None:
        return outcome("failed", "catalog lost the pending row")
    if row.status == "failed":
        return outcome("previously-failed", row.error)
    if row.status != "pending":
        return outcome("in-flight", f"status={row.status}")
    if queue_only:
        return outcome("queued")
    if not catalog.claim(key, getattr(backend, "name", "unknown")):
        return outcome("in-flight", "claimed by another worker")
    process_row(catalog, catalog.get(key) or row, backend, spark, verify_fn=verify_fn)
    final = catalog.get(key)
    if final is not None and final.status == "success":
        return outcome("transpiled", f"{final.target_kind} via {final.origin}/{final.model or '?'}")
    return outcome("failed", final.error if final is not None else None)


def prefill_paths(
    paths: Iterable[str],
    *,
    catalog: Any,
    backend: Any,
    spark: Any,
    spark_version: str,
    tracker: Optional[FileTracker] = None,
    queue_only: bool = False,
    rescan: bool = False,
    exclude: Iterable[str] = (),
    verify_fn: Optional[Callable[..., tuple[bool, Optional[str]]]] = None,
    log: Callable[[str], None] = print,
    verbose: bool = False,
) -> PrefillReport:
    """Extract and cross-compile every UDF under ``paths`` into ``catalog``."""
    from ai_udf_transpile.transpiler import _inputs_allowed

    report = PrefillReport()
    # Like poll_once: a run killed mid-claim leaves rows 'running'; past
    # claimTimeoutSeconds they go back to pending (or failed at maxRetries)
    # so this run can pick them up instead of reporting them in-flight forever.
    timeout = conf.get_int(conf.CLAIM_TIMEOUT, spark, int(conf.DEFAULTS[conf.CLAIM_TIMEOUT]))
    max_retries = conf.get_int(conf.MAX_RETRIES, spark, int(conf.DEFAULTS[conf.MAX_RETRIES]))
    catalog.reclaim_stale(timeout, max_retries)
    for path in iter_python_files(paths, exclude):
        try:
            extraction = extract_file(path)
        except OSError as exc:
            log(f"{path}: unreadable ({exc})")
            report.files_errored += 1
            continue
        if tracker is not None and not rescan:
            prior = tracker.done(str(path), extraction.sha256, spark_version)
            if prior is not None and _still_settled(catalog, loads_list(prior["udf_keys"]), spark):
                report.files_unchanged += 1
                if prior["path"] != str(path):
                    # Same bytes under a new path: inherit the finished scan.
                    tracker.record(
                        str(path),
                        extraction.sha256,
                        spark_version,
                        udf_keys=loads_list(prior["udf_keys"]),
                        skipped=loads_list(prior["skipped"]),
                        complete=True,
                        error=prior["error"],
                    )
                if verbose:
                    log(f"{path}: unchanged since last pre-fill, skipped")
                continue
        report.files_scanned += 1
        skipped = [{"line": s.lineno, "name": s.name, "reason": s.reason} for s in extraction.skipped]
        if extraction.error:
            report.files_errored += 1
            log(f"{path}: not parseable ({extraction.error})")
            if tracker is not None:
                # The same bytes will not parse next time either.
                tracker.record(
                    str(path),
                    extraction.sha256,
                    spark_version,
                    udf_keys=[],
                    skipped=[],
                    complete=True,
                    error=extraction.error,
                )
            continue
        report.udfs_found += len(extraction.udfs)
        report.udfs_skipped += len(extraction.skipped)
        if extraction.udfs or extraction.skipped:
            log(f"{path}: {len(extraction.udfs)} UDF(s), {len(extraction.skipped)} skipped")
        for s in extraction.skipped:
            if verbose:
                log(f"  skip  {s.name} (line {s.lineno}): {s.reason}")
        keys: list[str] = []
        settled = True
        for udf in extraction.udfs:
            key = udf.key(spark_version)
            if not _inputs_allowed(udf.input_categories, spark):
                # Not settled: opting the category in later must revisit this file.
                settled = False
                report.udfs_gated += 1
                log(f"  gated {udf.name} (line {udf.lineno}): inputs {udf.input_categories} not enabled")
                continue
            try:
                result = _prefill_udf(
                    udf,
                    key,
                    catalog=catalog,
                    backend=backend,
                    spark=spark,
                    spark_version=spark_version,
                    queue_only=queue_only,
                    verify_fn=verify_fn,
                )
            except Exception as exc:
                logger.exception("prefill failed for %s:%s", path, udf.name)
                why = f"{type(exc).__name__}: {exc}"
                result = UDFOutcome(udf.path, udf.lineno, udf.name, key, "failed", why)
            report.outcomes.append(result)
            key = result.udf_key  # the +no-rust twin when the hook would route there
            keys.append(key)
            detail = f" -- {result.detail}" if result.detail else ""
            log(f"  {result.outcome:<12} {udf.name} (line {udf.lineno}) {key[:12]}{detail}")
            settled = settled and _settled(catalog.get(key))
        if tracker is not None:
            tracker.record(
                str(path),
                extraction.sha256,
                spark_version,
                udf_keys=keys,
                skipped=skipped,
                complete=settled,
            )
    return report


def _parse_conf(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--conf expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out.append((key.strip(), value))
    return out


def _spark_session(master: str, confs: list[tuple[str, str]]) -> Any:
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master(master)
        .appName("ai-udf-transpile-prefill")
        .config("spark.ui.enabled", "false")
        # The runtime only transpiles under ANSI, and the smoke test
        # reconstructs through that same path.
        .config("spark.sql.ansi.enabled", "true")
    )
    for key, value in confs:
        builder = builder.config(key, value)
    return builder.getOrCreate()


def _dry_run(args: argparse.Namespace) -> int:
    version = args.spark_version or "unknown"
    found = skipped = 0
    for path in iter_python_files(args.paths, args.exclude):
        extraction = extract_file(path)
        if extraction.error:
            print(f"{path}: not parseable ({extraction.error})")
            continue
        for udf in extraction.udfs:
            found += 1
            params = ", ".join(f"{n}: {t}" for n, t in zip(udf.param_names, udf.input_types))
            key = udf.key(version)[:12]
            print(f"{path}:{udf.lineno}  {udf.name}({params}) -> {udf.return_type}  key={key}")
        for s in extraction.skipped:
            skipped += 1
            print(f"{path}:{s.lineno}  skip {s.name}: {s.reason}")
    print(f"{found} UDF(s) extracted, {skipped} skipped (keys computed for Spark {version})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ai_udf_transpile.prefill",
        description="Extract PySpark UDFs from Python files and pre-fill the transpile catalog.",
    )
    parser.add_argument("paths", nargs="+", help="Python files or directories (walked for *.py)")
    parser.add_argument("--sqlite-path", default=os.environ.get("AI_UDF_TRANSPILE_SQLITE"))
    parser.add_argument("--backend", default=None, help="coco, pi, claude, codex, cursor, ensemble, auto")
    parser.add_argument("--writeback-table", default=None, help="e.g. polaris.db.ai_udf_writeback")
    parser.add_argument("--writeback-format", default=None, choices=["iceberg", "parquet"])
    parser.add_argument("--writeback-threshold", type=int, default=None)
    parser.add_argument("--master", default="local[1]")
    parser.add_argument(
        "--conf",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Spark conf, repeatable (Iceberg catalog settings, aiUdfTranspile.* knobs)",
    )
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    parser.add_argument("--queue-only", action="store_true", help="queue pending rows for a worker")
    parser.add_argument("--rescan", action="store_true", help="ignore the file hash tracker")
    parser.add_argument("--dry-run", action="store_true", help="list extracted UDFs; no Spark, no catalog")
    parser.add_argument("--spark-version", default=None, help="key version for --dry-run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.dry_run:
        return _dry_run(args)
    if not args.sqlite_path:
        parser.error("--sqlite-path is required (or $AI_UDF_TRANSPILE_SQLITE)")
    try:
        confs = _parse_conf(args.conf)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    from ai_udf_transpile import enable, shutdown
    from ai_udf_transpile.backends import get_backend

    spark = _spark_session(args.master, confs)
    if args.writeback_table:
        conf.set_value(conf.WRITEBACK_TABLE, args.writeback_table, spark)
    if args.writeback_format:
        conf.set_value(conf.WRITEBACK_FORMAT, args.writeback_format, spark)
    if args.writeback_threshold is not None:
        conf.set_value(conf.WRITEBACK_THRESHOLD, args.writeback_threshold, spark)
    catalog = enable(spark, backend=args.backend, inline_worker=False, sqlite_path=args.sqlite_path)
    tracker = FileTracker(args.sqlite_path)
    try:
        # --queue-only never runs a backend: do not require its CLI on this machine.
        backend = None if args.queue_only else get_backend(args.backend, spark)
        report = prefill_paths(
            args.paths,
            catalog=catalog,
            backend=backend,
            spark=spark,
            spark_version=str(spark.version),
            tracker=tracker,
            queue_only=args.queue_only,
            rescan=args.rescan,
            exclude=args.exclude,
            verbose=args.verbose,
        )
        flush = getattr(catalog, "flush", None)
        if callable(flush):
            report.flushed = flush()
        print(report.summary())
    finally:
        tracker.close()
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
