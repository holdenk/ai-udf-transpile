# SPDX-License-Identifier: Apache-2.0
"""Ask the live backend for ONE rewrite kind per UDF, into its own catalog.

The examples catalog holds whichever kind the model picked first. The doc
wants all of them side by side, so this takes every UDF source from that
catalog and runs the real pipeline (backend, Hypothesis gate, smoke test)
once per kind, with `targets` pinned to it. A kind the model cannot honestly
produce lands as a failed row with the reason -- the rejection is the result.

  python examples/fill_targets.py --target catalyst   # SQL
  python examples/fill_targets.py --target java_udf   # Java
  python examples/fill_targets.py --target rust       # needs $COMET_JAR + cargo
  python examples/fill_targets.py --target all        # each kind in turn, its own process

Resumable: a row that already settled in the per-kind catalog is skipped.
One function at a time, on purpose.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ.pop(_key, None)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from bench_catalog import _function_name, _session  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.backends import get_backend  # noqa: E402
from ai_udf_transpile.catalog import CacheRow, infra_failure  # noqa: E402
from ai_udf_transpile.catalog.sqlite import SqliteCatalog, _row_from_sql  # noqa: E402
from ai_udf_transpile.targets import NO_RUST_SUFFIX  # noqa: E402
from ai_udf_transpile.worker import process_row  # noqa: E402

KINDS = ("catalyst", "java_udf", "rust")


def _comet_confs(jar: str) -> dict[str, str]:
    return {
        "spark.jars": jar,
        "spark.driver.extraClassPath": jar,
        "spark.executor.extraClassPath": jar,
        "spark.plugins": "org.apache.spark.CometPlugin",
        "spark.comet.enabled": "true",
        "spark.comet.shuffle.enabled": "false",
        "spark.comet.exec.range.enabled": "true",
        "spark.memory.offHeap.enabled": "true",
        "spark.memory.offHeap.size": "2g",
    }


def _fill_all(args: argparse.Namespace) -> int:
    """Every kind, one child process each: a kind's Spark session (Comet's jar
    for rust) is fixed when its JVM starts, so kinds cannot share one."""
    failed = 0
    for kind in KINDS:
        if kind == "rust" and not (
            os.environ.get("AI_UDF_TRANSPILE_COMET_JAR") or os.environ.get("COMET_JAR")
        ):
            print("##### rust: skipped (no $COMET_JAR; source .comet-env) #####")
            continue
        print(f"##### {kind} #####", flush=True)
        argv = [sys.executable, "-u", __file__, "--target", kind, "--source", args.source]
        for name in args.only:
            argv += ["--only", name]
        if subprocess.run(argv, check=False).returncode != 0:
            print(f"{kind}: fill failed", file=sys.stderr)
            failed = 1
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        choices=[*KINDS, "all"],
        help="the one rewrite kind to ask for; all = each kind separately (rust only with $COMET_JAR)",
    )
    parser.add_argument(
        "--source", default="target/examples_transpiled.sqlite", help="catalog holding the UDF sources"
    )
    parser.add_argument("--out", help="per-kind catalog (default target/targets/<kind>.sqlite)")
    parser.add_argument("--only", action="append", default=[], help="function name to run (repeatable)")
    args = parser.parse_args()
    if args.target == "all":
        if args.out:
            parser.error("--out names one kind's catalog; with --target all each kind uses its default")
        return _fill_all(args)
    out = Path(args.out or f"target/targets/{args.target}.sqlite")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.source).is_file():
        print(f"no source catalog at {args.source}", file=sys.stderr)
        return 1
    src = SqliteCatalog(Path(args.source))
    sources: dict[str, CacheRow] = {}
    for raw in src._conn.execute("SELECT * FROM cache").fetchall():
        row = _row_from_sql(raw)
        if not row.spark_version.endswith(NO_RUST_SUFFIX) and row.origin != "human":
            sources.setdefault(row.udf_key, row)
    rows = sorted(sources.values(), key=lambda r: (_function_name(r.source_text), r.udf_key))
    if args.only:
        rows = [r for r in rows if _function_name(r.source_text) in args.only]
    source_samples = {r.udf_key: src.samples_for(r.udf_key) for r in rows}
    src.close()
    if not rows:
        print("no matching UDF sources", file=sys.stderr)
        return 1

    extra = None
    if args.target == "rust":
        jar = os.environ.get("AI_UDF_TRANSPILE_COMET_JAR") or os.environ.get("COMET_JAR") or ""
        if not Path(jar).is_file():
            print("rust target needs $COMET_JAR (source .comet-env)", file=sys.stderr)
            return 1
        extra = _comet_confs(jar)
    spark = _session(f"ai-udf-transpile-fill-{args.target}", extra)
    catalog = enable(spark, sqlite_path=str(out), inline_worker=False)
    conf.set_value(conf.TARGETS, args.target, spark)
    conf.set_value(conf.PARTY_ON_STRINGS, "true", spark)
    conf.set_value(
        conf.INPUT_CATEGORIES, ",".join(sorted({c for r in rows for c in r.input_categories})), spark
    )
    backend = get_backend(spark=spark)
    failed = False
    try:
        for i, row in enumerate(rows, 1):
            name = _function_name(row.source_text) or row.udf_key[:12]
            have = catalog.get(row.udf_key)
            if have is not None and have.status in ("success", "failed") and not infra_failure(have):
                print(f"[{i}/{len(rows)}] {name}: already {have.status}")
                continue
            if have is not None and have.status != "pending":
                # A transient backend error: start the row over.
                catalog._conn.execute("DELETE FROM cache WHERE udf_key = ?", (row.udf_key,))
                catalog._conn.commit()
                have = None
            if have is None:
                catalog.insert_pending(
                    udf_key=row.udf_key,
                    source_text=row.source_text,
                    param_names=list(row.param_names),
                    input_types=list(row.input_types),
                    input_categories=list(row.input_categories),
                    return_type=row.return_type,
                    spark_version=row.spark_version,
                    closure_fingerprint=row.closure_fingerprint,
                    captures=dict(row.captures),
                )
            catalog.record_samples(row.udf_key, source_samples[row.udf_key])
            if not catalog.claim(row.udf_key, getattr(backend, "name", "unknown")):
                print(f"[{i}/{len(rows)}] {name}: claim failed", file=sys.stderr)
                failed = True
                continue
            process_row(catalog, catalog.get(row.udf_key), backend, spark)
            final = catalog.get(row.udf_key)
            if final.status not in ("success", "failed") or infra_failure(final):
                failed = True
            print(
                f"[{i}/{len(rows)}] {name}: {final.status} {final.target_kind or ''} "
                f"{(final.error or '')[:160]}"
            )
    finally:
        shutdown()
        spark.stop()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
