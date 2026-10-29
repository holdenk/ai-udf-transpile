# SPDX-License-Identifier: Apache-2.0
"""Benchmark every function in the filled catalog: interpreted python vs the
rewrite the pipeline actually serves. This is the coverage bench -- the
micro/tpcds benches go deep on their own functions, this one goes wide over
everything the examples queued, so docs/EXAMPLES_TRANSPILED.md gets numbers
for all of it.

  python examples/bench_catalog.py --sqlite target/examples_transpiled.sqlite
  python examples/bench_catalog.py --sqlite ... --csv out.csv --rows 100000

Fail-open by function: one benign sample value per input type, and any
function whose interpreted arm raises on it (kms_encrypt, ...) is skipped
loudly rather than given numbers.

When $AI_UDF_TRANSPILE_COMET_JAR (or $COMET_JAR) points at a Comet jar, a
second session runs the served SQL under Comet's native acceleration too,
writing comet_rows_per_second / comet_speedup_vs_python -- the doc's
"comet" arm. A function whose plan falls back to Spark gets no comet
number, said loudly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import os
import shutil
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A local bench must not call cloud services: ambient AWS variables plus
# ~/.aws credential sources would make service-calling fixtures (kms_encrypt)
# hit the real endpoint mid-bench. Scrub every source, like tests/conftest.py.
for _key in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_CA_BUNDLE",
    "AWS_PROFILE",
):
    os.environ.pop(_key, None)
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/dev/null"
os.environ["AWS_CONFIG_FILE"] = "/dev/null"

# Workers must run the same interpreter as the driver (cloudpickle shares
# bytecode); run_demos.sh does this for the other benches, so pin it here to
# stay correct standalone.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import _parse_datatype_string  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.catalog.sqlite import SqliteCatalog, _row_from_sql  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"

# One benign value per type. Strings want to satisfy the date-slicing
# fixtures too (udfb_extractyear/extractmonth/extractday all slice this).
_SAMPLES = {
    "bigint": lambda: 7,
    "int": lambda: 7,
    "double": lambda: 1.5,
    "float": lambda: 1.5,
    "string": lambda: "2015-08-17",
    "boolean": lambda: True,
    "timestamp": lambda: datetime(2020, 1, 2, 3, 4, 5),
    "date": lambda: date(2020, 1, 2),
}


def _values_for(input_types: list[str]) -> list | None:
    out: list = []
    for raw in input_types:
        t = (raw or "").strip().lower()
        if t in _SAMPLES:
            out.append(_SAMPLES[t]())
        elif t.startswith("array<") and t.endswith(">"):
            inner = _values_for([t[6:-1]])
            if inner is None:
                return None
            out.append(inner)
        elif t.startswith("map<") and t.endswith(">"):
            k, _, v = t[4:-1].partition(",")
            pair = _values_for([k, v])
            if pair is None:
                return None
            out.append({pair[0]: pair[1]})
        else:
            return None  # unknown type: no honest sample, skip the function
    return out


def _function_name(source_text: str) -> str:
    try:
        tree = ast.parse(source_text or "")
        fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return fn.name
    except Exception:
        return ""


# The plugin's type gate does inspect.getsource(func) at construction time,
# so imported sources must outlive the bench; one dir for the whole run.
_IMPORT_DIR: Path | None = None


def _import_dir() -> Path:
    global _IMPORT_DIR
    if _IMPORT_DIR is None:
        _IMPORT_DIR = Path(tempfile.mkdtemp(prefix="ai-udf-bench-catalog-"))
    return _IMPORT_DIR


def _import_function(source_text: str, name: str):
    """Exec row.source_text in its own module; the file stays on disk for the run."""
    path = _import_dir() / f"udf_{name}.py"
    path.write_text("from __future__ import annotations\n" + source_text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"bench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def _time_arm(df, udf, rounds: int) -> float | None:
    """Best (min) of N rounds; None when the function raises on the sample."""
    arm = df.select(udf(*[f"arg{i}" for i in range(len(df.columns))]))
    try:
        noop_seconds(arm)  # warmup doubles as the raise check
    except Exception:
        return None
    best = float("inf")
    for _ in range(rounds):
        try:
            best = min(best, noop_seconds(arm))
        except Exception:
            break  # a flaky late failure keeps the rounds that did land
    return best if best < float("inf") else None


def _comet_phase(args, rows, out: dict[str, dict], py_secs: dict[str, float], comet_jar: str) -> None:
    """The same served SQL, executed natively by Comet, in a fresh JVM: the jar
    must be on the driver classpath at launch, so the first session's gateway
    (and its JVM) is retired first."""
    from pyspark import SparkContext

    gateway = SparkContext._gateway
    if gateway is not None:
        proc = getattr(gateway, "proc", None)
        gateway.shutdown()
        if proc is not None:
            proc.terminate()
    SparkContext._gateway = None
    SparkContext._jvm = None
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench-catalog-comet")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.jars", comet_jar)
        .config("spark.driver.extraClassPath", comet_jar)
        .config("spark.executor.extraClassPath", comet_jar)
        .config("spark.plugins", "org.apache.spark.CometPlugin")
        .config("spark.comet.enabled", "true")
        # Without this the extension disables itself on session start.
        .config("spark.comet.shuffle.enabled", "false")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "1g")
        .getOrCreate()
    )
    scratch = tempfile.mkdtemp(prefix="bench-catalog-comet-")
    try:
        enable(spark, sqlite_path=args.sqlite, inline_worker=False)
        spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
        spark.conf.set(TRANSPILE_CONF, "true")
        cats = sorted({c for r in rows for c in r.input_categories})
        conf.set_value(conf.INPUT_CATEGORIES, ",".join(cats), spark)
        for row in rows:
            entry = out.get(row.udf_key)
            if entry is None or entry["status"] != "success" or not entry["transpiled_rows_per_second"]:
                continue
            name = entry["function"]
            values = _values_for(row.input_types)
            try:
                func = _import_function(row.source_text, name)
                # Comet has no native LocalTableScan, and an all-literal frame
                # would constant-fold the UDF away; a parquet scan is a real leaf.
                path = f"{scratch}/{row.udf_key}"
                spark.createDataFrame(
                    [tuple(values)] * args.rows,
                    schema=", ".join(f"arg{i} {t}" for i, t in enumerate(row.input_types)),
                ).write.parquet(path)
                df = spark.read.parquet(path)
            except Exception as exc:
                print(f"{name}: no comet frame for sample ({type(exc).__name__})")
                continue
            udf = UserDefinedFunction(func, _parse_datatype_string(row.return_type))
            if not udf.transpiled:
                continue
            arm = df.select(udf(*[f"arg{i}" for i in range(len(row.input_types))]))
            if "CometProject" not in arm._jdf.queryExecution().executedPlan().toString():
                print(f"{name}: comet fell back to Spark -- no native number")
                continue
            comet_secs = _time_arm(df, udf, args.rounds)
            if comet_secs:
                entry["comet_rows_per_second"] = round(args.rows / comet_secs)
                if row.udf_key in py_secs:
                    entry["comet_speedup_vs_python"] = round(py_secs[row.udf_key] / comet_secs, 1)
                print(
                    f"{name:24} comet={entry['comet_rows_per_second']:>10}"
                    f" ({entry['comet_speedup_vs_python'] or '-'}x vs python)"
                )
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        shutdown()
        spark.stop()
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=os.environ.get("AI_UDF_TRANSPILE_SQLITE", ""))
    parser.add_argument("--rows", type=int, default=100_000, help="rows per function")
    parser.add_argument("--rounds", type=int, default=3, help="timed rounds (min wins)")
    parser.add_argument("--csv", help="results CSV")
    args = parser.parse_args()
    if not args.sqlite:
        parser.error("need --sqlite or $AI_UDF_TRANSPILE_SQLITE (the filled catalog to bench)")
    if not Path(args.sqlite).exists():
        print(f"no catalog at {args.sqlite}", file=sys.stderr)
        return 1

    catalog = SqliteCatalog(Path(args.sqlite))
    rows = sorted(
        (_row_from_sql(raw) for raw in catalog._conn.execute("SELECT * FROM cache").fetchall()),
        key=lambda r: (_function_name(r.source_text), r.udf_key),
    )
    if not rows:
        print(f"{args.sqlite} has zero rows", file=sys.stderr)
        return 1

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench-catalog")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    # Cache hits only: no backend needed (and none consulted -- a CI env that
    # sets AI_UDF_TRANSPILE_BACKEND=fake must not break this, and a version
    # mismatch must not let a worker fire AI calls or write to the catalog
    # mid-bench).
    enable(spark, sqlite_path=args.sqlite, inline_worker=False)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    # The gate's default categories (numeric,string,bool,binary) would block
    # the timestamp/array rows the catalog holds; allow exactly what we bench.
    cats = sorted({c for r in rows for c in r.input_categories})
    conf.set_value(conf.INPUT_CATEGORIES, ",".join(cats), spark)
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)

    out: dict[str, dict] = {}  # by udf_key: two cache rows may share a function name
    py_secs: dict[str, float] = {}
    for row in rows:
        name = _function_name(row.source_text) or row.udf_key[:12]
        values = _values_for(row.input_types)
        func = None
        if values is not None:
            try:
                func = _import_function(row.source_text, name)
            except Exception as exc:
                print(f"{name}: import failed: {type(exc).__name__}: {exc}")
        entry = {
            "function": name,
            "rows": args.rows,
            "python_rows_per_second": "",
            "transpiled_rows_per_second": "",
            "transpiled_speedup_vs_python": "",
            "comet_rows_per_second": "",
            "comet_speedup_vs_python": "",
            "status": row.status,
            "target_kind": row.target_kind or "",
            "backend": row.backend or "",
            "model": row.model or "",
        }
        out[row.udf_key] = entry
        if values is None or func is None:
            print(f"{name}: skipped (no sample values or not importable)")
            continue
        df = spark.createDataFrame(
            [tuple(values)] * args.rows,
            schema=", ".join(f"arg{i} {t}" for i, t in enumerate(row.input_types)),
        )
        rtype = _parse_datatype_string(row.return_type)

        spark.conf.set(TRANSPILE_CONF, "false")
        py_t = _time_arm(df, UserDefinedFunction(func, rtype), args.rounds)
        tx_secs = None
        spark.conf.set(TRANSPILE_CONF, "true")
        if row.status == "success":
            udf = UserDefinedFunction(func, rtype)
            if udf.transpiled:
                tx_secs = _time_arm(df, udf, args.rounds)
            else:
                print(f"{name}: transpiled arm not served (no hit) -- python only")
        py_rate = round(args.rows / py_t) if py_t else ""
        tx_rate = round(args.rows / tx_secs) if tx_secs else ""
        entry["python_rows_per_second"] = py_rate
        entry["transpiled_rows_per_second"] = tx_rate
        entry["transpiled_speedup_vs_python"] = round(py_t / tx_secs, 1) if (py_t and tx_secs) else ""
        if py_t:
            py_secs[row.udf_key] = py_t
        note = "" if py_t else "python raises on the sample; skipped"
        print(
            f"{name:24} python={py_rate or '-':>10} transpiled={tx_rate or '-':>10}"
            f" speedup={entry['transpiled_speedup_vs_python'] or '-':>6} {note}"
        )

    spark.conf.set(TRANSPILE_CONF, "false")
    shutdown()
    spark.stop()

    comet_jar = os.environ.get("AI_UDF_TRANSPILE_COMET_JAR") or os.environ.get("COMET_JAR") or ""
    if comet_jar and not Path(comet_jar).exists():
        print(f"comet jar {comet_jar} not found; skipping comet arm", file=sys.stderr)
        comet_jar = ""
    if comet_jar:
        try:
            _comet_phase(args, rows, out, py_secs, comet_jar)
        except Exception as exc:  # fail open: the python/transpiled numbers still land
            print(f"comet arm failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(next(iter(out.values()))))
            writer.writeheader()
            writer.writerows(out.values())
        print(f"csv: wrote {len(out)} rows to {args.csv}")

    if _IMPORT_DIR is not None:
        shutil.rmtree(_IMPORT_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
