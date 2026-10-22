# SPDX-License-Identifier: Apache-2.0
"""TPC-DS-inspired UDF benchmark: Python UDF vs vanilla SQL vs transpiled SQL.

Each UDF below is a pattern extracted from a real TPC-DS query (source in the
comment).  We run three arms per pattern:

  python    the UDF as an interpreted Python UDF (transpilePyUDFs=false)
  vanilla   the native SQL expression the UDF is equivalent to
  transpiled  the UDF after AI transpilation (fake backend, inline worker)

The point: a Python UDF doing TPC-DS-like work is 5-10x slower than native
SQL, and transpilation closes that gap entirely -- the transpiled arm matches
the vanilla arm because it IS the vanilla SQL.

  export SPARK_HOME=~/spark
  python examples/bench_tpcds_udfs.py [rows] [--rounds N] [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.backends.fake import (  # noqa: E402
    common_c2f,
    common_discount_price,
    common_pct_change,
    common_url_decode,
    spark_add_one,
    spark_as_double,
    spark_to_lower,
    spark_to_upper,
    spark_trunc,
    tpcds_in_quarter,
    tpcds_safe_ratio,
    tpcds_sales_deviation,
    tpcds_zip5_mismatch,
)
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
PARTITIONS = os.cpu_count() or 16

# (name, func, return_type, input_cols, vanilla_sql)
# input_cols: list of (col_name, col_type) pairs for the synthetic frame.
# vanilla_sql: the native SQL expression (uses _udf_param_N placeholders,
#   which the bench substitutes with real column names).
PATTERNS = [
    # From apache/spark test_udf.py
    ("spark_add_one", spark_add_one, LongType(), [("x", "bigint")], "_udf_param_0 + 1"),
    ("spark_to_upper", spark_to_upper, StringType(), [("s", "string")], "upper(_udf_param_0)"),
    ("spark_to_lower", spark_to_lower, StringType(), [("s", "string")], "lower(_udf_param_0)"),
    ("spark_trunc", spark_trunc, LongType(), [("x", "double")], "cast(_udf_param_0 as bigint)"),
    ("spark_as_double", spark_as_double, DoubleType(), [("x", "bigint")], "cast(_udf_param_0 as double)"),
    # From TPC-DS modified queries
    (
        "tpcds_zip5_mismatch",
        tpcds_zip5_mismatch,
        BooleanType(),
        [("a", "string"), ("b", "string")],
        "substr(_udf_param_0, 1, 5) <> substr(_udf_param_1, 1, 5)",
    ),
    (
        "tpcds_safe_ratio",
        tpcds_safe_ratio,
        DoubleType(),
        [("num", "double"), ("denom", "double")],
        "CASE WHEN _udf_param_1 > 0 THEN _udf_param_0 / _udf_param_1 END",
    ),
    (
        "tpcds_sales_deviation",
        tpcds_sales_deviation,
        DoubleType(),
        [("actual", "double"), ("avg", "double")],
        "CASE WHEN _udf_param_1 > 0 THEN abs(_udf_param_0 - _udf_param_1) / _udf_param_1 END",
    ),
    (
        "tpcds_in_quarter",
        tpcds_in_quarter,
        BooleanType(),
        [("moy", "bigint"), ("start_moy", "bigint"), ("span", "bigint")],
        "CASE WHEN _udf_param_0 IS NULL OR _udf_param_1 IS NULL OR _udf_param_2 IS NULL THEN NULL "
        "ELSE _udf_param_0 BETWEEN _udf_param_1 AND _udf_param_1 + _udf_param_2 END",
    ),
    # Common real-world patterns
    (
        "common_discount_price",
        common_discount_price,
        DoubleType(),
        [("price", "double"), ("pct", "double")],
        "_udf_param_0 * (1 - _udf_param_1 / 100.0)",
    ),
    (
        "common_pct_change",
        common_pct_change,
        DoubleType(),
        [("old", "double"), ("new", "double")],
        "CASE WHEN _udf_param_0 <> 0 THEN (_udf_param_1 - _udf_param_0) / _udf_param_0 * 100.0 END",
    ),
    ("common_c2f", common_c2f, DoubleType(), [("c", "double")], "_udf_param_0 * 9.0 / 5.0 + 32.0"),
    (
        "common_url_decode",
        common_url_decode,
        StringType(),
        [("s", "string")],
        "replace(_udf_param_0, '%20', ' ')",
    ),
]


def _substitute(sql: str, cols: list[str]) -> str:
    """Replace _udf_param_N with actual column names."""
    for i, col in enumerate(cols):
        sql = sql.replace(f"_udf_param_{i}", col)
    return sql


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def wait_for_catalog(catalog, source_text, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with catalog._lock:
            row = catalog._conn.execute(
                "SELECT status, error FROM cache WHERE source_text = ?",
                (source_text,),
            ).fetchone()
        if row and row[0] in {"success", "failed"}:
            return row
        time.sleep(0.1)
    return ("timeout", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=int, default=5_000_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--csv", metavar="PATH", help="write results CSV")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-tpcds-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")
        .getOrCreate()
    )

    rows = args.rows
    print(f"=== TPC-DS UDF BENCHMARK ({rows:,} rows, min of {args.rounds} rounds) ===\n")

    # Build one cached frame per distinct column signature.
    frames: dict[tuple, object] = {}

    def get_frame(cols: list[tuple[str, str]]):
        key = tuple(cols)
        if key not in frames:
            base = spark.range(0, rows, numPartitions=PARTITIONS).toDF("x")
            sel = []
            for name, dtype in cols:
                if dtype == "string":
                    sel.append(F.col("x").cast("string").alias(name))
                elif dtype == "double":
                    sel.append((F.col("x") % 10000).cast("double").alias(name))
                else:
                    sel.append(F.col("x").alias(name))
            df = base.select(*sel).cache()
            df.count()
            frames[key] = df
        return frames[key]

    # -- Python arm: transpile off, all UDFs are interpreted python.
    py_udfs: dict[str, UserDefinedFunction] = {}
    for name, func, rtype, _cols, _sql in PATTERNS:
        py_udfs[name] = UserDefinedFunction(func, rtype)
        assert not py_udfs[name].transpiled

    # -- Vanilla SQL arm: native expressions, no UDF at all.
    vanilla_exprs: dict[str, str] = {}
    for name, _func, _rtype, cols, sql in PATTERNS:
        col_names = [c[0] for c in cols]
        vanilla_exprs[name] = _substitute(sql, col_names)

    # -- Transpiled arm: enable the plugin, let the fake backend fill each UDF.
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-tpcds-"), "cache.sqlite")
    catalog = enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)

    for name, func, rtype, _cols, _sql in PATTERNS:
        UserDefinedFunction(func, rtype)  # construction queues the pending row
        status, error = wait_for_catalog(catalog, canonical_source_from_func(func))
        assert status == "success", f"transpile failed for {name}: {error!r}"

    ai_udfs: dict[str, UserDefinedFunction] = {}
    for name, func, rtype, _cols, _sql in PATTERNS:
        ai_udfs[name] = UserDefinedFunction(func, rtype)
        assert ai_udfs[name].transpiled, f"{name} did not transpile"

    # -- Benchmark: interleave arms so warmup is fair.
    arms: list[tuple[str, object]] = []
    for name, _func, _rtype, cols, _sql in PATTERNS:
        df = get_frame(cols)
        col_names = [c[0] for c in cols]
        arms.append((f"python   {name}", df.select(py_udfs[name](*col_names))))
        arms.append((f"vanilla  {name}", df.select(F.expr(vanilla_exprs[name]).alias("out"))))
        arms.append((f"transpiled {name}", df.select(ai_udfs[name](*col_names))))

    for _name, arm_df in arms:  # warmup
        noop_seconds(arm_df)

    secs: dict[str, float] = {}
    for _ in range(args.rounds):
        for name, arm_df in arms:
            secs[name] = min(secs.get(name, float("inf")), noop_seconds(arm_df))

    # -- Print results.
    header = f"{'pattern':<30} {'python':>10} {'vanilla':>10} {'transpiled':>10} {'tx/py':>8} {'va/py':>8}"
    print(header)
    print("-" * len(header))
    csv_rows = []
    for name, _func, _rtype, _cols, _sql in PATTERNS:
        py_s = secs[f"python   {name}"]
        va_s = secs[f"vanilla  {name}"]
        tx_s = secs[f"transpiled {name}"]
        tx_speedup = py_s / tx_s if tx_s > 0 else 0
        va_speedup = py_s / va_s if va_s > 0 else 0
        print(f"{name:<30} {py_s:>9.2f}s {va_s:>9.2f}s {tx_s:>9.2f}s {tx_speedup:>9.1f}x {va_speedup:>9.1f}x")
        csv_rows.append(
            [
                name,
                rows,
                round(rows / py_s),
                round(rows / va_s),
                round(rows / tx_s),
                round(tx_speedup, 1),
                round(va_speedup, 1),
            ]
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "function",
                    "rows",
                    "python_rows_per_second",
                    "vanilla_rows_per_second",
                    "transpiled_rows_per_second",
                    "transpiled_speedup_vs_python",
                    "vanilla_speedup_vs_python",
                ]
            )
            w.writerows(csv_rows)
        print(f"\nCSV written to {args.csv}")

    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
