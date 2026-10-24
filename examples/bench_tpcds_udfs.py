# SPDX-License-Identifier: Apache-2.0
"""TPC-DS-inspired UDF benchmark: Python UDF vs vanilla SQL vs transpiled SQL.

Each UDF below is a pattern extracted from a real TPC-DS query (source in the
comment).  We run three arms per pattern:

  python    the UDF as an interpreted Python UDF (transpilePyUDFs=false)
  vanilla   the native SQL expression the UDF is equivalent to
  transpiled  the UDF after AI transpilation (fake backend, inline worker)

The point: a Python UDF doing TPC-DS-like work is 5-10x slower than native
SQL, and transpilation closes that gap for the patterns whose faithful
rewrite IS the vanilla SQL (most of them). The python-semantics rewrites
(spark_substr's clamped slice, spark_str's 'None', spark_eq's NULL-safe =)
carry guards the naive vanilla expression lacks, and run a little slower
than vanilla as a result; three UDFBench date parsers are declined outright
-- no faithful SQL exists for int() over a free-form prefix.

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
    common_full_name,
    common_is_adult,
    common_pct_change,
    common_round2,
    common_url_decode,
    spark_add_one,
    spark_add_two,
    spark_as_double,
    spark_eq,
    spark_len_plus,
    spark_str,
    spark_strlen,
    spark_substr,
    spark_to_lower,
    spark_to_upper,
    spark_trunc,
    tpcds_below_threshold,
    tpcds_day_sales,
    tpcds_in_quarter,
    tpcds_safe_ratio,
    tpcds_sales_deviation,
    tpcds_trunc30,
    tpcds_zip5_mismatch,
    udfb_converttoeuro,
    udfb_extractclass,
    udfb_extractday,
    udfb_extractfunder,
    udfb_extractid,
    udfb_extractmonth,
    udfb_extractyear,
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
    # More from apache/spark test_udf.py
    ("spark_add_two", spark_add_two, DoubleType(), [("x", "bigint")], "cast(_udf_param_0 + 2 as double)"),
    (
        "spark_substr",
        spark_substr,
        StringType(),
        [("s", "string"), ("start", "bigint"), ("end", "bigint")],
        "substr(_udf_param_0, _udf_param_1 + 1, _udf_param_2 - _udf_param_1)",
    ),
    ("spark_strlen", spark_strlen, LongType(), [("s", "string")], "length(_udf_param_0)"),
    (
        "spark_len_plus",
        spark_len_plus,
        LongType(),
        [("s", "string"), ("n", "bigint")],
        "length(_udf_param_0) + _udf_param_1",
    ),
    ("spark_str", spark_str, StringType(), [("x", "bigint")], "cast(_udf_param_0 as string)"),
    ("spark_eq", spark_eq, BooleanType(), [("a", "bigint"), ("b", "bigint")], "_udf_param_0 = _udf_param_1"),
    # More from TPC-DS modified queries
    (
        "tpcds_day_sales",
        tpcds_day_sales,
        DoubleType(),
        [("day_name", "string"), ("sales_price", "double")],
        "CASE WHEN _udf_param_0 = 'Sunday' THEN _udf_param_1 END",
    ),
    (
        "tpcds_below_threshold",
        tpcds_below_threshold,
        BooleanType(),
        [("revenue", "double"), ("avg", "double")],
        "CASE WHEN _udf_param_1 > 0 THEN _udf_param_0 <= 0.1 * _udf_param_1 END",
    ),
    ("tpcds_trunc30", tpcds_trunc30, StringType(), [("s", "string")], "substr(_udf_param_0, 1, 30)"),
    # More common real-world patterns
    (
        "common_full_name",
        common_full_name,
        StringType(),
        [("first", "string"), ("last", "string")],
        "_udf_param_0 || ' ' || _udf_param_1",
    ),
    ("common_round2", common_round2, DoubleType(), [("x", "double")], "round(_udf_param_0, 2)"),
    ("common_is_adult", common_is_adult, BooleanType(), [("age", "bigint")], "_udf_param_0 >= 18"),
    # From UDFBench (VLDB 2025, Foufoulas et al.)
    (
        "udfb_extractyear",
        udfb_extractyear,
        LongType(),
        [("s", "string")],
        "cast(substr(_udf_param_0, 1, 4) as int)",
    ),
    (
        "udfb_extractmonth",
        udfb_extractmonth,
        LongType(),
        [("s", "string")],
        "cast(substr(_udf_param_0, 6, 2) as int)",
    ),
    (
        "udfb_extractday",
        udfb_extractday,
        LongType(),
        [("s", "string")],
        "cast(substr(_udf_param_0, 9, 2) as int)",
    ),
    (
        "udfb_extractfunder",
        udfb_extractfunder,
        StringType(),
        [("s", "string")],
        "split_part(_udf_param_0, '::', 1)",
    ),
    (
        "udfb_extractclass",
        udfb_extractclass,
        StringType(),
        [("s", "string")],
        "split_part(_udf_param_0, '::', 2)",
    ),
    (
        "udfb_extractid",
        udfb_extractid,
        StringType(),
        [("s", "string")],
        "split_part(_udf_param_0, '::', 3)",
    ),
    (
        "udfb_converttoeuro",
        udfb_converttoeuro,
        DoubleType(),
        [("amount", "double"), ("currency", "string")],
        "CASE _udf_param_1 "
        "WHEN 'EUR' THEN _udf_param_0 / 1.0 "
        "WHEN 'USD' THEN _udf_param_0 / 1.09 "
        "WHEN 'GBP' THEN _udf_param_0 / 0.85 "
        "WHEN 'CHF' THEN _udf_param_0 / 0.98 END",
    ),
]


# Per-pattern column expressions. The default frame is numeric strings and
# small doubles, which is garbage for a date parser, a '::' splitter, or a
# currency lookup -- python itself would crash (or Hypothesis would see no
# signal) on rows that are not shaped like the pattern's real domain.
_SEP_EXPR = (
    "concat('funder', cast(x % 100 as string), '::', "
    "'class', cast(x % 10 as string), '::', 'proj', cast(x % 1000 as string))"
)
_DATE_EXPR = "date_format(make_date(2000 + (x % 50), 1 + (x % 12), 1 + (x % 28)), 'yyyy-MM-dd')"
FRAME_EXPRS: dict[str, dict[str, str]] = {
    "udfb_extractyear": {"s": _DATE_EXPR},
    "udfb_extractmonth": {"s": _DATE_EXPR},
    "udfb_extractday": {"s": _DATE_EXPR},
    "udfb_extractfunder": {"s": _SEP_EXPR},
    "udfb_extractclass": {"s": _SEP_EXPR},
    "udfb_extractid": {"s": _SEP_EXPR},
    "udfb_converttoeuro": {
        "currency": "element_at(array('EUR', 'USD', 'GBP', 'CHF'), cast(x % 4 as int) + 1)"
    },
}


def _frame(spark, pattern: str, cols: list[tuple[str, str]], n: int):
    exprs = FRAME_EXPRS.get(pattern, {})
    base = spark.range(0, n, numPartitions=PARTITIONS).toDF("x")
    sel = []
    for name, dtype in cols:
        if name in exprs:
            sel.append(F.expr(exprs[name]).alias(name))
        elif dtype == "string":
            sel.append(F.col("x").cast("string").alias(name))
        elif dtype == "double":
            sel.append((F.col("x") % 10000).cast("double").alias(name))
        else:
            sel.append(F.col("x").alias(name))
    df = base.select(*sel)
    return df.cache() if n > 8 else df


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

    def get_frame(cols: list[tuple[str, str]], name: str):
        key = (name, tuple(cols))
        if key not in frames:
            frames[key] = _frame(spark, name, cols, rows)
            frames[key].count()
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
    # strlen/upper raise on NULL. These patterns' strings are the values under test.
    conf.set_value(conf.PARTY_ON_STRINGS, "true", spark)
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)

    # -- Seed samples: run each UDF once on domain-shaped rows so Hypothesis
    # verifies against real inputs -- a '::' parser or currency lookup has no
    # signal on random text (python raises on every example, fail closed).
    # The worker polls slowly during seeding so a pending row is never
    # claimed before its samples land.
    conf.set_value(conf.POLL_INTERVAL, "60", spark)
    for name, func, rtype, cols, _sql in PATTERNS:
        seed_udf = UserDefinedFunction(func, rtype)
        seed = _frame(spark, name, cols, 8)
        col_names = [c[0] for c in cols]
        seed.select(seed_udf(*col_names)).collect()
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)

    declined: dict[str, bool] = {}
    for name, func, rtype, _cols, _sql in PATTERNS:
        UserDefinedFunction(func, rtype)  # construction queues the pending row
        status, error = wait_for_catalog(catalog, canonical_source_from_func(func))
        declined[name] = status != "success"
        if declined[name]:
            print(f"transpile declined/failed for {name}: {error!r}")
            continue

    ai_udfs: dict[str, UserDefinedFunction] = {}
    for name, func, rtype, _cols, _sql in PATTERNS:
        if declined[name]:
            continue
        ai_udfs[name] = UserDefinedFunction(func, rtype)
        assert ai_udfs[name].transpiled, f"{name} did not transpile"

    # -- Benchmark: interleave arms so warmup is fair.
    arms: list[tuple[str, object]] = []
    for name, _func, _rtype, cols, _sql in PATTERNS:
        df = get_frame(cols, name)
        col_names = [c[0] for c in cols]
        arms.append((f"python   {name}", df.select(py_udfs[name](*col_names))))
        arms.append((f"vanilla  {name}", df.select(F.expr(vanilla_exprs[name]).alias("out"))))
        # A declined pattern has no rewrite: its transpiled perf IS python's.
        arm_udf = ai_udfs.get(name, py_udfs[name])
        arms.append((f"transpiled {name}", df.select(arm_udf(*col_names))))

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
        mark = " (python fallback)" if declined[name] else ""
        print(
            f"{name:<30} {py_s:>9.2f}s {va_s:>9.2f}s {tx_s:>9.2f}s "
            f"{tx_speedup:>9.1f}x {va_speedup:>9.1f}x{mark}"
        )
        csv_rows.append(
            [
                name,
                declined[name],
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
                    "declined",
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
