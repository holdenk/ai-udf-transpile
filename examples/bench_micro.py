# SPDX-License-Identifier: Apache-2.0
"""Local micro benchmarks: Python UDF vs transpiled Catalyst SQL vs Java UDF.

Three numbers, all on one local SparkSession:

  throughput   rows/sec over a cached DataFrame, noop sink, min of --rounds rounds (default 3)
  definition   ms per UserDefinedFunction(...) creation (plugin hit vs plain)
  fill cost    one-time seconds for register_impl (Hypothesis + javac)

The throughput section also includes the atomic ``always_decline`` UDF whose
environment-dependent behavior stays on the Python path after the fake backend
declines.

    export SPARK_HOME=/path/to/spark-master   # same env as demo_quickstart.py
    python examples/bench_micro.py [rows] [--rounds N] [--csv PATH]
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
from pyspark.sql.types import LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import enable, register_impl, shutdown  # noqa: E402
from ai_udf_transpile.backends.fake import BACKWARDS_JAVA, FIXTURES, always_decline  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PARTITIONS = os.cpu_count() or 16


def plus_one(x: int) -> int:
    return x + 1


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def zfill3(s: str) -> str:
    # The non-trivial arm: python zfill never truncates and pads after a +/-
    # sign, so the rewrite is a 3-branch CASE -- bare lpad is wrong on both.
    # The type layer recognizes array-of-atomic returns, but Spark's ANSI
    # transpile path currently skips array-return UDFs before this hook. Keep
    # this bench atomic; it does not expand coverage to structs, maps, or
    # nested/non-atomic arrays.
    return s.zfill(3)


def edit_distance(a: str, b: str) -> int:
    # The algorithmic arm: a quadratic DP per row in interpreted python, one
    # native Catalyst builtin once transpiled. -1 on NULL, like a SQL citizen.
    if a is None or b is None:
        return -1
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


# The fake backend's fixture for this exact source: verified-faithful SQL.
ZFILL_SQL = FIXTURES[canonical_source_from_func(zfill3)].sql


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def wait_for_catalog_result(
    catalog, source_text: str, timeout: float = 10.0
) -> tuple[str, str | None, str | None]:
    deadline = time.monotonic() + timeout
    status = "missing"
    backend = None
    error = None
    while time.monotonic() < deadline:
        with catalog._lock:
            row = catalog._conn.execute(
                "SELECT status, backend, error FROM cache WHERE source_text = ?",
                (source_text,),
            ).fetchone()
        if row is not None:
            status, backend, error = row
            if status in {"failed", "success"}:
                return status, backend, error
        time.sleep(0.1)
    return status, backend, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=3, help="measured rounds per arm; min is reported")
    parser.add_argument(
        "--csv", metavar="PATH", help="also write results to PATH as section,name,metric,value"
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    rows = args.rows
    csv_rows: list[tuple[str, str, str, object]] = []
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")  # phase A must stay interpreted Python
        .getOrCreate()
    )

    # Cache the inputs so range/cast generation is not in the measurement.
    ints = spark.range(0, rows, numPartitions=PARTITIONS).toDF("x").cache()
    strs = ints.select(F.col("x").cast("string").alias("s")).cache()
    # Short strings with varying content: the DP cost is real but not so big
    # that the python arm dominates the wall clock.
    pairs = ints.select(
        (F.col("x") % 100003).cast("string").alias("a"),
        ((F.col("x") * 7 + 13) % 99991).cast("string").alias("b"),
    ).cache()
    ints.count()
    strs.count()
    pairs.count()
    # plus_one raises on None in interpreted Python, so the sample has no NULL.
    sample_ints = spark.createDataFrame([(1,), (41,)], ["x"])
    sample_strs = spark.createDataFrame([("holden",), (None,)], ["s"])
    # "-555" pins the CASE's branch order: sign-before-length would truncate.
    # No NULL row: interpreted python raises on None, the SQL returns NULL.
    sample_zfill = spark.createDataFrame([("5",), ("-5",), ("+7",), ("-555",), ("abcd",)], ["s"])
    sample_pairs = spark.createDataFrame(
        [("kitten", "sitting"), ("", "abc"), ("same", "same"), (None, "x"), ("x", None), (None, None)],
        ["a", "b"],
    )

    print(f"=== THROUGHPUT ({rows:,} rows, local[*], noop sink, min of {args.rounds} rounds) ===")
    # UserDefinedFunction is always SQL_BATCHED_UDF (the classic pickle path);
    # the Arrow conf only applies to the udf() helper. This is also the path
    # the transpiler hooks, so it is the honest baseline. The plan assertions
    # below guard against silently benchmarking the wrong operator.
    py_plus = UserDefinedFunction(plus_one, LongType())
    py_back = UserDefinedFunction(backwards, StringType())
    py_zfill = UserDefinedFunction(zfill3, StringType())
    py_edit = UserDefinedFunction(edit_distance, LongType())
    assert not any(u.transpiled for u in (py_plus, py_back, py_zfill, py_edit))
    assert "BatchEvalPython" in ints.select(py_plus("x"))._jdf.queryExecution().toString()
    py_plus_expect = [r[0] for r in sample_ints.select(py_plus("x")).collect()]
    py_back_expect = [r[0] for r in sample_strs.select(py_back("s")).collect()]
    py_zfill_expect = [r[0] for r in sample_zfill.select(py_zfill("s")).collect()]
    py_edit_expect = [r[0] for r in sample_pairs.select(py_edit("a", "b")).collect()]

    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-"), "cache.sqlite")
    catalog = enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    start = time.perf_counter()
    register_impl(spark, plus_one, kind="catalyst", catalyst_sql="_udf_param_0 + 1", return_type=LongType())
    fill_catalyst = time.perf_counter() - start
    start = time.perf_counter()
    register_impl(spark, backwards, kind="java_udf", impl_source=BACKWARDS_JAVA, return_type=StringType())
    fill_java = time.perf_counter() - start
    start = time.perf_counter()
    register_impl(spark, zfill3, kind="catalyst", catalyst_sql=ZFILL_SQL, return_type=StringType())
    fill_zfill = time.perf_counter() - start
    start = time.perf_counter()
    register_impl(
        spark,
        edit_distance,
        kind="catalyst",
        catalyst_sql="coalesce(levenshtein(_udf_param_0, _udf_param_1), -1)",
        return_type=LongType(),
    )
    fill_edit = time.perf_counter() - start

    ai_plus = UserDefinedFunction(plus_one, LongType())
    ai_back = UserDefinedFunction(backwards, StringType())
    ai_zfill = UserDefinedFunction(zfill3, StringType())
    ai_edit = UserDefinedFunction(edit_distance, LongType())
    UserDefinedFunction(always_decline, LongType())
    warm = (ai_plus, ai_back, ai_zfill, ai_edit)
    assert all(u.transpiled for u in warm), "warm catalog did not produce a hit"
    assert [r[0] for r in sample_ints.select(ai_plus("x")).collect()] == py_plus_expect
    assert [r[0] for r in sample_strs.select(ai_back("s")).collect()] == py_back_expect
    assert [r[0] for r in sample_zfill.select(ai_zfill("s")).collect()] == py_zfill_expect
    assert [r[0] for r in sample_pairs.select(ai_edit("a", "b")).collect()] == py_edit_expect
    for plan_df in (
        ints.select(ai_plus("x")),
        strs.select(ai_back("s")),
        strs.select(ai_zfill("s")),
        pairs.select(ai_edit("a", "b")),
    ):
        assert "EvalPython" not in plan_df._jdf.queryExecution().toString()
    decline_source = canonical_source_from_func(always_decline)
    decline_status, decline_backend, decline_error = wait_for_catalog_result(catalog, decline_source)
    assert (
        decline_status == "failed"
        and decline_backend == "fake"
        and decline_error is not None
        and decline_error.startswith("declined:")
    ), (
        "expected fake backend decline for always_decline, got "
        f"status={decline_status!r} backend={decline_backend!r} error={decline_error!r}"
    )
    fallback_decline = UserDefinedFunction(always_decline, LongType())
    assert not fallback_decline.transpiled, "declined UDF unexpectedly became a hit"
    cwd_df = ints.select(fallback_decline("x"))
    assert "BatchEvalPython" in cwd_df._jdf.queryExecution().toString()

    # All UDFs exist before any timing (the transpile gate is read at
    # construction), so rounds interleave fairly: nobody's warmup is deeper.
    arms = [
        ("python plus_one", ints.select(py_plus("x")), None),
        ("python backwards", strs.select(py_back("s")), None),
        ("python zfill3", strs.select(py_zfill("s")), None),
        ("python edit_distance", pairs.select(py_edit("a", "b")), None),
        ("catalyst plus_one", ints.select(ai_plus("x")), "python plus_one"),
        ("java udf backwards", strs.select(ai_back("s")), "python backwards"),
        ("catalyst zfill3", strs.select(ai_zfill("s")), "python zfill3"),
        ("catalyst edit_distance", pairs.select(ai_edit("a", "b")), "python edit_distance"),
        ("python fallback always_decline", cwd_df, None),
    ]
    for _, arm_df, _ in arms:  # warmup round for everyone
        noop_seconds(arm_df)
    secs: dict[str, float] = {}
    for _ in range(args.rounds):
        for name, arm_df, _ in arms:
            secs[name] = min(secs.get(name, float("inf")), noop_seconds(arm_df))
    for name, _, baseline in arms:
        speedup = f"  ({secs[baseline] / secs[name]:.1f}x python)" if baseline else ""
        print(f"{name + ':':24}{secs[name]:6.2f}s  {rows / secs[name]:>12,.0f} rows/s{speedup}")
        csv_rows.append(("throughput", name, "rows_per_second", round(rows / secs[name])))
        csv_rows.append(("throughput", name, "seconds", round(secs[name], 3)))
        if baseline:
            csv_rows.append(("throughput", name, "speedup_vs_python", round(secs[baseline] / secs[name], 1)))
    print()
    print("=== EXPECTED DECLINE ===")
    print(
        f"always_decline: catalog status={decline_status}, backend={decline_backend}; "
        "fresh UDF construction stays on the Python path"
    )
    csv_rows.append(("fallback", "python fallback always_decline", "transpile_status", decline_status))

    print()
    print("=== UDF DEFINITION OVERHEAD (avg of 300 creations) ===")
    creations = 300
    assert UserDefinedFunction(plus_one, LongType()).transpiled  # probe: really the hit path
    start = time.perf_counter()
    for _ in range(creations):
        UserDefinedFunction(plus_one, LongType())
    hit_ms = (time.perf_counter() - start) / creations * 1000
    spark.conf.set(TRANSPILE_CONF, "false")
    assert not UserDefinedFunction(plus_one, LongType()).transpiled  # probe: really the plain path
    start = time.perf_counter()
    for _ in range(creations):
        UserDefinedFunction(plus_one, LongType())
    plain_ms = (time.perf_counter() - start) / creations * 1000
    spark.conf.set(TRANSPILE_CONF, "true")
    print(f"plugin warm-hit: {hit_ms:.2f} ms/def   plain python udf: {plain_ms:.2f} ms/def")
    csv_rows.append(("definition_overhead", "plugin warm-hit", "ms_per_definition", round(hit_ms, 2)))
    csv_rows.append(("definition_overhead", "plain python udf", "ms_per_definition", round(plain_ms, 2)))

    print()
    print("=== ONE-TIME FILL COST (verify + publish, per rewrite) ===")
    fills = [
        ("catalyst plus_one (Hypothesis)", fill_catalyst),
        ("catalyst zfill3 (Hypothesis)", fill_zfill),
        ("catalyst edit_distance (Hypothesis)", fill_edit),
        ("java_udf backwards (Hypothesis + javac)", fill_java),
    ]
    for name, seconds in fills:
        print(f"register_impl {name}: {seconds:.2f}s")
        csv_rows.append(("fill_cost", name, "seconds", round(seconds, 2)))

    shutdown()
    spark.stop()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "name", "metric", "value"])
            writer.writerows(csv_rows)
        print(f"csv: wrote {len(csv_rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
