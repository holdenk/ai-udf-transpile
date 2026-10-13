# SPDX-License-Identifier: Apache-2.0
"""Local micro benchmarks: Python UDF vs transpiled Catalyst SQL vs Java UDF.

Three numbers, all on one local SparkSession:

  throughput   rows/sec over a cached DataFrame, noop sink, min of 3 rounds
  definition   ms per UserDefinedFunction(...) creation (plugin hit vs plain)
  fill cost    one-time seconds for register_impl (Hypothesis + javac)

    export SPARK_HOME=/path/to/spark-master   # same env as demo_quickstart.py
    python examples/bench_micro.py [rows]     # default 10,000,000
"""

from __future__ import annotations

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
from ai_udf_transpile.backends.fake import BACKWARDS_JAVA  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PARTITIONS = os.cpu_count() or 16


def plus_one(x: int) -> int:
    return x + 1


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def main() -> int:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
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
    ints.count()
    strs.count()
    # plus_one raises on None in interpreted Python, so the sample has no NULL.
    sample_ints = spark.createDataFrame([(1,), (41,)], ["x"])
    sample_strs = spark.createDataFrame([("holden",), (None,)], ["s"])

    print(f"=== THROUGHPUT ({rows:,} rows, local[*], noop sink, min of 3 rounds) ===")
    # UserDefinedFunction is always SQL_BATCHED_UDF (the classic pickle path);
    # the Arrow conf only applies to the udf() helper. This is also the path
    # the transpiler hooks, so it is the honest baseline. The plan assertions
    # below guard against silently benchmarking the wrong operator.
    py_plus = UserDefinedFunction(plus_one, LongType())
    py_back = UserDefinedFunction(backwards, StringType())
    assert not py_plus.transpiled and not py_back.transpiled
    assert "BatchEvalPython" in ints.select(py_plus("x"))._jdf.queryExecution().toString()
    py_plus_expect = [r[0] for r in sample_ints.select(py_plus("x")).collect()]
    py_back_expect = [r[0] for r in sample_strs.select(py_back("s")).collect()]

    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-"), "cache.sqlite")
    enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    start = time.perf_counter()
    register_impl(spark, plus_one, kind="catalyst", catalyst_sql="_udf_param_0 + 1", return_type=LongType())
    fill_catalyst = time.perf_counter() - start
    start = time.perf_counter()
    register_impl(spark, backwards, kind="java_udf", impl_source=BACKWARDS_JAVA, return_type=StringType())
    fill_java = time.perf_counter() - start

    ai_plus = UserDefinedFunction(plus_one, LongType())
    ai_back = UserDefinedFunction(backwards, StringType())
    assert ai_plus.transpiled and ai_back.transpiled, "warm catalog did not produce a hit"
    assert [r[0] for r in sample_ints.select(ai_plus("x")).collect()] == py_plus_expect
    assert [r[0] for r in sample_strs.select(ai_back("s")).collect()] == py_back_expect
    assert "EvalPython" not in ints.select(ai_plus("x"))._jdf.queryExecution().toString()
    assert "EvalPython" not in strs.select(ai_back("s"))._jdf.queryExecution().toString()

    # All four UDFs exist before any timing (the transpile gate is read at
    # construction), so rounds interleave fairly: nobody's warmup is deeper.
    arms = [
        ("python plus_one", ints.select(py_plus("x")), None),
        ("python backwards", strs.select(py_back("s")), None),
        ("catalyst plus_one", ints.select(ai_plus("x")), "python plus_one"),
        ("java udf backwards", strs.select(ai_back("s")), "python backwards"),
    ]
    for _, arm_df, _ in arms:  # warmup round for everyone
        noop_seconds(arm_df)
    secs: dict[str, float] = {}
    for _ in range(3):
        for name, arm_df, _ in arms:
            secs[name] = min(secs.get(name, float("inf")), noop_seconds(arm_df))
    for name, _, baseline in arms:
        speedup = f"  ({secs[baseline] / secs[name]:.1f}x python)" if baseline else ""
        print(f"{name + ':':24}{secs[name]:6.2f}s  {rows / secs[name]:>12,.0f} rows/s{speedup}")

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

    print()
    print("=== ONE-TIME FILL COST (verify + publish, per rewrite) ===")
    print(f"register_impl catalyst (Hypothesis): {fill_catalyst:.2f}s")
    print(f"register_impl java_udf (Hypothesis + javac): {fill_java:.2f}s")

    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
