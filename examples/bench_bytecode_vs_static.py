# SPDX-License-Identifier: Apache-2.0
"""Micro benchmark: bytecode backend vs Spark's built-in static transpiler.

Three arms per function on one local SparkSession:

  python    interpreted python (transpilePyUDFs=false) -- the baseline
  static    Spark's BUILT-IN catalyst transpiler (pyTranspilers=catalyst,
            plugin not enabled). It converts the simple single-expression
            bodies and declines the multi-statement one (is_none_branch).
            Correctness is checked per-sample rather than assumed.
  bytecode  the deterministic bytecode backend (pyTranspilers=ai, plugin on,
            backend="bytecode"). No AI round-trip, no Hypothesis fill latency
            in the hot path -- the catalog is filled once up front.

The point: where both convert (plus_one, both_positive, greet, mod_bucket)
the perf is comparable; where only bytecode converts (is_none_branch, which
is multi-statement) the speedup over interpreted python is the whole story
and static has no number. A wrong-but-converted rewrite gets no perf number
either -- the correctness boolean is the story.

  export SPARK_HOME=/path/to/spark-master
  python examples/bench_bytecode_vs_static.py [rows] [--rounds N] [--csv PATH]
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
from pyspark.sql.types import BooleanType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import enable, shutdown  # noqa: E402
from ai_udf_transpile.backends.bytecode import BytecodeBackend  # noqa: E402
from ai_udf_transpile.backends.fake import (  # noqa: E402
    both_positive,
    greet,
    is_none_branch,
    mod_bucket,
    plus_one,
)
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402
from ai_udf_transpile.worker import poll_once  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
PARTITIONS = os.cpu_count() or 16


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=int, default=2_000_000)
    parser.add_argument("--rounds", type=int, default=3, help="measured rounds per arm; min is reported")
    parser.add_argument("--csv", metavar="PATH", help="write the comparison table as CSV")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    rows = args.rows
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench-bytecode")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")  # python phase stays interpreted
        .getOrCreate()
    )

    # Cached inputs. is_none_branch needs NULLs; mod_bucket needs modulus > 0.
    ints = spark.range(0, rows, numPartitions=PARTITIONS).toDF("x").cache()
    ints_with_null = ints.select(F.when(F.col("x") % 100 == 0, None).otherwise(F.col("x")).alias("x")).cache()
    pairs = ints.select(F.col("x").alias("a"), ((F.col("x") * 7 + 13) % 1000).alias("b")).cache()
    strs = ints.select(F.col("x").cast("string").alias("s")).cache()
    mod_pairs = ints.select(
        F.col("x").alias("value"),
        ((F.col("x") % 99) + 1).alias("modulus"),  # 1..99, never zero
    ).cache()
    for df in (ints, ints_with_null, pairs, strs, mod_pairs):
        df.count()

    # name, fn, return type, input df, its cols, sample frame
    functions = [
        ("plus_one", plus_one, LongType(), ints, ["x"], spark.createDataFrame([(1,), (41,)], ["x"])),
        (
            "is_none_branch",
            is_none_branch,
            LongType(),
            ints_with_null,
            ["x"],
            spark.createDataFrame([(5,), (None,)], ["x"]),
        ),
        (
            "both_positive",
            both_positive,
            BooleanType(),
            pairs,
            ["a", "b"],
            spark.createDataFrame([(1, 2), (-1, 2), (1, -2)], ["a", "b"]),
        ),
        ("greet", greet, StringType(), strs, ["s"], spark.createDataFrame([("holden",), ("",)], ["s"])),
        (
            "mod_bucket",
            mod_bucket,
            LongType(),
            mod_pairs,
            ["value", "modulus"],
            spark.createDataFrame([(10, 3), (7, 100), (-7, 3)], ["value", "modulus"]),
        ),
    ]

    print(f"=== THROUGHPUT ({rows:,} rows, local[*], noop sink, min of {args.rounds} rounds) ===")

    # -- python phase: transpile off, every arm is interpreted python.
    py_udfs: dict[str, UserDefinedFunction] = {}
    py_expect: dict[str, list] = {}
    for name, fn, rtype, _df, _cols, sample in functions:
        udf = UserDefinedFunction(fn, rtype)
        assert not udf.transpiled
        py_udfs[name] = udf
        py_expect[name] = [r[0] for r in sample.select(udf(*sample.columns)).collect()]
    assert "BatchEvalPython" in ints.select(py_udfs["plus_one"]("x"))._jdf.queryExecution().toString()

    # -- static phase: built-in catalyst, plugin not enabled.
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    st_udfs: dict[str, UserDefinedFunction] = {}
    static_ok: dict[str, bool] = {}
    static_correct: dict[str, bool] = {}
    for name, fn, rtype, _df, _cols, _sample in functions:
        udf = UserDefinedFunction(fn, rtype)
        st_udfs[name] = udf
        static_ok[name] = bool(udf.transpiled)
    for name, _fn, _rtype, df, cols, sample in functions:
        if not static_ok[name]:
            static_correct[name] = False
            continue
        got = [r[0] for r in sample.select(st_udfs[name](*cols)).collect()]
        static_correct[name] = got == py_expect[name]
        if static_correct[name]:
            assert "EvalPython" not in df.select(st_udfs[name](*cols))._jdf.queryExecution().toString(), (
                f"static {name} claims transpiled but plan still has EvalPython"
            )

    # -- bytecode phase: plugin on, backend="bytecode", inline_worker=False so
    # the fill is an explicit poll_once (deterministic, no race).
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-bc-"), "cache.sqlite")
    catalog = enable(spark, sqlite_path=sqlite, backend="bytecode", inline_worker=False)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    bc_udfs: dict[str, UserDefinedFunction] = {}
    for name, fn, rtype, _df, _cols, _sample in functions:
        UserDefinedFunction(fn, rtype)  # construction queues the pending row
        assert poll_once(catalog, BytecodeBackend(), spark)
        with catalog._lock:
            status, error = catalog._conn.execute(
                "SELECT status, error FROM cache WHERE source_text = ?",
                (canonical_source_from_func(fn),),
            ).fetchone()
        assert status == "success", f"bytecode backend did not fill {name}: {status} {error}"
        udf = UserDefinedFunction(fn, rtype)
        assert udf.transpiled, f"bytecode fill did not hit for {name}"
        bc_udfs[name] = udf
    bytecode_correct: dict[str, bool] = {}
    for name, _fn, _rtype, df, cols, sample in functions:
        got = [r[0] for r in sample.select(bc_udfs[name](*cols)).collect()]
        bytecode_correct[name] = got == py_expect[name]
        if bytecode_correct[name]:
            assert "EvalPython" not in df.select(bc_udfs[name](*cols))._jdf.queryExecution().toString()

    # -- throughput: only correct arms are benched. A wrong rewrite (e.g. static
    # bare-`%` on negatives) gets no perf number -- the boolean tells the story.
    arms: list[tuple[str, object, str | None]] = []
    for name, _fn, _rtype, df, cols, _sample in functions:
        arms.append((f"python {name}", df.select(py_udfs[name](*cols)), None))
        if static_ok[name] and static_correct[name]:
            arms.append((f"static {name}", df.select(st_udfs[name](*cols)), f"python {name}"))
        if bytecode_correct[name]:
            arms.append((f"bytecode {name}", df.select(bc_udfs[name](*cols)), f"python {name}"))
    for _name, arm_df, _ in arms:  # warmup
        noop_seconds(arm_df)
    secs: dict[str, float] = {}
    for _ in range(args.rounds):
        for name, arm_df, _ in arms:
            secs[name] = min(secs.get(name, float("inf")), noop_seconds(arm_df))
    for name, _, baseline in arms:
        speedup = f"  ({secs[baseline] / secs[name]:.1f}x python)" if baseline else ""
        print(f"{name + ':':32}{secs[name]:6.2f}s  {rows / secs[name]:>12,.0f} rows/s{speedup}")

    print()
    print("=== PER-FUNCTION (the --csv table) ===")
    header = [
        "function",
        "static_converted",
        "static_correct",
        "bytecode_correct",
        "python_rows_per_second",
        "static_rows_per_second",
        "bytecode_rows_per_second",
        "static_speedup_vs_python",
        "bytecode_speedup_vs_python",
    ]
    wide: list[list] = []
    for name, _fn, _rtype, _df, _cols, _sample in functions:
        py_arm = f"python {name}"
        st_arm = f"static {name}" if (static_ok[name] and static_correct[name]) else None
        bc_arm = f"bytecode {name}" if bytecode_correct[name] else None
        wide.append(
            [
                name,
                static_ok[name],
                static_correct[name],
                bytecode_correct[name],
                round(rows / secs[py_arm]),
                round(rows / secs[st_arm]) if st_arm else "",
                round(rows / secs[bc_arm]) if bc_arm else "",
                round(secs[py_arm] / secs[st_arm], 1) if st_arm else "",
                round(secs[py_arm] / secs[bc_arm], 1) if bc_arm else "",
            ]
        )
    print(",".join(header))
    for row in wide:
        print(",".join(str(c) for c in row))

    print()
    st_conv = [n for n, ok in static_ok.items() if ok]
    st_corr = [n for n, ok in static_correct.items() if ok]
    bc_corr = [n for n, ok in bytecode_correct.items() if ok]
    print(
        f"static:  converted {len(st_conv)}/{len(functions)} ({', '.join(st_conv) or 'none'}); "
        f"correct {len(st_corr)}/{len(functions)} ({', '.join(st_corr) or 'none'})"
    )
    print(
        f"bytecode: converted {len(functions)}/{len(functions)}; "
        f"correct {len(bc_corr)}/{len(functions)} ({', '.join(bc_corr)}) -- deterministic, no AI round-trip"
    )

    shutdown()
    spark.stop()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(wide)
        print(f"csv: wrote {len(wide)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
