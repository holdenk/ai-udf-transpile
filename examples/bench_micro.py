# SPDX-License-Identifier: Apache-2.0
"""Local micro benchmarks: regular vs transpiled (static) vs transpiled (AI).

Three arms per function on one local SparkSession:

  regular   interpreted python, constructed before the plugin is enabled
  static    human-provided rewrite via register_impl (origin='human')
  ai        backend-generated rewrite (fake backend, through the inline worker)

The wide CSV (--csv; CI uploads it for slides) is one row per function with
perf for each arm, speedup vs regular, and transpile_succeeded booleans for
static and AI. always_decline has no faithful static rewrite (its result is
environment-dependent) and the backend declines it: AI perf falls back to
python, boolean False.

  throughput   rows/sec over a cached DataFrame, noop sink, min of --rounds rounds
  definition   ms per UserDefinedFunction(...) creation (plugin hit vs plain)
  fill cost    one-time seconds per rewrite (static: register_impl verify;
               ai: queue -> backend -> Hypothesis -> verified success)

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

from ai_udf_transpile import conf, enable, register_impl, shutdown  # noqa: E402
from ai_udf_transpile.backends.fake import BACKWARDS_JAVA, FIXTURES, always_decline  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
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


PLUS_SQL = "_udf_param_0 + 1"
EDIT_SQL = "coalesce(levenshtein(_udf_param_0, _udf_param_1), -1)"
ZFILL_SQL = FIXTURES[canonical_source_from_func(zfill3)].sql
for _f in (plus_one, backwards, zfill3, edit_distance):
    # The ai arm needs a fake fixture for each benched function; fail at
    # import rather than as a surprise decline mid-bench.
    assert canonical_source_from_func(_f) in FIXTURES, f"no fake fixture for {_f.__name__}"


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
        "--csv",
        metavar="PATH",
        help="write the wide per-function table (perf + transpile booleans)",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    rows = args.rows
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")  # the regular phase must stay interpreted python
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

    # -- regular phase: constructed before enable(), so the transpile hook
    # never runs (it is not even registered) and every arm is plain python.
    py_plus = UserDefinedFunction(plus_one, LongType())
    py_back = UserDefinedFunction(backwards, StringType())
    py_zfill = UserDefinedFunction(zfill3, StringType())
    py_edit = UserDefinedFunction(edit_distance, LongType())
    py_decline = UserDefinedFunction(always_decline, LongType())
    assert not any(u.transpiled for u in (py_plus, py_back, py_zfill, py_edit, py_decline))
    assert "BatchEvalPython" in ints.select(py_plus("x"))._jdf.queryExecution().toString()
    py_plus_expect = [r[0] for r in sample_ints.select(py_plus("x")).collect()]
    py_back_expect = [r[0] for r in sample_strs.select(py_back("s")).collect()]
    py_zfill_expect = [r[0] for r in sample_zfill.select(py_zfill("s")).collect()]
    py_edit_expect = [r[0] for r in sample_pairs.select(py_edit("a", "b")).collect()]

    # -- static phase: human-provided rewrites on their own catalog. No inline
    # worker needed -- register_impl verifies synchronously.
    static_sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-static-"), "cache.sqlite")
    enable(spark, sqlite_path=static_sqlite, backend="fake", inline_worker=False)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    static_fills: dict[str, float] = {}
    for name, fn, kwargs in [
        ("plus_one", plus_one, dict(kind="catalyst", catalyst_sql=PLUS_SQL, return_type=LongType())),
        (
            "backwards",
            backwards,
            dict(kind="java_udf", impl_source=BACKWARDS_JAVA, return_type=StringType()),
        ),
        ("zfill3", zfill3, dict(kind="catalyst", catalyst_sql=ZFILL_SQL, return_type=StringType())),
        (
            "edit_distance",
            edit_distance,
            dict(kind="catalyst", catalyst_sql=EDIT_SQL, return_type=LongType()),
        ),
    ]:
        start = time.perf_counter()
        register_impl(spark, fn, **kwargs)
        static_fills[name] = time.perf_counter() - start
    st_plus = UserDefinedFunction(plus_one, LongType())
    st_back = UserDefinedFunction(backwards, StringType())
    st_zfill = UserDefinedFunction(zfill3, StringType())
    st_edit = UserDefinedFunction(edit_distance, LongType())
    static_ok = {
        "plus_one": bool(st_plus.transpiled),
        "backwards": bool(st_back.transpiled),
        "zfill3": bool(st_zfill.transpiled),
        "edit_distance": bool(st_edit.transpiled),
    }
    assert all(static_ok.values()), f"static register_impl did not hit: {static_ok}"
    assert [r[0] for r in sample_ints.select(st_plus("x")).collect()] == py_plus_expect
    assert [r[0] for r in sample_strs.select(st_back("s")).collect()] == py_back_expect
    assert [r[0] for r in sample_zfill.select(st_zfill("s")).collect()] == py_zfill_expect
    assert [r[0] for r in sample_pairs.select(st_edit("a", "b")).collect()] == py_edit_expect
    for plan_df in (
        ints.select(st_plus("x")),
        strs.select(st_back("s")),
        strs.select(st_zfill("s")),
        pairs.select(st_edit("a", "b")),
    ):
        assert "EvalPython" not in plan_df._jdf.queryExecution().toString()

    # -- ai phase: a second enable() on a fresh sqlite path swaps the catalog
    # (none was running for the static phase, so the inline worker starts
    # here and polls only this one). The static plans above are already
    # resolved, so nothing re-reads the closed-over static catalog.
    ai_sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-ai-"), "cache.sqlite")
    catalog = enable(spark, sqlite_path=ai_sqlite, backend="fake", inline_worker=True)
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)
    ai_fills: dict[str, float] = {}
    decline_error: str | None = None
    for name, fn, rtype in [
        ("plus_one", plus_one, LongType()),
        ("backwards", backwards, StringType()),
        ("zfill3", zfill3, StringType()),
        ("edit_distance", edit_distance, LongType()),
        ("always_decline", always_decline, LongType()),
    ]:
        start = time.perf_counter()
        UserDefinedFunction(fn, rtype)  # construction queues the pending row
        status, backend, error = wait_for_catalog_result(catalog, canonical_source_from_func(fn), timeout=120)
        ai_fills[name] = time.perf_counter() - start
        if name == "always_decline":
            decline_error = error
            assert status == "failed" and backend == "fake" and (error or "").startswith("declined:"), (
                "expected fake backend decline for always_decline, got "
                f"status={status!r} backend={backend!r} error={error!r}"
            )
        else:
            assert status == "success", f"fake backend did not fill {name}: {error!r}"
    ai_plus = UserDefinedFunction(plus_one, LongType())
    ai_back = UserDefinedFunction(backwards, StringType())
    ai_zfill = UserDefinedFunction(zfill3, StringType())
    ai_edit = UserDefinedFunction(edit_distance, LongType())
    assert all(u.transpiled for u in (ai_plus, ai_back, ai_zfill, ai_edit)), "ai fills did not hit"
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
    fallback_decline = UserDefinedFunction(always_decline, LongType())
    assert not fallback_decline.transpiled, "declined UDF unexpectedly became a hit"
    cwd_df = ints.select(fallback_decline("x"))
    assert "BatchEvalPython" in cwd_df._jdf.queryExecution().toString()
    ai_ok = {
        "plus_one": bool(ai_plus.transpiled),
        "backwards": bool(ai_back.transpiled),
        "zfill3": bool(ai_zfill.transpiled),
        "edit_distance": bool(ai_edit.transpiled),
        "always_decline": bool(fallback_decline.transpiled),
    }

    # All arms exist before any timing, so rounds interleave fairly: nobody's
    # warmup is deeper. The plan assertions above guard against silently
    # benchmarking the wrong operator.
    arms = [
        ("python plus_one", ints.select(py_plus("x")), None),
        ("python backwards", strs.select(py_back("s")), None),
        ("python zfill3", strs.select(py_zfill("s")), None),
        ("python edit_distance", pairs.select(py_edit("a", "b")), None),
        ("python always_decline", ints.select(py_decline("x")), None),
        ("static plus_one", ints.select(st_plus("x")), "python plus_one"),
        ("static backwards", strs.select(st_back("s")), "python backwards"),
        ("static zfill3", strs.select(st_zfill("s")), "python zfill3"),
        ("static edit_distance", pairs.select(st_edit("a", "b")), "python edit_distance"),
        ("ai plus_one", ints.select(ai_plus("x")), "python plus_one"),
        ("ai backwards", strs.select(ai_back("s")), "python backwards"),
        ("ai zfill3", strs.select(ai_zfill("s")), "python zfill3"),
        ("ai edit_distance", pairs.select(ai_edit("a", "b")), "python edit_distance"),
        ("python fallback always_decline", cwd_df, "python always_decline"),
    ]
    for _, arm_df, _ in arms:  # warmup round for everyone
        noop_seconds(arm_df)
    secs: dict[str, float] = {}
    for _ in range(args.rounds):
        for name, arm_df, _ in arms:
            secs[name] = min(secs.get(name, float("inf")), noop_seconds(arm_df))
    for name, _, baseline in arms:
        speedup = f"  ({secs[baseline] / secs[name]:.1f}x python)" if baseline else ""
        print(f"{name + ':':30}{secs[name]:6.2f}s  {rows / secs[name]:>12,.0f} rows/s{speedup}")

    print()
    print("=== PER-FUNCTION (the --csv table) ===")
    header = [
        "function",
        "kind",
        "regular_rows_per_second",
        "transpiled_static_rows_per_second",
        "transpiled_ai_rows_per_second",
        "static_speedup_vs_regular",
        "ai_speedup_vs_regular",
        "transpile_succeeded_static",
        "transpile_succeeded_ai",
    ]
    wide: list[list] = []
    for fn_name, kind, regular_arm, static_arm, ai_arm in [
        ("plus_one", "catalyst", "python plus_one", "static plus_one", "ai plus_one"),
        ("backwards", "java_udf", "python backwards", "static backwards", "ai backwards"),
        ("zfill3", "catalyst", "python zfill3", "static zfill3", "ai zfill3"),
        ("edit_distance", "catalyst", "python edit_distance", "static edit_distance", "ai edit_distance"),
        ("always_decline", "declined", "python always_decline", None, "python fallback always_decline"),
    ]:
        wide.append(
            [
                fn_name,
                kind,
                round(rows / secs[regular_arm]),
                round(rows / secs[static_arm]) if static_arm else "",
                round(rows / secs[ai_arm]),
                round(secs[regular_arm] / secs[static_arm], 1) if static_arm else "",
                round(secs[regular_arm] / secs[ai_arm], 1),
                static_ok.get(fn_name, ""),
                ai_ok[fn_name],
            ]
        )
    print(",".join(header))
    for row in wide:
        print(",".join(str(c) for c in row))

    print()
    print(
        f"always_decline: backend declined ({decline_error}); fresh construction stays on the "
        "python path with unchanged perf -- the row above is the story"
    )

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
    print("=== ONE-TIME FILL COST (per rewrite) ===")
    print("static register_impl (Hypothesis + smoke test):")
    for name in ("plus_one", "backwards", "zfill3", "edit_distance"):
        print(f"  {name + ':':16}{static_fills[name]:6.2f}s")
    print("ai queue -> backend -> Hypothesis -> verified success (incl. claim latency):")
    for name in ("plus_one", "backwards", "zfill3", "edit_distance", "always_decline"):
        print(f"  {name + ':':16}{ai_fills[name]:6.2f}s")

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
