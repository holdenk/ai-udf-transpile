# SPDX-License-Identifier: Apache-2.0
"""Local micro benchmarks: regular vs transpiled (static) vs transpiled (AI).

Up to four arms per function on one local SparkSession:

  regular       interpreted python (transpilePyUDFs=false)
  static        Spark's BUILT-IN catalyst transpiler (pyTranspilers=catalyst,
                plugin not enabled) -- it converts simple operations only;
                anything it declines gets no static number, just the boolean
  ai catalyst   backend-generated Catalyst SQL (fake backend, inline worker)
  ai java udf   backend-generated Java UDF (compiled on the driver); the
                java target is a *_java twin -- same body, different def
                name, so its canonical source maps to a different catalog
                key and its own fixture. Slower per row than Catalyst SQL
                for cheap ops, but it is a target for every convertible
                benched UDF.

The wide CSV (--csv; CI uploads it for slides) is one row per function with
perf for each arm, speedup vs regular, and transpile_succeeded booleans for
static and both AI kinds. Static converts plus_one and declines the rest;
the backend converts four (both kinds) and declines always_decline
(env-dependent, no faithful rewrite), which falls back to python with
unchanged perf.

A second CSV (--transpilation-timing-csv, or auto-generated next to --csv)
records timing
details: one-time transpile fill cost and UDF lookup/definition overhead on
the plain vs AI paths.

  throughput   rows/sec over a cached DataFrame, noop sink, min of --rounds rounds
  definition   ms per UserDefinedFunction(...) creation (plugin hit vs plain)
  fill cost    ai: queue -> backend -> Hypothesis -> success or decline

    export SPARK_HOME=/path/to/spark-master   # same env as demo_quickstart.py
    python examples/bench_micro.py [rows] [--rounds N] [--csv PATH]
    python examples/bench_micro.py --transpilation-timing-csv bench_micro_transpilation_timing.csv
    python examples/bench_micro.py --rust-accel
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

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.backends.fake import (  # noqa: E402
    FIXTURES,
    always_decline,
    backwards_java,
    edit_distance_java,
    plus_one_java,
    zfill3_java,
)
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
PARTITIONS = os.cpu_count() or 16
RUST_COMET_CONFS = {
    "spark.plugins": "org.apache.comet.CometPlugin",
    "spark.comet.enabled": "true",
}


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


JAVA_TWINS = {
    "plus_one": plus_one_java,
    "backwards": backwards_java,
    "zfill3": zfill3_java,
    "edit_distance": edit_distance_java,
}

for _f in (
    plus_one,
    backwards,
    zfill3,
    edit_distance,
    plus_one_java,
    backwards_java,
    zfill3_java,
    edit_distance_java,
):
    # The ai arms need a fake fixture for each benched function (both target
    # kinds); fail at import rather than as a surprise decline mid-bench.
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


def parse_conf_overrides(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            parser.error(f"--rust-accel-conf expects key=value, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            parser.error(f"--rust-accel-conf expects key=value, got {raw!r}")
        out[key] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=int, default=10_000_000)
    parser.add_argument("--rounds", type=int, default=3, help="measured rounds per arm; min is reported")
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="write the wide per-function table (perf + transpile booleans)",
    )
    parser.add_argument(
        "--transpilation-timing-csv",
        "--timing-csv",
        dest="transpilation_timing_csv",
        metavar="PATH",
        help="write transpilation timing details (fill + lookup/definition cost) as a second CSV",
    )
    parser.add_argument(
        "--rust-accel",
        action="store_true",
        help="run with rust acceleration configs (default preset: comet)",
    )
    parser.add_argument(
        "--rust-accel-preset",
        choices=["comet", "none"],
        default=None,
        help="preset rust accelerator config set (implies --rust-accel)",
    )
    parser.add_argument(
        "--rust-accel-conf",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Spark config(s) when --rust-accel is on (repeatable)",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.rust_accel_preset is not None:
        args.rust_accel = True
    rust_extra = parse_conf_overrides(parser, args.rust_accel_conf)
    if rust_extra and not args.rust_accel:
        parser.error("--rust-accel-conf requires --rust-accel")
    timing_csv = args.transpilation_timing_csv
    if timing_csv is None and args.csv:
        p = Path(args.csv)
        suffix = p.suffix if p.suffix else ".csv"
        timing_csv = str(p.with_name(f"{p.stem}_transpilation_timing{suffix}"))
    if timing_csv and args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        timing_path = Path(timing_csv).expanduser().resolve()
        if csv_path == timing_path:
            parser.error(
                "--transpilation-timing-csv must differ from --csv "
                "(or omit it to use the default sibling path)"
            )
    rust_preset = args.rust_accel_preset or "comet"
    rows = args.rows
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")  # the regular phase must stay interpreted python
    )
    rust_confs: dict[str, str] = {}
    if args.rust_accel:
        if rust_preset == "comet":
            rust_confs.update(RUST_COMET_CONFS)
        rust_confs.update(rust_extra)
        for key, value in sorted(rust_confs.items()):
            builder = builder.config(key, value)
    spark = builder.getOrCreate()

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
    # name, fn, return type, cached input df, its columns, sample frame
    functions = [
        ("plus_one", plus_one, LongType(), ints, ["x"], sample_ints),
        ("backwards", backwards, StringType(), strs, ["s"], sample_strs),
        ("zfill3", zfill3, StringType(), strs, ["s"], sample_zfill),
        ("edit_distance", edit_distance, LongType(), pairs, ["a", "b"], sample_pairs),
        ("always_decline", always_decline, LongType(), ints, ["x"], None),
    ]

    if args.rust_accel:
        label = f"{rust_preset} preset"
        print(f"=== RUST ACCEL ({label}) ===")
        for key, value in sorted(rust_confs.items()):
            print(f"{key}={value}")
        print()

    print(f"=== THROUGHPUT ({rows:,} rows, local[*], noop sink, min of {args.rounds} rounds) ===")

    # -- regular phase: transpile off, so every arm is interpreted python.
    py_udfs = {}
    py_expect = {}
    for name, fn, rtype, df, cols, sample in functions:
        py_udfs[name] = UserDefinedFunction(fn, rtype)
        assert not py_udfs[name].transpiled
        if sample is not None:
            py_expect[name] = [r[0] for r in sample.select(py_udfs[name](*cols)).collect()]
    assert "BatchEvalPython" in ints.select(py_udfs["plus_one"]("x"))._jdf.queryExecution().toString()

    # -- static phase: Spark's BUILT-IN catalyst transpiler. enable() has not
    # run, so the ai plugin is neither registered nor named in pyTranspilers.
    # It converts simple operations only: plus_one converts; the multi-
    # statement bodies and unsupported method calls decline. A decline gets
    # NO static number (perf would just be the python fallback) -- the
    # boolean is the whole static story for it.
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    st_udfs = {}
    static_ok = {}
    for name, fn, rtype, _df, _cols, _sample in functions:
        st_udfs[name] = UserDefinedFunction(fn, rtype)
        static_ok[name] = bool(st_udfs[name].transpiled)
    # Pin the convert set: if the built-in ever learns more, the docstring's
    # "declines the rest" story is stale -- fail loudly instead.
    converted = {n for n, ok in static_ok.items() if ok}
    assert converted == {"plus_one"}, f"built-in catalyst convert set drifted: {sorted(converted)}"
    for name, _fn, _rtype, df, cols, sample in functions:
        if not static_ok[name]:
            continue
        got = [r[0] for r in sample.select(st_udfs[name](*cols)).collect()]
        assert got == py_expect[name], f"static {name} rewrote wrongly"
        assert "EvalPython" not in df.select(st_udfs[name](*cols))._jdf.queryExecution().toString()

    # -- ai phase: enable the plugin on a fresh catalog, drop the built-in
    # from pyTranspilers so the ai arms are isolated, and let the inline
    # worker fill each function from the backend (construction queues the
    # row). Two targets per function: the canonical name gets the catalyst
    # SQL fixture; a *_java twin (same body, different def name = a
    # different catalog key) gets the Java UDF fixture, compiled on the
    # driver during verification. always_decline declines whenever it is
    # asked; it has no java twin (nothing would serve it either way).
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-bench-"), "cache.sqlite")
    catalog = enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)
    fills: dict[tuple[str, str], float] = {}
    decline_error: str | None = None
    for name, fn, rtype, _df, _cols, _sample in functions:
        for kind, target in (("catalyst", fn), ("java", JAVA_TWINS.get(name))):
            if target is None:
                continue
            start = time.perf_counter()
            UserDefinedFunction(target, rtype)  # construction queues the pending row
            status, backend, error = wait_for_catalog_result(
                catalog, canonical_source_from_func(target), timeout=120
            )
            fills[(name, kind)] = time.perf_counter() - start
            if name == "always_decline":
                decline_error = error
                assert status == "failed" and backend == "fake" and (error or "").startswith("declined:"), (
                    "expected fake backend decline for always_decline, got "
                    f"status={status!r} backend={backend!r} error={error!r}"
                )
            else:
                assert status == "success", f"fake backend did not fill {name} ({kind}): {error!r}"
    ai_udfs = {}
    java_udfs = {}
    for name, fn, rtype, df, cols, sample in functions:
        if name == "always_decline":
            ai_udfs[name] = UserDefinedFunction(fn, rtype)
            assert not ai_udfs[name].transpiled, "declined UDF unexpectedly became a hit"
            continue
        # Twins share a body, so py_expect (from the original) vets both kinds.
        for label, udf_fn in (("ai", fn), ("java", JAVA_TWINS[name])):
            udf = UserDefinedFunction(udf_fn, rtype)
            assert udf.transpiled, f"{label} fill did not hit for {name}"
            got = [r[0] for r in sample.select(udf(*cols)).collect()]
            assert got == py_expect[name], f"{label} {name} rewrote wrongly"
            assert "EvalPython" not in df.select(udf(*cols))._jdf.queryExecution().toString()
            (ai_udfs if label == "ai" else java_udfs)[name] = udf
    ai_ok = {name: bool(u.transpiled) for name, u in ai_udfs.items()}
    java_ok = {name: bool(u.transpiled) for name, u in java_udfs.items()}
    fallback_df = ints.select(ai_udfs["always_decline"]("x"))
    assert "BatchEvalPython" in fallback_df._jdf.queryExecution().toString()

    # All arms exist before any timing, so rounds interleave fairly: nobody's
    # warmup is deeper. The plan assertions above guard against silently
    # benchmarking the wrong operator.
    arms: list[tuple[str, object, str | None]] = []
    for name, fn, rtype, df, cols, sample in functions:
        arms.append((f"python {name}", df.select(py_udfs[name](*cols)), None))
        if static_ok[name]:
            arms.append((f"static {name}", df.select(st_udfs[name](*cols)), f"python {name}"))
        if name == "always_decline":
            arms.append(("python fallback always_decline", fallback_df, "python always_decline"))
        else:
            arms.append((f"ai {name}", df.select(ai_udfs[name](*cols)), f"python {name}"))
            arms.append((f"java {name}", df.select(java_udfs[name](*cols)), f"python {name}"))
    for _name, arm_df, _ in arms:  # warmup round for everyone
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
        "kinds",
        "regular_rows_per_second",
        "transpiled_static_rows_per_second",
        "transpiled_ai_catalyst_rows_per_second",
        "transpiled_ai_java_rows_per_second",
        "static_speedup_vs_regular",
        "ai_catalyst_speedup_vs_regular",
        "ai_java_speedup_vs_regular",
        "transpile_succeeded_static",
        "transpile_succeeded_ai_catalyst",
        "transpile_succeeded_ai_java",
    ]
    wide: list[list] = []
    for name, _fn, _rtype, _df, _cols, _sample in functions:
        regular_arm = f"python {name}"
        static_arm = f"static {name}" if static_ok[name] else None
        ai_arm = f"ai {name}" if ai_ok[name] else None
        java_arm = f"java {name}" if java_ok.get(name) else None
        # always_decline's catalyst column is the fallback perf (the pipeline
        # outcome); the java column stays empty (no twin, nothing attempted).
        if name == "always_decline":
            ai_arm = "python fallback always_decline"
        # "+", not ",": the stdout row is comma-joined. Kinds come from the
        # success booleans, not arm presence, so a declined UDF reads "declined".
        kinds = (
            "+".join(k for k, ok in (("catalyst", ai_ok[name]), ("java", java_ok.get(name, False))) if ok)
            or "declined"
        )
        wide.append(
            [
                name,
                kinds,
                round(rows / secs[regular_arm]),
                round(rows / secs[static_arm]) if static_arm else "",
                round(rows / secs[ai_arm]) if ai_arm else "",
                round(rows / secs[java_arm]) if java_arm else "",
                round(secs[regular_arm] / secs[static_arm], 1) if static_arm else "",
                round(secs[regular_arm] / secs[ai_arm], 1) if ai_arm else "",
                round(secs[regular_arm] / secs[java_arm], 1) if java_arm else "",
                static_ok[name],
                ai_ok[name],
                java_ok.get(name, False),
            ]
        )
    print(",".join(header))
    for row in wide:
        print(",".join(str(c) for c in row))

    print()
    print(
        f"static: built-in catalyst converted {sum(static_ok.values())}/{len(functions)} "
        f"({', '.join(n for n, ok in static_ok.items() if ok)}); the rest have no static number"
    )
    print(
        "always_decline: backend declined "
        f"({decline_error}); fresh construction stays on the python path with unchanged perf"
    )

    print()
    print("=== ONE-TIME FILL COST (queue -> backend -> Hypothesis -> success or decline) ===")
    timing_header = ["section", "mode", "function", "kind", "metric", "value", "unit", "transpile_succeeded"]
    timing_rows: list[list] = []
    for (name, kind), fill_secs in fills.items():
        declined = "  (declined)" if name == "always_decline" else ""
        print(f"  {name + ' (' + kind + ')':24}{fill_secs:6.2f}s  (incl. claim latency){declined}")
        timing_rows.append(
            [
                "fill",
                "ai",
                name,
                kind,
                "fill_seconds",
                f"{fill_secs:.6f}",
                "seconds",
                str(name != "always_decline"),
            ]
        )

    print()
    print("=== LOOKUP COST (UDF definition; avg of 200 creations) ===")
    lookups = 200
    lookup_targets = []
    for name, fn, rtype, _df, _cols, _sample in functions:
        lookup_targets.append((name, "catalyst", fn, rtype, ai_ok[name]))
        twin = JAVA_TWINS.get(name)
        if twin is not None:
            lookup_targets.append((name, "java", twin, rtype, java_ok.get(name, False)))
    for mode, transpile_on in (("plain", False), ("ai", True)):
        spark.conf.set(TRANSPILE_CONF, "true" if transpile_on else "false")
        if transpile_on:
            spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
        for name, kind, fn, rtype, expected_hit in lookup_targets:
            start = time.perf_counter()
            transpiled = False
            for _ in range(lookups):
                transpiled = bool(UserDefinedFunction(fn, rtype).transpiled)
            ms = (time.perf_counter() - start) / lookups * 1000
            if transpile_on:
                assert transpiled == expected_hit, (
                    f"lookup mode drift for {name} ({kind}): transpiled={transpiled}, expected={expected_hit}"
                )
            else:
                assert not transpiled, f"plain lookup unexpectedly transpiled {name} ({kind})"
            suffix = " (hit)" if transpiled else " (fallback)"
            print(f"  {mode + ' ' + name + ' (' + kind + ')' + ':':30}{ms:6.2f} ms/def{suffix}")
            timing_rows.append(
                [
                    "lookup",
                    mode,
                    name,
                    kind,
                    "definition_ms",
                    f"{ms:.6f}",
                    "ms/def",
                    str(transpiled),
                ]
            )

    shutdown()
    spark.stop()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(wide)
        print(f"csv: wrote {len(wide)} rows to {args.csv}")
    if timing_csv:
        with open(timing_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(timing_header)
            writer.writerows(timing_rows)
        print(f"transpilation timing csv: wrote {len(timing_rows)} rows to {timing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
