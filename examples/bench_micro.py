# SPDX-License-Identifier: Apache-2.0
"""Local micro benchmarks: regular vs transpiled (static) vs transpiled (AI).

Up to five arms per function on one local SparkSession:

  regular       interpreted python (transpilePyUDFs=false)
  static        Spark's BUILT-IN catalyst transpiler (pyTranspilers=catalyst,
                plugin not enabled) -- it converts simple operations only;
                anything it declines gets no static number, just the boolean
  ai catalyst   backend-generated Catalyst SQL (inline worker)
  ai java udf   backend-generated Java UDF (compiled on the driver)
  ai rust udf   --rust-accel (comet preset) only: a Rust UDF run natively by
                Comet. Very new and not ready for real use; see the README.

Every AI arm is the SAME python function. Each rewrite kind is filled in its
own catalog with the `targets` conf pinned to that one kind, so the backend
is asked for (and the worker only accepts) a SQL rewrite, then separately a
Java one, then a Rust one. `--targets` picks the kinds; the default, `all`,
runs catalyst and java_udf, plus rust when the Comet preset is live.

A kind the backend cannot produce for a function (declined, failed the
Hypothesis gate, timed out) leaves that arm empty with the reason printed;
the run carries on and still writes its CSVs. The exit code is 1 when a
requested arm is missing, unless --allow-missing-arms.

The wide CSV (--csv; CI uploads it for slides) is one row per function with
perf for each arm, speedup vs regular, and transpile_succeeded booleans for
static and every AI kind. Static converts plus_one and declines the rest;
always_decline (env-dependent, no faithful rewrite) is declined by every
backend and falls back to python with unchanged perf.

A second CSV (--transpilation-timing-csv, or auto-generated next to --csv)
records timing details: one-time transpile fill cost per kind and UDF
lookup/definition overhead on the plain vs AI paths.

  throughput   rows/sec over a cached DataFrame, noop sink, min of --rounds rounds
  definition   ms per UserDefinedFunction(...) creation (plugin hit vs plain)
  fill cost    ai: queue -> backend -> Hypothesis -> success or decline

The per-kind catalogs live next to $AI_UDF_TRANSPILE_SQLITE when it is set
(<stem>.micro-<kind>.sqlite, so a second run is all cache hits), else in a
temporary directory. They are never the shared examples catalog: a row there
may already hold another kind's rewrite of the same source.

    export SPARK_HOME=/path/to/spark   # same env as demo_quickstart.py
    python examples/bench_micro.py [rows] [--rounds N] [--csv PATH] [--targets all|KIND,...]
    python examples/bench_micro.py --transpilation-timing-csv bench_micro_transpilation_timing.csv
    export COMET_JAR=/path/to/comet-spark-spark4.x_2.13-<version>.jar
    python examples/bench_micro.py --rust-accel
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import DataType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, rustc, shutdown  # noqa: E402
from ai_udf_transpile.backends import ai_backend  # noqa: E402
from ai_udf_transpile.backends.fake import always_decline  # noqa: E402
from ai_udf_transpile.catalog import Catalog  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, KIND_RUST  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
PARTITIONS = os.cpu_count() or 16
RUST_COMET_CONFS = {
    "spark.plugins": "org.apache.spark.CometPlugin",
    "spark.comet.enabled": "true",
    # No CometShuffleManager (the earlier tweak dropped it): without this
    # flag Comet disables its extension entirely on session start, and
    # --rust-accel silently measures plain Spark. These benches are
    # projections with no shuffle, so the default manager costs nothing.
    "spark.comet.shuffle.enabled": "false",
    # spark.range(...) is every bench frame's leaf scan.
    "spark.comet.exec.range.enabled": "true",
    # The frames are .cache()d: without a native cache scan Comet leaves the
    # projection above them to Spark, and the rust arm (which Spark cannot
    # evaluate) fails instead of falling back.
    "spark.comet.exec.inMemoryCache.enabled": "true",
    "spark.comet.explain.fallback.enabled": "true",
    "spark.memory.offHeap.enabled": "true",
    "spark.memory.offHeap.size": "2g",
}
COMET_JAR_ENV_KEYS = ("AI_UDF_TRANSPILE_COMET_JAR", "COMET_JAR")

# Rewrite kinds in fill order, and what each is called in the CSV columns
# (transpiled_<column>_rows_per_second) and on stdout.
KINDS = (KIND_CATALYST, KIND_JAVA_UDF, KIND_RUST)
KIND_COLUMNS = {KIND_CATALYST: "ai_catalyst", KIND_JAVA_UDF: "ai_java", KIND_RUST: "ai_rust"}
KIND_LABELS = {KIND_CATALYST: "ai", KIND_JAVA_UDF: "java", KIND_RUST: "rust"}
KIND_CELLS = {KIND_CATALYST: "catalyst", KIND_JAVA_UDF: "java", KIND_RUST: "rust"}  # the `kinds` column
DECLINER = "always_decline"


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


@dataclass(frozen=True)
class BenchFunction:
    name: str
    fn: Callable[..., Any]
    rtype: DataType
    df: DataFrame  # the cached input the throughput rounds read
    cols: list[str]
    sample: Optional[DataFrame]  # small frame whose python output vets each rewrite


@dataclass
class Fill:
    """How one (function, kind) fill settled."""

    status: str = "missing"
    backend: Optional[str] = None
    model: Optional[str] = None
    target_kind: Optional[str] = None
    error: Optional[str] = None
    seconds: float = 0.0


@dataclass
class KindRun:
    """One rewrite kind: its own catalog, its fills, and the UDFs it serves."""

    kind: str
    sqlite: str
    fills: dict[str, Fill] = field(default_factory=dict)
    udfs: dict[str, UserDefinedFunction] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)


_NATIVE_DIR: list[str] = []


def _native(spark: SparkSession, sample: DataFrame) -> DataFrame:
    """``sample`` as a parquet scan, which Comet reads natively (one dir per run, removed at exit)."""
    if not _NATIVE_DIR:
        _NATIVE_DIR.append(tempfile.mkdtemp(prefix="ai-udf-bench-rust-"))
    path = tempfile.mkdtemp(dir=_NATIVE_DIR[0])
    sample.coalesce(1).write.mode("overwrite").parquet(path)
    return spark.read.parquet(path)


def noop_seconds(df: DataFrame) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def _plan(df: DataFrame) -> str:
    return str(df._jdf.queryExecution().toString())


def wait_for_catalog_result(catalog: Catalog, source_text: str, timeout: float = 10.0) -> Fill:
    """The row's settled state (success or failed), or the last one seen at the deadline."""
    deadline = time.monotonic() + timeout
    fill = Fill()
    conn = catalog._conn  # type: ignore[attr-defined]  # sqlite catalog only
    lock = catalog._lock  # type: ignore[attr-defined]
    while time.monotonic() < deadline:
        with lock:
            got = conn.execute(
                "SELECT status, backend, model, target_kind, error FROM cache WHERE source_text = ?",
                (source_text,),
            ).fetchone()
        if got is not None:
            fill = Fill(*got)
            if fill.status in {"failed", "success"}:
                break
        time.sleep(0.1)
    return fill


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


def parse_targets(parser: argparse.ArgumentParser, raw: str) -> Optional[list[str]]:
    """The kinds --targets names, in fill order; None for `all` (decided once Comet is known)."""
    names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    if not names or names == ["all"]:
        return None
    unknown = sorted(set(names) - set(KINDS))
    if unknown:
        parser.error(f"--targets: unknown kind(s) {unknown}; use all or a subset of {list(KINDS)}")
    return [k for k in KINDS if k in names]


def kind_catalog_path(shared: Optional[str], workdir: str, kind: str) -> str:
    """Where one kind's fills live: beside the shared catalog (cached across runs), else a temp dir."""
    if shared:
        p = Path(shared)
        return str(p.with_name(f"{p.stem}.micro-{kind}{p.suffix or '.sqlite'}"))
    return os.path.join(workdir, f"{kind}.sqlite")


def _comet_jar_from_env() -> str:
    for key in COMET_JAR_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def use_kind(spark: SparkSession, run: KindRun, backend: str, *, inline_worker: bool) -> Catalog:
    """Point the plugin at ``run``'s catalog with ``targets`` pinned to its one kind.

    The inline worker is bound to the catalog it started with, so switching
    kinds is a full shutdown() + enable(). UDFs built under an earlier kind
    keep their rewrite: the expression (and any Java/Rust function it calls)
    is already in the session.
    """
    shutdown()
    catalog = enable(spark, sqlite_path=run.sqlite, backend=backend, inline_worker=inline_worker)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")  # the built-in has its own arm
    # zfill/backwards/edit_distance take text, and zfill raises on NULL.
    conf.set_value(conf.PARTY_ON_STRINGS, "true", spark)
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)
    conf.set_value(conf.TARGETS, run.kind, spark)
    return catalog


def fill_kind(
    spark: SparkSession,
    run: KindRun,
    functions: list[BenchFunction],
    py_expect: dict[str, list[Any]],
    backend: str,
    *,
    with_decliner: bool,
    timeout: float,
) -> None:
    """Ask for ``run.kind`` for every function, then vet each served rewrite against python."""
    catalog = use_kind(spark, run, backend, inline_worker=True)
    for f in functions:
        if f.name == DECLINER and not with_decliner:
            continue
        start = time.perf_counter()
        UserDefinedFunction(f.fn, f.rtype)  # construction queues the pending row
        fill = wait_for_catalog_result(catalog, canonical_source_from_func(f.fn), timeout=timeout)
        fill.seconds = time.perf_counter() - start
        run.fills[f.name] = fill
        if f.name == DECLINER:
            if fill.status != "failed":
                run.problems[f.name] = f"expected a decline, got status={fill.status} ({fill.error})"
            continue
        if fill.status != "success":
            run.problems[f.name] = f"{fill.status}: {fill.error or 'no reason recorded'}"
            continue
        if fill.target_kind != run.kind:
            # The worker rejects any kind it did not ask for, so this is a row
            # filled by an earlier run with other targets: not this arm.
            run.problems[f.name] = f"catalog row is {fill.target_kind}, not {run.kind}; use a fresh catalog"
            continue
        udf = UserDefinedFunction(f.fn, f.rtype)
        if not udf.transpiled:
            run.problems[f.name] = "filled, but the cache lookup did not hit"
            continue
        # A Rust UDF runs only inside Comet: its sample is a parquet scan (a
        # local relation is evaluated by Spark itself).
        assert f.sample is not None
        check = _native(spark, f.sample) if run.kind == KIND_RUST else f.sample
        got = [r[0] for r in check.select(udf(*f.cols)).collect()]
        if got != py_expect[f.name]:
            run.problems[f.name] = f"wrong result on the sample: {got!r} != python {py_expect[f.name]!r}"
            continue
        if "EvalPython" in _plan(f.df.select(udf(*f.cols))):
            run.problems[f.name] = "served, but the plan still evaluates python"
            continue
        run.udfs[f.name] = udf


def time_lookups(
    spark: SparkSession, runs: list[KindRun], functions: list[BenchFunction], backend: str, lookups: int
) -> tuple[list[list[str]], list[str]]:
    """ms per UserDefinedFunction(...) creation: plain (plugin off), then each kind's catalog."""
    rows: list[list[str]] = []
    drift: list[str] = []

    def measure(mode: str, kind: str, f: BenchFunction, expected_hit: bool) -> None:
        start = time.perf_counter()
        transpiled = False
        for _ in range(lookups):
            transpiled = bool(UserDefinedFunction(f.fn, f.rtype).transpiled)
        ms = (time.perf_counter() - start) / lookups * 1000
        if transpiled != expected_hit:
            drift.append(
                f"{mode} lookup of {f.name} ({kind}): transpiled={transpiled}, expected={expected_hit}"
            )
        suffix = " (hit)" if transpiled else " (fallback)"
        print(f"  {mode + ' ' + f.name + ' (' + kind + ')' + ':':34}{ms:6.2f} ms/def{suffix}")
        rows.append(["lookup", mode, f.name, kind, "definition_ms", f"{ms:.6f}", "ms/def", str(transpiled)])

    spark.conf.set(TRANSPILE_CONF, "false")
    for f in functions:
        measure("plain", "-", f, False)
    for run in runs:
        use_kind(spark, run, backend, inline_worker=False)
        spark.conf.set(TRANSPILE_CONF, "true")
        for f in functions:
            if f.name in run.fills:
                measure("ai", run.kind, f, f.name in run.udfs)
    return rows, drift


def build_session(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[SparkSession, dict[str, str]]:
    rust_extra = parse_conf_overrides(parser, args.rust_accel_conf)
    if rust_extra and not args.rust_accel:
        parser.error("--rust-accel-conf requires --rust-accel")
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
        if args.rust_accel_preset == "comet":
            rust_confs.update(RUST_COMET_CONFS)
            comet_jar = _comet_jar_from_env()
            if comet_jar and not Path(comet_jar).is_file():
                parser.error(f"Comet jar does not exist: {comet_jar}")
            classpath_override = {"spark.jars", "spark.driver.extraClassPath"} & rust_extra.keys()
            if not comet_jar and not classpath_override:
                parser.error(
                    "--rust-accel-preset comet requires an existing jar via "
                    "AI_UDF_TRANSPILE_COMET_JAR/COMET_JAR or spark.jars via --rust-accel-conf"
                )
            if comet_jar:
                rust_confs.setdefault("spark.jars", comet_jar)
                rust_confs.setdefault("spark.driver.extraClassPath", comet_jar)
                rust_confs.setdefault("spark.executor.extraClassPath", comet_jar)
        rust_confs.update(rust_extra)
        for key, value in sorted(rust_confs.items()):
            builder = builder.config(key, value)
    return builder.getOrCreate(), rust_confs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        "--targets",
        default="all",
        metavar="all|KIND[,KIND...]",
        help=(
            "rewrite kinds to fill and time, each in its own catalog with only that kind asked for: "
            "catalyst, java_udf, rust. all (default) = catalyst,java_udf, plus rust under the Comet preset"
        ),
    )
    parser.add_argument(
        "--fill-timeout",
        type=float,
        default=900.0,
        metavar="SECONDS",
        help="how long one (function, kind) fill may take before its arm is reported missing",
    )
    parser.add_argument(
        "--allow-missing-arms",
        action="store_true",
        help="exit 0 even when a requested kind could not be filled for some function",
    )
    parser.add_argument(
        "--rust-accel",
        action="store_true",
        help=(
            "run with rust acceleration configs (default preset: comet; "
            "jar from AI_UDF_TRANSPILE_COMET_JAR/COMET_JAR)"
        ),
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
    args.rust_accel_preset = args.rust_accel_preset or "comet"
    requested = parse_targets(parser, args.targets)
    comet_preset = args.rust_accel and args.rust_accel_preset == "comet"
    if requested and KIND_RUST in requested and not comet_preset:
        parser.error(
            "--targets rust needs --rust-accel with the comet preset (a Rust UDF runs only in Comet)"
        )
    timing_csv = args.transpilation_timing_csv
    if timing_csv is None and args.csv:
        p = Path(args.csv)
        suffix = p.suffix if p.suffix else ".csv"
        timing_csv = str(p.with_name(f"{p.stem}_transpilation_timing{suffix}"))
    if timing_csv and args.csv:
        if Path(args.csv).expanduser().resolve() == Path(timing_csv).expanduser().resolve():
            parser.error(
                "--transpilation-timing-csv must differ from --csv "
                "(or omit it to use the default sibling path)"
            )
    rows = args.rows
    spark, rust_confs = build_session(args, parser)

    # Cache the inputs so range/cast generation is not in the measurement.
    ints = spark.range(0, rows, numPartitions=PARTITIONS).toDF("x").cache()
    strs = ints.select(F.col("x").cast("string").alias("s")).cache()
    # Short strings with varying content: the DP cost is real but not so big
    # that the python arm dominates the wall clock.
    pairs = ints.select(
        (F.col("x") % 100003).cast("string").alias("a"),
        ((F.col("x") * 7 + 13) % 99991).cast("string").alias("b"),
    ).cache()
    for frame in (ints, strs, pairs):
        frame.count()
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
    functions = [
        BenchFunction("plus_one", plus_one, LongType(), ints, ["x"], sample_ints),
        BenchFunction("backwards", backwards, StringType(), strs, ["s"], sample_strs),
        BenchFunction("zfill3", zfill3, StringType(), strs, ["s"], sample_zfill),
        BenchFunction("edit_distance", edit_distance, LongType(), pairs, ["a", "b"], sample_pairs),
        BenchFunction(DECLINER, always_decline, LongType(), ints, ["x"], None),
    ]
    convertible = [f for f in functions if f.name != DECLINER]

    if args.rust_accel:
        print(f"=== RUST ACCEL ({args.rust_accel_preset} preset) ===")
        for key, value in sorted(rust_confs.items()):
            print(f"{key}={value}")
        print()

    # -- regular phase: transpile off, so every arm is interpreted python.
    py_udfs: dict[str, UserDefinedFunction] = {}
    py_expect: dict[str, list[Any]] = {}
    for f in functions:
        py_udfs[f.name] = UserDefinedFunction(f.fn, f.rtype)
        assert not py_udfs[f.name].transpiled
        if f.sample is not None:
            py_expect[f.name] = [r[0] for r in f.sample.select(py_udfs[f.name](*f.cols)).collect()]
    assert "BatchEvalPython" in _plan(ints.select(py_udfs["plus_one"]("x")))

    # -- static phase: Spark's BUILT-IN catalyst transpiler. enable() has not
    # run, so the ai plugin is neither registered nor named in pyTranspilers.
    # It converts simple operations only. A decline gets NO static number
    # (perf would just be the python fallback) -- the boolean is the whole
    # static story for it.
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    st_udfs: dict[str, UserDefinedFunction] = {}
    static_ok: dict[str, bool] = {}
    for f in functions:
        st_udfs[f.name] = UserDefinedFunction(f.fn, f.rtype)
        static_ok[f.name] = bool(st_udfs[f.name].transpiled)
    converted = {n for n, ok in static_ok.items() if ok}
    if converted != {"plus_one"}:
        # The docstring's "converts plus_one, declines the rest" is stale:
        # worth knowing, not worth losing the run over.
        print(f"warning: built-in catalyst convert set drifted: {sorted(converted)}", file=sys.stderr)
    for f in functions:
        if not static_ok[f.name] or f.sample is None:
            continue
        got = [r[0] for r in f.sample.select(st_udfs[f.name](*f.cols)).collect()]
        if got != py_expect[f.name] or "EvalPython" in _plan(f.df.select(st_udfs[f.name](*f.cols))):
            print(
                f"warning: built-in static rewrite of {f.name} is wrong or not native; no arm",
                file=sys.stderr,
            )
            static_ok[f.name] = False

    # -- ai phase: one catalog per kind, targets pinned to it. The decliner is
    # asked once, in the first kind's catalog: every kind declines it.
    backend = ai_backend(spark)
    kinds = requested or [KIND_CATALYST, KIND_JAVA_UDF]
    if requested is None and comet_preset:
        kinds = [*kinds, KIND_RUST]
    if KIND_RUST in kinds:
        # The probes read the targets conf; the rust run pins it the same way.
        conf.set_value(conf.TARGETS, KIND_RUST, spark)
        reason = rustc.rust_unservable(spark)
        if not reason and not rustc.rust_offered(spark, ["bigint"], "bigint"):
            reason = "no cargo or no comet-udf-sdk to build against"
        if reason:
            print(f"warning: no rust arm: {reason}", file=sys.stderr)
            kinds = [k for k in kinds if k != KIND_RUST]
    shared = os.environ.get("AI_UDF_TRANSPILE_SQLITE")
    workdir = tempfile.mkdtemp(prefix="ai-udf-bench-")
    runs = [KindRun(kind, kind_catalog_path(shared, workdir, kind)) for kind in kinds]
    for i, run in enumerate(runs):
        print(f"--- filling {run.kind} (targets={run.kind}, catalog {run.sqlite}) ---")
        fill_kind(spark, run, functions, py_expect, backend, with_decliner=i == 0, timeout=args.fill_timeout)
        for name, problem in run.problems.items():
            print(f"warning: no {run.kind} arm for {name}: {problem}", file=sys.stderr)
    shutdown()
    spark.conf.set(TRANSPILE_CONF, "true")
    fallback_df = ints.select(py_udfs[DECLINER]("x"))

    # All arms exist before any timing, so rounds interleave fairly: nobody's
    # warmup is deeper. The plan checks above guard against silently
    # benchmarking the wrong operator.
    print()
    print(f"=== THROUGHPUT ({rows:,} rows, local[*], noop sink, min of {args.rounds} rounds) ===")
    arms: list[tuple[str, DataFrame, Optional[str]]] = []
    for f in functions:
        arms.append((f"python {f.name}", f.df.select(py_udfs[f.name](*f.cols)), None))
        if static_ok[f.name]:
            arms.append((f"static {f.name}", f.df.select(st_udfs[f.name](*f.cols)), f"python {f.name}"))
        if f.name == DECLINER:
            arms.append((f"python fallback {f.name}", fallback_df, f"python {f.name}"))
            continue
        for run in runs:
            if f.name in run.udfs:
                label = f"{KIND_LABELS[run.kind]} {f.name}"
                arms.append((label, f.df.select(run.udfs[f.name](*f.cols)), f"python {f.name}"))
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
        *(f"transpiled_{KIND_COLUMNS[k]}_rows_per_second" for k in KINDS),
        "static_speedup_vs_regular",
        *(f"{KIND_COLUMNS[k]}_speedup_vs_regular" for k in KINDS),
        "transpile_succeeded_static",
        *(f"transpile_succeeded_{KIND_COLUMNS[k]}" for k in KINDS),
        "ai_backend",
        "ai_model",
    ]
    by_kind = {run.kind: run for run in runs}
    wide: list[list[Any]] = []
    for f in functions:
        regular = secs[f"python {f.name}"]
        static_arm = f"static {f.name}" if static_ok[f.name] else None
        served = {k: f.name in by_kind[k].udfs if k in by_kind else False for k in KINDS}
        arm_for: dict[str, Optional[str]] = {
            k: f"{KIND_LABELS[k]} {f.name}" if served[k] else None for k in KINDS
        }
        if f.name == DECLINER:
            # The decliner's catalyst column is the fallback perf (the
            # pipeline outcome); its other columns stay empty.
            arm_for[KIND_CATALYST] = f"python fallback {f.name}"
        # "+", not ",": the stdout row is comma-joined.
        kinds_cell = "+".join(KIND_CELLS[k] for k in KINDS if served[k]) or "declined"
        fills = [by_kind[k].fills[f.name] for k in KINDS if k in by_kind and f.name in by_kind[k].fills]
        first_fill = fills[0] if fills else Fill()
        wide.append(
            [
                f.name,
                kinds_cell,
                round(rows / regular),
                round(rows / secs[static_arm]) if static_arm else "",
                *(round(rows / secs[a]) if (a := arm_for[k]) else "" for k in KINDS),
                round(regular / secs[static_arm], 1) if static_arm else "",
                *(round(regular / secs[a], 1) if (a := arm_for[k]) else "" for k in KINDS),
                static_ok[f.name],
                *(served[k] for k in KINDS),
                first_fill.backend or "",
                first_fill.model or "",
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
    decline = runs[0].fills.get(DECLINER) if runs else None
    if decline is not None:
        print(
            f"{DECLINER}: backend declined ({decline.error}); fresh construction stays on the python "
            "path with unchanged perf"
        )

    print()
    print("=== ONE-TIME FILL COST (queue -> backend -> Hypothesis -> success or decline) ===")
    timing_header = ["section", "mode", "function", "kind", "metric", "value", "unit", "transpile_succeeded"]
    timing_rows: list[list[str]] = []
    for run in runs:
        for name, fill in run.fills.items():
            ok = name in run.udfs
            note = "" if ok else f"  ({fill.status})"
            print(f"  {name + ' (' + run.kind + ')':30}{fill.seconds:7.2f}s  (incl. claim latency){note}")
            timing_rows.append(
                ["fill", "ai", name, run.kind, "fill_seconds", f"{fill.seconds:.6f}", "seconds", str(ok)]
            )

    print()
    lookups = 200
    print(f"=== LOOKUP COST (UDF definition; avg of {lookups} creations) ===")
    lookup_rows, drift = time_lookups(spark, runs, functions, backend, lookups)
    timing_rows.extend(lookup_rows)
    for message in drift:
        print(f"warning: {message}", file=sys.stderr)

    shutdown()
    spark.stop()
    if _NATIVE_DIR:
        shutil.rmtree(_NATIVE_DIR[0], ignore_errors=True)
    shutil.rmtree(workdir, ignore_errors=True)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(wide)
        print(f"csv: wrote {len(wide)} rows to {args.csv}")
    if timing_csv:
        with open(timing_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(timing_header)
            writer.writerows(timing_rows)
        print(f"transpilation timing csv: wrote {len(timing_rows)} rows to {timing_csv}")

    missing = [f"{f.name} ({run.kind})" for run in runs for f in convertible if f.name not in run.udfs]
    decliner_problem = runs[0].problems.get(DECLINER) if runs else None
    if decliner_problem:
        missing.append(f"{DECLINER}: {decliner_problem}")
    if missing:
        print(f"missing arms: {', '.join(missing)}", file=sys.stderr)
        if not args.allow_missing_arms:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
