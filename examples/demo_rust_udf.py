# SPDX-License-Identifier: Apache-2.0
"""Rust target demo: Python UDFs with no SQL form, rewritten to Rust, run by Comet.

Each UDF below is a data-dependent loop. SQL is still the backend's first
choice, and a model will sometimes find one (unrolling a bounded loop into
arithmetic); where it cannot, with the `rust` target on it answers in Rust
rather than a Java UDF. The demo prints whichever rewrite was served, the
physical plan (a Rust call sits inside CometProject: no JVM or Python hop
per row), and the throughput against the interpreted Python UDF.

Needs, on top of the usual Spark setup (see demo_quickstart.py):

    examples/setup_comet.sh --write-env-file .comet-env && source .comet-env
    # COMET_JAR: the Comet jar. COMET_HOME: the checkout it was built from
    # (its native/comet-udf-sdk is what the UDF libraries build against).
    # With a released jar and no checkout, cargo fetches the SDK by tag.
    python examples/demo_rust_udf.py [--rows 2000000]

Backend: $AI_UDF_TRANSPILE_BACKEND, else auto (first AI CLI on PATH). The
first Rust build against a Comet SDK compiles arrow (a couple of minutes);
after that a UDF builds in about a second.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import BooleanType, LongType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, rustc, shutdown  # noqa: E402
from ai_udf_transpile.backends import ai_backend  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402

COMET_JAR_ENV_KEYS = ("AI_UDF_TRANSPILE_COMET_JAR", "COMET_JAR")
COMET_CONFS = {
    "spark.plugins": "org.apache.spark.CometPlugin",
    "spark.comet.enabled": "true",
    # Without Comet's shuffle manager the extension disables itself unless
    # Comet shuffle is explicitly off.
    "spark.comet.shuffle.enabled": "false",
    # spark.range is the demo's scan: generate it natively so the projection
    # holding the UDF is Comet's. A Rust UDF above a Spark operator is a
    # query error, not a fallback.
    "spark.comet.exec.range.enabled": "true",
    "spark.memory.offHeap.enabled": "true",
    "spark.memory.offHeap.size": "2g",
}


def collatz_steps(n: int) -> int:
    if n is None or n < 1:
        return None
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def is_prime(n: int) -> bool:
    # Trial division: the loop bound depends on the value.
    if n is None:
        return None
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def longest_digit_run(n: int) -> int:
    # Length of the longest run of one repeated decimal digit in n.
    if n is None or n < 0:
        return None
    best = 0
    run = 0
    prev = -1
    while True:
        digit = n % 10
        run = run + 1 if digit == prev else 1
        best = max(best, run)
        prev = digit
        n //= 10
        if n == 0:
            break
    return best


UDFS = [
    (collatz_steps, LongType()),
    (is_prime, BooleanType()),
    (longest_digit_run, LongType()),
]


def _row(func):
    """This session's row for ``func``: same source AND same Spark version.

    A shared cache can hold the same source under another Spark version, or
    the "+no-rust" twin a session without Comet queued; neither is ours.
    """
    catalog = get_catalog()
    spark = SparkSession.getActiveSession()
    with catalog._lock:
        return catalog._conn.execute(
            "SELECT status, target_kind, impl_source, catalyst_sql, model, error FROM cache "
            "WHERE source_text = ? AND spark_version = ?",
            (canonical_source_from_func(func), str(spark.version)),
        ).fetchone()


def wait_for_row(func, rtype, timeout: float):
    """Re-create the UDF until the worker has answered (success or failed)."""
    deadline = time.time() + timeout
    udf = UserDefinedFunction(func, rtype)
    while time.time() < deadline and not udf.transpiled:
        row = _row(func)
        if row is not None and row[0] == "failed":
            break
        time.sleep(1.0)
        udf = UserDefinedFunction(func, rtype)
    return udf


def rows_per_second(df, rows: int, rounds: int = 3) -> float:
    best = float("inf")
    for _ in range(rounds):
        started = time.perf_counter()
        df.write.format("noop").mode("overwrite").save()
        best = min(best, time.perf_counter() - started)
    return rows / best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds to wait per UDF")
    parser.add_argument(
        "--skip-if-unavailable",
        action="store_true",
        help="exit 0 with a note when this machine cannot build or run Rust UDFs (run_demos.sh)",
    )
    args = parser.parse_args()
    unavailable = 0 if args.skip_if_unavailable else 2

    jar = next((os.environ[k] for k in COMET_JAR_ENV_KEYS if os.environ.get(k)), "")
    if not jar or not Path(jar).exists():
        print("set COMET_JAR to a Comet 1.2+ jar (examples/setup_comet.sh builds one)", file=sys.stderr)
        return unavailable

    builder = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-rust")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.jars", jar)
        .config("spark.driver.extraClassPath", jar)
        .config("spark.executor.extraClassPath", jar)
    )
    for key, value in COMET_CONFS.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    sqlite = os.environ.get("AI_UDF_TRANSPILE_SQLITE") or os.path.join(
        tempfile.mkdtemp(prefix="ai-udf-rust-"), "cache.sqlite"
    )
    enable(spark, sqlite_path=sqlite, backend=ai_backend(spark), inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    # The opt-in. Without it no backend is asked for Rust and no Rust is served.
    conf.set_value(conf.TARGETS, "catalyst,java_udf,rust", spark)

    reason = rustc.rust_unservable(spark)
    if not reason and not rustc.rust_offered(spark, ["bigint"], "bigint"):
        reason = "no cargo or no comet-udf-sdk to build against (see the warning above)"
    if reason:
        print(f"this session cannot run Rust UDFs: {reason}", file=sys.stderr)
        shutdown()
        spark.stop()
        return unavailable
    print(f"comet-udf-sdk: {rustc.resolve_sdk(spark).ident}")
    print(f"build cache:   {rustc.build_dir(spark)}")

    # Positive ids only: every UDF here returns None below its domain.
    frame = spark.range(1, args.rows + 1)
    ok = True
    for func, rtype in UDFS:
        name = func.__name__
        print(f"\n=== {name} ===")
        plain = UserDefinedFunction(func, rtype)
        state = "cache hit" if plain.transpiled else "miss -> Python UDF while the worker fills the cache"
        print(f"first definition: transpiled={bool(plain.transpiled)} ({state})")
        served = wait_for_row(func, rtype, args.timeout)
        status, kind, source, sql, model, error = _row(func) or ("missing", None, None, None, None, None)
        if not served.transpiled:
            print(f"no rewrite served: status={status} error={error}")
            ok = False
            continue
        print(f"served as kind={kind} by {model}")
        print((source or sql or "").rstrip())
        out = frame.select(served("id").alias("r"))
        plan = out._jdf.queryExecution().executedPlan().toString()
        native = "CometProject" in plan and "PythonUDF" not in plan
        print(f"plan ({'native' if native else 'NOT native'}):\n{plan.rstrip()}")

        sample = [1, 2, 7, 27, 97, 871, 6171, 1000003, 7772111]
        check = spark.createDataFrame([(n,) for n in sample], "id bigint").coalesce(1)
        path = os.path.join(tempfile.mkdtemp(prefix="ai-udf-rust-"), "sample")
        check.write.parquet(path)  # a Parquet scan is native; an in-memory frame is not
        got = dict(spark.read.parquet(path).select("id", served("id")).collect())
        want = {n: func(n) for n in sample}
        print(f"matches python on {sample}: {got == want}")
        ok = ok and got == want

        # The interpreted arm: same function object, transpilation switched off.
        spark.conf.set("spark.sql.experimental.optimizer.transpilePyUDFs", "false")
        python_udf = UserDefinedFunction(func, rtype)
        python_rate = rows_per_second(frame.select(python_udf("id").alias("r")), args.rows)
        spark.conf.set("spark.sql.experimental.optimizer.transpilePyUDFs", "true")
        rust_rate = rows_per_second(out, args.rows)
        print(
            f"python {python_rate:,.0f} rows/s | {kind} {rust_rate:,.0f} rows/s | "
            f"{rust_rate / python_rate:.1f}x"
        )

    shutdown()
    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
