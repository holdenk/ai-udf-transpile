# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: working rewrites and deliberate declines.

Run with a packaged Spark master:

    export SPARK_HOME=/path/to/spark-master
    export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip"
    export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"   # workers must match the driver
    export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
    python examples/demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import BooleanType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import enable, register_impl, shutdown  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402


def plus_one(x: int) -> int:
    return x + 1


def greet(name: str) -> str:
    return "hi " + name


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def untyped(x):
    return x + 1


def uses_helper(x: int) -> int:
    return plus_one(x)


def always_decline(x: int) -> int:
    import os

    return len(os.getcwd())


def wait_for_status(catalog, key, want, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = catalog.get(key)
        if row is not None and row.status == want:
            return row
        time.sleep(0.5)
    return catalog.get(key)


def main() -> int:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ai-udf-transpile-demo")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-demo-"), "cache.sqlite")
    enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    catalog = get_catalog()
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    print("=== WORKING CASES (fake backend fixtures, Hypothesis-verified) ===")
    df = spark.createDataFrame([(1,), (41,), (None,)], ["x"])

    for func, rtype in [
        (plus_one, LongType()),
        (is_none_branch, LongType()),
    ]:
        first = UserDefinedFunction(func, rtype)
        print(f"{func.__name__}: first call transpiled={bool(first.transpiled)} (cache miss -> Python)")
        # wait for the inline worker
        deadline = time.time() + 120
        while time.time() < deadline:
            second = UserDefinedFunction(func, rtype)
            if second.transpiled:
                break
            time.sleep(1.0)
        print(f"{func.__name__}: second call transpiled={bool(second.transpiled)} (cache hit)")
        rows = df.filter("x is not null").select(second("x").alias("r")).collect()
        print(f"{func.__name__}: results={[r[0] for r in rows]}")

    g = UserDefinedFunction(greet, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        g = UserDefinedFunction(greet, StringType())
        if g.transpiled:
            break
        time.sleep(1.0)
    names = spark.createDataFrame([("bo",), ("holden",)], ["name"])
    greet_rows = [r[0] for r in names.select(g("name")).collect()]
    print(f"greet: transpiled={bool(g.transpiled)} results={greet_rows}")

    print()
    print("=== JAVA UDF TARGET (compiled on the driver, Hypothesis-verified) ===")
    bw = UserDefinedFunction(backwards, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        bw = UserDefinedFunction(backwards, StringType())
        if bw.transpiled:
            break
        time.sleep(1.0)
    bw_rows = [r[0] for r in names.select(bw("name")).collect()]
    print(f"backwards: transpiled={bool(bw.transpiled)} results={bw_rows} (Java UDF from cache)")

    b = UserDefinedFunction(both_positive, BooleanType())
    deadline = time.time() + 120
    while time.time() < deadline:
        b = UserDefinedFunction(both_positive, BooleanType())
        if b.transpiled:
            break
        time.sleep(1.0)
    pairs = spark.createDataFrame([(1, 2), (-1, 2)], ["a", "b"])
    print(
        f"both_positive: transpiled={bool(b.transpiled)} "
        f"results={[r[0] for r in pairs.select(b('a', 'b')).collect()]}"
    )

    print()
    print("=== HUMAN-PROVIDED IMPL (register_impl) ===")

    def times_ten(x: int) -> int:
        return x * 10

    register_impl(
        spark,
        times_ten,
        kind="catalyst",
        catalyst_sql="_udf_param_0 * 10",
        return_type=LongType(),
    )
    t = UserDefinedFunction(times_ten, LongType())
    print(f"times_ten: transpiled={bool(t.transpiled)} (human impl, Hypothesis-verified)")
    print(f"times_ten: results={[r[0] for r in df.filter('x is not null').select(t('x')).collect()]}")

    print()
    print("=== NOT WORKING / BY-DESIGN DECLINES ===")

    u = UserDefinedFunction(untyped, LongType())
    print(
        f"untyped UDF: transpiled={bool(u.transpiled)}, catalog rows={catalog.count()} "
        "(no type annotations -> never queued)"
    )

    h = UserDefinedFunction(uses_helper, LongType())
    print(
        f"closure over a Python function: transpiled={bool(h.transpiled)} (cannot key captures -> declined)"
    )

    UserDefinedFunction(always_decline, LongType())
    cur = catalog._conn.execute("SELECT udf_key FROM cache WHERE source_text LIKE '%always_decline%'")
    found = cur.fetchone()
    if found:
        row = wait_for_status(catalog, found[0], "failed", timeout=60)
        print(
            f"always_decline: status={row.status} error={row.error!r} "
            "(no fixture -> failed, cooldown applies)"
        )
    before = catalog.count()
    UserDefinedFunction(always_decline, LongType())
    print(f"always_decline second call: catalog rows {before} -> {catalog.count()} (cooldown, no re-queue)")

    print()
    print("=== WRONG REWRITE REJECTED BY HYPOTHESIS ===")

    def plus_two(x: int) -> int:
        return x + 2

    try:
        register_impl(
            spark,
            plus_two,
            kind="catalyst",
            catalyst_sql="_udf_param_0 + 99",  # wrong on purpose
            return_type=LongType(),
        )
        print("BUG: wrong impl was accepted")
    except ValueError as exc:
        print(f"register_impl rejected wrong SQL: {exc}")

    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
