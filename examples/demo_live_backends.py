# SPDX-License-Identifier: Apache-2.0
"""Demo against the real AI CLI backends (coco / cursor / claude).

Each backend gets a fresh SQLite catalog, transpiles two UDFs for real
(cache miss -> CLI -> Hypothesis verify -> cache hit), and we time it.

    export SPARK_HOME=/path/to/spark-master
    export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip"
    export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"
    export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
    python examples/demo_live_backends.py              # all backends found on PATH
    python examples/demo_live_backends.py cursor       # just one
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402

BACKEND_BINARIES = {"coco": "cortex", "cursor": "agent", "claude": "claude"}


def plus_one(x: int) -> int:
    return x + 1


def greet(name: str) -> str:
    return "hi " + name


def wait_for_transpile(func, rtype, timeout=900):
    """Re-create the UDF until the catalog hit flips transpiled=True."""
    deadline = time.time() + timeout
    udf = UserDefinedFunction(func, rtype)
    while time.time() < deadline:
        udf = UserDefinedFunction(func, rtype)
        if udf.transpiled:
            return udf, True
        time.sleep(2)
    return udf, False


def run_backend(spark, backend: str) -> dict:
    sqlite = os.path.join(tempfile.mkdtemp(prefix=f"ai-udf-live-{backend}-"), "cache.sqlite")
    enable(spark, sqlite_path=sqlite, backend=backend, inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    catalog = get_catalog()
    out = {"backend": backend}

    df = spark.createDataFrame([(41,), (None,)], ["x"])
    names = spark.createDataFrame([("bo",), (None,)], ["name"])

    start = time.time()
    first = UserDefinedFunction(plus_one, LongType())
    assert not first.transpiled
    udf, ok = wait_for_transpile(plus_one, LongType())
    out["plus_one_seconds"] = round(time.time() - start, 1)
    out["plus_one"] = ok
    if ok:
        rows = [r[0] for r in df.select(udf("x")).collect()]
        out["plus_one_results"] = rows

    start = time.time()
    udf, ok = wait_for_transpile(greet, StringType())
    out["greet_seconds"] = round(time.time() - start, 1)
    out["greet"] = ok
    if ok:
        out["greet_results"] = [r[0] for r in names.select(udf("name")).collect()]

    rows = catalog._conn.execute(
        "SELECT target_kind, status, origin, hypothesis_passed, error FROM cache"
    ).fetchall()
    out["catalog"] = rows
    shutdown()
    return out


def main() -> int:
    wanted = sys.argv[1:] or [name for name, binary in BACKEND_BINARIES.items() if shutil.which(binary)]
    if not wanted:
        print("no backends found on PATH (looked for cortex, agent, claude)")
        return 1
    conf.set_value(conf.CLI_TIMEOUT, "600")  # coco can take minutes

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ai-udf-transpile-live-demo")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )

    summaries = []
    for backend in wanted:
        print(f"=== {backend} ({BACKEND_BINARIES.get(backend, '?')}) ===", flush=True)
        try:
            summaries.append(run_backend(spark, backend))
        except Exception as exc:
            summaries.append({"backend": backend, "error": f"{type(exc).__name__}: {exc}"})
        last = summaries[-1]
        if "error" in last:
            print(f"  ERROR: {last['error']}")
        else:
            print(
                f"  plus_one: transpiled={last['plus_one']} in {last['plus_one_seconds']}s"
                f" results={last.get('plus_one_results')}"
            )
            print(
                f"  greet:    transpiled={last['greet']} in {last['greet_seconds']}s"
                f" results={last.get('greet_results')}"
            )
            for kind, status, origin, hyp, error in last["catalog"]:
                print(
                    f"  catalog: kind={kind} status={status} origin={origin} hypothesis={hyp} {error or ''}"
                )
        print(flush=True)

    spark.stop()
    failed = [s["backend"] for s in summaries if "error" in s or not s.get("plus_one")]
    print("=== SUMMARY ===")
    for s in summaries:
        if "error" in s:
            print(f"{s['backend']}: ERROR {s['error']}")
        else:
            print(
                f"{s['backend']}: plus_one={s['plus_one']} ({s['plus_one_seconds']}s)"
                f" greet={s['greet']} ({s['greet_seconds']}s)"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
