# SPDX-License-Identifier: Apache-2.0
"""The 60-second demo: one Catalyst rewrite and one Java UDF, fake backend.

export SPARK_HOME=/path/to/spark-master
export PYTHONPATH="$SPARK_HOME/python:$(ls $SPARK_HOME/python/lib/py4j-*-src.zip | head -n1)"
export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"   # workers must match the driver
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
python examples/demo_quickstart.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import enable, shutdown  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402


def plus_one(x: int) -> int:
    return x + 1


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def wait_for_hit(func, rtype, timeout=120):
    """Re-create the UDF until the catalog hit flips transpiled=True."""
    deadline = time.time() + timeout
    udf = UserDefinedFunction(func, rtype)
    while time.time() < deadline and not udf.transpiled:
        time.sleep(1.0)
        udf = UserDefinedFunction(func, rtype)
    return udf


def main() -> int:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ai-udf-transpile-quickstart")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-quickstart-"), "cache.sqlite")
    enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    df = spark.createDataFrame([(1,), (41,), (None,)], ["x"])
    names = spark.createDataFrame([("holden",), (None,)], ["name"])

    # First definition is a cache miss: runs as a plain Python UDF while the
    # inline worker asks the backend, verifies the rewrite, and fills the cache.
    ok = True
    first = UserDefinedFunction(plus_one, LongType())
    print(f"plus_one: first call transpiled={bool(first.transpiled)} (miss -> Python path)")
    hit = wait_for_hit(plus_one, LongType())
    ok = ok and hit.transpiled
    rows = [r[0] for r in df.select(hit("x")).collect()]
    label = " (Catalyst SQL)" if hit.transpiled else " (TIMED OUT, still Python)"
    print(f"plus_one: second call transpiled={bool(hit.transpiled)} results={rows}{label}")

    first = UserDefinedFunction(backwards, StringType())
    print(f"backwards: first call transpiled={bool(first.transpiled)} (miss -> Python path)")
    hit = wait_for_hit(backwards, StringType())
    ok = ok and hit.transpiled
    rows = [r[0] for r in names.select(hit("name")).collect()]
    label = " (Java UDF)" if hit.transpiled else " (TIMED OUT, still Python)"
    print(f"backwards: second call transpiled={bool(hit.transpiled)} results={rows}{label}")

    try:  # _conn is sqlite-catalog-only; skip the dump on wrapped catalogs
        catalog_rows = get_catalog()._conn.execute(
            "SELECT target_kind, status, origin, hypothesis_passed FROM cache ORDER BY udf_key"
        )
        for kind, status, origin, hyp in catalog_rows:
            print(f"catalog: kind={kind} status={status} origin={origin} hypothesis_passed={hyp}")
    except AttributeError:
        pass

    shutdown()
    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
