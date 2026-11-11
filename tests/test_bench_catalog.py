# SPDX-License-Identifier: Apache-2.0
"""bench_catalog.py: coverage bench over every cached function."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import enable, shutdown  # noqa: E402
from ai_udf_transpile.backends.fake import PLUS_ONE_RS, always_decline, plus_one  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402

pytestmark = pytest.mark.spark

BENCH = Path(__file__).resolve().parents[1] / "examples" / "bench_catalog.py"


def _settled(catalog, statuses=("success", "failed"), timeout=60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = catalog._conn.execute("SELECT status FROM cache").fetchall()
        if len(rows) == 2 and all(r[0] in statuses for r in rows):
            return
        time.sleep(0.5)
    pytest.fail(f"catalog did not settle: {rows!r}")


@pytest.fixture
def catalog(spark, sqlite_path) -> str:
    """plus_one filled by the fake backend; always_decline declined by it."""
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(plus_one, LongType())
    UserDefinedFunction(always_decline, LongType())
    _settled(get_catalog())
    shutdown()
    return sqlite_path


def test_bench_catalog_writes_rates_and_speedups(catalog, tmp_path):
    out = tmp_path / "bench_catalog.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH),
            "--sqlite",
            catalog,
            "--rows",
            "1000",
            "--rounds",
            "1",
            "--csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={k: v for k, v in os.environ.items() if k not in ("AI_UDF_TRANSPILE_COMET_JAR", "COMET_JAR")},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = {r["function"]: r for r in csv.DictReader(out.open())}
    assert rows["plus_one"]["python_rows_per_second"]
    assert rows["plus_one"]["transpiled_rows_per_second"]
    assert float(rows["plus_one"]["transpiled_speedup_vs_python"]) >= 1.0
    assert rows["plus_one"]["status"] == "success"
    # failed row: python arm only, nothing pretending to be transpiled
    assert rows["always_decline"]["python_rows_per_second"]
    assert rows["always_decline"]["transpiled_rows_per_second"] == ""
    assert rows["always_decline"]["status"] == "failed"


def test_bench_catalog_missing_sqlite(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--sqlite", str(tmp_path / "nope.sqlite")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0


def test_bench_catalog_comet_arm(catalog, tmp_path):
    """CI's comet job (AI_UDF_TRANSPILE_TEST_COMET_JAR set) gets a native arm."""
    jar = os.environ.get("AI_UDF_TRANSPILE_TEST_COMET_JAR", "")
    if not jar or not Path(jar).exists():
        pytest.skip("set AI_UDF_TRANSPILE_TEST_COMET_JAR to a Comet jar to run the comet arm")
    out = tmp_path / "bench_catalog_comet.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH),
            "--sqlite",
            catalog,
            "--rows",
            "1000",
            "--rounds",
            "1",
            "--csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "AI_UDF_TRANSPILE_COMET_JAR": jar},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = {r["function"]: r for r in csv.DictReader(out.open())}
    assert rows["plus_one"]["comet_rows_per_second"]
    assert float(rows["plus_one"]["comet_speedup_vs_python"]) > 0
    # always_decline has no served rewrite: no comet number either
    assert rows["always_decline"]["comet_rows_per_second"] == ""


def test_bench_catalog_benches_a_rust_row_under_comet_only(catalog, tmp_path):
    """A rust rewrite is never served by the plain session; the comet one times it."""
    import shutil
    import sqlite3

    jar = os.environ.get("AI_UDF_TRANSPILE_TEST_COMET_JAR", "")
    if not jar or not Path(jar).exists():
        pytest.skip("set AI_UDF_TRANSPILE_TEST_COMET_JAR to a Comet jar to run the comet arm")
    if not shutil.which("cargo") or not os.environ.get("COMET_HOME"):
        pytest.skip("needs cargo and $COMET_HOME (comet-udf-sdk) to build the rust row")
    # Turn plus_one's verified SQL row into a rust row: the bench serves what
    # the catalog says, and this row function computes the same thing.
    with sqlite3.connect(catalog) as conn:
        conn.execute(
            "UPDATE cache SET target_kind = 'rust', catalyst_sql = NULL, impl_source = ? "
            "WHERE status = 'success'",
            (PLUS_ONE_RS,),
        )
    out = tmp_path / "bench_catalog_rust.csv"
    env = {**os.environ, "AI_UDF_TRANSPILE_COMET_JAR": jar}
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH),
            "--sqlite",
            catalog,
            "--rows",
            "1000",
            "--rounds",
            "1",
            "--csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=1800,  # a cold build compiles arrow
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "rust rewrite -- runs only under Comet" in proc.stdout
    row = {r["function"]: r for r in csv.DictReader(out.open())}["plus_one"]
    assert row["target_kind"] == "rust"
    assert row["transpiled_rows_per_second"] and float(row["transpiled_speedup_vs_python"]) > 0
    assert row["comet_rows_per_second"] == ""  # its one number is the transpiled one
