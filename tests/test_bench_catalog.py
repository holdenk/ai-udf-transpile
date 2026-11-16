# SPDX-License-Identifier: Apache-2.0
"""bench_catalog.py: coverage bench over every cached function."""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


def test_strict_comet_requires_a_configured_jar(tmp_path):
    source = tmp_path / "source.sqlite"
    source.touch()
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--sqlite", str(source), "--strict-comet"],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k not in ("AI_UDF_TRANSPILE_COMET_JAR", "COMET_JAR")},
    )
    assert proc.returncode != 0
    assert "strict Comet benchmark" in proc.stderr


def test_strict_comet_propagates_rust_compile_failures(tmp_path, monkeypatch):
    from pyspark import SparkContext

    spec = importlib.util.spec_from_file_location("bench_catalog", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spark = SimpleNamespace(conf=SimpleNamespace(set=lambda *a: None), stop=lambda: None)
    monkeypatch.setattr(SparkContext, "_gateway", None)
    monkeypatch.setattr(SparkContext, "_jvm", None)
    monkeypatch.setattr(module, "_session", lambda *a: spark)
    monkeypatch.setattr(module, "enable", lambda *a, **k: None)

    def failed_compile(*args):
        raise RuntimeError("cargo unavailable")

    monkeypatch.setattr(module.rustc, "compile_rust", failed_compile)
    row = SimpleNamespace(
        udf_key="k",
        target_kind="rust",
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        impl_source="source",
    )
    out = {"k": {"function": "plus_one", "status": "success", "transpiled_rows_per_second": ""}}
    args = SimpleNamespace(sqlite=str(tmp_path / "cache.sqlite"), strict_comet=True)
    with pytest.raises(RuntimeError, match="Rust benchmarks failed for: plus_one"):
        module._comet_phase(args, [row], out, {}, "comet.jar")


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

    from ai_udf_transpile.catalog import loads_list
    from ai_udf_transpile.keys import udf_key
    from ai_udf_transpile.targets import NO_RUST_SUFFIX

    jar = os.environ.get("AI_UDF_TRANSPILE_TEST_COMET_JAR", "")
    if not jar or not Path(jar).exists():
        pytest.skip("set AI_UDF_TRANSPILE_TEST_COMET_JAR to a Comet jar to run the comet arm")
    if not shutil.which("cargo") or not os.environ.get("COMET_HOME"):
        pytest.skip("needs cargo and $COMET_HOME (comet-udf-sdk) to build the rust row")
    # Turn plus_one's verified SQL row into a rust row: the bench serves what
    # the catalog says, and this row function computes the same thing.
    with sqlite3.connect(catalog) as conn:
        # Its no-Rust twin keeps the SQL rewrite. The Comet session can run
        # Rust, so the function resolves to the Rust row there: the twin must
        # not be credited with the Rust number.
        conn.row_factory = sqlite3.Row
        base = dict(conn.execute("SELECT * FROM cache WHERE status = 'success'").fetchone())
        twin_version = base["spark_version"] + NO_RUST_SUFFIX
        twin_key = udf_key(
            base["source_text"],
            loads_list(base["param_names"]),
            loads_list(base["input_types"]),
            base["return_type"],
            twin_version,
            base["closure_fingerprint"] or "",
        )
        twin = {**base, "udf_key": twin_key, "spark_version": twin_version}
        conn.execute(
            f"INSERT INTO cache ({', '.join(twin)}) VALUES ({', '.join('?' * len(twin))})",
            tuple(twin.values()),
        )
        conn.execute(
            "UPDATE cache SET target_kind = 'rust', catalyst_sql = NULL, impl_source = ? "
            "WHERE status = 'success' AND udf_key = ?",
            (PLUS_ONE_RS, base["udf_key"]),
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
    assert "plus_one: no-Rust twin -- this session serves the Rust row" in proc.stdout
    by_kind = {r["target_kind"]: r for r in csv.DictReader(out.open()) if r["function"] == "plus_one"}
    assert by_kind["catalyst"]["comet_rows_per_second"] == ""
    row = by_kind["rust"]
    assert row["transpiled_rows_per_second"] and float(row["transpiled_speedup_vs_python"]) > 0
    assert row["comet_rows_per_second"] == ""  # its one number is the transpiled one
