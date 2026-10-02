# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures. Spark tests skip cleanly when no classic session can start."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

from ai_udf_transpile import shutdown

_REPO = Path(__file__).resolve().parents[1]


def _bootstrap_spark_home() -> None:
    home = os.environ.get("SPARK_HOME")
    if not home:
        sibling = Path.home() / "spark"
        if (sibling / "python" / "pyspark" / "sql" / "transpile.py").exists():
            home = str(sibling)
            os.environ["SPARK_HOME"] = home
    if not home:
        return
    py = str(Path(home) / "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    lib = Path(home) / "python" / "lib"
    for z in glob.glob(str(lib / "py4j-*.zip")):
        if z not in sys.path:
            sys.path.insert(0, z)


_bootstrap_spark_home()


def _clear_session_writeback_conf() -> None:
    """The SparkSession is process-wide. A write-back conf set by one test
    would otherwise make the next test's enable() open a WritebackCatalog.
    """
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return
    spark = SparkSession.getActiveSession()
    if spark is None:
        return
    from ai_udf_transpile import conf

    for key in (conf.WRITEBACK_TABLE, conf.WRITEBACK_FORMAT, conf.WRITEBACK_THRESHOLD):
        try:
            spark.conf.unset(key)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _reset_transpile_state():
    shutdown()
    _clear_session_writeback_conf()
    yield
    shutdown()
    _clear_session_writeback_conf()


@pytest.fixture
def sqlite_path(tmp_path):
    return str(tmp_path / "ai_udf_transpile.sqlite")


@pytest.fixture(scope="session")
def spark():
    _bootstrap_spark_home()
    pytest.importorskip("pyspark")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    # Workers must run the same interpreter as the driver (cloudpickle shares
    # bytecode); the system default python3 may be too old for Spark master.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.transpile import AbstractTranspiler  # noqa: F401
    except Exception as exc:
        pytest.skip(f"pyspark transpile surface missing: {exc}")
    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("ai-udf-transpile-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.sql.ansi.enabled", "true")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .getOrCreate()
        )
        session.range(1).count()
    except Exception as exc:
        pytest.skip(f"Spark session unavailable: {exc}")
    yield session
    session.stop()
