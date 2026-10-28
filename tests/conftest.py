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

# Ambient AWS variables break the test env two ways: a classic session
# cannot start on hosts where they force a failing InetAddress.getLocalHost()
# (the gateway dies before its port), and botocore would stall on IMDS
# lookups. Scrub them before anything imports Spark or boto3.
for _key in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_CA_BUNDLE",
    "AWS_PROFILE",
):
    os.environ.pop(_key, None)
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
# Env scrubbing is not enough: botocore also resolves credentials from
# ~/.aws/credentials, ~/.aws/config, and the SSO cli cache -- a developer
# box with any of those would make the kms fixture call the real service
# mid-verification. Point every file source at nothing.
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/dev/null"
os.environ["AWS_CONFIG_FILE"] = "/dev/null"


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
    jars = []
    for z in glob.glob(str(lib / "py4j-*.zip")):
        if z not in sys.path:
            sys.path.insert(0, z)
        jars.append(z)
    if jars:
        # Subprocess tests (dump_results, bench_catalog) re-exec python and
        # need pyspark importable there too -- conftest's sys.path surgery
        # does not survive exec. Mirror CI's PYTHONPATH so local runs match.
        env_path = py + os.pathsep + os.pathsep.join(jars)
        existing = os.environ.get("PYTHONPATH", "")
        if env_path not in existing:
            os.environ["PYTHONPATH"] = env_path + (os.pathsep + existing if existing else "")


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
    # With SPARK_HOME explicitly provided (CI does), a session that cannot
    # start is a failure, not a silent skip -- otherwise the CI job goes
    # green with zero Spark coverage, which is exactly the breakage it
    # exists to catch. Without it, skip cleanly (no Spark here at all).
    have_home = bool(os.environ.get("SPARK_HOME"))
    outcome = pytest.fail if have_home else pytest.skip
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.transpile import AbstractTranspiler  # noqa: F401
    except Exception as exc:
        outcome(f"pyspark transpile surface missing (SPARK_HOME={os.environ.get('SPARK_HOME')}): {exc}")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    # Workers must run the same interpreter as the driver (cloudpickle shares
    # bytecode); the system default python3 may be too old for Spark master.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    try:
        builder = (
            SparkSession.builder.master("local[1]")
            .appName("ai-udf-transpile-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.sql.ansi.enabled", "true")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
        )
        for key, value in _comet_configs():
            builder = builder.config(key, value)
        session = builder.getOrCreate()
        session.range(1).count()
    except Exception as exc:
        outcome(f"Spark session unavailable: {exc}")
    yield session
    session.stop()


def _comet_configs() -> list[tuple[str, str]]:
    """CI's comet child points the test session at a Comet jar via env.

    Unset (the default everywhere) or dangling jar: no comet, plain Spark.
    """
    jar = os.environ.get("AI_UDF_TRANSPILE_TEST_COMET_JAR", "")
    if not jar or not Path(jar).exists():
        return []
    return [
        ("spark.jars", jar),
        ("spark.plugins", "org.apache.spark.CometPlugin"),
        ("spark.comet.enabled", "true"),
        ("spark.memory.offHeap.enabled", "true"),
        ("spark.memory.offHeap.size", "1g"),
    ]
