# SPDX-License-Identifier: Apache-2.0
"""The comet test hook: env-gated, fail-safe to plain Spark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import _comet_configs


def test_unset_env_means_no_comet(monkeypatch):
    monkeypatch.delenv("AI_UDF_TRANSPILE_TEST_COMET_JAR", raising=False)
    assert _comet_configs() == []


def test_dangling_jar_means_no_comet(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_UDF_TRANSPILE_TEST_COMET_JAR", str(tmp_path / "nope.jar"))
    assert _comet_configs() == []


def test_real_jar_enables_comet(monkeypatch, tmp_path):
    jar = tmp_path / "comet.jar"
    jar.write_bytes(b"PK")
    monkeypatch.setenv("AI_UDF_TRANSPILE_TEST_COMET_JAR", str(jar))
    configs = dict(_comet_configs())
    assert configs["spark.jars"] == str(jar)
    assert configs["spark.comet.enabled"] == "true"
    assert configs["spark.plugins"] == "org.apache.spark.CometPlugin"


@pytest.mark.spark
def test_micro_comet_requires_a_real_jar(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "bench_micro.py"
    for jar in ("", str(tmp_path / "missing.jar")):
        env = {k: v for k, v in os.environ.items() if k not in ("AI_UDF_TRANSPILE_COMET_JAR", "COMET_JAR")}
        if jar:
            env["COMET_JAR"] = jar
        proc = subprocess.run(
            [sys.executable, str(script), "1", "--rounds", "1", "--rust-accel"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert proc.returncode == 2, proc.stderr[-2000:]
        assert "jar" in proc.stderr
        assert "usage:" in proc.stderr
