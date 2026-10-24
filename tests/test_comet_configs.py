# SPDX-License-Identifier: Apache-2.0
"""The comet test hook: env-gated, fail-safe to plain Spark."""

from __future__ import annotations

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
