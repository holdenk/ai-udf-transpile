# SPDX-License-Identifier: Apache-2.0
"""Per-target regeneration must distinguish rejected rewrites from failed runs."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_udf_transpile.catalog.sqlite import SqliteCatalog

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [("declined: no faithful rewrite", 0), ("BackendError: timed out", 1)],
)
def test_fill_reports_infrastructure_errors(tmp_path, monkeypatch, error, exit_code):
    pytest.importorskip("pyspark")
    spec = importlib.util.spec_from_file_location("fill_targets", ROOT / "examples/fill_targets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.sqlite"
    cat = SqliteCatalog(source)
    cat.insert_pending(
        udf_key="k",
        source_text="def f(x: str) -> int:\n    return len(x)\n",
        param_names=["x"],
        input_types=["string"],
        input_categories=["string"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    cat.record_samples("k", [["example"]])
    cat.close()
    output = SqliteCatalog(tmp_path / "out.sqlite")
    monkeypatch.setattr(module, "_session", lambda *a: SimpleNamespace(stop=lambda: None))
    monkeypatch.setattr(module, "enable", lambda *a, **k: output)
    monkeypatch.setattr(module, "get_backend", lambda **k: SimpleNamespace(name="coco"))
    monkeypatch.setattr(module.conf, "set_value", lambda *a: None)
    monkeypatch.setattr(module, "shutdown", lambda: None)
    monkeypatch.setattr(
        module, "process_row", lambda catalog, row, *a: catalog.mark_failed(row.udf_key, error)
    )
    monkeypatch.setattr(sys, "argv", ["fill_targets.py", "--target", "catalyst", "--source", str(source)])
    assert module.main() == exit_code
    assert output.samples_for("k") == [["example"]]
    output.close()


@pytest.mark.spark
def test_fill_rejects_a_rewrite_that_only_matches_null_strings(spark, tmp_path, monkeypatch):
    from ai_udf_transpile.targets import TranspileResult

    spec = importlib.util.spec_from_file_location("fill_targets", ROOT / "examples/fill_targets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.sqlite"
    out = tmp_path / "out.sqlite"
    cat = SqliteCatalog(source)
    cat.insert_pending(
        udf_key="k",
        source_text="def length(s: str) -> int:\n    return 0 if s is None else len(s)\n",
        param_names=["s"],
        input_types=["string"],
        input_categories=["string"],
        return_type="bigint",
        spark_version=spark.version,
        closure_fingerprint="",
        captures={},
    )
    cat.close()
    backend = SimpleNamespace(
        name="coco", run=lambda *a: TranspileResult(kind="catalyst", sql="0", model="stub")
    )
    session = spark.newSession()
    monkeypatch.setattr(module, "_session", lambda *a: session)
    monkeypatch.setattr(session, "stop", lambda: None)
    monkeypatch.setattr(module, "get_backend", lambda **k: backend)
    monkeypatch.setattr(
        sys, "argv", ["fill_targets.py", "--target", "catalyst", "--source", str(source), "--out", str(out)]
    )
    assert module.main() == 0
    cat = SqliteCatalog(out)
    row = cat.get("k")
    assert row.status == "failed" and "mismatch" in row.error
    cat.close()


def test_regeneration_propagates_dump_failure(tmp_path):
    spark = tmp_path / "spark"
    (spark / "python/lib").mkdir(parents=True)
    (spark / "python/lib/py4j-test-src.zip").touch()
    python = tmp_path / "python"
    python.write_text(
        '#!/usr/bin/env bash\ncase "$*" in *dump_results.py*) exit 7;; esac\nexit 0\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    targets = tmp_path / "targets"
    targets.mkdir()
    for kind in ("catalyst", "java_udf", "rust"):
        (targets / f"{kind}_bench.csv").write_text("stale timings", encoding="utf-8")
    env = dict(os.environ, PYTHON=str(python), SPARK_HOME=str(spark), TARGETS_DIR=str(targets))
    proc = subprocess.run(
        ["bash", str(ROOT / "examples/regen_targets_doc.sh"), "--only", "f"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert not list(targets.glob("*_bench.csv"))
