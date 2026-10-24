# SPDX-License-Identifier: Apache-2.0
"""dump_results.py writes the what-transpiles-to-what markdown from a live sqlite catalog."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.keys import canonical_source_text
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileResult

pytestmark = pytest.mark.spark

DUMP = Path(__file__).resolve().parents[1] / "examples" / "dump_results.py"

PLUS_ONE_SRC = canonical_source_text("def plus_one(x: int) -> int:\n    return x + 1\n")
BACKWARDS_SRC = canonical_source_text(
    "def backwards(name: str) -> str:\n    if name is None:\n        return None\n    return name[::-1]\n"
)
HAS_ITEM_SRC = canonical_source_text("def has_item(x: int, xs: list) -> bool:\n    return x in xs\n")
ALWAYS_DECLINE_SRC = canonical_source_text(
    "def always_decline(x: int) -> int:\n    import os\n    return len(os.getcwd())\n"
)

BACKWARDS_JAVA = "package ai_udf;\n\npublic class Backwards {}\n"


def _insert(cat, key, source_text, param_names, input_types, input_categories, return_type) -> None:
    cat.insert_pending(
        udf_key=key,
        source_text=source_text,
        param_names=param_names,
        input_types=input_types,
        input_categories=input_categories,
        return_type=return_type,
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )


def _fill_catalog(path: Path) -> None:
    cat = SqliteCatalog(path)

    _insert(cat, "k_plus_one", PLUS_ONE_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_plus_one", "coco")
    cat.mark_success(
        "k_plus_one",
        TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 1", model="coco-model"),
        origin="coco",
    )

    _insert(cat, "k_backwards", BACKWARDS_SRC, ["name"], ["string"], ["string"], "string")
    cat.claim("k_backwards", "coco")
    cat.mark_success(
        "k_backwards",
        TranspileResult(
            kind=KIND_JAVA_UDF, java_source=BACKWARDS_JAVA, class_name="ai_udf.Backwards", model="coco-model"
        ),
        origin="coco",
    )

    # Bytecode declines on CONTAINS_OP (the `in` operator); the AI row still succeeds.
    _insert(
        cat,
        "k_has_item",
        HAS_ITEM_SRC,
        ["x", "xs"],
        ["bigint", "array<string>"],
        ["numeric", "other"],
        "boolean",
    )
    cat.claim("k_has_item", "coco")
    cat.mark_success(
        "k_has_item",
        TranspileResult(kind=KIND_CATALYST, sql="array_contains(_udf_param_1, _udf_param_0)"),
        origin="coco",
    )

    _insert(cat, "k_always_decline", ALWAYS_DECLINE_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_always_decline", "coco")
    cat.mark_failed("k_always_decline", "model declined: no faithful rewrite", origin="coco")
    cat.close()


def _run_dump(sqlite_path: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(DUMP),
            "--sqlite",
            str(sqlite_path),
            "--out",
            str(out),
            "--backend",
            "coco",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_dump_writes_all_three_paths(spark, tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    out = tmp_path / "dump.md"
    proc = _run_dump(sqlite_path, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = out.read_text(encoding="utf-8")
    assert "## plus_one" in text
    assert "CAST(+(_udf_param_0, 1) AS BIGINT)" in text  # built-in static
    assert "(_udf_param_0 + 1)" in text  # bytecode -> SQL
    assert "declined: unsupported bytecode: CONTAINS_OP" in text  # has_item
    assert "java UDF" in text
    assert "public class Backwards" in text
    assert "failed: model declined: no faithful rewrite" in text
    assert "AI (coco / coco-model)" in text
    assert "AI (coco)" in text  # k_has_item has no model


def test_dump_rejects_rows_from_another_backend(tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    cat = SqliteCatalog(sqlite_path)
    _insert(cat, "k_fake", PLUS_ONE_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_fake", "fake")
    cat.mark_success("k_fake", TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 1"), origin="fake")
    cat.close()
    out = tmp_path / "dump.md"
    proc = _run_dump(sqlite_path, out)
    assert proc.returncode != 0
    assert "fake" in proc.stderr
    assert not out.exists()


def test_dump_rejects_backend_fake(tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    out = tmp_path / "dump.md"
    proc = _run_dump(sqlite_path, out, "--backend", "fake")
    assert proc.returncode != 0
    assert not out.exists()


def test_dump_rejects_missing_sqlite(tmp_path):
    out = tmp_path / "dump.md"
    proc = _run_dump(tmp_path / "nope.sqlite", out)
    assert proc.returncode != 0
    assert not out.exists()
