# SPDX-License-Identifier: Apache-2.0
"""dump_results.py writes the what-transpiles-to-what markdown from a live sqlite catalog."""

from __future__ import annotations

import importlib.util
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


MICRO_CSV_HEADER = (
    "function,kinds,regular_rows_per_second,transpiled_static_rows_per_second,"
    "transpiled_ai_catalyst_rows_per_second,transpiled_ai_java_rows_per_second,"
    "static_speedup_vs_regular,ai_catalyst_speedup_vs_regular,ai_java_speedup_vs_regular,"
    "transpile_succeeded_static,transpile_succeeded_ai_catalyst,transpile_succeeded_ai_java,"
    "ai_backend,ai_model"
)
MICRO_CSV_ROWS = (
    "plus_one,catalyst+java,2000000,3000000,8000000,4000000,1.5,4.0,2.0,True,True,True,coco,auto\n"
    "always_decline,declined,1000000,,1000000,,,1.0,,False,False,False,coco,auto\n"
)
TPCDS_CSV = (
    "function,declined,rows,python_rows_per_second,vanilla_rows_per_second,"
    "transpiled_rows_per_second,transpiled_speedup_vs_python,vanilla_speedup_vs_python,"
    "ai_backend,ai_model\n"
    "common_c2f,False,1000,1000000,5000000,4000000,4.0,5.0,coco,auto\n"
    "tpcds_safe_ratio,True,1000,900000,4500000,900000,1.0,5.0,coco,auto\n"
)


def _load_dump_module():
    spec = importlib.util.spec_from_file_location("dump_results", DUMP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perf_lines_unit(tmp_path):
    micro = tmp_path / "micro.csv"
    micro.write_text(MICRO_CSV_HEADER + "\n" + MICRO_CSV_ROWS, encoding="utf-8")
    tpcds = tmp_path / "tpcds.csv"
    tpcds.write_text(TPCDS_CSV, encoding="utf-8")
    perf = _load_dump_module()._perf_lines([str(micro), str(tpcds)])
    assert perf["plus_one"] == (
        "- **perf (micro bench)**: python 2.0M rows/s · static 3.0M (1.5x) · ai catalyst 8.0M (4.0x)"
    )
    assert perf["plus_one_java"] == "- **perf (micro bench)**: python 2.0M rows/s · ai java 4.0M (2.0x)"
    # fallback shows no speedup -- there is no rewrite to speed up
    assert perf["always_decline"] == "- **perf (micro bench)**: python 1.0M rows/s · python fallback 1.0M"
    assert perf["common_c2f"] == (
        "- **perf (tpcds bench)**: python 1.0M rows/s · vanilla 5.0M (5.0x) · transpiled 4.0M (4.0x)"
    )
    assert "python fallback (900k rows/s; declined)" in perf["tpcds_safe_ratio"]


def test_perf_lines_missing_csv_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="cannot read perf csv"):
        _load_dump_module()._perf_lines([str(tmp_path / "nope.csv")])


def test_dump_writes_perf_bullets(spark, tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    micro = tmp_path / "micro.csv"
    micro.write_text(MICRO_CSV_HEADER + "\n" + MICRO_CSV_ROWS, encoding="utf-8")
    out = tmp_path / "dump.md"
    proc = _run_dump(sqlite_path, out, "--perf-csv", str(micro))
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = out.read_text(encoding="utf-8")
    plus_one = text.split("## plus_one")[1].split("## ")[0]
    assert "- **perf (micro bench)**: python 2.0M rows/s" in plus_one
    assert "ai catalyst 8.0M (4.0x)" in plus_one
    always = text.split("## always_decline")[1].split("## ")[0]
    assert "python fallback 1.0M" in always
    # not benched -> no perf bullet
    backwards = text.split("## backwards")[1].split("## ")[0]
    assert "perf" not in backwards
