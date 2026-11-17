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
REJECTED_SRC = canonical_source_text("def rejected_cand(x: int) -> int:\n    return x + 1\n")

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

    # a hypothesis failure keeps the rewrite that was rejected
    _insert(cat, "k_rejected", REJECTED_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_rejected", "coco")
    cat.mark_failed(
        "k_rejected",
        "mismatch on (3,): python=4 sql=5",
        origin="coco",
        model="auto",
        result=TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 2"),
    )
    # ... and a rejected java UDF keeps its source, in a fence
    _insert(cat, "k_rejected_java", REJECTED_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_rejected_java", "coco")
    cat.mark_failed(
        "k_rejected_java",
        "JavaCompileError: not a class",
        origin="coco",
        model="auto",
        result=TranspileResult(
            kind=KIND_JAVA_UDF,
            java_source="package ai_udf;\npublic class Nope {}\n",
            class_name="ai_udf.Nope",
        ),
    )
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


def test_dump_writes_every_transpiled_path(spark, tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    out = tmp_path / "dump.md"
    proc = _run_dump(sqlite_path, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = out.read_text(encoding="utf-8")
    assert "## plus_one" in text
    assert "CAST(+(_udf_param_0, 1) AS BIGINT)" in text  # built-in static
    assert "(_udf_param_0 + 1)" in text  # bytecode -> SQL
    assert "bytecode → Java eval" in text
    assert "ai_udf.Bc" in text
    assert "declined: unsupported bytecode: CONTAINS_OP" in text  # has_item
    assert "java UDF" in text
    assert "public class Backwards" in text
    assert "failed: model declined: no faithful rewrite" in text
    assert "AI (coco / coco-model)" in text
    assert "AI (coco)" in text  # k_has_item has no model
    # hypothesis failure: the error AND the rejected rewrite
    assert "failed: mismatch on (3,): python=4 sql=5" in text
    assert "- **rejected rewrite**: `_udf_param_0 + 2`" in text
    # a rejected java UDF keeps its source, in a fence
    assert "failed: JavaCompileError: not a class" in text
    assert "public class Nope {}" in text
    # a decline produced nothing: no rejected-rewrite bullet
    always = text.split("## always_decline")[1].split("## ")[0]
    assert "rejected rewrite" not in always


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
BYTECODE_CSV = (
    "function,static_converted,static_correct,bytecode_static_correct,java_eval_correct,"
    "python_rows_per_second,static_rows_per_second,bytecode_static_rows_per_second,"
    "java_eval_rows_per_second,static_speedup_vs_python,bytecode_static_speedup_vs_python,"
    "java_eval_speedup_vs_python\n"
    "plus_one,True,True,True,True,2000000,3500000,7000000,2500000,1.8,3.5,1.3\n"
)
CATALOG_CSV = (
    "function,rows,python_rows_per_second,static_rows_per_second,static_speedup_vs_python,"
    "bytecode_static_rows_per_second,bytecode_static_speedup_vs_python,"
    "java_eval_rows_per_second,java_eval_speedup_vs_python,transpiled_rows_per_second,"
    "transpiled_speedup_vs_python,comet_rows_per_second,comet_speedup_vs_python,"
    "status,target_kind,backend,model\n"
    "has_item,1000,2000000,3000000,1.5,4000000,2.0,1000000,0.5,6000000,3.0,"
    "9000000,4.5,success,catalyst,ensemble,coco/auto\n"
    "plus_two,1000,1500000,2500000,1.7,,,,,,,,,failed,,human,\n"
    "kms_encrypt,1000,,,,,,,,,,,,failed,,ensemble,coco/auto\n"
)


def _load_dump_module():
    spec = importlib.util.spec_from_file_location("dump_results", DUMP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dump_targets_shows_each_kind_and_marks_backend_errors_unavailable(spark, tmp_path):
    targets = tmp_path / "targets"
    targets.mkdir()
    _fill_catalog(targets / "catalyst.sqlite")
    _fill_catalog(targets / "java_udf.sqlite")
    cat = SqliteCatalog(targets / "rust.sqlite")
    _insert(cat, "k_plus_one", PLUS_ONE_SRC, ["x"], ["bigint"], ["numeric"], "bigint")
    cat.claim("k_plus_one", "coco")
    cat.mark_failed("k_plus_one", "BackendError: timed out", origin="coco")
    cat.close()
    out = tmp_path / "dump.md"
    assert _load_dump_module().dump_targets(spark, str(targets), str(out), "coco") == 0
    text = out.read_text(encoding="utf-8")
    assert "| UDF | python | SQL | Java | Rust |" in text
    assert "'Rust': {'unavailable': 1}" in text
    section = text.split("## plus_one\n", 1)[1].split("## ", 1)[0]
    assert "**SQL (" in section and "**Java (" in section
    assert "**Rust (coco)**: unavailable: BackendError: timed out" in section


def test_empty_target_catalog_preserves_existing_document(spark, tmp_path):
    targets = tmp_path / "targets"
    targets.mkdir()
    SqliteCatalog(targets / "catalyst.sqlite").close()
    out = tmp_path / "dump.md"
    out.write_text("existing document", encoding="utf-8")
    assert _load_dump_module().dump_targets(spark, str(targets), str(out), "coco") == 1
    assert out.read_text(encoding="utf-8") == "existing document"


def test_failed_write_preserves_existing_document(tmp_path, monkeypatch):
    out = tmp_path / "dump.md"
    out.write_text("existing document", encoding="utf-8")

    def failed_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", failed_write)
    with pytest.raises(OSError, match="disk full"):
        _load_dump_module()._write_doc(str(out), ["replacement"])
    assert out.read_text(encoding="utf-8") == "existing document"


def test_perf_lines_unit(tmp_path):
    micro = tmp_path / "micro.csv"
    micro.write_text(MICRO_CSV_HEADER + "\n" + MICRO_CSV_ROWS, encoding="utf-8")
    tpcds = tmp_path / "tpcds.csv"
    tpcds.write_text(TPCDS_CSV, encoding="utf-8")
    perf = _load_dump_module()._perf_lines([str(micro), str(tpcds)])
    assert perf["plus_one"] == {
        "top": "- **perf (micro bench)**: interpreted python 2.0M rows/s",
        "static": "3.0M rows/s (1.5x vs python)",
        "bytecode": "",
        "java_eval": "",
        "ai": "",
        # One function, each kind filled separately: keyed by the kind it was served as.
        "ai:catalyst": "8.0M rows/s (4.0x vs python)",
        "ai:java_udf": "4.0M rows/s (2.0x vs python)",
    }
    assert "plus_one_java" not in perf
    # always_decline: the fallback perf is the top line's story; no rewrite was served
    assert perf["always_decline"]["top"].endswith("interpreted python 1.0M rows/s")
    assert perf["always_decline"]["static"] == ""
    assert perf["always_decline"]["ai"] == ""
    assert perf["common_c2f"] == {
        "top": "- **perf (tpcds bench)**: interpreted python 1.0M rows/s · native vanilla 5.0M (5.0x)",
        "static": "",
        "bytecode": "",
        "java_eval": "",
        "ai": "4.0M rows/s (4.0x vs python)",
    }
    assert perf["tpcds_safe_ratio"]["ai"] == ""  # declined: python fallback serves, no suffix
    # catalog CSV: every transpiled arm survives, including static on a failed AI row
    cat = tmp_path / "catalog.csv"
    cat.write_text(CATALOG_CSV, encoding="utf-8")
    perf = _load_dump_module()._perf_lines([str(cat)])
    assert perf["has_item"] == {
        "top": "- **perf (catalog bench)**: interpreted python 2.0M rows/s",
        "static": "3.0M rows/s (1.5x vs python)",
        "bytecode": "4.0M rows/s (2.0x vs python)",
        "java_eval": "1.0M rows/s (0.5x vs python)",
        "ai": "6.0M rows/s (3.0x vs python) · comet 9.0M rows/s (4.5x vs python)",
    }
    # bench_fhi_udfs.py writes the same schema and names itself in a bench column
    fhi = tmp_path / "fhi.csv"
    fhi.write_text(
        CATALOG_CSV.replace("status,target_kind,backend,model\n", "status,target_kind,backend,model,bench\n")
        .replace("coco/auto\n", "coco/auto,fhi\n")
        .replace("human,\n", "human,,fhi\n"),
        encoding="utf-8",
    )
    assert _load_dump_module()._perf_lines([str(fhi)])["has_item"]["top"] == (
        "- **perf (fhi bench)**: interpreted python 2.0M rows/s"
    )
    # ...and passed after the catalog CSV it takes over the function's line,
    # but the comet number only the catalog bench measured stays, labeled.
    fhi.write_text(fhi.read_text().replace("3.0,9000000,4.5,success", "3.0,,,success"), encoding="utf-8")
    merged = _load_dump_module()._perf_lines([str(cat), str(fhi)])["has_item"]
    assert merged["top"].startswith("- **perf (fhi bench)**")
    assert merged["ai"] == (
        "6.0M rows/s (3.0x vs python) · comet (catalog bench) 9.0M rows/s (4.5x vs python)"
    )
    # bench_fhi_udfs rows for its pinned arms and its non-null column are
    # skipped: the doc's AI line is the live arm on the nullable column.
    arms = tmp_path / "arms.csv"
    arms.write_text(
        "function,arm,column_nullable,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,status,target_kind,backend,model,bench\n"
        "f,naive,True,1000,1000000,300000,0.3,success,catalyst,pinned,,fhi\n"
        "f,ai,False,1000,1000000,9000000,9.0,success,catalyst,claude,m,fhi\n"
        "f,ai,True,1000,1000000,1500000,1.5,success,catalyst,claude,m,fhi\n"
        "f,tuned,True,1000,1000000,1900000,1.9,success,catalyst,pinned,,fhi\n",
        encoding="utf-8",
    )
    assert _load_dump_module()._perf_lines([str(arms)])["f"]["ai"] == "1.5M rows/s (1.5x vs python)"
    # python-only row (no transpiled arm served) keeps the top, no ai suffix
    assert perf["plus_two"]["ai"] == ""
    assert perf["plus_two"]["static"] == "2.5M rows/s (1.7x vs python)"
    assert perf["plus_two"]["top"].endswith("interpreted python 1.5M rows/s")
    # skipped function (no arm produced a rate): no entry at all
    assert "kms_encrypt" not in perf
    # python-skipped but transpiled-measured: the rate survives, labeled
    widg = tmp_path / "widg.csv"
    widg.write_text(
        "function,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,status,target_kind,backend,model\n"
        "widget_name,1000,,884683,,success,catalyst,ensemble,coco/auto\n",
        encoding="utf-8",
    )
    perf = _load_dump_module()._perf_lines([str(widg)])
    assert perf["widget_name"] == {
        "top": "",
        "static": "",
        "bytecode": "",
        "java_eval": "",
        "ai": "885k rows/s (no python baseline)",
    }
    # a rust rewrite's number is Comet's, labeled as such, with no comet suffix
    rust_cat = tmp_path / "rust_catalog.csv"
    rust_cat.write_text(
        "function,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,comet_rows_per_second,comet_speedup_vs_python,"
        "status,target_kind,backend,model\n"
        "collatz_steps,1000,300000,6000000,20.0,,,success,rust,claude,m\n",
        encoding="utf-8",
    )
    perf = _load_dump_module()._perf_lines([str(rust_cat)])
    assert perf["collatz_steps"]["ai"] == "6.0M rows/s (20.0x vs python, Comet native)"
    # A rust row and its no-rust twin share a name: each keeps its own numbers by udf_key.
    twins = tmp_path / "twins.csv"
    twins.write_text(
        "function,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,comet_rows_per_second,comet_speedup_vs_python,"
        "status,target_kind,backend,model,udf_key\n"
        "collatz_steps,1000,300000,600000,2.0,,,success,java_udf,claude,m,kjava\n"
        "collatz_steps,1000,300000,6000000,20.0,,,success,rust,claude,m,krust\n",
        encoding="utf-8",
    )
    perf = _load_dump_module()._perf_lines([str(twins)])
    assert perf["kjava"]["ai"] == "600k rows/s (2.0x vs python)"
    assert perf["krust"]["ai"] == "6.0M rows/s (20.0x vs python, Comet native)"
    # Only the Rust row was measured: its numbers are still bound to its key,
    # so the dump does not put them (and "Comet native") on the twin.
    only_rust = tmp_path / "only_rust.csv"
    only_rust.write_text(
        "function,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,comet_rows_per_second,comet_speedup_vs_python,"
        "status,target_kind,backend,model,udf_key\n"
        "collatz_steps,1000,300000,6000000,20.0,,,success,rust,claude,m,krust\n",
        encoding="utf-8",
    )
    dump = _load_dump_module()
    perf = dump._perf_lines([str(only_rust)])
    assert perf["collatz_steps"]["udf_key"] == "krust"
    assert dump._perf_for(perf, "collatz_steps", "krust")["ai"].endswith("Comet native)")
    assert dump._perf_for(perf, "collatz_steps", "kjava") == {}  # never on the twin
    # A keyed catalog entry does not shadow a later, deeper micro row by name.
    keyed = tmp_path / "keyed_catalog.csv"
    keyed.write_text(
        "function,rows,python_rows_per_second,transpiled_rows_per_second,"
        "transpiled_speedup_vs_python,comet_rows_per_second,comet_speedup_vs_python,"
        "status,target_kind,backend,model,udf_key\n"
        "plus_one,1000,1000000,1500000,1.5,,,success,catalyst,claude,m,kplus\n",
        encoding="utf-8",
    )
    perf = dump._perf_lines([str(keyed), str(micro)])
    assert dump._perf_for(perf, "plus_one", "kplus")["static"] == "3.0M rows/s (1.5x vs python)"  # micro won
    # micro: the rust arm is keyed by kind on the function itself
    rust_micro = tmp_path / "rust_micro.csv"
    rust_micro.write_text(
        "function,kinds,regular_rows_per_second,transpiled_static_rows_per_second,"
        "transpiled_ai_catalyst_rows_per_second,transpiled_ai_java_rows_per_second,"
        "transpiled_ai_rust_rows_per_second,static_speedup_vs_regular,ai_catalyst_speedup_vs_regular,"
        "ai_java_speedup_vs_regular,ai_rust_speedup_vs_regular,transpile_succeeded_static,"
        "transpile_succeeded_ai_catalyst,transpile_succeeded_ai_java,transpile_succeeded_ai_rust,"
        "ai_backend,ai_model\n"
        "edit_distance,catalyst+rust,100000,,9000000,,2000000,,90.0,,20.0,False,True,False,True,claude,m\n",
        encoding="utf-8",
    )
    perf = _load_dump_module()._perf_lines([str(rust_micro)])
    assert perf["edit_distance"]["ai:rust"] == "2.0M rows/s (20.0x vs python, Comet native)"
    assert perf["edit_distance"]["ai:catalyst"] == "9.0M rows/s (90.0x vs python)"
    assert "ai:java_udf" not in perf["edit_distance"]
    assert "edit_distance_rust" not in perf
    # merge precedence: later files take over per function with their deeper arms
    perf = _load_dump_module()._perf_lines([str(cat), str(micro)])
    assert perf["has_item"]["top"].endswith("interpreted python 2.0M rows/s")  # catalog-only
    assert perf["plus_one"]["static"] == "3.0M rows/s (1.5x vs python)"  # micro wins
    # Later files replace only arms they measured; bytecode/Java eval survive micro.
    bc = tmp_path / "bytecode.csv"
    bc.write_text(BYTECODE_CSV, encoding="utf-8")
    perf = _load_dump_module()._perf_lines([str(cat), str(bc), str(micro)])
    assert perf["has_item"]["top"].endswith("interpreted python 2.0M rows/s")
    assert perf["plus_one"]["top"].endswith("interpreted python 2.0M rows/s")
    assert perf["plus_one"]["static"] == "3.0M rows/s (1.5x vs python)"
    assert perf["plus_one"]["bytecode"] == "7.0M rows/s (3.5x vs python)"
    assert perf["plus_one"]["java_eval"] == "2.5M rows/s (1.3x vs python)"


def test_perf_lines_missing_csv_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="cannot read perf csv"):
        _load_dump_module()._perf_lines([str(tmp_path / "nope.csv")])


def test_dump_writes_perf_bullets(spark, tmp_path):
    sqlite_path = tmp_path / "catalog.sqlite"
    _fill_catalog(sqlite_path)
    micro = tmp_path / "micro.csv"
    micro.write_text(MICRO_CSV_HEADER + "\n" + MICRO_CSV_ROWS, encoding="utf-8")
    bytecode = tmp_path / "bytecode.csv"
    bytecode.write_text(BYTECODE_CSV, encoding="utf-8")
    out = tmp_path / "dump.md"
    proc = _run_dump(
        sqlite_path,
        out,
        "--perf-csv",
        str(bytecode),
        "--perf-csv",
        str(micro),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = out.read_text(encoding="utf-8")
    plus_one = text.split("## plus_one")[1].split("## ")[0]
    assert (
        "- **perf (micro bench)**: interpreted python 2.0M rows/s" in plus_one
    )  # top of section, before the paths
    assert plus_one.index("perf (micro bench)") < plus_one.index("built-in static (AST)")
    assert (
        "- **built-in static (AST)**: `Column<'CAST(+(_udf_param_0, 1) AS BIGINT)'>`"
        " — 3.0M rows/s (1.5x vs python)" in plus_one
    )
    assert "- **bytecode → SQL**: `(_udf_param_0 + 1)` — 7.0M rows/s (3.5x vs python)" in plus_one
    assert "bytecode → Java eval" in plus_one
    assert " — 2.5M rows/s (1.3x vs python)" in plus_one
    assert (
        "- **AI (coco / coco-model)**: `_udf_param_0 + 1`"
        " (attempt_count=1, hypothesis_passed=True) — 8.0M rows/s (4.0x vs python)" in plus_one
    )
    always = text.split("## always_decline")[1].split("## ")[0]
    assert "interpreted python 1.0M rows/s" in always
    assert "vs python)" not in always  # declined: no transpiled arm to measure
    # not benched -> no perf anywhere
    backwards = text.split("## backwards")[1].split("## ")[0]
    assert "perf" not in backwards
