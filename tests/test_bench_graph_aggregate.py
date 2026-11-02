# SPDX-License-Identifier: Apache-2.0
"""bench_graph_aggregate.py rolls every bench CSV into one speedup chart."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

AGGREGATE = Path(__file__).resolve().parents[1] / "examples" / "bench_graph_aggregate.py"


def test_aggregate_collects_and_excludes_fallbacks(tmp_path):
    # A micro-schema CSV: one real static rewrite and one python fallback
    # (transpile_succeeded=False) -- the fallback point must not count.
    with open(tmp_path / "bench_micro.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "function",
                "kinds",
                "regular_rows_per_second",
                "transpiled_static_rows_per_second",
                "transpiled_ai_catalyst_rows_per_second",
                "transpiled_ai_java_rows_per_second",
                "static_speedup_vs_regular",
                "ai_catalyst_speedup_vs_regular",
                "ai_java_speedup_vs_regular",
                "transpile_succeeded_static",
                "transpile_succeeded_ai_catalyst",
                "transpile_succeeded_ai_java",
            ]
        )
        w.writerow(["ok", "static", 1_000_000, 5_000_000, "", "", 5.0, "", "", True, False, False])
        w.writerow(["fell", "static", 1_000_000, 1_000_000, "", "", 1.0, "", "", False, False, False])
    # A comet run of the same schema: its points land on separate arms.
    with open(tmp_path / "bench_micro_comet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "function",
                "kinds",
                "regular_rows_per_second",
                "transpiled_static_rows_per_second",
                "transpiled_ai_catalyst_rows_per_second",
                "transpiled_ai_java_rows_per_second",
                "static_speedup_vs_regular",
                "ai_catalyst_speedup_vs_regular",
                "ai_java_speedup_vs_regular",
                "transpile_succeeded_static",
                "transpile_succeeded_ai_catalyst",
                "transpile_succeeded_ai_java",
            ]
        )
        w.writerow(["ok", "static", 1_000_000, 20_000_000, "", "", 20.0, "", "", True, False, False])
    # A bytecode-schema CSV.
    with open(tmp_path / "bench_bc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "function",
                "static_converted",
                "static_correct",
                "bytecode_static_correct",
                "java_eval_correct",
                "python_rows_per_second",
                "static_rows_per_second",
                "bytecode_static_rows_per_second",
                "java_eval_rows_per_second",
                "static_speedup_vs_python",
                "bytecode_static_speedup_vs_python",
                "java_eval_speedup_vs_python",
            ]
        )
        w.writerow(["p1", True, True, True, True, 10, 100, 120, 90, 10.0, 12.0, 9.0])
    png = tmp_path / "agg.png"
    proc = subprocess.run(
        [sys.executable, str(AGGREGATE), str(tmp_path), "--png", str(png)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert png.stat().st_size > 0
    assert png.with_name("agg_log.png").stat().st_size > 0
    # static arm = 2 CSV points, not 3: the fallback row is a decline.
    assert "static (built-in AST)" in proc.stdout
    assert "2 points, best 10.0x" in proc.stdout
    assert "bytecode → SQL (static)" in proc.stdout
    assert "1 points, best 12.0x" in proc.stdout
    # The comet CSV lands on its own arm, not mixed into the plain one.
    assert "static (built-in AST, Comet)" in proc.stdout
    assert "best 20.0x" in proc.stdout


def test_aggregate_rejects_unrecognized_csv(tmp_path):
    (tmp_path / "bench_nonsense.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(AGGREGATE), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_aggregate_collects_every_catalog_arm(tmp_path):
    path = tmp_path / "bench_catalog.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "function",
                "status",
                "target_kind",
                "static_speedup_vs_python",
                "bytecode_static_speedup_vs_python",
                "java_eval_speedup_vs_python",
                "transpiled_speedup_vs_python",
                "comet_speedup_vs_python",
            ]
        )
        writer.writerow(["cat", "success", "catalyst", 2.0, 3.0, 4.0, 5.0, 6.0])
        writer.writerow(["java", "success", "java_udf", "", "", "", 7.0, 8.0])

    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_graph_aggregate", AGGREGATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = module.collect([path])

    assert points["static"] == [("catalog/cat", 2.0)]
    assert points["bytecode_static"] == [("catalog/cat", 3.0)]
    assert points["java_eval"] == [("catalog/cat", 4.0)]
    assert points["ai_catalyst"] == [("catalog/cat", 5.0)]
    assert points["ai_catalyst_comet"] == [("catalog/cat", 6.0)]
    assert points["ai_java"] == [("catalog/java", 7.0)]
    assert points["ai_java_comet"] == [("catalog/java", 8.0)]
