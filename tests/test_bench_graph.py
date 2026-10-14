# SPDX-License-Identifier: Apache-2.0
"""The README perf graph regenerates from the bench CSV (rot guard)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

BENCH_GRAPH = Path(__file__).resolve().parents[1] / "examples" / "bench_graph.py"

# The bench_micro --csv wide schema.
HEADER = [
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


def test_graph_renders_from_wide_csv(tmp_path):
    csv_path = tmp_path / "bench.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        # Millions, like a real run, so a reintroduced axis floor clips visibly.
        writer.writerow(
            [
                "plus_one",
                "catalyst+java",
                2_000_000,
                23_000_000,
                23_000_000,
                18_000_000,
                11.5,
                11.5,
                9.0,
                True,
                True,
                True,
            ]
        )
        # The honest gaps: no static rewrite and no java rewrite for a
        # declined UDF -- those cells are empty, not zero. The ai column
        # carries the fallback perf with transpile_succeeded=False, which the
        # graph must draw as hatched (fallback), not a plain orange bar.
        writer.writerow(
            ["always_decline", "declined", 1_000_000, "", 1_000_000, "", "", 1.0, "", False, False, False]
        )
    png_path = tmp_path / "bench.png"
    proc = subprocess.run(
        [sys.executable, str(BENCH_GRAPH), str(csv_path), str(png_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert png_path.stat().st_size > 0


def test_graph_rejects_wrong_csv(tmp_path):
    csv_path = tmp_path / "not_bench.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BENCH_GRAPH), str(csv_path), str(tmp_path / "x.png")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
