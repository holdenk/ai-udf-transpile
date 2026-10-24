# SPDX-License-Identifier: Apache-2.0
"""Aggregate every benchmark CSV into one speedup chart.

  python examples/bench_graph_aggregate.py questionable_benchmark_results/

Reads every bench*.csv in the directory (bench_micro, bench_bytecode_vs_static,
bench_tpcds_udfs), normalizes each arm to a speedup vs interpreted python, and
draws one horizontal strip per path: every point is one (benchmark, function)
pair, so the honest picture includes the stragglers. Fallback rows (a declined
pattern running python perf under the transpiled label) are excluded -- they
are declines, not rewrites. Writes bench_aggregate.png plus a _log sibling.

No AI in the loop, ever: CSV in, PNG out.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (fallback flag column, value) -> skip the row: python fallback perf
# wearing the transpiled label is a decline, not a rewrite.

ARMS = {
    # normalized arm key -> legend label, color
    "static": ("static (built-in AST)", "tab:blue"),
    "bytecode_static": ("bytecode \u2192 SQL (static)", "tab:orange"),
    "java_eval": ("bytecode eval (Java UDF)", "tab:red"),
    "ai_catalyst": ("AI: Catalyst SQL", "tab:purple"),
    "ai_java": ("AI: Java UDF", "tab:green"),
    "vanilla": ("vanilla SQL (tpcds patterns)", "tab:gray"),
    "transpiled": ("transpiled (tpcds patterns)", "tab:brown"),
}

# csv signature column -> [(arm, speedup column, (fallback flag col, skip value))]
_SCHEMAS = {
    "regular_rows_per_second": [
        ("static", "static_speedup_vs_regular", ("transpile_succeeded_static", "False")),
        (
            "ai_catalyst",
            "ai_catalyst_speedup_vs_regular",
            ("transpile_succeeded_ai_catalyst", "False"),
        ),
        ("ai_java", "ai_java_speedup_vs_regular", ("transpile_succeeded_ai_java", "False")),
    ],
    "java_eval_rows_per_second": [
        ("static", "static_speedup_vs_python", None),
        ("bytecode_static", "bytecode_static_speedup_vs_python", None),
        ("java_eval", "java_eval_speedup_vs_python", None),
    ],
    "transpiled_rows_per_second": [
        ("vanilla", "vanilla_speedup_vs_python", None),
        ("transpiled", "transpiled_speedup_vs_python", ("declined", "True")),
    ],
}


def collect(paths: list[Path]) -> dict[str, list[tuple[str, float]]]:
    """arm -> [(benchmark, function), speedup]"""
    points: dict[str, list[tuple[str, float]]] = {arm: [] for arm in ARMS}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        specs = next((s for sig, s in _SCHEMAS.items() if sig in rows[0]), None)
        if specs is None:
            print(f"skipping {path.name}: unrecognized schema", file=sys.stderr)
            continue
        bench = path.stem.replace("bench_", "").replace("_", " ")
        for row in rows:
            for arm, speedup_col, guard in specs:
                raw = row.get(speedup_col, "")
                if not raw or float(raw) <= 0:
                    continue  # declined/failed arm: no number, no point
                if guard and row.get(guard[0]) == guard[1]:
                    continue  # python fallback perf, not a rewrite
                points[arm].append((f"{bench}/{row['function']}", float(raw)))
    return points


def render(points: dict[str, list[tuple[str, float]]], png_path: str, *, log_scale: bool) -> None:
    import matplotlib.pyplot as plt

    order = sorted(points, key=lambda a: sorted(v for _, v in points[a])[len(points[a]) // 2])
    fig, ax = plt.subplots(figsize=(10.5, 0.9 * len(order) + 1.6), dpi=150)
    for y, arm in enumerate(order):
        label, color = ARMS[arm]
        xs = [v for _, v in points[arm]]
        med = sorted(xs)[len(xs) // 2]
        ax.scatter(xs, [y] * len(xs), s=26, color=color, alpha=0.75, zorder=3)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=color, lw=2.5, zorder=4)
        ax.annotate(
            f"{med:.1f}x med ({len(xs)})",
            (max(xs), y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ARMS[a][0] for a in order], fontsize=9)
    if log_scale:
        ax.set_xscale("log")
    ax.axvline(1.0, color="black", lw=1, ls="--")
    ax.set_xlabel("speedup vs interpreted python (log scale)" if log_scale else "speedup vs python")
    scale = "log scale" if log_scale else "linear scale"
    ax.set_title(f"Aggregate UDF benchmark: per-path speedup across all CSVs ({scale})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="directory of bench*.csv CSVs (or a single CSV)")
    parser.add_argument("--png", help="output PNG (default: <results>/bench_aggregate.png)")
    args = parser.parse_args()

    root = Path(args.results)
    paths = sorted(root.glob("bench*.csv")) if root.is_dir() else [root]
    if not paths:
        parser.error(f"no bench*.csv under {args.results}")
    points = collect(paths)
    live = {a: pts for a, pts in points.items() if pts}
    if not live:
        parser.error("no speedup points collected")
    for arm, pts in live.items():
        best = max(v for _, v in pts)
        print(f"{ARMS[arm][0]:34}{len(pts):4} points, best {best:.1f}x")

    import matplotlib

    matplotlib.use("Agg")
    png = args.png or str(root / "bench_aggregate.png")
    render(live, png, log_scale=False)
    log_path = Path(png).with_name(f"{Path(png).stem}_log{Path(png).suffix}")
    render(live, str(log_path), log_scale=True)
    print(f"wrote {png}")
    print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
