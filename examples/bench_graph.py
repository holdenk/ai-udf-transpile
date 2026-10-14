# SPDX-License-Identifier: Apache-2.0
"""Render the bench_micro wide CSV as a simple perf bar chart.

    python examples/bench_graph.py bench_micro.csv docs/bench_perf.png

Deterministic CSV-in, PNG-out (matplotlib Agg): grouped bars per function,
log scale, one bar per arm. Empty cells draw nothing -- always_decline has
no static arm and no java arm, and the gaps in its group are the honest
picture. An arm with transpile_succeeded=False is the python fallback, not
a rewrite: it draws hatched and labeled as such.
"""

from __future__ import annotations

import argparse
import csv

# arm key, legend label, bar color
ARMS = [
    ("regular", "regular (python)", "tab:gray"),
    ("static", "transpiled (static)", "tab:blue"),
    ("ai_catalyst", "transpiled (AI: Catalyst SQL)", "tab:orange"),
    ("ai_java", "transpiled (AI: Java UDF)", "tab:red"),
]


def value_column(arm: str) -> str:
    return "regular_rows_per_second" if arm == "regular" else f"transpiled_{arm}_rows_per_second"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="the bench_micro --csv wide table")
    parser.add_argument("png_path", help="output PNG")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error(f"{args.csv_path} has no data rows")
    for _arm, _label, _color in ARMS:
        if value_column(_arm) not in rows[0]:
            parser.error(f"{args.csv_path} is missing the {value_column(_arm)} column")

    functions = [r["function"] for r in rows]
    n = len(functions)
    width = 0.8 / len(ARMS)
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=150)
    any_declined = False
    for i, (arm, label, color) in enumerate(ARMS):
        column = value_column(arm)
        xs = np.arange(n) + (i - (len(ARMS) - 1) / 2) * width
        ys = [float(r[column]) if r[column] != "" else np.nan for r in rows]
        declined = [arm != "regular" and r[f"transpile_succeeded_{arm}"] == "False" for r in rows]
        any_declined = any_declined or any(d and y == y for d, y in zip(declined, ys))
        ax.bar(
            xs,
            ys,
            width,
            label=label,
            color=color,
            hatch=["//" if d else "" for d in declined],
        )
        for x, y, d in zip(xs, ys, declined):
            if y == y:  # not NaN: this arm has a number
                suffix = " (fallback)" if d else ""
                ax.annotate(
                    f"{y / 1e6:.0f}M{suffix}",
                    (x, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(functions)
    ax.set_ylabel("rows / second (log scale)")
    ax.set_title("AI UDF transpile: throughput by function and arm (higher is better)")
    handles, _labels = ax.get_legend_handles_labels()
    if any_declined:
        handles.append(
            Patch(facecolor="white", edgecolor="black", hatch="///", label="declined (python fallback perf)")
        )
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.png_path)
    print(f"wrote {args.png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
