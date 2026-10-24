# SPDX-License-Identifier: Apache-2.0
"""Render a bench wide CSV as perf bar charts, linear and log.

  python examples/bench_graph.py bench_micro.csv docs/bench_perf.png
  python examples/bench_graph.py bench_bytecode_vs_static.csv docs/bench_bytecode.png

The CSV schema is sniffed from the header: the bench_micro wide table
(`regular_*` / `transpiled_*_rows_per_second`) or the bytecode-vs-static
table (`python_*` / `bytecode_static_*` / `java_eval_*`). Writes the
linear PNG at the given path and a `_log` sibling. Deterministic CSV-in,
PNG-out (matplotlib Agg): grouped bars per function, one bar per arm.
Empty cells draw nothing -- always_decline has no static arm and no java
arm, and the gaps in its group are the honest picture. In the micro
schema an arm with transpile_succeeded=False is the python fallback, not
a rewrite: it draws hatched and labeled as such.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable


class _Schema:
    def __init__(
        self,
        sniff_column: str,
        arms: list[tuple[str, str, str]],
        value_column: Callable[[str], str],
        declined: Callable[[dict[str, str], str], bool],
        title: str,
    ):
        self.sniff_column = sniff_column
        self.arms = arms
        self.value_column = value_column
        self.declined = declined
        self.title = title


def _micro_declined(row: dict[str, str], arm: str) -> bool:
    return arm != "regular" and row[f"transpile_succeeded_{arm}"] == "False"


def _never_declined(_row: dict[str, str], _arm: str) -> bool:
    # The bytecode bench gives a declined arm no number at all, so there is
    # no fallback bar to hatch.
    return False


MICRO_SCHEMA = _Schema(
    sniff_column="regular_rows_per_second",
    arms=[
        ("regular", "regular (python)", "tab:gray"),
        ("static", "transpiled (static)", "tab:blue"),
        ("ai_catalyst", "transpiled (AI: Catalyst SQL)", "tab:orange"),
        ("ai_java", "transpiled (AI: Java UDF)", "tab:red"),
    ],
    value_column=lambda arm: (
        "regular_rows_per_second" if arm == "regular" else f"transpiled_{arm}_rows_per_second"
    ),
    declined=_micro_declined,
    title="AI UDF transpile",
)

BYTECODE_SCHEMA = _Schema(
    sniff_column="java_eval_rows_per_second",
    arms=[
        ("python", "python (interpreted)", "tab:gray"),
        ("static", "static transpile (built-in AST)", "tab:blue"),
        ("bytecode_static", "bytecode \u2192 SQL (static)", "tab:orange"),
        ("java_eval", "bytecode eval (Java UDF)", "tab:red"),
    ],
    value_column=lambda arm: f"{arm}_rows_per_second",
    declined=_never_declined,
    title="Bytecode transpile paths",
)


def log_png_path(png_path: str) -> Path:
    path = Path(png_path)
    return path.with_name(f"{path.stem}_log{path.suffix}")


def schema_for(header: list[str]) -> _Schema:
    for schema in (MICRO_SCHEMA, BYTECODE_SCHEMA):
        if schema.sniff_column in header:
            return schema
    raise SystemExit(
        "unrecognized bench CSV: expected a bench_micro or "
        "bench_bytecode_vs_static wide table "
        f"(looked for {MICRO_SCHEMA.sniff_column!r} or {BYTECODE_SCHEMA.sniff_column!r})"
    )


def render(rows: list[dict[str, str]], png_path: str, *, log_scale: bool, schema: _Schema) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    functions = [r["function"] for r in rows]
    n = len(functions)
    width = 0.8 / len(schema.arms)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=150)
    any_declined = False
    for i, (arm, label, color) in enumerate(schema.arms):
        column = schema.value_column(arm)
        xs = np.arange(n) + (i - (len(schema.arms) - 1) / 2) * width
        ys = [float(r[column]) if r[column] != "" else np.nan for r in rows]
        declined = [schema.declined(r, arm) for r in rows]
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
                    va="bottom",
                    rotation=0 if log_scale else 90,
                    fontsize=7,
                )
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1e6:.0f}M"))
        ax.margins(y=0.18)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(functions)
    ax.set_ylabel("rows / second (log scale)" if log_scale else "rows / second")
    scale = "log scale" if log_scale else "linear scale"
    ax.set_title(f"{schema.title}: throughput by function and arm ({scale}, higher is better)")
    handles, _labels = ax.get_legend_handles_labels()
    if any_declined:
        handles.append(
            Patch(facecolor="white", edgecolor="black", hatch="///", label="declined (python fallback perf)")
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="a bench --csv wide table")
    parser.add_argument("png_path", help="linear-scale PNG; the log-scale PNG is written beside it")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    if not rows:
        parser.error(f"{args.csv_path} has no data rows")
    schema = schema_for(header)
    for _arm, _label, _color in schema.arms:
        if schema.value_column(_arm) not in header:
            parser.error(f"{args.csv_path} is missing the {schema.value_column(_arm)} column")

    render(rows, args.png_path, log_scale=False, schema=schema)
    log_path = log_png_path(args.png_path)
    render(rows, str(log_path), log_scale=True, schema=schema)
    print(f"wrote {args.png_path}")
    print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
