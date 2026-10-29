# SPDX-License-Identifier: Apache-2.0
"""Dump what every transpiler does to every UDF the examples filled into a
live sqlite catalog, as markdown.

One section per cached row, in three paths:

  built-in static (AST)  Spark's own catalyst transpiler (pyTranspilers=catalyst)
  bytecode -> SQL        the deterministic bytecode backend
  AI (<backend>)         whatever the live backend produced (Hypothesis-verified
                          before it was cached)

Declines are shown as declines, with the reason -- that is half the story.

  python examples/dump_results.py [--sqlite target/examples_transpiled.sqlite]
                                   [--out docs/EXAMPLES_TRANSPILED.md]
                                   [--backend coco]
                                   [--perf-csv bench_micro.csv] [--perf-csv bench_tpcds.csv]
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import os
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import _parse_datatype_string  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile.backends import ai_backend  # noqa: E402
from ai_udf_transpile.backends.base import BackendDecline, BackendError  # noqa: E402
from ai_udf_transpile.backends.bytecode import BytecodeBackend  # noqa: E402
from ai_udf_transpile.catalog import CacheRow  # noqa: E402
from ai_udf_transpile.catalog.sqlite import SqliteCatalog, _row_from_sql  # noqa: E402
from ai_udf_transpile.sandbox import make_sandbox  # noqa: E402
from ai_udf_transpile.targets import KIND_JAVA_UDF  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"


def _all_rows(catalog: SqliteCatalog) -> list[CacheRow]:
    cur = catalog._conn.execute("SELECT * FROM cache")
    return [_row_from_sql(raw) for raw in cur.fetchall()]


def _function_name(source_text: str) -> str:
    try:
        tree = ast.parse(source_text or "")
        fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        return fn.name
    except Exception:
        return ""


def _bad_backend_rows(rows: list[CacheRow], backend: str) -> list[CacheRow]:
    """Rows whose backend/origin disagrees with --backend. origin='human' is exempt."""
    return [
        row
        for row in rows
        if row.origin != "human"
        and ((row.backend and row.backend != backend) or (row.origin and row.origin != backend))
    ]


def _safe(label: str, fn) -> list[str]:
    """Run one transpiler path; a crash is a dump finding, not a dump death."""
    try:
        return [f"- **{label}**: `{fn()}`"]
    except Exception as exc:
        return [f"- **{label}**: error: {type(exc).__name__}: {exc}"]


def _fmt_rate(value: str) -> str:
    n = float(value)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def _perf_lines(paths: list[str]) -> dict[str, dict[str, str]]:
    """Section name -> per-arm perf strings, from the bench CSVs.

    "top" rides under the types line (the interpreted python baseline, plus
    the native vanilla arm for tpcds patterns); "static" and "ai" are
    suffixed onto the matching transpiled-path line. Only benched sections
    get numbers; a declined pattern serves python, so its AI line gets no
    suffix and the top line tells that story. Files overwrite per function,
    last one wins: pass the wide bench_catalog CSV first, then the deeper
    micro/tpcds CSVs whose extra arms (static, vanilla, java) take over.
    """
    perf: dict[str, dict[str, str]] = {}
    for path in paths:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except OSError as exc:
            raise SystemExit(f"cannot read perf csv {path}: {exc}")
        if not rows:
            continue
        if "target_kind" in rows[0]:  # bench_catalog CSV: every cached function
            for r in rows:
                py_rate = r.get("python_rows_per_second")
                tx_rate = r.get("transpiled_rows_per_second")
                if not py_rate and not tx_rate:
                    continue  # skipped function (python raises on the sample)
                if not py_rate:
                    # Interpreted arm could not run on the sample; the
                    # transpiled rate is still a real measurement.
                    perf[r["function"]] = {
                        "top": "",
                        "static": "",
                        "ai": f"{_fmt_rate(tx_rate)} rows/s (no python baseline)",
                    }
                    continue
                arms = {
                    "top": f"- **perf (catalog bench)**: interpreted python {_fmt_rate(py_rate)} rows/s",
                    "static": "",
                    "ai": "",
                }
                if tx_rate and r["status"] == "success":
                    arms["ai"] = (
                        f"{_fmt_rate(tx_rate)} rows/s ({r['transpiled_speedup_vs_python']}x vs python)"
                    )
                comet = r.get("comet_rows_per_second")
                if comet and arms["ai"] and r.get("comet_speedup_vs_python"):
                    # The same served SQL, executed natively by Comet.
                    arms["ai"] += (
                        f" · comet {_fmt_rate(comet)} rows/s ({r['comet_speedup_vs_python']}x vs python)"
                    )
                perf[r["function"]] = arms
        elif "kinds" in rows[0]:  # bench_micro wide CSV
            for r in rows:
                py = _fmt_rate(r["regular_rows_per_second"])
                top = f"- **perf (micro bench)**: interpreted python {py} rows/s"
                arms: dict[str, str] = {"top": top, "static": "", "ai": ""}
                static = r.get("transpiled_static_rows_per_second")
                if static:
                    arms["static"] = (
                        f"{_fmt_rate(static)} rows/s ({r['static_speedup_vs_regular']}x vs python)"
                    )
                catalyst = r.get("transpiled_ai_catalyst_rows_per_second")
                if catalyst and r["transpile_succeeded_ai_catalyst"] == "True":
                    arms["ai"] = (
                        f"{_fmt_rate(catalyst)} rows/s ({r['ai_catalyst_speedup_vs_regular']}x vs python)"
                    )
                perf[r["function"]] = arms
                java = r.get("transpiled_ai_java_rows_per_second")
                if java:
                    perf[r["function"] + "_java"] = {
                        "top": top,
                        "static": "",
                        "ai": f"{_fmt_rate(java)} rows/s ({r['ai_java_speedup_vs_regular']}x vs python)",
                    }
        elif "declined" in rows[0]:  # bench_tpcds CSV
            for r in rows:
                top = (
                    f"- **perf (tpcds bench)**: interpreted python"
                    f" {_fmt_rate(r['python_rows_per_second'])} rows/s · native vanilla"
                    f" {_fmt_rate(r['vanilla_rows_per_second'])} ({r['vanilla_speedup_vs_python']}x)"
                )
                ai = ""
                if r["declined"] != "True":
                    ai = (
                        f"{_fmt_rate(r['transpiled_rows_per_second'])} rows/s"
                        f" ({r['transpiled_speedup_vs_python']}x vs python)"
                    )
                perf[r["function"]] = {"top": top, "static": "", "ai": ai}
    return perf


def _builtin_result(row: CacheRow) -> str:
    name = _function_name(row.source_text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "udf.py"
        path.write_text("from __future__ import annotations\n" + row.source_text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("dump_row", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        udf = UserDefinedFunction(getattr(module, name), _parse_datatype_string(row.return_type))
    if not udf.transpiled:
        return "declined (the construction-time UserWarning carries the reason)"
    return "  |  ".join(str(sql) for sql in udf.transpiled)


def _bytecode_result(row: CacheRow) -> str:
    job = row.as_job()
    try:
        with make_sandbox(job) as sandbox:
            return BytecodeBackend().run(job, sandbox).sql
    except BackendDecline as exc:
        return f"declined: {exc}"


def _ai_section(row: CacheRow) -> list[str]:
    label = row.origin or row.backend or "never claimed"
    if row.model:
        label = f"{label} / {row.model}"
    tail = f" (attempt_count={row.attempt_count}, hypothesis_passed={row.hypothesis_passed})"
    if row.status == "success":
        if row.target_kind == KIND_JAVA_UDF:
            return [f"- **AI ({label})**: java UDF:{tail}", "", "```java", row.impl_source or "", "```"]
        return [f"- **AI ({label})**: `{row.catalyst_sql}`{tail}"]
    if row.status == "failed":
        lines = [f"- **AI ({label})**: failed: {row.error}{tail}"]
        # The rejected candidate is half the story: show it when the row kept it.
        if row.target_kind == KIND_JAVA_UDF and row.impl_source:
            return lines + ["", "```java", row.impl_source, "```"]
        if row.catalyst_sql:
            lines.append(f"- **rejected rewrite**: `{row.catalyst_sql}`")
        return lines
    return [f"- **AI ({label})**: status={row.status}{tail}"]


def dump(spark, rows: list[CacheRow], out_path: str, backend: str, perf: dict[str, dict[str, str]]) -> int:
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    by_status = dict(Counter(row.status for row in rows))
    lines = [
        "# What transpiles to what",
        "",
        f"Generated by `examples/dump_results.py` from the catalog the examples filled "
        f"with backend {backend!r}, against Spark {spark.version} on "
        f"CPython {platform.python_version()}.",
        f"{len(rows)} rows, by status: {by_status}.",
        "Regenerate with `examples/regen_examples_doc.sh`.",
        "Perf bullets are the local single-node benches (bench_catalog.py for",
        "every cached function, bench_micro.py / bench_tpcds_udfs.py for the",
        "deeper arms): interpreted python rows/s at the top of each section,",
        "and each transpiled arm's rows/s at the end of its own line -- with a",
        "comet arm when the catalog bench ran against a Comet build; no perf",
        "= not benched or the interpreted arm raises on the bench sample.",
        "",
    ]
    try:
        for row in rows:
            name = _function_name(row.source_text) or row.udf_key[:12]
            p = perf.get(name) or {}
            head = [f"- **types**: {', '.join(row.input_types)} -> {row.return_type}"]
            if p.get("top"):
                head.append(p["top"])
            lines += [
                f"## {name}",
                "",
                "```python",
                row.source_text,
                "```",
                "",
                *head,
                "",
            ]
            static_lines = _safe("built-in static (AST)", lambda r=row: _builtin_result(r))
            # No perf suffix on an error or a failed row: a stale CSV must
            # not put a happy rate next to a crash or a decline.
            if p.get("static") and not static_lines[0].startswith("- **built-in static (AST)**: error"):
                static_lines[0] += f" — {p['static']}"
            lines += static_lines
            lines += _safe("bytecode → SQL", lambda r=row: _bytecode_result(r))
            ai_lines = _ai_section(row)
            if p.get("ai") and row.status == "success":
                ai_lines[0] += f" — {p['ai']}"
            lines += ai_lines
            lines.append("")
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default="target/examples_transpiled.sqlite", help="catalog sqlite path")
    parser.add_argument("--out", default="docs/EXAMPLES_TRANSPILED.md", help="markdown output path")
    parser.add_argument(
        "--backend",
        help="backend every AI row came from (default: $AI_UDF_TRANSPILE_BACKEND, else auto)",
    )
    parser.add_argument(
        "--perf-csv",
        action="append",
        default=[],
        metavar="PATH",
        help="bench CSV to pull per-function perf bullets from (micro and/or tpcds); repeatable",
    )
    args = parser.parse_args()
    if args.backend:
        os.environ["AI_UDF_TRANSPILE_BACKEND"] = args.backend
    try:
        args.backend = ai_backend()
    except BackendError as exc:
        print(exc, file=sys.stderr)
        return 1
    perf = _perf_lines(args.perf_csv) if args.perf_csv else {}
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        print(f"no sqlite catalog at {sqlite_path}", file=sys.stderr)
        return 1
    catalog = SqliteCatalog(sqlite_path)
    rows = _all_rows(catalog)
    if not rows:
        print(f"{sqlite_path} has zero rows", file=sys.stderr)
        return 1
    bad = _bad_backend_rows(rows, args.backend)
    if bad:
        detail = ", ".join(f"{r.udf_key[:12]}(backend={r.backend!r},origin={r.origin!r})" for r in bad[:5])
        print(f"rows not produced by backend {args.backend!r}: {detail}", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (_function_name(r.source_text), r.udf_key))
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("ai-udf-transpile-dump")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    try:
        return dump(spark, rows, args.out, args.backend, perf)
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
