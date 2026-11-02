# SPDX-License-Identifier: Apache-2.0
"""Dump what every transpiler does to every UDF the examples filled into a
live sqlite catalog, as markdown.

One section per cached row, covering every transpiled path:

  built-in static (AST)  Spark's own catalyst transpiler (pyTranspilers=catalyst)
  bytecode -> SQL        the deterministic bytecode backend
  bytecode -> Java eval  the same bytecode subset in the fixed Java interpreter
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
from ai_udf_transpile.backends.bytecode_eval import emit_java_eval_source  # noqa: E402
from ai_udf_transpile.catalog import CacheRow, infra_failure  # noqa: E402
from ai_udf_transpile.catalog.sqlite import SqliteCatalog, _row_from_sql  # noqa: E402
from ai_udf_transpile.sandbox import make_sandbox  # noqa: E402
from ai_udf_transpile.targets import KIND_GPU_KERNEL, KIND_JAVA_UDF, KIND_RUST, NO_RUST_SUFFIX  # noqa: E402

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


def _empty_perf() -> dict[str, str]:
    return {"top": "", "static": "", "bytecode": "", "java_eval": "", "ai": ""}


def _record_perf(perf: dict[str, dict[str, str]], name: str, arms: dict[str, str]) -> None:
    current = perf.setdefault(name, {})
    for arm, value in arms.items():
        if value or arm not in current:
            current[arm] = value


def _rate_pair(rate: str, speedup: str) -> str:
    if not rate:
        return ""
    detail = f"{speedup}x vs python" if speedup else "no python baseline"
    return f"{_fmt_rate(rate)} rows/s ({detail})"


def _perf_lines(paths: list[str]) -> dict[str, dict[str, str]]:
    """Section name -> per-arm perf strings, merged without dropping arms.

    Pass the wide catalog CSV first, then bytecode and the deeper micro/TPC-DS
    CSVs. A later non-empty measurement replaces the same arm; other arms and
    the Python baseline from earlier files remain intact.
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
        if "target_kind" in rows[0]:  # bench_catalog / bench_fhi_udfs CSV
            # bench_fhi_udfs also times its pinned naive/tuned rewrites and a
            # non-nullable column; the doc's AI line is the live arm on the
            # ordinary (nullable) column.
            live = [
                r for r in rows if r.get("arm", "ai") == "ai" and r.get("column_nullable", "True") == "True"
            ]
            # A name two catalog rows share (a rust row and its "+no-rust"
            # twin) is keyed by udf_key as well, so each section gets its own
            # row's numbers rather than whichever came last.
            names = Counter(r["function"] for r in live)
            for r in live:
                key = r.get("udf_key") or ""
                # By udf_key whenever the CSV has one; by name only for a
                # name no other catalog row in this CSV shares. dump() checks
                # the key too, so a name entry never lands on another row.
                keys = [key] if key else []
                if names[r["function"]] == 1 or not key:
                    keys.append(r["function"])
                arms = _empty_perf()
                py_rate = r.get("python_rows_per_second", "")
                bench = r.get("bench") or "catalog"  # bench_fhi_udfs.py names itself
                if py_rate:
                    arms["top"] = f"- **perf ({bench} bench)**: interpreted python {_fmt_rate(py_rate)} rows/s"
                for arm, prefix in (
                    ("static", "static"),
                    ("bytecode", "bytecode_static"),
                    ("java_eval", "java_eval"),
                    ("ai", "transpiled"),
                ):
                    arms[arm] = _rate_pair(
                        r.get(f"{prefix}_rows_per_second", ""), r.get(f"{prefix}_speedup_vs_python", "")
                    )
                if r["status"] != "success":
                    arms["ai"] = ""
                elif r.get("target_kind") == "rust" and arms["ai"]:
                    # A rust rewrite's only arm is Comet's: say where it ran.
                    arms["ai"] = arms["ai"].replace("x vs python", "x vs python, Comet native")
                comet = r.get("comet_rows_per_second", "")
                if comet and arms["ai"] and r.get("comet_speedup_vs_python"):
                    arms["ai"] += (
                        f" · comet {_fmt_rate(comet)} rows/s ({r['comet_speedup_vs_python']}x vs python)"
                    )
                prev_ai = perf.get(r["function"], {}).get("ai", "")
                if arms["ai"] and not comet and " · comet " in prev_ai:
                    # A later bench on this schema (fhi) has no comet arm:
                    # keep the catalog bench's, labeled, since its speedup is
                    # against that bench's own python baseline.
                    arms["ai"] += " · comet (catalog bench) " + prev_ai.split(" · comet ", 1)[1]
                if key:
                    arms["udf_key"] = key
                if any(arms.values()):
                    for k in keys:
                        perf[k] = dict(arms)
        elif "java_eval_rows_per_second" in rows[0]:  # bench_bytecode_vs_static CSV
            for r in rows:
                arms = _empty_perf()
                py_rate = r.get("python_rows_per_second", "")
                if py_rate:
                    arms["top"] = (
                        f"- **perf (bytecode bench)**: interpreted python {_fmt_rate(py_rate)} rows/s"
                    )
                for arm, prefix in (
                    ("static", "static"),
                    ("bytecode", "bytecode_static"),
                    ("java_eval", "java_eval"),
                ):
                    arms[arm] = _rate_pair(
                        r.get(f"{prefix}_rows_per_second", ""), r.get(f"{prefix}_speedup_vs_python", "")
                    )
                _record_perf(perf, r["function"], arms)
        elif "kinds" in rows[0]:  # bench_micro wide CSV
            for r in rows:
                py = _fmt_rate(r["regular_rows_per_second"])
                top = f"- **perf (micro bench)**: interpreted python {py} rows/s"
                arms = _empty_perf()
                arms["top"] = top
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
                _record_perf(perf, r["function"], arms)
                java = r.get("transpiled_ai_java_rows_per_second")
                if java:
                    java_arms = _empty_perf()
                    java_arms["top"] = top
                    java_arms["ai"] = (
                        f"{_fmt_rate(java)} rows/s ({r['ai_java_speedup_vs_regular']}x vs python)"
                    )
                    _record_perf(perf, r["function"] + "_java", java_arms)
                rust = r.get("transpiled_ai_rust_rows_per_second")
                if rust:
                    rust_arms = _empty_perf()
                    rust_arms["top"] = top
                    rust_arms["ai"] = (
                        f"{_fmt_rate(rust)} rows/s"
                        f" ({r['ai_rust_speedup_vs_regular']}x vs python, Comet native)"
                    )
                    _record_perf(perf, r["function"] + "_rust", rust_arms)
        elif "declined" in rows[0]:  # bench_tpcds CSV
            for r in rows:
                arms = _empty_perf()
                arms["top"] = (
                    f"- **perf (tpcds bench)**: interpreted python"
                    f" {_fmt_rate(r['python_rows_per_second'])} rows/s · native vanilla"
                    f" {_fmt_rate(r['vanilla_rows_per_second'])} ({r['vanilla_speedup_vs_python']}x)"
                )
                if r["declined"] != "True":
                    arms["ai"] = (
                        f"{_fmt_rate(r['transpiled_rows_per_second'])} rows/s"
                        f" ({r['transpiled_speedup_vs_python']}x vs python)"
                    )
                _record_perf(perf, r["function"], arms)
    return perf


def _perf_for(perf: dict[str, dict[str, str]], name: str, udf_key: str) -> dict[str, str]:
    """The perf bullets for one catalog row.

    By name first: later --perf-csv files (micro/tpcds/fhi, deeper arms)
    override per function by name. A name entry that carries another catalog
    row's udf_key (a Rust row's, for its "+no-rust" twin) is not this row's;
    then only this row's own keyed entry, if any, applies.
    """
    p = perf.get(name) or {}
    if not p or (p.get("udf_key") and p["udf_key"] != udf_key):
        p = perf.get(udf_key) or {}
    return p


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


def _java_eval_result(row: CacheRow) -> str:
    try:
        emitted = emit_java_eval_source(row.source_text)
        return f"{emitted.class_name} (fixed BytecodeEval interpreter)"
    except BackendDecline as exc:
        return f"declined: {exc}"


def _ai_section(row: CacheRow) -> list[str]:
    label = row.origin or row.backend or "never claimed"
    if row.model:
        label = f"{label} / {row.model}"
    tail = f" (attempt_count={row.attempt_count}, hypothesis_passed={row.hypothesis_passed})"
    if row.status == "success":
        if row.target_kind == KIND_GPU_KERNEL:
            return [
                f"- **AI ({label})**: CUDA kernel (GPU) with a java UDF (CPU):{tail}",
                "",
                "```cpp",
                row.impl_entry or "",
                "```",
                "",
                "```java",
                row.impl_source or "",
                "```",
            ]
        if row.target_kind == KIND_JAVA_UDF:
            return [f"- **AI ({label})**: java UDF:{tail}", "", "```java", row.impl_source or "", "```"]
        if row.target_kind == KIND_RUST:
            head = f"- **AI ({label})**: rust UDF (Comet native):{tail}"
            return [head, "", "```rust", row.impl_source or "", "```"]
        return [f"- **AI ({label})**: `{row.catalyst_sql}`{tail}"]
    if row.status == "failed":
        lines = [f"- **AI ({label})**: failed: {row.error}{tail}"]
        # The rejected candidate is half the story: show it when the row kept it.
        if row.target_kind == KIND_JAVA_UDF and row.impl_source:
            return lines + ["", "```java", row.impl_source, "```"]
        if row.target_kind == KIND_RUST and row.impl_source:
            return lines + ["", "```rust", row.impl_source, "```"]
        if row.catalyst_sql:
            lines.append(f"- **rejected rewrite**: `{row.catalyst_sql}`")
        return lines
    return [f"- **AI ({label})**: status={row.status}{tail}"]


def _write_doc(out_path: str, lines: list[str]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as directory:
        temporary = Path(directory) / path.name
        temporary.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temporary, path)


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
        "every cached function, bench_bytecode_vs_static.py / bench_micro.py /",
        "bench_tpcds_udfs.py for deeper arms): interpreted python rows/s at the",
        "top of each section, and every transpiled arm's rows/s on its own line",
        "-- with a comet arm when configured. No perf suffix means that arm was",
        "not benched, declined, or failed; a missing Python baseline is labeled.",
        "Rust rewrites (the opt-in `rust` target) run only inside Comet, so",
        "their one number is Comet's and says so.",
        "",
    ]
    try:
        for row in rows:
            name = _function_name(row.source_text) or row.udf_key[:12]
            p = _perf_for(perf, name, row.udf_key)
            title = name
            if (row.spark_version or "").endswith(NO_RUST_SUFFIX):
                # The rewrite a session that cannot run Rust got for a UDF
                # whose main row is Rust: same source, its own key.
                title = f"{name} (no-Rust twin)"
            head = [f"- **types**: {', '.join(row.input_types)} -> {row.return_type}"]
            if p.get("top"):
                head.append(p["top"])
            lines += [
                f"## {title}",
                "",
                "```python",
                row.source_text,
                "```",
                "",
                *head,
                "",
            ]
            static_lines = _safe("built-in static (AST)", lambda r=row: _builtin_result(r))
            # No perf suffix on an error or a decline: a stale CSV must not put
            # a happy rate next to a rewrite that was not actually produced.
            if p.get("static") and not any(x in static_lines[0] for x in ("error:", "declined")):
                static_lines[0] += f" — {p['static']}"
            lines += static_lines
            bytecode_lines = _safe("bytecode → SQL", lambda r=row: _bytecode_result(r))
            if p.get("bytecode") and not any(x in bytecode_lines[0] for x in ("error:", "declined")):
                bytecode_lines[0] += f" — {p['bytecode']}"
            lines += bytecode_lines
            java_eval_lines = _safe("bytecode → Java eval", lambda r=row: _java_eval_result(r))
            if p.get("java_eval") and not any(x in java_eval_lines[0] for x in ("error:", "declined")):
                java_eval_lines[0] += f" — {p['java_eval']}"
            lines += java_eval_lines
            ai_lines = _ai_section(row)
            if p.get("ai") and row.status == "success":
                ai_lines[0] += f" — {p['ai']}"
            lines += ai_lines
            lines.append("")
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)
    _write_doc(out_path, lines)
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


KIND_LABELS = {"catalyst": "SQL", "java_udf": "Java", "rust": "Rust"}


def _load_targets(targets_dir: str) -> dict[str, dict[str, tuple[CacheRow, dict[str, str]]]]:
    """kind -> udf_key -> (row, its bench_catalog CSV line), from fill_targets.py's output."""
    out: dict[str, dict[str, tuple[CacheRow, dict[str, str]]]] = {}
    for kind in KIND_LABELS:
        path = Path(targets_dir) / f"{kind}.sqlite"
        if not path.exists():
            continue
        bench: dict[str, dict[str, str]] = {}
        csv_path = Path(targets_dir) / f"{kind}_bench.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                bench = {r["udf_key"]: r for r in csv.DictReader(f)}
        # bench_catalog's Comet session queues "+no-rust" twins for misses; not a UDF of its own.
        out[kind] = {
            r.udf_key: (r, bench.get(r.udf_key, {}))
            for r in _all_rows(SqliteCatalog(path))
            if not r.spark_version.endswith(NO_RUST_SUFFIX)
        }
    return out


def _bench_suffix(kind: str, b: dict[str, str]) -> str:
    parts = []
    if b.get("transpiled_rows_per_second"):
        where = ", Comet native" if kind == "rust" else ""
        speedup = (
            f"{b['transpiled_speedup_vs_python']}x vs python"
            if b.get("transpiled_speedup_vs_python")
            else "no python baseline"
        )
        parts.append(f"{_fmt_rate(b['transpiled_rows_per_second'])} rows/s ({speedup}{where})")
    if b.get("comet_rows_per_second") and kind != "rust":
        parts.append(
            f"comet {_fmt_rate(b['comet_rows_per_second'])} rows/s "
            f"({b.get('comet_speedup_vs_python') or '?'}x vs python)"
        )
    return " · ".join(parts)


def _target_section(kind: str, entry: tuple[CacheRow, dict[str, str]] | None) -> list[str]:
    label = KIND_LABELS[kind]
    if entry is None:
        return [f"- **{label}**: not attempted"]
    row, bench = entry
    lines = _ai_section(row)
    lines[0] = lines[0].replace("**AI (", f"**{label} (", 1).replace(")**: failed:", ")**: rejected:", 1)
    if infra_failure(row):
        lines[0] = lines[0].replace(")**: rejected:", ")**: unavailable:", 1)
    if row.status == "success":
        lines[0] += " — accepted"
        if suffix := _bench_suffix(kind, bench):
            lines[0] += f" — {suffix}"
        else:
            lines[0] += " — not benched (no sample, or not served in a plain session)"
    return [*lines, ""] if lines and lines[-1] == "```" else lines


def _verdict(entry: tuple[CacheRow, dict[str, str]] | None) -> str:
    if entry is None:
        return "-"
    row, bench = entry
    if infra_failure(row):
        return "unavailable"
    if row.status != "success":
        return "rejected" if row.status == "failed" else row.status
    rate = bench.get("transpiled_rows_per_second")
    return f"{_fmt_rate(rate)}/s" if rate else "accepted"


def dump_targets(spark, targets_dir: str, out_path: str, backend: str) -> int:
    """One section per UDF: python, the two non-AI transpilers, then SQL / Java / Rust each."""
    by_kind = _load_targets(targets_dir)
    if not any(by_kind.values()):
        print(f"no per-target catalogs under {targets_dir}", file=sys.stderr)
        return 1
    bad = _bad_backend_rows([r for rows in by_kind.values() for r, _ in rows.values()], backend)
    if bad:
        print(f"rows not produced by backend {backend!r}: {len(bad)}", file=sys.stderr)
        return 1
    keys: dict[str, CacheRow] = {}
    for rows in by_kind.values():
        for key, (row, _) in rows.items():
            keys.setdefault(key, row)
    ordered = sorted(keys.values(), key=lambda r: (_function_name(r.source_text), r.udf_key))
    tally = {
        KIND_LABELS[k]: dict(
            Counter("accepted" if r.status == "success" else _verdict((r, {})) for r, _ in rows.values())
        )
        for k, rows in by_kind.items()
    }
    lines = [
        "# What transpiles to what",
        "",
        f"Generated by `examples/dump_results.py --targets-dir` from catalogs filled by "
        f"`examples/fill_targets.py` with backend {backend!r}, against Spark {spark.version} "
        f"on CPython {platform.python_version()}.",
        f"{len(ordered)} UDFs. Per rewrite kind: {tally}.",
        "Regenerate with `examples/regen_targets_doc.sh`.",
        "",
        "Every UDF is asked for each rewrite kind separately (SQL, Java, Rust), so one",
        "can be accepted while another is rejected; a rejection carries the reason and",
        "the rejected candidate when the model produced one. Unavailable means the",
        "backend failed before a candidate could be assessed. Accepted means the",
        "Hypothesis gate and the reconstruction smoke test passed. String inputs include",
        "generated non-null values, as in the demos. Perf is the local single-node",
        "`bench_catalog.py` run over each kind's own catalog (rows/s,",
        "speedup vs the interpreted python UDF); Rust only runs inside Comet, so its",
        "number is Comet's. No perf = not benched, or python raises on the bench sample.",
        "Java listings retain the model's source; Janino fallback can accept generic",
        "declarations that the JDK compiler rejects.",
        "",
        "| UDF | python | SQL | Java | Rust |",
        "|---|---|---|---|---|",
    ]
    for row in ordered:
        name = _function_name(row.source_text) or row.udf_key[:12]
        py = ""
        for kind in KIND_LABELS:
            b = by_kind.get(kind, {}).get(row.udf_key, (None, {}))[1]
            if b.get("python_rows_per_second"):
                py = f"{_fmt_rate(b['python_rows_per_second'])}/s"
                break
        cells = [_verdict(by_kind.get(k, {}).get(row.udf_key)) for k in KIND_LABELS]
        lines.append(f"| [{name}](#{name.lower()}) | {py or '-'} | " + " | ".join(cells) + " |")
    lines.append("")
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    try:
        for row in ordered:
            name = _function_name(row.source_text) or row.udf_key[:12]
            entries = {k: by_kind.get(k, {}).get(row.udf_key) for k in KIND_LABELS}
            lines += [
                f"## {name}",
                "",
                "```python",
                row.source_text,
                "```",
                "",
                f"- **types**: {', '.join(row.input_types)} -> {row.return_type}",
            ]
            for _, b in filter(None, entries.values()):
                if b.get("python_rows_per_second"):
                    lines.append(
                        "- **perf (catalog bench)**: interpreted python "
                        f"{_fmt_rate(b['python_rows_per_second'])} rows/s"
                    )
                    break
            lines.append("")
            lines += _safe("built-in static (AST)", lambda r=row: _builtin_result(r))
            lines += _safe("bytecode → SQL", lambda r=row: _bytecode_result(r))
            for kind in KIND_LABELS:
                lines += _target_section(kind, entries[kind])
            lines.append("")
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)
    _write_doc(out_path, lines)
    print(f"wrote {out_path} ({len(ordered)} UDFs)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default="target/examples_transpiled.sqlite", help="catalog sqlite path")
    parser.add_argument("--out", default="docs/EXAMPLES_TRANSPILED.md", help="markdown output path")
    parser.add_argument(
        "--targets-dir", help="dir of per-kind catalogs from fill_targets.py: dump SQL/Java/Rust side by side"
    )
    parser.add_argument(
        "--backend",
        help="backend every AI row came from (default: $AI_UDF_TRANSPILE_BACKEND, else auto)",
    )
    parser.add_argument(
        "--perf-csv",
        action="append",
        default=[],
        metavar="PATH",
        help="bench CSV to pull per-arm perf bullets from; repeatable",
    )
    args = parser.parse_args()
    if args.backend:
        os.environ["AI_UDF_TRANSPILE_BACKEND"] = args.backend
    try:
        args.backend = ai_backend()
    except BackendError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.targets_dir:
        spark = (
            SparkSession.builder.master("local[1]")
            .appName("ai-udf-transpile-dump")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.ansi.enabled", "true")
            .getOrCreate()
        )
        try:
            return dump_targets(spark, args.targets_dir, args.out, args.backend)
        finally:
            spark.stop()
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
