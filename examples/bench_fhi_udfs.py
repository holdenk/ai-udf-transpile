# SPDX-License-Identifier: Apache-2.0
"""Real pipeline UDF benchmark: when is a faithful rewrite a regression?

The functions are the scalar text UDFs of Fight Health Insurance's
training-data pipeline (ai_udf_transpile/backends/fhi_udfs.py): verdict
parsing, entity cleanup, the findings regex extractors, and the regex PII
scrubber. Nobody wrote them to be transpiled, and every one is regex or a
line loop over free text, so this is the bench where the rewrite has to get
Python's `re`, `strip()` and `splitlines()` right in Catalyst SQL or Java.

Getting them right is not the same as getting them fast. Each function has
two pinned rewrites (backends/fhi_rewrites.py), both faithful:

  python   the UDF as an interpreted Python UDF (transpilePyUDFs=false)
  naive    verbatim what an AI backend served on its first verified
           attempt -- for most of these, slower than the python
  tuned    hand-written with what the naive one turned out to cost in mind
           (native trim instead of a regex strip, a Java line loop instead
           of regexes over the whole text, an ASCII fast path)
  ai       (--ai) whatever the AI chain serves today, through the plugin

and each arm runs over the same rows twice: once in a column Spark must
treat as nullable and once in a column it can prove non-null. That is one
of the regression conditions on its own -- a rewrite that opens with
`CASE WHEN x IS NULL THEN NULL` loses Spark's subexpression sharing on a
nullable column and gets it back on a non-nullable one, where the optimizer
deletes the guard.

Next to every measured ratio the table prints what
ai_udf_transpile.regress guessed from the two texts alone (ok / at-risk /
likely-regression / not-served), so each run is also a check of that guess.

A rewrite that verified but that Spark will not serve gets no number: an
expression with a higher-order-function lambda in it is dropped at plan
time and the python UDF runs instead, silently. Each arm's physical plan is
checked before it is timed.

With --ai, strings are the values under test, so partyOnStrings is on by
default; --sampled-strings-only verifies on the seeded rows alone. The
string_gate column says which gate this run verified a rewrite under, or
`cached` when the rewrite was already in $AI_UDF_TRANSPILE_SQLITE and this
run verified nothing.

  export SPARK_HOME=~/spark
  python examples/bench_fhi_udfs.py [rows] [--rounds N] [--csv PATH]
  python examples/bench_fhi_udfs.py --ai                     # + the live AI arm
  python examples/bench_fhi_udfs.py --only fhi_clean_health_history --nullability both
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Workers must run the driver's interpreter (cloudpickle shares bytecode);
# run_demos.sh pins this for the other benches, so stay correct standalone.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import _parse_datatype_string  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.backends import ai_backend  # noqa: E402
from ai_udf_transpile.backends.fhi_rewrites import VARIANTS, fhi_rewrite  # noqa: E402
from ai_udf_transpile.backends.fhi_udfs import FHI_UDFS  # noqa: E402
from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402
from ai_udf_transpile.regress import NOT_SERVED, regression_risk  # noqa: E402
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult  # noqa: E402

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"
PARTITIONS = os.cpu_count() or 16


def _pick(options: list[str], by: str) -> str:
    """SQL: one of `options`, cycling on the integer expression `by`."""
    quoted = ", ".join("'" + o.replace("\\", "\\\\").replace("'", "\\'") + "'" for o in options)
    return f"element_at(array({quoted}), cast(({by}) % {len(options)} as int) + 1)"


# Synthetic rows shaped like the real columns. Nothing here is pipeline data:
# the verdict block is the layout appeal_grounding.render_verdict writes, the
# findings sentences are in the register of public IMR decisions, and the
# letter carries one of each thing the scrubber looks for.
_VERDICT_EXPR = (
    "concat("
    + _pick(
        [
            "VERDICT: VIABLE",
            "- **VERDICT: NOT_VIABLE**",
            "1. VERDICT: VIABLE (dates unverified)",
            "Verdict: non-viable",
            "VERDICT: VIABLE or NOT_VIABLE",
            "SUMMARY: no decision line",
            # Two spellings of one answer: the seed that catches a rewrite
            # comparing matched words instead of what they mean.
            "VERDICT: UNVIABLE and NON VIABLE",
        ],
        "x",
    )
    + r", '\nREASON: claim ', cast(x as string), ' is supported by the cited evidence.\nJUDGE: VIABLE\n', "
    + _pick(
        ["LEGAL: OK", "LEGAL: UNCHECKED", "LEGAL:", "  LEGAL: ERISA 503 deadline cited", "legal: ok"],
        "x div 7",
    )
    + ")"
)
_HISTORY_EXPR = (
    "CASE cast(x % 6 as int)"
    " WHEN 0 THEN 'None.' WHEN 1 THEN 'UNKNOWN' WHEN 2 THEN ' none ' WHEN 3 THEN ''"
    " WHEN 4 THEN concat('  Type 2 diabetes since ', cast(1990 + x % 30 as string), '; on metformin.  ')"
    " ELSE concat('Prior lumbar fusion (L4-L5), case ', cast(x as string)) END"
)
_FINDINGS_EXPR = (
    "concat("
    + _pick(
        [
            "The patient requested coverage for 12 sessions of physical therapy."
            " The reviewer found the request reasonable.",
            "At issue in this case is whether Humira 40mg is medically necessary for the patient.",
            "The Health Plan determined the lumbar MRI was not medically necessary. The reviewer disagrees.",
            "The patient was admitted to the hospital for acute pancreatitis."
            " The stay was denied as not necessary.",
            "The treating physicians recommended inpatient rehabilitation following surgery.",
            "The reviewer upheld the denial based on the plan documents",
        ],
        "x",
    )
    + ", ' Case ', cast(x as string), '.')"
)
_ENTITY_EXPR = (
    "concat("
    + _pick(
        [
            "**The diagnosis is** <answer>Type 2 diabetes</answer>",
            "Treatment: physical therapy ---",
            "The insurance company is Acme Health",
            "<answer>**Lumbar MRI**</answer>",
            "Procedure: knee arthroscopy",
            "UNKNOWN",
        ],
        "x",
    )
    + ", ' ', cast(x % 1000 as string))"
)
_LETTER_EXPR = (
    "concat("
    r"'Dear Appeals Department,\n\nI am writing on behalf of {{FIRST_NAME}} {{LAST_NAME}} regarding claim ', "
    "cast(1000000000000 + x as string), "
    "'. You can reach me at (415) 555-', lpad(cast(x % 10000 as string), 4, '0'), "
    r"' or jo', cast(x % 97 as string), '@example.org.\nMember ID: AB', "
    "cast(10000000 + x % 89999999 as string), "
    r"'\nDOB: 0', cast(1 + x % 9 as string), '/1', cast(x % 10 as string), "
    "'/19', cast(60 + x % 40 as string), "
    r"'\nSSN: ', cast(100 + x % 900 as string), '-', cast(10 + x % 90 as string), '-', "
    "lpad(cast(x % 10000 as string), 4, '0'), "
    r"'\nAddress: ', cast(1 + x % 999 as string), ' Main Street\n\n', "
    "'The denial states that the requested treatment is not medically necessary. [REDACTED] My treating "
    "physician disagrees: the treatment is the standard of care for my condition, I have tried and failed "
    "the alternatives the plan lists, and the plan documents cover it. Please reverse the denial and approve "
    r"the claim.\n\nSincerely,\n{{NAME}}'"
    ")"
)

# name -> {column: SQL expression over the bigint column x}
FRAME_EXPRS: dict[str, dict[str, str]] = {
    "fhi_verdict_has_legal_check": {"verdict_text": _VERDICT_EXPR},
    "fhi_parse_verdict": {"text": _VERDICT_EXPR},
    "fhi_clean_health_history": {"text": _HISTORY_EXPR},
    "fhi_treatment_from_findings": {
        "findings": _FINDINGS_EXPR,
        "tc": _pick(["Pharmacy", "Other", "Mental Health", "Orthopedics"], "x"),
        "tsc": _pick(["Other", "Outpatient", "Imaging"], "x"),
    },
    "fhi_diagnosis_from_findings": {
        "findings": _FINDINGS_EXPR,
        "dc": _pick(["Digestive System", "Other", "Mental Disorder", "Orthopedic"], "x"),
        "dsc": _pick(["Other", "Pancreatitis", "Depression"], "x"),
    },
    "fhi_normalize_entity_text": {"text": _ENTITY_EXPR},
    "fhi_deidentify_text": {"text": _LETTER_EXPR},
}


# Same values, different static nullability: Spark cannot prove nullif(...)
# non-null and can prove coalesce(..., '') non-null. No row is actually NULL
# in either, so the two columns differ only in what the optimizer may assume.
FLAVORS = {"nullable": "nullif({e}, '\\u0000')", "nonnull": "coalesce({e}, '')"}

# The regex-heavy functions cost microseconds a row in every arm; scale
# their frames down so one function does not own the wall clock.
ROW_SCALE = {
    "fhi_deidentify_text": 0.1,
    "fhi_treatment_from_findings": 0.25,
    "fhi_normalize_entity_text": 0.25,
}


def frame(spark, name: str, n: int, flavor: str = "nullable"):
    base = spark.range(0, n, numPartitions=min(PARTITIONS, max(1, n))).toDF("x")
    wrap = FLAVORS[flavor]
    df = base.select(*[F.expr(wrap.format(e=expr)).alias(col) for col, expr in FRAME_EXPRS[name].items()])
    return df.cache() if n > 64 else df


def rewrite_column(spark, result: TranspileResult, cols: list[str], rtype, tag: str):
    """A pinned rewrite as a Column over `cols`, the way the plugin would
    reconstruct it: SQL with its placeholders bound, or a registered Java UDF."""
    if result.kind == KIND_CATALYST:
        sql = result.sql
        for i, col in enumerate(cols):
            sql = re.sub(rf"_udf_param_{i}(?!\d)", f"`{col}`", sql)
        return F.expr(sql).cast(rtype)
    compiled = compile_java(spark, result.java_source, result.class_name)
    fname = verify_function_name(f"fhi_{tag}", compiled.jar_bytes)
    register_java_udf(spark, fname, compiled.class_name, compiled.jar_bytes, rtype, janino=compiled.janino)
    return F.expr(f"{fname}({', '.join(f'`{c}`' for c in cols)})").cast(rtype)


def noop_seconds(df) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


_ROW_SQL = (
    "SELECT status, error, backend, model, target_kind, attempt_count FROM cache "
    "WHERE source_text = ? AND spark_version = ? ORDER BY updated_at DESC"
)


def catalog_row(catalog, source_text, spark_version):
    """This session's row for a function. The key includes the Spark version,
    so a shared catalog can hold the same source under another key; that row
    must not answer for the one this session queued."""
    with catalog._lock:
        return catalog._conn.execute(_ROW_SQL, (source_text, spark_version)).fetchone()


def wait_for_catalog(catalog, source_text, spark_version, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = catalog_row(catalog, source_text, spark_version)
        if row and row[0] in {"success", "failed"}:
            return row
        time.sleep(0.1)
    return ("timeout", None, None, None, None, None)


def fill_from_ai(spark, args, names, funcs, rtypes, cols):
    """The live arm: enable the plugin, let the backend fill each UDF, and
    return {name: (served udf or None, status, kind, backend, model, string_gate)}."""
    sqlite = os.environ.get("AI_UDF_TRANSPILE_SQLITE") or os.path.join(
        tempfile.mkdtemp(prefix="ai-udf-fhi-"), "cache.sqlite"
    )
    catalog = enable(spark, sqlite_path=sqlite, backend=ai_backend(spark), inline_worker=True)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    spark.conf.set(TRANSPILE_CONF, "true")
    conf.set_value(conf.PARTY_ON_STRINGS, "false" if args.sampled_strings_only else "true", spark)
    string_gate = "sampled" if args.sampled_strings_only else "party"
    version = str(spark.version)
    sources = {n: canonical_source_from_func(funcs[n]) for n in names}
    # A row already terminal in a reused catalog is served as it is: this run
    # verifies nothing for it, under either gate.
    attempts_before = {n: (catalog_row(catalog, sources[n], version) or (None,) * 6)[5] for n in names}

    # -- Seed samples: run each UDF once on domain-shaped rows so Hypothesis
    # checks real verdict blocks and letters, not just generated text. The
    # worker polls slowly meanwhile so a pending row is never claimed before
    # its samples land.
    conf.set_value(conf.POLL_INTERVAL, "60", spark)
    for n in names:
        seed_udf = UserDefinedFunction(funcs[n], rtypes[n])
        frame(spark, n, 48).select(seed_udf(*cols[n])).collect()
    conf.set_value(conf.POLL_INTERVAL, "0.2", spark)

    out = {}
    for n in names:
        UserDefinedFunction(funcs[n], rtypes[n])  # construction queues the pending row
        status, error, backend, model, kind, attempts = wait_for_catalog(
            catalog, sources[n], version, args.timeout
        )
        udf = None
        if status != "success":
            print(f"no rewrite for {n} ({status}): {error!r}")
        else:
            udf = UserDefinedFunction(funcs[n], rtypes[n])
            if not udf.transpiled:
                print(f"{n}: catalog row is success but the UDF was not offered a rewrite -- python only")
                udf = None
        verified_now = attempts is not None and attempts != attempts_before[n]
        out[n] = (
            udf,
            status,
            kind or "",
            backend or "",
            model or "",
            string_gate if verified_now else "cached",
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rows", nargs="?", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--csv", metavar="PATH", help="write results CSV (one row per function, arm and column)"
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME", help="bench just this function (repeatable)"
    )
    parser.add_argument(
        "--nullability",
        choices=["both", *FLAVORS],
        default="both",
        help="which input column to time each arm over (default: both)",
    )
    parser.add_argument("--ai", action="store_true", help="also time whatever the AI chain serves, live")
    parser.add_argument(
        "--sampled-strings-only",
        action="store_true",
        help="with --ai: verify on the seeded domain rows alone (partyOnStrings off): a weaker gate",
    )
    parser.add_argument("--timeout", type=int, default=900, help="with --ai: seconds to wait for each fill")
    args = parser.parse_args()
    unknown = [n for n in args.only if n not in FHI_UDFS]
    if unknown:
        parser.error(f"unknown function(s): {', '.join(unknown)}; have {', '.join(FHI_UDFS)}")
    names = args.only or list(FHI_UDFS)
    flavors = list(FLAVORS) if args.nullability == "both" else [args.nullability]

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-fhi-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")
        # The letter frame is ~700 chars a row; cached, it does not fit the
        # 1g default heap reliably (it OOMed on one run in two).
        .config("spark.driver.memory", os.environ.get("AI_UDF_TRANSPILE_BENCH_DRIVER_MEMORY", "4g"))
        .getOrCreate()
    )
    print(f"=== FHI PIPELINE UDF BENCHMARK ({args.rows:,} rows base, min of {args.rounds} rounds) ===\n")

    funcs = {n: FHI_UDFS[n][0] for n in names}
    rtypes = {n: _parse_datatype_string(FHI_UDFS[n][1]) for n in names}
    cols = {n: list(FRAME_EXPRS[n]) for n in names}

    # -- Python arm: built while transpile is off, so it stays interpreted.
    py_udfs = {n: UserDefinedFunction(funcs[n], rtypes[n]) for n in names}
    for n in names:
        assert not py_udfs[n].transpiled
    live = fill_from_ai(spark, args, names, funcs, rtypes, cols) if args.ai else {}

    def runs_python(df) -> bool:
        return "EvalPython" in df._jdf.queryExecution().executedPlan().toString()

    out = []
    for n in names:
        rows = max(1000, int(args.rows * ROW_SCALE.get(n, 1.0)))
        source = canonical_source_from_func(funcs[n])
        for flavor in flavors:
            df = frame(spark, n, rows, flavor)
            df.count()
            py_df = df.select(py_udfs[n](*cols[n]))
            assert runs_python(py_df), f"{n}: python arm has no python eval node"
            # arm -> (frame or None, status, kind, backend, model, string_gate, static risk)
            arms: dict[str, tuple] = {}
            for variant in VARIANTS:
                result = fhi_rewrite(n, variant)
                if result is None:
                    continue
                risk = regression_risk(source, result)
                if risk.verdict == NOT_SERVED:
                    # As plain SQL this would run; through the transpile hook
                    # Spark keeps the python UDF, so there is no number to give.
                    arms[variant] = (None, "not_served", result.kind, "pinned", "", "", risk)
                    continue
                arm_df = df.select(rewrite_column(spark, result, cols[n], rtypes[n], variant))
                assert not runs_python(arm_df), f"{n}: pinned {variant} arm evaluates python"
                arms[variant] = (arm_df, "success", result.kind, "pinned", "", "", risk)
            if n in live:
                udf, status, kind, backend, model, gate = live[n]
                arm_df = df.select(udf(*cols[n])) if udf is not None else None
                if arm_df is not None and runs_python(arm_df):
                    print(f"{n}: rewrite verified, but Spark still plans the python UDF -- python only")
                    arm_df, status = None, "not_served"
                arms["ai"] = (arm_df, status, kind, backend, model, gate, None)

            timed = [("python", py_df)] + [(a, v[0]) for a, v in arms.items() if v[0] is not None]
            for _arm, arm_df in timed:  # warmup
                noop_seconds(arm_df)
            secs: dict[str, float] = {}
            for _ in range(args.rounds):  # interleaved, so drift hits every arm alike
                for arm, arm_df in timed:
                    secs[arm] = min(secs.get(arm, float("inf")), noop_seconds(arm_df))
            df.unpersist()

            py_s = secs["python"]
            print(f"{n}  [{flavor} column, {rows:,} rows]  python {py_s:.2f}s")
            for arm, (arm_df, status, kind, backend, model, gate, risk) in arms.items():
                tx_s = secs.get(arm)
                speedup = round(py_s / tx_s, 1) if tx_s else ""
                measured = f"{tx_s:6.2f}s {speedup:>5}x" if tx_s else f"{'-':>7} {'(' + status + ')':>6}"
                guess = f"   static guess: {risk}" if risk is not None else ""
                print(f"    {arm:<6} {kind or '-':<9} {measured}{guess}")
                out.append(
                    {
                        "function": n,
                        "arm": arm,
                        "column_nullable": flavor == "nullable",
                        "rows": rows,
                        "python_rows_per_second": round(rows / py_s),
                        "transpiled_rows_per_second": round(rows / tx_s) if tx_s else "",
                        "transpiled_speedup_vs_python": speedup,
                        "comet_rows_per_second": "",
                        "comet_speedup_vs_python": "",
                        "status": status,
                        "target_kind": kind,
                        "backend": backend,
                        "model": model,
                        "bench": "fhi",
                        "string_gate": gate,
                        "static_risk": risk.verdict if risk is not None else "",
                        "static_flags": " ".join(risk.flags) if risk is not None else "",
                    }
                )

    if args.csv and out:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)
        print(f"\nCSV written to {args.csv}")

    spark.conf.set(TRANSPILE_CONF, "false")
    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
