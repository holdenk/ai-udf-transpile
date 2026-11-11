# SPDX-License-Identifier: Apache-2.0
"""Real pipeline UDF benchmark: interpreted python vs whatever the AI chain serves.

The functions are the scalar text UDFs of Fight Health Insurance's
training-data pipeline (ai_udf_transpile/backends/fhi_udfs.py): verdict
parsing, entity cleanup, the findings regex extractors, and the regex PII
scrubber. Nobody wrote them to be transpiled, and every one is regex or a
line loop over free text, so this is the bench where the rewrite has to get
Python's `re`, `strip()` and `splitlines()` right in Catalyst SQL or Java.

Two arms per function, over synthetic rows shaped like the real columns
(verdict blocks, reviewer findings, a letter with PII in it):

  python      the UDF as an interpreted Python UDF (transpilePyUDFs=false)
  transpiled  the UDF after AI transpilation (inline worker), Catalyst SQL
              or a Java UDF -- the backend picks; the CSV says which

A function the backend declines, or whose rewrite Hypothesis rejects, keeps
serving python: it gets a python number and no transpiled one. So does a
rewrite that verified but that Spark will not serve: each arm's physical
plan is checked before it is timed, and a "transpiled" arm that still has a
python eval node is reported as not_served (Spark drops a rewrite with a
higher-order-function lambda in it and runs the python instead, silently).

Strings are the values under test, so partyOnStrings is on by default: the
gate generates arbitrary text (unicode whitespace, odd line terminators) on
top of the seeded domain rows. --sampled-strings-only turns that off and
verifies on the seeded rows alone -- a weaker gate. The CSV's string_gate
column says which gate this run verified a rewrite under, or `cached` when
the rewrite was already in $AI_UDF_TRANSPILE_SQLITE and this run verified
nothing (the catalog does not record the gate a row was earned under).

  export SPARK_HOME=~/spark
  python examples/bench_fhi_udfs.py [rows] [--rounds N] [--csv PATH]
  python examples/bench_fhi_udfs.py --only fhi_deidentify_text --sampled-strings-only
"""

from __future__ import annotations

import argparse
import csv
import os
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
from ai_udf_transpile.backends.fhi_udfs import FHI_UDFS  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402

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
        ],
        "x",
    )
    + r", '\nREASON: claim ', cast(x as string), ' is supported by the cited evidence.\nJUDGE: VIABLE\n', "
    + _pick(
        ["LEGAL: OK", "LEGAL: UNCHECKED", "LEGAL:", "  LEGAL: ERISA 503 deadline cited", "legal: ok"],
        "x div 6",
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


def frame(spark, name: str, n: int):
    base = spark.range(0, n, numPartitions=min(PARTITIONS, max(1, n))).toDF("x")
    df = base.select(*[F.expr(expr).alias(col) for col, expr in FRAME_EXPRS[name].items()])
    return df.cache() if n > 64 else df


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--csv", metavar="PATH", help="write results CSV")
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME", help="bench just this function (repeatable)"
    )
    parser.add_argument(
        "--sampled-strings-only",
        action="store_true",
        help="verify on the seeded domain rows alone (partyOnStrings off): a weaker gate",
    )
    parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for each fill")
    args = parser.parse_args()
    unknown = [n for n in args.only if n not in FHI_UDFS]
    if unknown:
        parser.error(f"unknown function(s): {', '.join(unknown)}; have {', '.join(FHI_UDFS)}")
    names = args.only or list(FHI_UDFS)
    rows = args.rows

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-fhi-bench")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(PARTITIONS))
        .config("spark.sql.ansi.enabled", "true")
        .config(TRANSPILE_CONF, "false")
        .getOrCreate()
    )
    string_gate = "sampled" if args.sampled_strings_only else "party"
    print(
        f"=== FHI PIPELINE UDF BENCHMARK ({rows:,} rows, min of {args.rounds} rounds, "
        f"string gate: {string_gate}) ===\n"
    )

    funcs = {n: FHI_UDFS[n][0] for n in names}
    rtypes = {n: _parse_datatype_string(FHI_UDFS[n][1]) for n in names}
    cols = {n: list(FRAME_EXPRS[n]) for n in names}
    frames = {n: frame(spark, n, rows) for n in names}
    for df in frames.values():
        df.count()

    # -- Python arm: transpile off, every UDF is interpreted python.
    py_udfs = {n: UserDefinedFunction(funcs[n], rtypes[n]) for n in names}
    for n in names:
        assert not py_udfs[n].transpiled

    # -- Transpiled arm: enable the plugin, let the backend fill each UDF.
    sqlite = os.environ.get("AI_UDF_TRANSPILE_SQLITE") or os.path.join(
        tempfile.mkdtemp(prefix="ai-udf-fhi-"), "cache.sqlite"
    )
    catalog = enable(spark, sqlite_path=sqlite, backend=ai_backend(spark), inline_worker=True)
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
    spark.conf.set(TRANSPILE_CONF, "true")
    conf.set_value(conf.PARTY_ON_STRINGS, "false" if args.sampled_strings_only else "true", spark)
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

    filled: dict[str, tuple] = {}
    for n in names:
        UserDefinedFunction(funcs[n], rtypes[n])  # construction queues the pending row
        filled[n] = wait_for_catalog(catalog, sources[n], version, args.timeout)
        status, error = filled[n][0], filled[n][1]
        if status != "success":
            print(f"no rewrite for {n} ({status}): {error!r}")

    ai_udfs: dict[str, UserDefinedFunction] = {}
    for n in names:
        if filled[n][0] != "success":
            continue
        udf = UserDefinedFunction(funcs[n], rtypes[n])
        if udf.transpiled:
            ai_udfs[n] = udf
        else:
            print(f"{n}: catalog row is success but the UDF was not served -- python only")

    # -- Benchmark: interleave arms so warmup is fair. The plans are checked
    # first: a "transpiled" number that quietly ran in a python worker would
    # be the python number again.
    def runs_python(df) -> bool:
        return "EvalPython" in df._jdf.queryExecution().executedPlan().toString()

    arms: list[tuple[str, str, object]] = []
    not_served: set[str] = set()
    for n in names:
        py_df = frames[n].select(py_udfs[n](*cols[n]))
        assert runs_python(py_df), f"{n}: python arm has no python eval node"
        arms.append(("python", n, py_df))
        if n in ai_udfs:
            tx_df = frames[n].select(ai_udfs[n](*cols[n]))
            if runs_python(tx_df):
                not_served.add(n)
                print(f"{n}: rewrite verified, but Spark still plans the python UDF -- python only")
                continue
            arms.append(("transpiled", n, tx_df))
    for _arm, _n, arm_df in arms:  # warmup
        noop_seconds(arm_df)
    secs: dict[tuple[str, str], float] = {}
    for _ in range(args.rounds):
        for arm, n, arm_df in arms:
            secs[(arm, n)] = min(secs.get((arm, n), float("inf")), noop_seconds(arm_df))

    header = f"{'function':<30} {'python':>10} {'transpiled':>10} {'tx/py':>8}  served as"
    print(header)
    print("-" * len(header))
    out = []
    for n in names:
        status, _error, backend, model, kind, attempts = filled[n]
        if n in not_served:
            status = "not_served"
        verified_now = attempts is not None and attempts != attempts_before[n]
        py_s = secs[("python", n)]
        tx_s = secs.get(("transpiled", n))
        speedup = round(py_s / tx_s, 1) if tx_s else ""
        served = f"{kind} ({backend}/{model})" if tx_s else f"python ({status})"
        print(
            f"{n:<30} {py_s:>9.2f}s {(f'{tx_s:.2f}s' if tx_s else '-'):>10} "
            f"{(f'{speedup}x' if tx_s else '-'):>8}  {served}"
        )
        out.append(
            {
                "function": n,
                "rows": rows,
                "python_rows_per_second": round(rows / py_s),
                "transpiled_rows_per_second": round(rows / tx_s) if tx_s else "",
                "transpiled_speedup_vs_python": speedup,
                "comet_rows_per_second": "",
                "comet_speedup_vs_python": "",
                "status": status,
                "target_kind": kind or "",
                "backend": backend or "",
                "model": model or "",
                "bench": "fhi",
                "string_gate": string_gate if verified_now else "cached",
            }
        )

    if args.csv:
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
