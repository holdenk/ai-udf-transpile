# SPDX-License-Identifier: Apache-2.0
"""Multi-model demo: run UDFs through every available AI backend, show
agreement/divergence, success rates, and pick the fastest rewrite.

Shows which models produce the same SQL, which differ, and which decline.

  export SPARK_HOME=~/spark
  python examples/demo_multi_model.py         # $AI_UDF_TRANSPILE_BACKEND, else auto
  python examples/demo_multi_model.py --all   # every AI CLI on PATH
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from ai_udf_transpile.backends import ai_backend, get_backend  # noqa: E402
from ai_udf_transpile.backends.base import BackendDecline, BackendError  # noqa: E402
from ai_udf_transpile.backends.fake import (  # noqa: E402
    both_positive,
    clean_phone,
    greet,
    is_none_branch,
    math_floor,
    mod_bucket,
    plus_one,
    snake_case,
    title_case,
)
from ai_udf_transpile.benchmark import benchmark_all  # noqa: E402
from ai_udf_transpile.catalog.sqlite import SqliteCatalog  # noqa: E402
from ai_udf_transpile.keys import canonical_source_from_func  # noqa: E402
from ai_udf_transpile.model_stats import collect_stats, format_stats_table  # noqa: E402
from ai_udf_transpile.sandbox import write_sandbox  # noqa: E402
from ai_udf_transpile.targets import TranspileJob  # noqa: E402

# (name, func, return_type_str, input_type_strs, input_df_key)
UDFS = [
    ("plus_one", plus_one, "bigint", ["bigint"], "ints"),
    ("is_none_branch", is_none_branch, "bigint", ["bigint"], "ints"),
    ("both_positive", both_positive, "boolean", ["bigint", "bigint"], "pairs"),
    ("greet", greet, "string", ["string"], "strs"),
    ("mod_bucket", mod_bucket, "bigint", ["bigint", "bigint"], "pairs"),
    ("snake_case", snake_case, "string", ["string"], "strs"),
    ("title_case", title_case, "string", ["string"], "strs"),
    ("clean_phone", clean_phone, "string", ["string"], "strs"),
    ("math_floor", math_floor, "bigint", ["double"], "dbls"),
]

# Real AI backends to try (name, binary).
LIVE_BACKENDS = [
    ("claude", "claude"),
    ("codex", "codex"),
    ("cursor", "agent"),
    ("pi", "pi"),
    ("coco", "cortex"),
]


def _available_backends() -> list[str]:
    """Return backend names whose CLI binary is on PATH."""
    return [name for name, bin in LIVE_BACKENDS if shutil.which(bin)]


def _job_for(func, input_types: list[str], return_type: str) -> TranspileJob:
    src = canonical_source_from_func(func)
    params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    return TranspileJob(
        udf_key="demo",
        source_text=src,
        param_names=params,
        input_types=input_types,
        input_categories=["numeric"] * len(params),
        return_type=return_type,
        captures={},
    )


def _try_backend(backend, job, sandbox):
    """Run one backend in its sandbox; return (name, model, result) or None."""
    name = getattr(backend, "name", "?")
    try:
        result = backend.run(job, sandbox)
        model = result.model or f"{name}-unknown"
        return (name, model, result)
    except BackendDecline:
        return None
    except BackendError:
        return None


def _short(text, width=72):
    if text is None:
        return "(none)"
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000, help="benchmark rows")
    parser.add_argument("--all", action="store_true", help="every AI CLI on PATH, not just one")
    args = parser.parse_args()

    backend_names = _available_backends() if args.all else [ai_backend()]
    if not backend_names:
        print("No AI CLIs on PATH")
        return 1

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ai-udf-transpile-multi-model")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )

    print(f"=== MULTI-MODEL TRANSPILATION ({', '.join(backend_names)}) ===\n")

    # Cached input DataFrames for benchmarking.
    n = args.rows
    ints = spark.range(0, n, numPartitions=16).toDF("x").cache()
    strs = ints.select(F.col("x").cast("string").alias("s")).cache()
    pairs = ints.select(F.col("x").alias("a"), ((F.col("x") % 99) + 1).alias("b")).cache()
    dbls = ints.select(F.col("x").cast("double").alias("d")).cache()
    for df in (ints, strs, pairs, dbls):
        df.count()
    input_dfs = {
        "ints": (ints, ["x"]),
        "strs": (strs, ["s"]),
        "pairs": (pairs, ["a", "b"]),
        "dbls": (dbls, ["d"]),
    }

    all_candidates: list[tuple[str, str, str, str]] = []  # (udf_name, backend, model, sql)

    with tempfile.TemporaryDirectory(prefix="ai-udf-mm-") as tmp:
        root = Path(tmp)
        for name, func, rtype, itypes, df_key in UDFS:
            job = _job_for(func, itypes, rtype)
            print(f"--- {name} ---")
            candidates: list[tuple[str, str, str]] = []  # (backend, model, sql)

            for bname in backend_names:
                backend = get_backend(bname, spark)
                sub = root / f"{name}_{bname}"
                write_sandbox(sub, job)
                got = _try_backend(backend, job, sub)
                if got is None:
                    print(f"  {bname:<10} DECLINED")
                    continue
                _, model, result = got
                sql = result.sql or "(java)"
                print(f"  {bname:<10} model={model:<20} {_short(sql)}")
                candidates.append((bname, model, sql))
                all_candidates.append((name, bname, model, sql))

            if not candidates:
                print("  (all backends declined)\n")
                continue

            sqls = [c[2] for c in candidates if c[2] != "(java)"]
            unique = len(set(sqls))
            if unique == 1:
                print(f"  >> all {len(candidates)} model(s) AGREE\n")
            else:
                print(f"  >> {len(candidates)} models, {unique} distinct SQL(s) -- DIVERGENCE\n")

    # --- success rate table, from the catalog the other examples filled ---
    shared = os.environ.get("AI_UDF_TRANSPILE_SQLITE")
    if shared and os.path.exists(shared):
        print(f"=== MODEL SUCCESS RATES ({shared}) ===")
        print(format_stats_table(collect_stats(SqliteCatalog(shared))))
        print()

    # --- benchmark candidates and pick best ---
    print(f"=== BENCHMARK: choose best rewrite ({n:,} rows, min of 3 rounds) ===")
    for name, _func, _rtype, _itypes, df_key in UDFS:
        cands = [(bn, md, sql) for (n2, bn, md, sql) in all_candidates if n2 == name]
        if not cands:
            continue
        df, cols = input_dfs[df_key]
        results = benchmark_all(spark, cands, df, cols)
        best = results[0]
        print(f"  {name}:")
        for r in results:
            marker = " <-- BEST" if r is best else ""
            print(f"    {r.backend:<10} {r.model:<20} {r.rows_per_second:>12,.0f} rows/s{marker}")
        print()

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
