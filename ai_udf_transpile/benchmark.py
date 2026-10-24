# SPDX-License-Identifier: Apache-2.0
"""Benchmark multiple SQL candidates on sample data and pick the fastest.

Given several AI-generated rewrites for the same UDF, run each on a small
DataFrame and measure rows/sec.  Returns the fastest (sql, model) so the
caller can serve it instead of whichever happened to finish first.

This is the "choose best" half of the multi-model story: demo_multi_model
collects one rewrite per backend, this module benchmarks them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

_PARAM_RE = re.compile(r"_udf_param_(\d+)")


@dataclass(frozen=True)
class CandidateResult:
    backend: str
    model: str
    sql: str
    rows_per_second: float


def _substitute_params(sql: str, cols: list[str]) -> str:
    """Replace _udf_param_N placeholders with actual column names."""

    def repl(m):
        idx = int(m.group(1))
        return cols[idx] if idx < len(cols) else m.group(0)

    return _PARAM_RE.sub(repl, sql)


def _noop_seconds(spark: SparkSession, df: Any) -> float:
    start = time.perf_counter()
    df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - start


def _benchmark_one(spark, backend, model, sql, input_df, cols, rows, rounds):
    """Returns CandidateResult, or None if unparseable."""
    real_sql = _substitute_params(sql, cols)
    try:
        expr = F.expr(real_sql)
        df = input_df.select(expr.alias("out"))
        _noop_seconds(spark, df)  # warmup (also triggers analysis/parse)
    except Exception:
        return None
    best = float("inf")
    for _ in range(rounds):
        best = min(best, _noop_seconds(spark, df))
    rps = rows / best if best > 0 else 0.0
    return CandidateResult(backend, model, sql, rps)


def benchmark_all(
    spark: SparkSession,
    candidates: list[tuple[str, str, str]],
    input_df: Any,
    cols: list[str],
    rounds: int = 3,
) -> list[CandidateResult]:
    """Benchmark each (backend, model, sql) candidate, return all sorted by speed."""
    rows = input_df.count()
    results: list[CandidateResult] = []
    for backend, model, sql in candidates:
        r = _benchmark_one(spark, backend, model, sql, input_df, cols, rows, rounds)
        if r is None:
            results.append(CandidateResult(backend, model, sql, 0.0))
        else:
            results.append(r)
    results.sort(key=lambda r: r.rows_per_second, reverse=True)
    return results


def choose_best(
    spark: SparkSession,
    candidates: list[tuple[str, str, str]],
    input_df: Any,
    cols: list[str],
    rounds: int = 3,
) -> CandidateResult:
    """Benchmark each candidate, return the fastest.  Raises if all unparseable."""
    results = benchmark_all(spark, candidates, input_df, cols, rounds)
    parseable = [r for r in results if r.rows_per_second > 0]
    if not parseable:
        raise ValueError("no benchmarkable candidate (all failed to parse)")
    return results[0]
