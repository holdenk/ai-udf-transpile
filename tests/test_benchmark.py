# SPDX-License-Identifier: Apache-2.0
"""benchmark: choose_best and benchmark_all."""

from __future__ import annotations

import pytest

from ai_udf_transpile.benchmark import benchmark_all, choose_best

pytestmark = pytest.mark.spark


def test_choose_best_returns_a_valid_candidate(spark):
    df = spark.range(0, 100_000, numPartitions=8).toDF("x").cache()
    df.count()
    candidates = [
        ("fake", "model-a", "_udf_param_0 + 1"),
        ("fake", "model-b", "_udf_param_0 + 2"),
    ]
    best = choose_best(spark, candidates, df, ["x"])
    assert best.backend == "fake"
    assert best.model in ("model-a", "model-b")
    assert best.rows_per_second > 0


def test_benchmark_all_returns_sorted(spark):
    df = spark.range(0, 100_000, numPartitions=8).toDF("x").cache()
    df.count()
    # One lightweight, one deliberately heavy (repeated expr).
    candidates = [
        ("fake", "light", "_udf_param_0 + 1"),
        ("fake", "heavy", "repeat(_udf_param_0 + 1, 100)"),
    ]
    results = benchmark_all(spark, candidates, df, ["x"])
    assert len(results) == 2
    # Sorted by rows_per_second descending
    assert results[0].rows_per_second >= results[1].rows_per_second
    # The lightweight one should be faster than the one that repeats 100x.
    light = [r for r in results if r.model == "light"][0]
    heavy = [r for r in results if r.model == "heavy"][0]
    assert light.rows_per_second > heavy.rows_per_second


def test_benchmark_all_skips_unparseable(spark):
    df = spark.range(0, 1000, numPartitions=4).toDF("x").cache()
    df.count()
    candidates = [
        ("fake", "model-a", "_udf_param_0 + 1"),
        ("fake", "model-bad", "this is not sql !!!"),
    ]
    results = benchmark_all(spark, candidates, df, ["x"])
    assert len(results) == 2
    bad = [r for r in results if r.model == "model-bad"][0]
    assert bad.rows_per_second == 0.0
    good = [r for r in results if r.model == "model-a"][0]
    assert good.rows_per_second > 0


def test_choose_best_raises_on_all_unparseable(spark):
    df = spark.range(0, 100, numPartitions=2).toDF("x").cache()
    df.count()
    candidates = [("fake", "model-bad", "!!! not sql !!!")]
    with pytest.raises(ValueError):
        choose_best(spark, candidates, df, ["x"])
