# SPDX-License-Identifier: Apache-2.0
"""contains_ingredient: the boolean NULL trap and the LIKE metachar trap.

    @udf(returnType=BooleanType())
    def contains_ingredient(recipe, ingredient):
        if recipe is not None:
            return ingredient in recipe.lower()
        return False

The naive rewrite instr(lower(r), i) > 0 returns NULL (not false) when
recipe is NULL -- in a WHERE clause NULL filters rows exactly like false,
so the bug is invisible until the column is selected or composed. The
LIKE-based rewrite additionally treats % and _ in the ingredient as
wildcards. Both look gucci; neither is.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import BooleanType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable, register_impl
from ai_udf_transpile.backends.fake import contains_ingredient
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

CORRECT_SQL = "coalesce(instr(lower(_udf_param_0), _udf_param_1) > 0, false)"
NAIVE_SQL = "instr(lower(_udf_param_0), _udf_param_1) > 0"  # NULL trap
LIKE_SQL = "lower(_udf_param_0) LIKE concat('%', _udf_param_1, '%')"  # metachar trap

RECIPE_SAMPLES = [
    ["2 cups flour, 1 tsp salt, 100% butter", "salt"],
    ["Salt-free broth", "salt"],  # case-insensitive hit
    ["100% whole wheat", "%"],  # LIKE metacharacter as a literal
    ["eggs_milk", "_"],  # LIKE metacharacter as a literal
    [None, "salt"],  # python False, never NULL
]


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(contains_ingredient),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string", "string"],
        return_type="boolean",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_correct_rewrite_passes(spark):
    ok, err = _check(CORRECT_SQL, spark, samples=RECIPE_SAMPLES)
    assert ok, err


def test_naive_instr_rejected_null_trap(spark):
    # (None, 'salt'): python False, naive SQL NULL. Caught even without
    # samples: built-in string examples put None in the recipe position.
    ok, err = _check(NAIVE_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "")
    assert "None" in (err or "")


def test_like_rewrite_rejected_metachar_trap(spark):
    # ingredient '%': python checks the literal character, LIKE treats it
    # as a wildcard and matches every non-null recipe.
    ok, err = _check(LIKE_SQL, spark, samples=[["100% whole wheat", "%"]])
    assert not ok
    assert "mismatch" in (err or "")


def test_register_impl_rejects_naive_instr(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            contains_ingredient,
            kind="catalyst",
            catalyst_sql=NAIVE_SQL,
            return_type=BooleanType(),
        )


def test_end_to_end_transpile_with_null_recipe(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    df = spark.createDataFrame(
        [(r[0], r[1]) for r in RECIPE_SAMPLES], ["recipe", "ingredient"]
    )
    expected = [contains_ingredient(r[0], r[1]) for r in RECIPE_SAMPLES]

    first = UserDefinedFunction(contains_ingredient, BooleanType())
    assert not first.transpiled
    assert [r[0] for r in df.select(first("recipe", "ingredient")).collect()] == expected

    deadline = time.time() + 60
    second = UserDefinedFunction(contains_ingredient, BooleanType())
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(contains_ingredient, BooleanType())
    assert second.transpiled, "fake backend never landed the coalesce instr rewrite"
    assert [r[0] for r in df.select(second("recipe", "ingredient")).collect()] == expected
    row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
    assert row[0] == "success"
    assert row[1] == CORRECT_SQL
