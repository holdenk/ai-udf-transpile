# SPDX-License-Identifier: Apache-2.0
"""Mutation testing of the Hypothesis gate.

For each UDF: the correct rewrite must verify, and every mutant -- a
plausible wrong rewrite of the kind a backend actually produces -- must be
rejected. Each mutant here is killed by a *deterministic* explicit example
(a built-in edge value or a cross-param combination), not by a lucky random
draw, so these tests do not flake: if one starts passing, the example that
used to kill it was lost.

The integer-width and decimal cases also guard the other direction. The gate
used to hand narrow-int and decimal columns values they cannot hold, so a
*correct* rewrite of any int / smallint / tinyint / decimal UDF was rejected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

P0, P1 = "_udf_param_0", "_udf_param_1"

# (id, python source, input types, return type, correct sql, {mutant id: (sql, kill reason)})
CASES = [
    (
        "floor_div",
        "def f(x: int) -> int:\n    return None if x is None else x // 2\n",
        ["bigint"],
        "bigint",
        f"CASE WHEN {P0} < 0 AND {P0} % 2 != 0 THEN {P0} DIV 2 - 1 ELSE {P0} DIV 2 END",
        {
            # Truncates toward zero: -7 // 2 is -4 in python, -3 here.
            "truncating_div": (f"{P0} DIV 2", "mismatch"),
        },
    ),
    (
        "coalesce",
        "def f(x: int, y: int) -> int:\n    return x if x is not None else y\n",
        ["bigint", "bigint"],
        "bigint",
        f"coalesce({P0}, {P1})",
        {
            # (None, n): python returns n.
            "drops_fallback": (P0, "mismatch"),
            # Cross-param combo (0, 1): python returns 0.
            "swapped_args": (f"coalesce({P1}, {P0})", "mismatch"),
        },
    ),
    (
        "strict_and",
        "def f(a: bool, b: bool) -> bool:\n"
        "    if a is None or b is None:\n"
        "        return None\n"
        "    return a and b\n",
        ["boolean", "boolean"],
        "boolean",
        f"CASE WHEN {P0} IS NULL OR {P1} IS NULL THEN NULL ELSE {P0} AND {P1} END",
        {
            # Three-valued logic: NULL AND FALSE is FALSE in SQL, None here.
            "sql_three_valued_and": (f"{P0} AND {P1}", "mismatch"),
        },
    ),
    (
        "int_plus_one_widened",
        "def f(x: int) -> int:\n    return None if x is None else x + 1\n",
        ["int"],
        "bigint",
        f"CAST({P0} AS BIGINT) + 1",
        {
            # int + 1 overflows under ANSI at 2^31 - 1; python's int does not.
            "int_overflow": (f"{P0} + 1", "sql raised"),
        },
    ),
    (
        "tinyint_double",
        "def f(x: int) -> int:\n    return None if x is None else x * 2\n",
        ["tinyint"],
        "int",
        f"CAST({P0} AS INT) * 2",
        {
            # tinyint * tinyint stays tinyint: overflows at the type's own edge
            # (127), which the gate only tries since it began adding each
            # narrow type's edges to the built-in examples.
            "tinyint_overflow": (f"{P0} * CAST(2 AS TINYINT)", "sql raised"),
        },
    ),
    (
        "smallint_negate",
        "def f(x: int) -> int:\n    return None if x is None else -x\n",
        ["smallint"],
        "int",
        f"-CAST({P0} AS INT)",
        {
            # -(-32768) does not fit smallint.
            "smallint_overflow": (f"-{P0}", "sql raised"),
        },
    ),
    (
        "decimal_bankers_round",
        "def f(x):\n"
        "    import decimal\n"
        "    if x is None:\n"
        "        return None\n"
        "    return x.quantize(decimal.Decimal(1), rounding=decimal.ROUND_HALF_EVEN)\n",
        ["decimal(10,2)"],
        "decimal(10,0)",
        f"bround({P0}, 0)",
        {
            # HALF_UP: 2.5 -> 3, python -> 2. 2.5 is a built-in decimal example.
            "half_up_round": (f"round({P0}, 0)", "mismatch"),
            # floor agrees on 2.5 -> 2 and 0.5 -> 0; -0.5 -> -1 vs python 0.
            "floor": (f"floor({P0})", "mismatch"),
        },
    ),
    (
        "decimal_identity",
        "def f(x):\n    return x\n",
        ["decimal(38,18)"],
        "decimal(38,18)",
        P0,
        {
            # Rounds away the scale: 0.50 -> 1 (HALF_UP). At the column's edge
            # the rounded value no longer fits decimal(38,18), and the range
            # check may reject it before the comparison does: either is a kill.
            "drops_scale": (f"CAST({P0} AS DECIMAL(38,0))", "mismatch|does not fit"),
        },
    ),
]


def _check(spark, source, sql, input_types, return_type, max_examples=30):
    return hypothesis_check(
        source_text=source,
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=input_types,
        return_type=return_type,
        spark=spark,
        max_examples=max_examples,
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_correct_rewrite_verifies(spark, case):
    _, source, input_types, return_type, correct, _ = case
    ok, err = _check(spark, source, correct, input_types, return_type)
    assert ok, err


MUTANTS = [
    pytest.param(case, mutant_id, sql, reason, id=f"{case[0]}-{mutant_id}")
    for case in CASES
    for mutant_id, (sql, reason) in case[5].items()
]


@pytest.mark.parametrize("case, mutant_id, sql, reason", MUTANTS)
def test_mutant_rejected(spark, case, mutant_id, sql, reason):
    _, source, input_types, return_type, _, _ = case
    # One random draw: the explicit examples (always run) must do the
    # killing, so a pass here never depends on what Hypothesis happened to draw.
    ok, err = _check(spark, source, sql, input_types, return_type, max_examples=1)
    assert not ok, f"mutant {mutant_id} survived: {sql}"
    # `reason` is one expected substring, or several separated by "|".
    assert any(r in (err or "") for r in reason.split("|")), err
