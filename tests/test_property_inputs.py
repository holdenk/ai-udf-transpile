# SPDX-License-Identifier: Apache-2.0
"""Property tests of the verifier's own input generators (no JVM).

The differential gate is only as good as the inputs it feeds both sides.
An input the declared column type cannot hold is worse than useless: Spark
refuses to build the row, the gate reads that as "sql raised" while python
returned a value, and a *correct* rewrite is rejected. Every int / smallint
/ tinyint / decimal UDF used to fail that way. These tests pin the contract
"every generated or built-in input is a value the column can hold".
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_udf_transpile.verify import (
    _INT_INPUT_RANGES,
    _decimal_bound,
    _decimal_examples_for,
    _decimal_shape,
    _int_examples_for,
    _representable,
    _strategy_for,
)

INT_TYPES = sorted(_INT_INPUT_RANGES)


def _exact_at_scale(value: Decimal, scale: int) -> bool:
    with localcontext() as ctx:
        ctx.prec = 60  # decimal(38, s) is past the default 28-digit context
        return value == value.quantize(Decimal(f"1E-{scale}"))


DECIMAL_TYPES = ["decimal", "decimal(5,0)", "decimal(10,2)", "decimal(38,18)", "decimal(3,3)"]


@pytest.mark.parametrize("stype", INT_TYPES)
def test_integer_draws_fit_the_column(stype):
    lo, hi = _INT_INPUT_RANGES[stype]

    @settings(max_examples=300, database=None)
    @given(_strategy_for(stype))
    def prop(value):
        assert value is None or lo <= value <= hi

    prop()


@pytest.mark.parametrize("stype", INT_TYPES)
def test_integer_builtins_fit_and_cover_the_edges(stype):
    lo, hi = _INT_INPUT_RANGES[stype]
    examples = _int_examples_for(stype)
    assert all(lo <= n <= hi for n in examples)
    # The edges are where a rewrite's arithmetic overflows the input type
    # under ANSI while python's unbounded int does not.
    assert lo in examples and hi in examples
    # Sign and zero are always present: truncation vs flooring lives there.
    assert {0, 1, -1} <= set(examples)


def test_bigint_builtins_keep_the_float_precision_cliff():
    # (2^53, 2^53 + 1) is where int and float math diverge; narrowing the
    # integer examples must not drop it for the 64-bit type.
    assert {2**53, 2**53 + 1} <= set(_int_examples_for("bigint"))


@pytest.mark.parametrize(
    "spelling, shape",
    [
        ("decimal", (10, 0)),
        ("decimal(5)", (5, 0)),
        ("decimal(10,2)", (10, 2)),
        ("DECIMAL( 38 , 18 )", (38, 18)),
        ("decimal(39,0)", None),
        ("decimal(2,3)", None),
        ("decimalish", None),
    ],
)
def test_decimal_shape(spelling, shape):
    assert _decimal_shape(spelling) == shape


@pytest.mark.parametrize("stype", DECIMAL_TYPES)
def test_decimal_draws_are_exact_at_the_column_scale(stype):
    precision, scale = _decimal_shape(stype)
    bound = _decimal_bound(precision, scale)

    @settings(max_examples=200, database=None)
    @given(_strategy_for(stype))
    def prop(value):
        if value is None:
            return
        assert isinstance(value, Decimal)
        assert value.is_finite()
        assert value.copy_abs() <= bound
        # Exactly representable: no rounding between what python sees and
        # what the column stores.
        assert _exact_at_scale(value, scale)

    prop()


@pytest.mark.parametrize("stype", DECIMAL_TYPES)
def test_decimal_builtins_fit_and_include_the_extremes(stype):
    precision, scale = _decimal_shape(stype)
    bound = _decimal_bound(precision, scale)
    examples = _decimal_examples_for(stype)
    assert bound in examples and bound.copy_negate() in examples
    for d in examples:
        assert d.copy_abs() <= bound
        assert _exact_at_scale(d, scale)


def test_decimal_bound_is_exact_at_full_precision():
    assert str(_decimal_bound(38, 18)) == "9" * 20 + "." + "9" * 18
    assert _decimal_bound(10, 2) == Decimal("99999999.99")
    assert _decimal_bound(3, 3) == Decimal("0.999")


def test_decimal_builtins_carry_halves_only_where_the_scale_allows():
    # 2.5 is the banker's-rounding probe; it only exists with a fractional scale.
    assert Decimal("2.5") in _decimal_examples_for("decimal(10,2)")
    assert all(d == d.to_integral_value() for d in _decimal_examples_for("decimal(5,0)"))


@given(st.integers(min_value=-(2**70), max_value=2**70))
def test_representable_matches_the_integer_ranges(n):
    for stype, (lo, hi) in _INT_INPUT_RANGES.items():
        assert _representable(n, stype) == (lo <= n <= hi)


def test_representable_decimal_integer_digits():
    assert _representable(Decimal("99999999.99"), "decimal(10,2)")
    assert not _representable(Decimal("100000000"), "decimal(10,2)")
    assert not _representable(10**8, "decimal(10,2)")
    # Extra scale is rounded by Spark, not an overflow.
    assert _representable(Decimal("1.23456"), "decimal(10,2)")


def test_representable_ignores_bools_and_other_types():
    assert _representable(True, "tinyint")
    assert _representable("not a number", "int")
    assert _representable(1.5e300, "bigint")
