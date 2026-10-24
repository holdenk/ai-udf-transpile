# SPDX-License-Identifier: Apache-2.0
"""Static-bytecode parity: on the shared supported subset, the deterministic
bytecode backend's SQL must match Python wherever it converts -- the same
standard of correctness the built-in Catalyst (AST) transpiler is held to,
which this suite checks per function too."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import TranspileJob

pytestmark = pytest.mark.spark

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"


# --- corpus: all inside the bytecode backend's supported subset ---


def plus_one(x: int) -> int:
    return x + 1


def add_mul(x: int, y: int) -> int:
    return x * y + 1


def sub_neg(x: int) -> int:
    return 1 - -x


def truediv(x: int, y: int) -> float:
    return x / y


def mod_bucket(x: int, y: int) -> int:
    return x % y


def eq_zero(x: int) -> bool:
    return x == 0


def ne_zero(x: int) -> bool:
    return x != 0


def less_five(x: int) -> bool:
    return x < 5


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def py_and(a: bool, b: bool) -> bool:
    return a and b


def py_or(a: bool, b: bool) -> bool:
    return a or b


def not_x(a: bool) -> bool:
    return not a


def ternary(x: int) -> int:
    return 1 if x else 2


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def nested_if(x: int, y: int) -> int:
    if x > 0:
        if y > 0:
            return 1
        return 2
    return 3


def chained(x: int) -> bool:
    return 0 < x < 10


def greet(name: str) -> str:
    return "hi " + name


def str_lt(a: str, b: str) -> bool:
    return a < b


def str_eq(a: str, b: str) -> bool:
    return a == b


def str_or(s: str, t: str) -> str:
    return s or t


# (fn, spark return type, input types, ddl schema, grid rows)
CORPUS = [
    (plus_one, LongType(), ["bigint"], "x long", [(0,), (1,), (-1,), (None,)]),
    (add_mul, LongType(), ["bigint", "bigint"], "x long, y long", [(2, 3), (0, 5), (-2, -3), (None, 4)]),
    (sub_neg, LongType(), ["bigint"], "x long", [(5,), (0,), (-5,), (None,)]),
    (
        truediv,
        DoubleType(),
        ["bigint", "bigint"],
        "x long, y long",
        [(7, 2), (9, 3), (0, 5), (-7, 2), (1, 1), (None, 3)],
    ),
    (
        mod_bucket,
        LongType(),
        ["bigint", "bigint"],
        "x long, y long",
        [(10, 3), (7, 100), (-7, 3), (7, -3), (-7, -3), (0, 5), (None, 5)],
    ),
    (eq_zero, BooleanType(), ["bigint"], "x long", [(0,), (5,), (-5,), (None,)]),
    (ne_zero, BooleanType(), ["bigint"], "x long", [(0,), (5,), (None,)]),
    (less_five, BooleanType(), ["bigint"], "x long", [(4,), (5,), (None,)]),
    (
        both_positive,
        BooleanType(),
        ["bigint", "bigint"],
        "x long, y long",
        [(1, 2), (-1, 2), (1, -2), (0, 1)],
    ),
    (
        py_and,
        BooleanType(),
        ["boolean", "boolean"],
        "a boolean, b boolean",
        [(True, True), (True, False), (False, True), (False, False), (None, True), (True, None)],
    ),
    (
        py_or,
        BooleanType(),
        ["boolean", "boolean"],
        "a boolean, b boolean",
        [(True, True), (True, False), (False, True), (False, False), (None, True), (False, None)],
    ),
    (not_x, BooleanType(), ["boolean"], "a boolean", [(True,), (False,), (None,)]),
    (ternary, LongType(), ["bigint"], "x long", [(1,), (0,), (-3,), (None,)]),
    (is_none_branch, LongType(), ["bigint"], "x long", [(5,), (0,), (None,)]),
    (nested_if, LongType(), ["bigint", "bigint"], "x long, y long", [(1, 1), (1, -1), (-1, 1), (0, 0)]),
    (chained, BooleanType(), ["bigint"], "x long", [(5,), (0,), (10,), (-1,), (None,)]),
    (greet, StringType(), ["string"], "name string", [("holden",), ("",), (None,)]),
    (
        str_lt,
        BooleanType(),
        ["string", "string"],
        "a string, b string",
        [("a", "z"), ("z", "a"), ("é", "z"), ("a", "a")],
    ),
    (
        str_eq,
        BooleanType(),
        ["string", "string"],
        "a string, b string",
        [("a", "a"), ("a", "b"), (None, None), ("a", None)],
    ),
    (
        str_or,
        StringType(),
        ["string", "string"],
        "a string, b string",
        [("a", "b"), ("", "b"), (None, "b"), ("a", None), (None, None)],
    ),
]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _expected(fn, rows) -> list:
    """(raised, value) per row. A raised row puts no constraint on the SQL."""
    out = []
    for row in rows:
        try:
            out.append((False, fn(*row)))
        except Exception:
            out.append((True, None))
    return out


def _bytecode_sql(fn, input_types, return_type) -> str:
    job = TranspileJob(
        udf_key="k",
        source_text=canonical_source_from_func(fn),
        param_names=list(fn.__code__.co_varnames[: fn.__code__.co_argcount]),
        input_types=input_types,
        input_categories=["numeric"] * len(input_types),
        return_type=return_type,
        captures={},
    )
    with make_sandbox(job) as sandbox:
        return BytecodeBackend().run(job, sandbox).sql


@pytest.mark.parametrize("fn,rtype,input_types,schema,rows", CORPUS)
def test_bytecode_static_matches_python(spark, fn, rtype, input_types, schema, rows):
    sql = _bytecode_sql(fn, input_types, rtype.simpleString())
    assert sql, f"{fn.__name__} did not convert"
    cols = [c.split()[0] for c in schema.split(", ")]
    expected = _expected(fn, rows)
    # One df for the rows python returns on; a row where python raises puts
    # no constraint on the SQL, and ANSI Spark may raise on it too (e.g. a
    # future divide-by-zero row), so those rows are evaluated apart.
    ok_rows = [row for row, (raised, _) in zip(rows, expected) if not raised]
    bad_rows = [row for row, (raised, _) in zip(rows, expected) if raised]
    expr = re.sub(r"_udf_param_(\d+)", lambda m: cols[int(m.group(1))], sql)
    got = [r[0] for r in spark.createDataFrame(ok_rows, schema).selectExpr(f"({expr}) AS r").collect()]
    if bad_rows:
        try:
            spark.createDataFrame(bad_rows, schema).selectExpr(f"({expr}) AS r").collect()
        except Exception:
            pass
    for i, (raised, value) in enumerate(expected):
        if raised:
            continue
        assert _same(got[ok_rows.index(rows[i])], value), (
            f"{fn.__name__}{rows[i]}: sql={got[ok_rows.index(rows[i])]!r} python={value!r}"
        )


@pytest.mark.parametrize("fn,rtype,input_types,schema,rows", CORPUS)
def test_builtin_static_matches_python_where_it_converts(spark, fn, rtype, input_types, schema, rows):
    del input_types
    cols = [c.split()[0] for c in schema.split(", ")]
    expected = _expected(fn, rows)
    # Where Python raises, the built-in rewrite raises the whole query too
    # (its ordered compares fail on NULL like Python's TypeError) -- which
    # differential testing allows, so those rows get no value constraint.
    ok_rows = [row for row, (raised, _) in zip(rows, expected) if not raised]
    bad_rows = [row for row, (raised, _) in zip(rows, expected) if raised]
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    try:
        udf = UserDefinedFunction(fn, rtype)
        if not udf.transpiled:
            pytest.skip("built-in catalyst transpiler declined")
        got = [r[0] for r in spark.createDataFrame(ok_rows, schema).select(udf(*cols)).collect()]
        if bad_rows:
            try:
                spark.createDataFrame(bad_rows, schema).select(udf(*cols)).collect()
            except Exception:
                pass
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)
    for i, (raised, value) in enumerate(expected):
        if raised:
            continue
        assert _same(got[ok_rows.index(rows[i])], value), (
            f"{fn.__name__}{rows[i]}: static={got[ok_rows.index(rows[i])]!r} python={value!r}"
        )


@pytest.mark.parametrize("fn,rtype,input_types,schema,rows", CORPUS)
def test_bytecode_converts_everything_the_builtin_converts(spark, fn, rtype, input_types, schema, rows):
    del schema, rows
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    try:
        builtin_converted = bool(UserDefinedFunction(fn, rtype).transpiled)
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)
    if not builtin_converted:
        pytest.skip("built-in declined; no superset claim")
    assert _bytecode_sql(fn, input_types, rtype.simpleString()), (
        f"built-in converted {fn.__name__} but bytecode declined"
    )
