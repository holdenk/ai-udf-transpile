# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic bytecode backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.backends.fake import (
    both_positive,
    greet,
    is_none_branch,
    mod_bucket,
    plus_one,
    round_half,
    upper_useragent,
    weekday_of,
)
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import TranspileJob


def _job_for(func, **kwargs) -> TranspileJob:
    src = canonical_source_from_func(func)
    params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    return TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=params,
        input_types=kwargs.get("input_types", ["bigint"] * len(params)),
        input_categories=kwargs.get("input_categories", ["numeric"] * len(params)),
        return_type=kwargs.get("return_type", "bigint"),
        captures={},
    )


def _run(func, **kwargs) -> str:
    backend = BytecodeBackend()
    with make_sandbox(_job_for(func, **kwargs)) as sandbox:
        return backend.run(_job_for(func, **kwargs), sandbox).sql


def test_plus_one():
    assert _run(plus_one) == "(_udf_param_0 + 1)"


def test_is_none_branch():
    assert _run(is_none_branch) == ("CASE WHEN _udf_param_0 IS NULL THEN -1 ELSE _udf_param_0 END")


def test_greet():
    assert _run(greet, input_types=["string"], return_type="string") == "concat('hi ', _udf_param_0)"


def test_both_positive():
    assert _run(both_positive, return_type="boolean") == (
        "CASE WHEN (_udf_param_0 > 0) THEN (_udf_param_1 > 0) ELSE (_udf_param_0 > 0) END"
    )


def test_mod_bucket_floored():
    assert _run(mod_bucket) == (
        "((_udf_param_0 % _udf_param_1) + CASE WHEN "
        "(_udf_param_0 % _udf_param_1) <> 0 AND "
        "((_udf_param_0 < 0) <> (_udf_param_1 < 0)) "
        "THEN _udf_param_1 ELSE 0 END)"
    )


# --- precedence: every composed expression is parenthesised ---


def test_precedence_mul():
    def f(x: int, y: int) -> int:
        return (x + 1) * y

    assert _run(f) == "((_udf_param_0 + 1) * _udf_param_1)"


def test_precedence_sub():
    def f(x: int, y: int, z: int) -> int:
        return x - (y - z)

    assert _run(f) == "(_udf_param_0 - (_udf_param_1 - _udf_param_2))"


def test_precedence_unary():
    def f(x: int) -> int:
        return -(x + 1)

    assert _run(f) == "(-(_udf_param_0 + 1))"


def test_double_negation():
    def f(x: int) -> int:
        return -(-x)

    assert _run(f) == "(-(-_udf_param_0))"


# --- join point: shared code after the jump is seen by both branches ---


def test_not_of_and():
    def f(a: bool, b: bool) -> bool:
        return not (a and b)

    sql = _run(f, input_types=["boolean", "boolean"], return_type="boolean")
    # The NOT appears in BOTH branch arms (the branch split happens before
    # UNARY_NOT), each as a null-safe CASE -- `not None` is True in Python,
    # and a bare NOT (NULL) would emit NULL.
    assert sql == (
        "CASE WHEN _udf_param_0 THEN "
        "CASE WHEN _udf_param_1 THEN false ELSE true END "
        "ELSE CASE WHEN _udf_param_0 THEN false ELSE true END END"
    )


def test_or():
    def f(a: bool, b: bool) -> bool:
        return a or b

    sql = _run(f, input_types=["boolean", "boolean"], return_type="boolean")
    assert "CASE WHEN" in sql


def test_is_not_none():
    def f(x: int) -> int:
        if x is not None:
            return x
        return 0

    assert _run(f) == "CASE WHEN _udf_param_0 IS NULL THEN 0 ELSE _udf_param_0 END"


def test_nested_if():
    def f(x: int, y: int) -> int:
        if x > 0:
            if y > 0:
                return 1
            return 2
        return 3

    sql = _run(f)
    assert sql.count("CASE WHEN") == 2
    assert "_udf_param_0 > 0" in sql and "_udf_param_1 > 0" in sql


def test_chained_compare():
    def f(x: int) -> bool:
        return 0 < x < 10

    sql = _run(f, return_type="boolean")
    assert "CASE WHEN" in sql
    assert "_udf_param_0" in sql


# --- null-safe equality ---


def test_eq_null_safe():
    def f(x: int) -> bool:
        return x == 0

    assert _run(f, return_type="boolean") == "(_udf_param_0 <=> 0)"


def test_neq_null_safe():
    def f(x: int) -> bool:
        return x != 0

    assert _run(f, return_type="boolean") == "(NOT (_udf_param_0 <=> 0))"


def test_truthiness_on_non_bool_operands():
    # `if x` on an int is a jump condition, not a boolean column: the SQL
    # must spell Python truthiness (None and 0 falsy) or fail analysis.
    def f(x: int) -> int:
        return 1 if x else 2

    assert _run(f) == ("CASE WHEN (_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0) THEN 1 ELSE 2 END")


def test_and_on_ints_keeps_python_value_semantics():
    def f(x: int, y: int) -> int:
        return x and y

    sql = _run(f)
    assert sql == (
        "CASE WHEN (_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0) THEN _udf_param_1 ELSE _udf_param_0 END"
    )


def test_not_null_is_true():
    def f(a: bool) -> bool:
        return not a

    assert _run(f, input_types=["boolean"], return_type="boolean") == (
        "CASE WHEN _udf_param_0 THEN false ELSE true END"
    )


def test_not_on_int_spells_truthiness():
    def f(x: int) -> bool:
        return not x

    assert _run(f, return_type="boolean") == (
        "CASE WHEN (_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0) THEN false ELSE true END"
    )


def test_string_truthiness():
    def f(s: str, t: str) -> str:
        return s or t

    sql = _run(f, input_types=["string", "string"], return_type="string")
    assert sql == (
        "CASE WHEN (_udf_param_0 IS NOT NULL AND length(_udf_param_0) > 0) "
        "THEN _udf_param_0 ELSE _udf_param_1 END"
    )


def test_arithmetic_on_string_declined():
    def f(s: str, n: int) -> str:
        return s * n

    with pytest.raises(BackendDecline):
        _run(f, input_types=["string", "bigint"], return_type="string")


def test_cross_kind_equality_declined():
    # Spark would coerce '5' = 5 to true; python says False. Not a rewrite.
    def f(s: str, n: int) -> bool:
        return s == n

    with pytest.raises(BackendDecline, match="comparison"):
        _run(f, input_types=["string", "bigint"], return_type="boolean")


def test_cross_kind_inequality_declined():
    def f(s: str, n: int) -> bool:
        return s != n

    with pytest.raises(BackendDecline, match="comparison"):
        _run(f, input_types=["string", "bigint"], return_type="boolean")


def test_bool_int_compare_declined():
    def f(a: bool, n: int) -> bool:
        return a == n

    with pytest.raises(BackendDecline, match="comparison"):
        _run(f, input_types=["boolean", "bigint"], return_type="boolean")


def test_null_folds_into_any_compare():
    def f(x: int) -> bool:
        return x == None  # noqa: E711  -- the exact shape the null fold covers

    assert _run(f, return_type="boolean") == "(_udf_param_0 <=> NULL)"


def test_infinite_float_const_emits_cast():
    def f(x: int) -> float:
        return x * 1e500

    assert _run(f, return_type="double") == "(_udf_param_0 * cast('Infinity' as double))"


def test_nan_float_const_emits_cast():
    # `float("nan")` in a body is a LOAD_GLOBAL decline, so exercise the
    # const renderer directly: a nan const must not become the bare
    # identifier `nan` (UNRESOLVED_COLUMN).
    from ai_udf_transpile.backends.bytecode import _Evaluator

    code = compile("def _f(): pass", "<t>", "exec")
    sql, kind = _Evaluator([], code, [])._const_sql(float("nan"))
    assert sql == "cast('NaN' as double)"
    assert kind == "numeric"


# --- declines ---


@pytest.mark.parametrize(
    "func,input_types,rtype,op",
    [
        (upper_useragent, ["string"], "string", "LOAD_ATTR"),
        (round_half, ["double"], "bigint", "LOAD_GLOBAL"),
        (weekday_of, ["timestamp"], "bigint", "LOAD_ATTR"),
    ],
)
def test_declined(func, input_types, rtype, op):
    with pytest.raises(BackendDecline, match=op):
        _run(func, input_types=input_types, return_type=rtype)


def test_floor_div_declined():
    def f(x: int, y: int) -> int:
        return x // y

    with pytest.raises(BackendDecline):
        _run(f)


def test_power_declined():
    def f(x: int, y: int) -> int:
        return x**y

    with pytest.raises(BackendDecline):
        _run(f)


def test_try_except_declined():
    src = "def f(x: int) -> int:\n    try:\n        return x + 1\n    except Exception:\n        return 0\n"
    job = TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        captures={},
    )
    with pytest.raises(BackendDecline, match="try/except"):
        BytecodeBackend().run(job, make_sandbox(job).__enter__())


def test_backslash_string_declined():
    def f(s: str) -> str:
        return s + "\\n"

    with pytest.raises(BackendDecline, match="backslash"):
        _run(f, input_types=["string"], return_type="string")


def test_mixed_kind_branch_arms_convert():
    def f(x: int, s: str):
        if x:
            return s
        return 0

    sql = _run(f, input_types=["bigint", "string"], return_type="string")
    assert sql == ("CASE WHEN (_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0) THEN _udf_param_1 ELSE 0 END")


def test_bool_or_string_join_is_mixed_but_convertible():
    def f(a: bool, s: str) -> str:
        return a or s

    # Mixed-kind joins that flow straight to RETURN convert: the CASE
    # returns whichever arm python would. (Consumption of a mixed join by
    # arithmetic cannot be pinned -- CPython duplicates the consumer into
    # both arms, so the join never reaches a BINARY_OP.)
    sql = _run(f, input_types=["boolean", "string"], return_type="string")
    assert sql == "CASE WHEN _udf_param_0 THEN _udf_param_0 ELSE _udf_param_1 END"


def test_parameterized_decimal_stays_numeric():
    def f(d) -> int:
        return 1 if d else 0

    sql = _run(f, input_types=["decimal(10,2)"], return_type="bigint")
    assert "_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0" in sql


def test_off_313_declines_fail_closed(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    with pytest.raises(BackendDecline, match="3.13"):
        _run(plus_one)


def test_out_sql_written():
    backend = BytecodeBackend()
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        backend.run(job, sandbox)
        assert (Path(sandbox) / "OUT.sql").read_text().strip() == "(_udf_param_0 + 1)"
