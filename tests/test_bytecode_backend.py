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
    # The NOT must wrap the whole (a and b) result, not just one branch.
    assert "NOT" in sql
    assert sql.count("CASE WHEN") == 1  # one CASE, not a dangling NOT on one branch


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


def test_out_sql_written():
    backend = BytecodeBackend()
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        backend.run(job, sandbox)
        assert (Path(sandbox) / "OUT.sql").read_text().strip() == "(_udf_param_0 + 1)"
