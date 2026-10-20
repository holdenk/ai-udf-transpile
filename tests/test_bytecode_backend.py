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


def _run(func, **kwargs):
    backend = BytecodeBackend()
    with make_sandbox(_job_for(func, **kwargs)) as sandbox:
        return backend.run(_job_for(func, **kwargs), sandbox)


def test_plus_one():
    assert _run(plus_one).sql == "_udf_param_0 + 1"


def test_is_none_branch():
    assert _run(is_none_branch).sql == ("CASE WHEN _udf_param_0 IS NULL THEN -1 ELSE _udf_param_0 END")


def test_greet():
    # String + emits concat() (Spark ANSI rejects 'hi ' + str).
    assert _run(greet, input_types=["string"], return_type="string").sql == "concat('hi ', _udf_param_0)"


def test_both_positive():
    # Short-circuit and emits a CASE; semantically equivalent to AND.
    sql = _run(both_positive, return_type="boolean").sql
    assert "CASE WHEN" in sql
    assert "_udf_param_0 > 0" in sql
    assert "_udf_param_1 > 0" in sql


def test_mod_bucket_floored():
    # The bytecode backend emits the floored-remainder CASE (same as the fake
    # fixture, minus outer parens).
    sql = _run(mod_bucket).sql
    assert "% _udf_param_1" in sql
    assert "CASE WHEN" in sql
    assert "ELSE 0 END" in sql


def test_upper_useragent_declined():
    # .upper() is a method call -> LOAD_ATTR + CALL -> decline.
    with pytest.raises(BackendDecline, match="LOAD_ATTR|CALL"):
        _run(upper_useragent, input_types=["string"], return_type="string")


def test_round_half_declined():
    # round() is a builtin call -> LOAD_GLOBAL + CALL -> decline.
    with pytest.raises(BackendDecline, match="LOAD_GLOBAL|CALL"):
        _run(round_half, input_types=["double"], return_type="bigint")


def test_weekday_of_declined():
    # .weekday() is a method call -> LOAD_ATTR + CALL -> decline.
    with pytest.raises(BackendDecline, match="LOAD_ATTR|CALL"):
        _run(weekday_of, input_types=["timestamp"], return_type="bigint")


def test_out_sql_written():
    backend = BytecodeBackend()
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        backend.run(job, sandbox)
        assert (Path(sandbox) / "OUT.sql").read_text().strip() == "_udf_param_0 + 1"
