# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from ai_udf_transpile.verify import VerifyFailed, exec_udf_source, load_python_udf


def test_import_inside_function():
    src = """
def uses_math(x: float) -> float:
    import math
    return math.sin(x)
"""
    ns = exec_udf_source(src)
    assert ns["uses_math"](0.0) == 0.0


def test_module_level_math_from_captures():
    src = """
def uses_math(x: float) -> float:
    return math.sin(x)
"""
    ns = exec_udf_source(src, captures={"math": {"__module__": "math"}})
    assert ns["uses_math"](0.0) == 0.0


def test_on_demand_allowlist_import():
    src = """
def uses_math(x: float) -> float:
    return math.cos(x)
"""
    fn = load_python_udf(src)
    assert fn(0.0) == 1.0


def test_unknown_name_is_name_error():
    src = """
def uses_missing(x: int) -> int:
    return definitely_not_a_module.foo(x)
"""
    fn = load_python_udf(src)
    with pytest.raises(NameError):
        fn(1)


def test_numeric_tolerance_does_not_fuzzy_match_strings():
    from ai_udf_transpile.verify import _within_tolerance

    assert _within_tolerance(1.0, 1.0 + 1e-8, 1e-6)
    assert not _within_tolerance(1.0, 1.0 + 1e-4, 1e-6)
    assert not _within_tolerance(1.0, 1.0 + 1e-8, 0.0)
    assert not _within_tolerance("a", "b", 10.0)
    assert not _within_tolerance(True, False, 10.0)


def test_no_function_raises():
    with pytest.raises(VerifyFailed):
        load_python_udf("x = 1")


def test_lambda_placeholder_scan():
    from ai_udf_transpile.verify import lambda_captures_placeholder

    outside = "transform(sequence(_udf_param_0, _udf_param_1), x -> date_format(x, 'yyyy-MM-dd'))"
    inside = "transform(sequence(0, 1), i -> concat(i, _udf_param_0))"
    assert not lambda_captures_placeholder(outside)
    assert lambda_captures_placeholder(inside)
    assert not lambda_captures_placeholder("_udf_param_0 + 1")
    # A quoted arrow must not start a lambda, and a quoted placeholder inside
    # a lambda is text, not a column reference.
    assert not lambda_captures_placeholder("concat('->', _udf_param_0)")
    assert not lambda_captures_placeholder("transform(arr, x -> concat(x, '_udf_param_0'))")


def test_reflect_and_process_builder_rejected_without_spark():
    from ai_udf_transpile.targets import TranspileResult
    from ai_udf_transpile.verify import hypothesis_check

    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(kind="catalyst", sql="reflect('java.lang.Math', 'abs', _udf_param_0)"),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert not ok
    assert "reflect" in (err or "")

    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(kind="catalyst", sql="`reflect` /* not a comment trick */ (1)"),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert not ok
    assert "reflect" in (err or "")

    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(
            kind="java_udf",
            java_source="class T { ProcessBuilder p; }",
            class_name="T",
        ),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert not ok
    assert "ProcessBuilder" in (err or "")
