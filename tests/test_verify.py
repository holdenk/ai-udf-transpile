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


def test_no_function_raises():
    with pytest.raises(VerifyFailed):
        load_python_udf("x = 1")
