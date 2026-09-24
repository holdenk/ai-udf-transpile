# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import inspect
import textwrap

from ai_udf_transpile.keys import (
    canonical_source,
    closure_fingerprint,
    extract_captures,
    udf_key,
)


def _ast(func) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]


def test_offset_10_vs_20_are_different_keys():
    def make(offset: int):
        def add_offset(x: int) -> int:
            return x + offset

        return add_offset

    a = make(10)
    b = make(20)
    ast_a, ast_b = _ast(a), _ast(b)
    cap_a = extract_captures(a, ast_a, ["x"])
    cap_b = extract_captures(b, ast_b, ["x"])
    assert cap_a is not None and cap_b is not None
    assert cap_a != cap_b
    key_a = udf_key(
        canonical_source(ast_a),
        ["x"],
        ["bigint"],
        "bigint",
        "5.0.0",
        closure_fingerprint(cap_a),
    )
    key_b = udf_key(
        canonical_source(ast_b),
        ["x"],
        ["bigint"],
        "bigint",
        "5.0.0",
        closure_fingerprint(cap_b),
    )
    assert key_a != key_b


def test_same_offset_same_key():
    def make(offset: int):
        def add_offset(x: int) -> int:
            return x + offset

        return add_offset

    a = make(10)
    b = make(10)
    ast_a, ast_b = _ast(a), _ast(b)
    cap_a = extract_captures(a, ast_a, ["x"])
    cap_b = extract_captures(b, ast_b, ["x"])
    key_a = udf_key(canonical_source(ast_a), ["x"], ["bigint"], "bigint", "5.0.0", closure_fingerprint(cap_a))
    key_b = udf_key(canonical_source(ast_b), ["x"], ["bigint"], "bigint", "5.0.0", closure_fingerprint(cap_b))
    assert key_a == key_b


def test_closing_over_function_declines():
    def helper(x):
        return x

    def uses_helper(x: int) -> int:
        return helper(x)

    captured = extract_captures(uses_helper, _ast(uses_helper), ["x"])
    assert captured is None


def test_stdlib_module_is_allowed():
    import math

    def uses_math(x: float) -> float:
        return math.sin(x)

    captured = extract_captures(uses_math, _ast(uses_math), ["x"])
    assert captured is not None
    assert captured["math"]["__module__"] == "math"


def test_input_types_in_key():
    src = "def f(x: int) -> int:\n    return x"
    k1 = udf_key(src, ["x"], ["bigint"], "bigint", "5.0.0", "")
    k2 = udf_key(src, ["x"], ["bigint"], "int", "5.0.0", "")
    assert k1 != k2
