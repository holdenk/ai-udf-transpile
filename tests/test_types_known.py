# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast

from ai_udf_transpile.keys import canonical_source_text
from ai_udf_transpile.types import input_spark_types, return_spark_type, types_known


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(src).body[0]


def test_untyped_params_are_not_known():
    fn = _fn("def f(x):\n    return x + 1")
    assert types_known(fn, ["x"], "bigint") is False
    assert input_spark_types(fn, ["x"]) is None


def test_annotated_int_maps_to_bigint():
    fn = _fn("def f(x: int) -> int:\n    return x + 1")
    assert types_known(fn, ["x"], "bigint") is True
    assert input_spark_types(fn, ["x"]) == ["bigint"]


def test_stringized_annotation():
    fn = _fn('def f(x: "str") -> str:\n    return x')
    assert input_spark_types(fn, ["x"]) == ["string"]
    assert types_known(fn, ["x"], "string") is True


def test_missing_return_type_declines():
    fn = _fn("def f(x: int) -> int:\n    return x")
    assert types_known(fn, ["x"], None) is False
    assert return_spark_type(None) is None


def test_non_atomic_return_declines():
    fn = _fn("def f(x: int) -> int:\n    return x")
    assert types_known(fn, ["x"], "struct<a:int>") is False
    assert types_known(fn, ["x"], "map<string,string>") is False
    assert return_spark_type("array<struct<a:int>>") is None


def test_array_of_atomic_return_accepted():
    fn = _fn("def f(x: int) -> int:\n    return x")
    assert types_known(fn, ["x"], "array<int>") is True
    assert return_spark_type("array<string>") == "array<string>"


def test_return_type_from_simpleString_object():
    class IntegerType:
        def simpleString(self):
            return "int"

    fn = _fn("def f(x: int) -> int:\n    return x")
    assert types_known(fn, ["x"], IntegerType()) is True
    assert return_spark_type(IntegerType()) == "int"


def test_canonical_source_is_stable():
    a = canonical_source_text("def plus_one(x: int) -> int: return x + 1")
    b = canonical_source_text("def plus_one(x: int) -> int:\n    return x + 1\n")
    assert a == b
