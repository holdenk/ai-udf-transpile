# SPDX-License-Identifier: Apache-2.0
"""Type-known gate: we only queue UDFs whose inputs and output are concrete atomics."""

from __future__ import annotations

import ast
from typing import Any, Optional

# Python annotation → (input category, Spark simpleString used in the cache key / DDL)
ANNOTATION_MAP: dict[str, tuple[str, str]] = {
    "int": ("numeric", "bigint"),
    "float": ("numeric", "double"),
    "str": ("string", "string"),
    "bool": ("bool", "boolean"),
    "bytes": ("binary", "binary"),
}

ATOMIC_RETURN_PREFIXES = (
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    "decimal",
    "string",
    "boolean",
    "binary",
    "byte",
    "short",
    "long",
)

_NUMERIC_RETURN_TYPE_NAMES = {
    "ByteType",
    "ShortType",
    "IntegerType",
    "LongType",
    "FloatType",
    "DoubleType",
    "DecimalType",
    "NumericType",
}
_ATOMIC_RETURN_TYPE_NAMES = _NUMERIC_RETURN_TYPE_NAMES | {
    "StringType",
    "BooleanType",
    "BinaryType",
}


def annotation_name(annotation: Optional[ast.AST]) -> Optional[str]:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None


def _public_args(function_ast: ast.FunctionDef, public_params: list[str]) -> list[ast.arg]:
    n = len(public_params)
    return function_ast.args.args[-n:] if n else []


def input_categories(function_ast: ast.FunctionDef, public_params: list[str]) -> Optional[list[str]]:
    cats: list[str] = []
    for arg in _public_args(function_ast, public_params):
        mapped = ANNOTATION_MAP.get(annotation_name(arg.annotation) or "")
        if mapped is None:
            return None
        cats.append(mapped[0])
    return cats


def input_spark_types(function_ast: ast.FunctionDef, public_params: list[str]) -> Optional[list[str]]:
    types: list[str] = []
    for arg in _public_args(function_ast, public_params):
        mapped = ANNOTATION_MAP.get(annotation_name(arg.annotation) or "")
        if mapped is None:
            return None
        types.append(mapped[1])
    return types


def return_spark_type(return_type: Any) -> Optional[str]:
    if return_type is None:
        return None
    if isinstance(return_type, str):
        simple = return_type.strip().lower()
        if any(simple == p or simple.startswith(p + "(") for p in ATOMIC_RETURN_PREFIXES):
            return return_type.strip()
        return None
    simple_fn = getattr(return_type, "simpleString", None)
    if callable(simple_fn):
        try:
            simple = str(simple_fn())
        except Exception:
            simple = ""
        if any(simple.lower() == p or simple.lower().startswith(p + "(") for p in ATOMIC_RETURN_PREFIXES):
            return simple
    name = type(return_type).__name__
    if name in _ATOMIC_RETURN_TYPE_NAMES:
        if callable(simple_fn):
            try:
                return str(simple_fn())
            except Exception:
                pass
        return name
    return None


def types_known(
    function_ast: ast.FunctionDef,
    public_params: list[str],
    return_type: Any,
) -> bool:
    if input_spark_types(function_ast, public_params) is None:
        return False
    return return_spark_type(return_type) is not None
