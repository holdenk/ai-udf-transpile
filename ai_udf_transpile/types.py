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
    # `datetime` and `datetime.datetime` both resolve to the name "datetime".
    "datetime": ("timestamp", "timestamp"),
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


def _annotation_mapping(annotation: Optional[ast.AST]) -> Optional[tuple[str, str]]:
    """(category, Spark simpleString) for an annotation, or None when unknown."""
    name = annotation_name(annotation)
    if name is not None:
        return ANNOTATION_MAP.get(name)
    if isinstance(annotation, ast.Subscript) and annotation_name(annotation.value) in {
        "dict",
        "Dict",
    }:
        slice_ = annotation.slice
        elts = slice_.elts if isinstance(slice_, ast.Tuple) else [slice_]
        if len(elts) == 2 and all(annotation_name(e) == "str" for e in elts):
            return ("map", "map<string,string>")
    return None


def _public_args(function_ast: ast.FunctionDef, public_params: list[str]) -> list[ast.arg]:
    n = len(public_params)
    return function_ast.args.args[-n:] if n else []


def input_categories(function_ast: ast.FunctionDef, public_params: list[str]) -> Optional[list[str]]:
    cats: list[str] = []
    for arg in _public_args(function_ast, public_params):
        mapped = _annotation_mapping(arg.annotation)
        if mapped is None:
            return None
        cats.append(mapped[0])
    return cats


def input_spark_types(function_ast: ast.FunctionDef, public_params: list[str]) -> Optional[list[str]]:
    types: list[str] = []
    for arg in _public_args(function_ast, public_params):
        mapped = _annotation_mapping(arg.annotation)
        if mapped is None:
            return None
        types.append(mapped[1])
    return types


def _is_atomic_simple(simple: str) -> bool:
    return any(simple == p or simple.startswith(p + "(") for p in ATOMIC_RETURN_PREFIXES)


def _is_supported_return_simple(simple: str) -> bool:
    if _is_atomic_simple(simple):
        return True
    # array of an atomic (e.g. array<string>) -- scatter-style UDFs.
    if simple.startswith("array<") and simple.endswith(">"):
        return _is_atomic_simple(simple[6:-1].strip())
    return False


def return_spark_type(return_type: Any) -> Optional[str]:
    if return_type is None:
        return None
    if isinstance(return_type, str):
        simple = return_type.strip().lower()
        if _is_supported_return_simple(simple):
            return return_type.strip()
        return None
    simple_fn = getattr(return_type, "simpleString", None)
    if callable(simple_fn):
        try:
            simple = str(simple_fn())
        except Exception:
            simple = ""
        if _is_supported_return_simple(simple.lower()):
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
