# SPDX-License-Identifier: Apache-2.0
"""Hypothesis differential check plus an import-aware exec sandbox."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any, Callable, Optional

STDLIB_ALLOWLIST = frozenset(
    {
        "math",
        "cmath",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "operator",
        "statistics",
        "string",
        "re",
        "json",
        "hashlib",
        "collections",
        "typing",
        "dataclasses",
        "numbers",
        "os",
        "sys",
        "copy",
        "random",
    }
)

_SENTINEL_RAISED = object()


class VerifyFailed(Exception):
    """Hypothesis or sandbox determined the rewrite is not equivalent."""


def collect_imported_module_names(source_text: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _copy_defining_module_imports(func: Any, ns: dict[str, Any]) -> None:
    globals_dict = getattr(func, "__globals__", None) or {}
    for key, value in globals_dict.items():
        if inspect.ismodule(value) and key not in ns:
            ns[key] = value
    try:
        srcfile = inspect.getsourcefile(func)
    except TypeError:
        srcfile = None
    if not srcfile:
        return
    try:
        file_src = open(srcfile, encoding="utf-8").read()
        tree = ast.parse(file_src)
    except Exception:
        return
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in globals_dict:
                    ns.setdefault(bound, globals_dict[bound])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in globals_dict:
                    ns.setdefault(bound, globals_dict[bound])


def _hydrate_captures(captures: Optional[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in (captures or {}).items():
        if isinstance(value, dict) and "__module__" in value:
            out[name] = importlib.import_module(value["__module__"])
        elif isinstance(value, dict) and "__bytes__" in value:
            out[name] = bytes.fromhex(value["__bytes__"])
        else:
            out[name] = value
    return out


class _OnDemandModules(dict):
    def __missing__(self, key: str) -> Any:
        if key in STDLIB_ALLOWLIST:
            mod = importlib.import_module(key)
            self[key] = mod
            return mod
        raise KeyError(key)


def exec_udf_source(
    source_text: str,
    captures: Optional[dict[str, Any]] = None,
    func: Any = None,
) -> dict[str, Any]:
    """Exec ``source_text`` in an import-aware namespace. Raises NameError/ImportError."""
    ns: _OnDemandModules = _OnDemandModules()
    ns.update(_hydrate_captures(captures))
    if func is not None:
        _copy_defining_module_imports(func, ns)
    for name in collect_imported_module_names(source_text):
        if name in STDLIB_ALLOWLIST and name not in ns:
            ns[name] = importlib.import_module(name)
    exec(textwrap.dedent(source_text), ns, ns)  # noqa: S102 — isolated verify sandbox
    return ns


def load_python_udf(
    source_text: str,
    captures: Optional[dict[str, Any]] = None,
    func: Any = None,
) -> Callable[..., Any]:
    ns = exec_udf_source(source_text, captures, func)
    callables = [
        v for v in ns.values() if inspect.isfunction(v) and getattr(v, "__name__", None) != "<lambda>"
    ]
    if not callables:
        raise VerifyFailed("no function defined in source_text")
    return callables[-1]


def _strategy_for(spark_type: str):
    from hypothesis import strategies as st

    t = spark_type.strip().lower()
    if t in {"bigint", "long", "int", "integer", "smallint", "tinyint", "byte", "short"}:
        return st.one_of(st.none(), st.integers(min_value=-(2**31), max_value=2**31 - 1))
    if t.startswith("decimal"):
        return st.one_of(st.none(), st.decimals(allow_nan=False, allow_infinity=False, places=4))
    if t in {"double", "float"}:
        return st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False, width=32),
        )
    if t == "string":
        return st.one_of(st.none(), st.text(max_size=16))
    if t in {"boolean", "bool"}:
        return st.one_of(st.none(), st.booleans())
    if t == "binary":
        return st.one_of(st.none(), st.binary(max_size=16))
    raise VerifyFailed(f"unsupported input type for verify: {spark_type}")


def _spark_type(simple: str):
    from pyspark.sql.types import (
        BinaryType,
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
    )

    t = simple.strip().lower()
    return {
        "tinyint": ByteType(),
        "byte": ByteType(),
        "smallint": ShortType(),
        "short": ShortType(),
        "int": IntegerType(),
        "integer": IntegerType(),
        "bigint": LongType(),
        "long": LongType(),
        "float": FloatType(),
        "double": DoubleType(),
        "string": StringType(),
        "boolean": BooleanType(),
        "bool": BooleanType(),
        "binary": BinaryType(),
    }.get(t, LongType() if t.startswith("decimal") else StringType())


def _eval_sql(spark: Any, sql: str, args: tuple, input_types: list[str], return_type: str):
    from pyspark.sql import Row
    from pyspark.sql.types import StructField, StructType

    fields = [
        StructField(f"_udf_param_{i}", _spark_type(input_types[i]), True) for i in range(len(input_types))
    ]
    schema = StructType(fields)
    row_kwargs = {f"_udf_param_{i}": args[i] for i in range(len(args))}
    df = spark.createDataFrame([Row(**row_kwargs)], schema=schema)
    return df.selectExpr(f"({sql}) AS result").collect()[0][0]


def _java_udf_expr(
    spark: Any,
    result: Any,
    input_types: list[str],
    return_type: str,
) -> Optional[str]:
    """Compile (if needed) and register the Java UDF under a content-addressed name."""
    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    binary = getattr(result, "binary", None)
    class_name = getattr(result, "class_name", None)
    java_source = getattr(result, "java_source", None)
    janino: Optional[bool] = None
    if not binary and java_source:
        compiled = compile_java(spark, java_source, class_name)
        binary, class_name, janino = compiled.jar_bytes, compiled.class_name, compiled.janino
        result.binary = binary
        result.class_name = class_name
    if not (binary and class_name):
        return None
    fname = verify_function_name("ai_udf_verify", binary)
    register_java_udf(spark, fname, class_name, binary, _spark_type(return_type), janino=janino)
    args = ", ".join(f"_udf_param_{i}" for i in range(len(input_types)))
    return f"{fname}({args})"


def hypothesis_check(
    *,
    source_text: str,
    captures: dict[str, Any],
    result: Any,
    input_types: list[str],
    return_type: str,
    spark: Any,
    max_examples: int = 20,
    func: Any = None,
) -> tuple[bool, Optional[str]]:
    """Return (ok, error). Analysis failure / mismatch → (False, msg). Never raises to the worker."""
    try:
        python_fn = load_python_udf(source_text, captures, func)
    except (NameError, ImportError, SyntaxError, VerifyFailed) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"sandbox exec failed: {exc}"

    kind = getattr(result, "kind", "catalyst")
    sql = getattr(result, "sql", None)
    if kind == "catalyst" and sql:
        eval_expr = sql
    elif kind == "java_udf":
        if spark is None:
            return False, "java_udf verify requires Spark"
        try:
            eval_expr = _java_udf_expr(spark, result, input_types, return_type)
        except Exception as exc:
            return False, f"java_udf verify setup failed: {exc}"
        if eval_expr is None:
            return False, "java_udf result has no class/binary to verify"
    else:
        return False, f"verify of target_kind={kind} is not implemented"

    if spark is None:
        return False, "hypothesis_check requires a SparkSession"

    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    strategies = [_strategy_for(t) for t in input_types]
    mismatch: list[str] = []

    @settings(
        max_examples=max_examples,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        database=None,
    )
    @given(st.tuples(*strategies) if strategies else st.just(()))
    def _check(args: tuple) -> None:
        py_exc: Optional[BaseException] = None
        try:
            py_value = python_fn(*args)
        except (NameError, ImportError) as exc:
            # Sandbox couldn't resolve a name/import — not a per-row Python error.
            mismatch.append(f"{type(exc).__name__}: {exc} on {args!r}")
            return
        except Exception as exc:  # Python UDF raised
            py_exc = exc
            py_value = _SENTINEL_RAISED
        try:
            sql_value = _eval_sql(spark, eval_expr, args, input_types, return_type)
            sql_exc = None
        except Exception as exc:
            sql_value = _SENTINEL_RAISED
            sql_exc = exc
        if sql_exc is not None:
            if py_exc is None:
                mismatch.append(f"sql raised {sql_exc!r} on {args!r} but python returned {py_value!r}")
            return
        if py_exc is not None:
            # Spark's own hypothesis policy: python raise + sql value is allowed.
            return
        if py_value != sql_value:
            mismatch.append(f"mismatch on {args!r}: python={py_value!r} sql={sql_value!r}")

    try:
        _check()
    except Exception as exc:
        return False, f"hypothesis failed: {exc}"
    if mismatch:
        return False, mismatch[0]
    return True, None
