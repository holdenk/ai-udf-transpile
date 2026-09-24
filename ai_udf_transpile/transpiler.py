# SPDX-License-Identifier: Apache-2.0
"""Spark AbstractTranspiler plugin (variety='ai')."""

from __future__ import annotations

import ast
import logging
import weakref
from typing import Any, List, Optional

from ai_udf_transpile.catalog import HIT, MISS, Catalog
from ai_udf_transpile.keys import (
    canonical_source,
    closure_fingerprint,
    extract_captures,
    func_from_spark_stack,
    udf_key,
)
from ai_udf_transpile.types import (
    input_categories,
    input_spark_types,
    return_spark_type,
    types_known,
)

logger = logging.getLogger(__name__)

_SESSION_REF: Optional[weakref.ReferenceType] = None
_CATALOG: Optional[Catalog] = None
_TRANSPILER_CLS: Any = None


def set_session(spark: Any) -> None:
    global _SESSION_REF
    _SESSION_REF = weakref.ref(spark) if spark is not None else None


def get_session() -> Any:
    return _SESSION_REF() if _SESSION_REF is not None else None


def set_catalog(catalog: Optional[Catalog]) -> None:
    global _CATALOG
    _CATALOG = catalog


def get_catalog() -> Optional[Catalog]:
    return _CATALOG


def _pin_session(spark: Any) -> None:
    if spark is None:
        return
    try:
        from pyspark.sql import SparkSession

        SparkSession._activeSession = spark
        SparkSession._instantiatedSession = spark
        jvm = getattr(spark, "_jvm", None)
        jspark = getattr(spark, "_jsparkSession", None)
        if jvm is not None and jspark is not None:
            SparkSession._get_j_spark_session_class(jvm).setActiveSession(jspark)
    except Exception:
        logger.debug("could not pin SparkSession to this thread", exc_info=True)


def _reconstruct_column(row: Any, return_type: Any, spark: Any) -> Any:
    import pyspark.sql.functions as F

    _pin_session(spark)
    if row.target_kind == "catalyst" and row.catalyst_sql:
        return F.expr(row.catalyst_sql).cast(return_type)
    if row.target_kind == "java_udf":
        n = len(row.param_names)
        args = ", ".join(f"_udf_param_{i}" for i in range(n))
        if row.impl_class and spark is not None:
            fname = f"ai_udf_{row.udf_key[:16]}"
            spark.udf.registerJavaFunction(fname, row.impl_class, return_type)
            return F.expr(f"{fname}({args})").cast(return_type)
        if row.impl_source:
            logger.debug("java_udf impl_source without impl_class; cannot reconstruct")
            return None
    return None


def _spark_version(spark: Any) -> str:
    if spark is None:
        return "unknown"
    try:
        return str(spark.version)
    except Exception:
        return "unknown"


def get_transpiler_class() -> type:
    global _TRANSPILER_CLS
    if _TRANSPILER_CLS is not None:
        return _TRANSPILER_CLS
    from pyspark.sql.transpile import AbstractTranspiler

    class AITranspiler(AbstractTranspiler):
        variety = "ai"

        def _transpile_from_ast(
            self,
            src: Optional[str],
            ast_info: ast.AST,
            function_ast: ast.FunctionDef,
            params: List[str],
            returnType: Any,
            param_categories: Optional[dict] = None,
        ) -> Optional[Any]:
            del src, ast_info, param_categories
            spark = get_session()
            try:
                if not types_known(function_ast, params, returnType):
                    return None
                in_types = input_spark_types(function_ast, params)
                in_cats = input_categories(function_ast, params)
                out_type = return_spark_type(returnType)
                if in_types is None or in_cats is None or out_type is None:
                    return None
                func = func_from_spark_stack()
                captures = extract_captures(func, function_ast, params)
                if captures is None:
                    return None
                source = canonical_source(function_ast)
                fingerprint = closure_fingerprint(captures)
                version = _spark_version(spark)
                key = udf_key(source, params, in_types, out_type, version, fingerprint)
                catalog = get_catalog()
                if catalog is None:
                    return None
                kind, row = catalog.lookup(key, spark=spark)
                if kind == HIT and row is not None:
                    try:
                        return _reconstruct_column(row, returnType, spark)
                    except Exception:
                        logger.debug("Column reconstruction failed", exc_info=True)
                        return None
                if kind == MISS:
                    catalog.insert_pending(
                        udf_key=key,
                        source_text=source,
                        param_names=list(params),
                        input_types=in_types,
                        input_categories=in_cats,
                        return_type=out_type,
                        spark_version=version,
                        closure_fingerprint=fingerprint,
                        captures=captures,
                    )
                return None
            except Exception:
                logger.debug("AITranspiler declined due to exception", exc_info=True)
                return None

    _TRANSPILER_CLS = AITranspiler
    return AITranspiler


def register_transpiler() -> type:
    cls = get_transpiler_class()
    cls.register()
    return cls
