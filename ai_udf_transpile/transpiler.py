# SPDX-License-Identifier: Apache-2.0
"""Spark AbstractTranspiler plugin (variety='ai')."""

from __future__ import annotations

import ast
import contextlib
import contextvars
import logging
import weakref
from typing import Any, List, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.catalog import HIT, MISS, Catalog
from ai_udf_transpile.keys import (
    canonical_source,
    closure_fingerprint,
    extract_captures,
    func_from_spark_stack,
    udf_function_ast,
    udf_key,
)
from ai_udf_transpile.targets import NO_RUST_SUFFIX
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
    if row.target_kind in ("java_udf", "gpu_kernel"):
        n = len(row.param_names)
        args = ", ".join(f"_udf_param_{i}" for i in range(n))
        if spark is not None and row.impl_class and (row.impl_binary or row.impl_source):
            from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

            binary = row.impl_binary
            class_name = row.impl_class
            if binary is None and row.impl_source:
                # Compiled under its content-addressed name (javac.unique_class_name),
                # which a row stored before that rename does not carry.
                compiled = compile_java(spark, row.impl_source, row.impl_class)
                binary, class_name = compiled.jar_bytes, compiled.class_name
            if row.target_kind == "gpu_kernel":
                gpu = _gpu_kernel_column(row, class_name, binary, args, return_type, spark)
                if gpu is not None:
                    return gpu
                # No RAPIDS here, the conf is off, or the wrapper would not
                # load: the row's java class is the verified CPU path.
            # Content-addressed like the verify path: a key-addressed name
            # would keep serving a stale jar after a re-verify stored new
            # bytes for the same key (the artifact cache refuses overwrite
            # and the classloader will not reload the old class name).
            fname = verify_function_name("ai_udf", binary)
            register_java_udf(spark, fname, class_name, binary, return_type)
            return F.expr(f"{fname}({args})").cast(return_type)
        if row.impl_source:
            logger.debug("java_udf impl_source without impl_class; cannot reconstruct")
            return None
    if row.target_kind == "rust":
        return _reconstruct_rust(row, return_type, spark)
    return None


def _reconstruct_rust(row: Any, return_type: Any, spark: Any) -> Any:
    """A Rust rewrite, or None wherever attaching it could fail the query.

    Spark cannot evaluate a Comet native UDF, so "not sure" means the Python
    UDF: target switched off, Comet not live, a master whose executors may
    not see the library. A library that is not built yet (a row from the
    shared table, a wiped cache) starts a background build and keeps Python
    for now -- UDF definition never waits on cargo.
    """
    from ai_udf_transpile import rustc

    if not row.impl_source or not rustc.supports_types(row.input_types, row.return_type):
        return None
    # A row from the shared table was verified by whoever wrote it, not here.
    # SQL and jars are served as-is; source about to become native code in
    # this process gets the scan again first.
    from ai_udf_transpile.verify import rejected_rewrite

    if rejected_rewrite(row.as_result()):
        logger.debug("rust rewrite for %s not served: forbidden construct", row.udf_key[:12])
        return None
    reason = rustc.rust_unservable(spark)
    if reason:
        logger.debug("rust rewrite for %s not served: %s", row.udf_key[:12], reason)
        return None
    compiled = rustc.cached_library(spark, row.impl_source, row.input_types, row.return_type)
    if compiled is None:
        rustc.compile_in_background(spark, row.impl_source, row.input_types, row.return_type)
        return None
    import pyspark.sql.functions as F

    rustc.register_rust_udf(spark, compiled, row.input_types, row.return_type)
    return F.expr(rustc.call_expr(compiled.name, row.input_types)).cast(return_type)


_NO_RUST_ROUTING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ai_udf_no_rust_routing", default=False
)


@contextlib.contextmanager
def no_rust_routing():
    """Resolve UDFs to their no-Rust twin key inside the block (the worker's
    smoke test of a twin row must reconstruct that row, not the Rust one)."""
    token = _NO_RUST_ROUTING.set(True)
    try:
        yield
    finally:
        _NO_RUST_ROUTING.reset(token)


def _rust_elsewhere(row: Any, spark: Any) -> bool:
    """A served Rust row this session could never attach (no target, no Comet)."""
    if row.target_kind != "rust":
        return False
    from ai_udf_transpile.rustc import rust_unservable

    return rust_unservable(spark) is not None


def _gpu_kernel_column(
    row: Any, class_name: str, binary: Any, args: str, return_type: Any, spark: Any
) -> Any:
    """The RapidsUDF wrapper of a gpu_kernel row as a Column, or None to serve the java class."""
    import pyspark.sql.functions as F

    from ai_udf_transpile.javac import register_java_udf, verify_function_name
    from ai_udf_transpile.rapids import serving_class

    wrapper = serving_class(spark, class_name, binary)
    if wrapper is None:
        return None
    try:
        fname = verify_function_name("ai_udf_gpu", binary)
        register_java_udf(spark, fname, wrapper, binary, return_type)
        return F.expr(f"{fname}({args})").cast(return_type)
    except Exception:
        logger.debug("gpu_kernel wrapper did not register; serving the java class", exc_info=True)
        return None


def _spark_version(spark: Any) -> str:
    if spark is None:
        return "unknown"
    try:
        return str(spark.version)
    except Exception:
        return "unknown"


def _inputs_allowed(in_cats: list[str], spark: Any) -> bool:
    allowed = conf.get_csv(conf.INPUT_CATEGORIES, spark, conf.DEFAULTS[conf.INPUT_CATEGORIES])
    return all(cat in allowed for cat in in_cats)


def _maybe_wrap_sampling(spark: Any, key: str, in_cats: list[str]) -> None:
    """Wrap the UDF-under-construction's func so real rows land in the samples table."""
    if spark is None or not ({"string", "map", "timestamp", "array"} & set(in_cats)):
        return
    try:
        if not conf.get_bool(conf.SAMPLING, spark, True):
            return
        if conf.get_value(conf.CATALOG, spark, conf.DEFAULTS[conf.CATALOG]).strip().lower() != "sqlite":
            return
        from ai_udf_transpile.sampling import udf_instance_from_spark_stack, wrap_for_sampling

        instance = udf_instance_from_spark_stack()
        func = getattr(instance, "func", None)
        if instance is None or func is None:
            return
        sqlite_path = conf.get_value(conf.SQLITE_PATH, spark, "")
        max_samples = conf.get_int(conf.MAX_SAMPLES, spark, int(conf.DEFAULTS[conf.MAX_SAMPLES]))
        instance.func = wrap_for_sampling(func, key, sqlite_path, max_samples)
    except Exception:
        logger.debug("could not wrap UDF for sampling", exc_info=True)


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
                func = func_from_spark_stack()
                function_ast = udf_function_ast(func, function_ast)
                if not types_known(function_ast, params, returnType):
                    return None
                in_types = input_spark_types(function_ast, params)
                in_cats = input_categories(function_ast, params)
                out_type = return_spark_type(returnType)
                if in_types is None or in_cats is None or out_type is None:
                    return None
                if not _inputs_allowed(in_cats, spark):
                    logger.debug("input categories %s gated out by conf", in_cats)
                    return None
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
                # A Rust row this session cannot run (no Comet, no target) gets
                # a twin key with its own SQL/Java rewrite rather than pin the
                # UDF to Python here. Decided on a side-effect-free read, before
                # lookup: lookup may park the row for approval or re-queue it,
                # and a session that never serves the row must not change it.
                peek = catalog.get(key)
                if _NO_RUST_ROUTING.get() or (peek is not None and _rust_elsewhere(peek, spark)):
                    version += NO_RUST_SUFFIX
                    key = udf_key(source, params, in_types, out_type, version, fingerprint)
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
                    _maybe_wrap_sampling(spark, key, in_cats)
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
