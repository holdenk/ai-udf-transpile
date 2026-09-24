# SPDX-License-Identifier: Apache-2.0
"""AI-backed Python UDF transpilation for Apache Spark."""

from __future__ import annotations

import ast
import inspect
import logging
import os
import textwrap
import warnings
from typing import Any, Callable, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.keys import (
    canonical_source,
    canonical_source_from_func,
    closure_fingerprint,
    extract_captures,
    udf_key,
)
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileResult
from ai_udf_transpile.types import (
    input_categories,
    input_spark_types,
    return_spark_type,
    types_known,
)

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _function_ast(func: Callable[..., Any]) -> ast.FunctionDef:
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return fn


def _public_params(func: Callable[..., Any], function_ast: ast.FunctionDef) -> list[str]:
    params = [arg.arg for arg in function_ast.args.args]
    if inspect.ismethod(func):
        return params[1:]
    return params


def _default_sqlite_path(spark: Any) -> str:
    local_dir = "/tmp"
    if spark is not None:
        try:
            local_dir = spark.conf.get("spark.local.dir", "/tmp") or "/tmp"
        except Exception:
            local_dir = "/tmp"
    if "," in local_dir:
        local_dir = local_dir.split(",")[0].strip()
    return os.path.join(local_dir, "ai_udf_transpile.sqlite")


def _ansi_on(spark: Any) -> bool:
    try:
        value = spark.conf.get("spark.sql.ansi.enabled")
        return value is not None and str(value).lower() == "true"
    except Exception:
        return False


def enable(
    spark: Any,
    *,
    catalog: Optional[str] = None,
    table: Optional[str] = None,
    backend: Optional[str] = None,
    inline_worker: Optional[bool] = None,
    sqlite_path: Optional[str] = None,
) -> Any:
    """Register the ``ai`` transpiler, open the catalog, optionally start the inline worker."""
    from ai_udf_transpile.catalog import open_catalog
    from ai_udf_transpile.catalog.delta import require_delta
    from ai_udf_transpile.transpiler import register_transpiler, set_catalog, set_session
    from ai_udf_transpile.worker import maybe_start_inline, stop_inline

    register_transpiler()
    set_session(spark)

    kind = (catalog or conf.DEFAULTS[conf.CATALOG]).strip().lower()
    if kind == "delta":
        require_delta(spark)
    conf.set_value(conf.CATALOG, kind, spark)

    path = str(sqlite_path) if sqlite_path is not None else _default_sqlite_path(spark)
    conf.set_value(conf.SQLITE_PATH, path, spark)
    if table is not None:
        conf.set_value(conf.TABLE, table, spark)
    chosen_backend = backend if backend is not None else conf.default_backend()
    conf.set_value(conf.BACKEND, chosen_backend, spark)
    conf.set_value(conf.MAX_EXAMPLES, conf.default_max_examples(), spark)

    try:
        spark.conf.set("spark.sql.experimental.optimizer.transpilePyUDFs", "true")
    except Exception:
        logger.debug("could not set transpilePyUDFs", exc_info=True)

    if not _ansi_on(spark):
        warnings.warn(
            "Python UDF transpilation requires spark.sql.ansi.enabled=true; "
            "the AI hook is registered but Spark will skip transpilation until ANSI is on.",
            RuntimeWarning,
        )

    try:
        current = spark.conf.get("spark.sql.experimental.optimizer.pyTranspilers", "catalyst") or "catalyst"
    except Exception:
        current = "catalyst"
    names = [n.strip() for n in current.split(",") if n.strip()]
    if "catalyst" not in names:
        names.insert(0, "catalyst")
    if "ai" not in names:
        names.append("ai")
    try:
        spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", ",".join(names))
    except Exception:
        logger.debug("could not set pyTranspilers", exc_info=True)

    opened = open_catalog(spark, sqlite_path=path)
    set_catalog(opened)

    use_inline = (
        conf.get_bool(conf.INLINE_WORKER, spark, True) if inline_worker is None else bool(inline_worker)
    )
    conf.set_value(conf.INLINE_WORKER, "true" if use_inline else "false", spark)
    if use_inline:
        maybe_start_inline(spark, opened)
    else:
        stop_inline()
    return opened


def shutdown() -> None:
    """Stop the inline worker and drop session/catalog globals (for tests)."""
    from ai_udf_transpile.transpiler import set_catalog, set_session
    from ai_udf_transpile.worker import stop_inline

    stop_inline()
    set_catalog(None)
    set_session(None)
    conf.reset_runtime()


def register_impl(
    spark: Any,
    func: Callable[..., Any],
    *,
    kind: str = KIND_CATALYST,
    catalyst_sql: Optional[str] = None,
    impl_source: Optional[str] = None,
    impl_class: Optional[str] = None,
    impl_binary: Optional[bytes] = None,
    impl_entry: Optional[str] = None,
    input_types: Optional[list[str]] = None,
    return_type: Any = None,
    verify: bool = True,
) -> Any:
    """Write a human-provided rewrite into the catalog (origin='human')."""
    from ai_udf_transpile.backends.human import ORIGIN
    from ai_udf_transpile.transpiler import get_catalog
    from ai_udf_transpile.verify import hypothesis_check

    catalog = get_catalog()
    if catalog is None:
        raise RuntimeError("call ai_udf_transpile.enable(spark) before register_impl")

    function_ast = _function_ast(func)
    params = _public_params(func, function_ast)
    declared_return = return_type
    if declared_return is None:
        raise ValueError("register_impl requires return_type (Spark DataType or simpleString)")
    if not types_known(function_ast, params, declared_return) and not (
        input_types and return_spark_type(declared_return)
    ):
        raise ValueError("register_impl requires known input and output types")
    in_types = input_types or input_spark_types(function_ast, params)
    in_cats = input_categories(function_ast, params) or ["numeric"] * len(params)
    out_type = return_spark_type(declared_return)
    if in_types is None or out_type is None:
        raise ValueError("register_impl requires known input_types and return_type")
    captures = extract_captures(func, function_ast, params)
    if captures is None:
        raise ValueError("register_impl cannot key this UDF (unsupported captures)")
    source = canonical_source(function_ast)
    fingerprint = closure_fingerprint(captures)
    version = str(getattr(spark, "version", "unknown"))
    key = udf_key(source, params, in_types, out_type, version, fingerprint)
    result = TranspileResult(
        kind=kind,
        sql=catalyst_sql,
        java_source=impl_source,
        class_name=impl_class,
        binary=impl_binary,
        entry=impl_entry,
    )
    if kind not in {KIND_CATALYST, KIND_JAVA_UDF}:
        raise ValueError(f"v1 register_impl kind must be catalyst or java_udf, not {kind}")
    if not result.reconstructable():
        raise ValueError("register_impl needs catalyst_sql or a java_udf payload")

    hyp: Optional[bool] = None
    if verify:
        ok, err = hypothesis_check(
            source_text=source,
            captures=captures,
            result=result,
            input_types=in_types,
            return_type=out_type,
            spark=spark,
            max_examples=conf.get_int(conf.MAX_EXAMPLES, spark, int(conf.default_max_examples())),
            func=func,
        )
        if not ok:
            if catalog.get(key) is None:
                catalog.insert_pending(
                    udf_key=key,
                    source_text=source,
                    param_names=params,
                    input_types=in_types,
                    input_categories=in_cats,
                    return_type=out_type,
                    spark_version=version,
                    closure_fingerprint=fingerprint,
                    captures=captures,
                )
            catalog.mark_failed(key, err or "register_impl verify failed")
            raise ValueError(f"register_impl Hypothesis failed: {err}")
        hyp = True

    catalog.upsert_success(
        udf_key=key,
        source_text=source,
        param_names=params,
        input_types=in_types,
        input_categories=in_cats,
        return_type=out_type,
        spark_version=version,
        closure_fingerprint=fingerprint,
        captures=captures,
        result=result,
        origin=ORIGIN,
        hypothesis_passed=hyp,
    )
    return key


__all__ = [
    "__version__",
    "enable",
    "register_impl",
    "shutdown",
    "canonical_source_from_func",
]
