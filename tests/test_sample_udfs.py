# SPDX-License-Identifier: Apache-2.0
"""Tests for sample UDFs from Spark tests, TPC-DS, and common patterns."""

from __future__ import annotations

import pytest

from ai_udf_transpile.backends.fake import FakeBackend, canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob

# (name, func, input_types, return_type, expected_sql)
EXAMPLES = [
    # From apache/spark test_udf.py
    ("spark_add_one", ["bigint"], "bigint", "CASE WHEN _udf_param_0 IS NOT NULL THEN _udf_param_0 + 1 END"),
    ("spark_to_upper", ["string"], "string", "upper(_udf_param_0)"),
    ("spark_to_lower", ["string"], "string", "lower(_udf_param_0)"),
    ("spark_trunc", ["double"], "bigint", "cast(_udf_param_0 as bigint)"),
    ("spark_as_double", ["bigint"], "double", "cast(_udf_param_0 as double)"),
    # From TPC-DS modified queries
    (
        "tpcds_zip5_mismatch",
        ["string", "string"],
        "boolean",
        "substr(_udf_param_0, 1, 5) <> substr(_udf_param_1, 1, 5)",
    ),
    (
        "tpcds_safe_ratio",
        ["double", "double"],
        "double",
        "CASE WHEN _udf_param_1 > 0 THEN _udf_param_0 / _udf_param_1 END",
    ),
    (
        "tpcds_sales_deviation",
        ["double", "double"],
        "double",
        "CASE WHEN _udf_param_1 > 0 THEN abs(_udf_param_0 - _udf_param_1) / _udf_param_1 END",
    ),
    (
        "tpcds_in_quarter",
        ["bigint", "bigint", "bigint"],
        "boolean",
        "CASE WHEN _udf_param_0 IS NULL OR _udf_param_1 IS NULL OR _udf_param_2 IS NULL THEN NULL "
        "ELSE _udf_param_0 BETWEEN _udf_param_1 AND _udf_param_1 + _udf_param_2 END",
    ),
    # Common real-world patterns
    ("common_discount_price", ["double", "double"], "double", "_udf_param_0 * (1 - _udf_param_1 / 100.0)"),
    (
        "common_pct_change",
        ["double", "double"],
        "double",
        "CASE WHEN _udf_param_0 <> 0 THEN (_udf_param_1 - _udf_param_0) / _udf_param_0 * 100.0 END",
    ),
    ("common_c2f", ["double"], "double", "_udf_param_0 * 9.0 / 5.0 + 32.0"),
    ("common_url_decode", ["string"], "string", "replace(_udf_param_0, '%20', ' ')"),
]


def _func(name):
    from ai_udf_transpile.backends import fake as fake_mod

    return getattr(fake_mod, name)


def _job_for(func, input_types, return_type):
    src = canonical_source_from_func(func)
    params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    return TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=params,
        input_types=input_types,
        input_categories=["numeric"] * len(params),
        return_type=return_type,
        captures={},
    )


@pytest.mark.parametrize("name,itype,rtype,sql", EXAMPLES)
def test_fake_backend_produces_expected_sql(name, itype, rtype, sql):
    func = _func(name)
    backend = FakeBackend()
    job = _job_for(func, itype, rtype)
    with make_sandbox(job) as sandbox:
        result = backend.run(job, sandbox)
    assert result.kind == KIND_CATALYST
    assert result.sql == sql
