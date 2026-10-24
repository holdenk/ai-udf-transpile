# SPDX-License-Identifier: Apache-2.0
"""AI-only examples, no JVM: bytecode declines them, the fake backend
produces their SQL. These used to live in the spark-marked module and
never ran in the no-JVM CI leg."""

from __future__ import annotations

import pytest

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.backends.fake import FakeBackend, canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob

# (name, input_types, return_type, expected_sql, needs_numpy)
EXAMPLES = [
    ("title_case", "string", "string", "initcap(_udf_param_0)", False),
    ("trim_whitespace", "string", "string", "trim(BOTH ' \t\n\r\f\v' FROM _udf_param_0)", False),
    ("snake_case", "string", "string", "replace(_udf_param_0, ' ', '_')", False),
    ("lower_useragent", "string", "string", "lower(_udf_param_0)", False),
    (
        "pad_left5",
        "string",
        "string",
        "CASE WHEN length(_udf_param_0) >= 5 THEN _udf_param_0 ELSE lpad(_udf_param_0, 5, '0') END",
        False,
    ),
    ("clean_phone", "string", "string", "regexp_replace(_udf_param_0, '[^0-9]', '')", False),
    ("extract_domain", "string", "string", "regexp_extract(_udf_param_0, '://([^/]+)', 1)", False),
    ("is_email", "string", "boolean", "_udf_param_0 rlike '^[^@]+@[^@]+\\\\.[^@]+$'", False),
    ("count_vowels", "string", "bigint", "length(regexp_replace(_udf_param_0, '[^aeiouAEIOU]', ''))", False),
    ("np_sqrt", "double", "double", "sqrt(_udf_param_0)", True),
    ("np_log", "double", "double", "ln(_udf_param_0)", True),
    (
        "np_clip",
        ["double", "double", "double"],
        "double",
        "least(greatest(_udf_param_0, _udf_param_1), _udf_param_2)",
        True,
    ),
    ("np_exp", "double", "double", "exp(_udf_param_0)", True),
    ("math_floor", "double", "bigint", "floor(_udf_param_0)", False),
    ("math_ceil", "double", "bigint", "ceil(_udf_param_0)", False),
    ("math_log10", "double", "double", "log10(_udf_param_0)", False),
    ("to_iso_date", "timestamp", "string", "date_format(_udf_param_0, 'yyyy-MM-dd')", False),
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
        input_types=input_types if isinstance(input_types, list) else [input_types],
        input_categories=["numeric"] * len(params),
        return_type=return_type,
        captures={},
    )


def _spark_type(rtype):
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        LongType,
        StringType,
        TimestampType,
    )

    return {
        "string": StringType(),
        "bigint": LongType(),
        "double": DoubleType(),
        "boolean": BooleanType(),
        "timestamp": TimestampType(),
    }[rtype]


@pytest.mark.parametrize("name,itype,rtype,sql,needs_np", EXAMPLES)
def test_bytecode_declines(name, itype, rtype, sql, needs_np):
    if needs_np:
        pytest.importorskip("numpy")
    func = _func(name)
    backend = BytecodeBackend()
    job = _job_for(func, itype, rtype)
    with pytest.raises(BackendDecline):
        with make_sandbox(job) as sandbox:
            backend.run(job, sandbox)


@pytest.mark.parametrize("name,itype,rtype,sql,needs_np", EXAMPLES)
def test_fake_backend_produces_sql(name, itype, rtype, sql, needs_np):
    if needs_np:
        pytest.importorskip("numpy")
    func = _func(name)
    backend = FakeBackend()
    job = _job_for(func, itype, rtype)
    with make_sandbox(job) as sandbox:
        result = backend.run(job, sandbox)
    assert result.kind == KIND_CATALYST
    assert result.sql == sql
