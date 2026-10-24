# SPDX-License-Identifier: Apache-2.0
"""AI-only examples: UDFs that only the AI backend can transpile.

Every function below uses method calls (.title, .strip, .replace, ...),
imports (re, math, numpy), or module-level functions that neither the
static transpiler (no rules for them) nor the bytecode backend
(LOAD_ATTR / IMPORT_NAME / CALL unsupported) can handle.
"""

from __future__ import annotations

import pytest

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.backends.fake import FakeBackend, canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob, TranspileResult

# (name, func, input_types, return_type, expected_sql, needs_numpy)
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


# --- Unit tests (no Spark): bytecode declines, fake backend produces SQL ---


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


# --- Spark tests: static declines, AI SQL matches Python via Hypothesis ---

pytestmark = pytest.mark.spark

# Hypothesis-friendly subset: functions where the SQL faithfully matches
# Python for all generated inputs.  Excluded:
#   title_case   initcap disagrees on apostrophes (word boundary differs)
#   trim_ws      Python strip() handles more whitespace (e.g. \x1d) than Spark trim
#   np_log/log10  log of <=0 diverges: numpy nan vs Spark NULL
#   np_sqrt       nan != nan in IEEE 754 comparison
#   np_exp        numpy vs Spark float precision mismatch
#   to_iso_date   datetime input type not handled by the transpile hook
HYPOTHESIS_OK = [
    e
    for e in EXAMPLES
    if e[0]
    not in ("title_case", "trim_whitespace", "np_log", "math_log10", "np_sqrt", "np_exp", "to_iso_date")
]


@pytest.mark.parametrize("name,itype,rtype,sql,needs_np", EXAMPLES)
def test_static_transpiler_declines(spark, name, itype, rtype, sql, needs_np):
    if needs_np:
        pytest.importorskip("numpy")
    from pyspark.sql.udf import UserDefinedFunction

    func = _func(name)
    spark.conf.set("spark.sql.experimental.optimizer.transpilePyUDFs", "true")
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "catalyst")
    udf = UserDefinedFunction(func, _spark_type(rtype))
    assert not udf.transpiled, f"static transpiler unexpectedly converted {name}"


@pytest.mark.parametrize("name,itype,rtype,sql,needs_np", HYPOTHESIS_OK)
def test_ai_sql_matches_python(spark, sqlite_path, name, itype, rtype, sql, needs_np):
    if needs_np:
        pytest.importorskip("numpy")
    from pyspark.sql.udf import UserDefinedFunction

    from ai_udf_transpile import conf, enable
    from ai_udf_transpile.verify import hypothesis_check
    from ai_udf_transpile.worker import poll_once

    func = _func(name)
    catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    # The fixture SQL was written against string contents, not just NULL.
    conf.set_value(conf.PARTY_ON_STRINGS, "true")
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(func, _spark_type(rtype))
    assert poll_once(catalog, FakeBackend(), spark)
    ok, err = hypothesis_check(
        source_text=canonical_source_from_func(func),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=[itype] if not isinstance(itype, list) else itype,
        return_type=rtype,
        spark=spark,
        max_examples=50,
        func=func,
        party_on_strings=True,
    )
    assert ok, f"AI SQL for {name} failed Hypothesis: {err}"


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
        "boolean": BooleanType(),
        "double": DoubleType(),
        "timestamp": TimestampType(),
    }[rtype]
