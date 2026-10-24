# SPDX-License-Identifier: Apache-2.0
"""AI-only examples, Spark half: the built-in static transpiler declines
them and the fake SQL matches Python under Hypothesis. The no-JVM half
(bytecode declines, fake SQL shape) lives in test_ai_only_examples_unit.py
so the lint/unit CI leg runs it."""

from __future__ import annotations

import pytest
from test_ai_only_examples_unit import EXAMPLES, _func, _spark_type

from ai_udf_transpile import conf, enable
from ai_udf_transpile.backends.fake import FakeBackend, canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.verify import hypothesis_check
from ai_udf_transpile.worker import poll_once

pytestmark = pytest.mark.spark

TRANSPILE_CONF = "spark.sql.experimental.optimizer.transpilePyUDFs"
PY_TRANSPLIERS_CONF = "spark.sql.experimental.optimizer.pyTranspilers"

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
    spark.conf.set(TRANSPILE_CONF, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "catalyst")
    try:
        udf = UserDefinedFunction(func, _spark_type(rtype))
        assert not udf.transpiled, f"static transpiler unexpectedly converted {name}"
    finally:
        spark.conf.set(TRANSPILE_CONF, "false")
        spark.conf.unset(PY_TRANSPLIERS_CONF)


@pytest.mark.parametrize("name,itype,rtype,sql,needs_np", HYPOTHESIS_OK)
def test_ai_sql_matches_python(spark, sqlite_path, name, itype, rtype, sql, needs_np):
    if needs_np:
        pytest.importorskip("numpy")
    from pyspark.sql.udf import UserDefinedFunction

    func = _func(name)
    catalog = enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    # The fixture SQL was written against string contents, not just NULL.
    conf.set_value(conf.PARTY_ON_STRINGS, "true")
    spark.conf.set(PY_TRANSPLIERS_CONF, "ai")
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
