# SPDX-License-Identifier: Apache-2.0
"""SPARK-21935: the trivial upper() UDF that killed executors.

The reporter's Python UDF was just ``useragent.upper()`` over a Parquet
column -- but the Python worker memory overhead OOM'd every executor while
the Scala equivalent ran fine. It is the motivating case for this plugin:
transpiling to ``upper(col)`` removes the Python worker entirely.

It also has a semantic trap we test both sides of: Spark master's ``upper``
does full Unicode case mapping (ß -> SS), matching Python on realistic
data -- but codepoints assigned only in the JVM's newer Unicode (e.g.
U+A7D5, mapped to U+A7D4 by Unicode-16 JDKs, identity in Python 3.13's
15.1) genuinely diverge, and a sampled row containing one must reject the
rewrite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable
from ai_udf_transpile.backends.fake import upper_useragent
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

REAL_UA_SAMPLES = [
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"],
    ["curl/7.68.0"],
    ["Mozilla/5.0 (iPhone; CPU iPhone OS 13_5 like Mac OS X) AppleWebKit/605.1.15"],
    ["Apache-HttpClient/4.5.12 (Java/11.0.7)"],
]


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(upper_useragent),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string"],
        return_type="string",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_upper_rewrite_passes_with_real_user_agents(spark):
    ok, err = _check("upper(_udf_param_0)", spark, samples=REAL_UA_SAMPLES)
    assert ok, err


def test_initcap_lookalike_rejected(spark):
    ok, err = _check("initcap(_udf_param_0)", spark, samples=REAL_UA_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_lower_rejected(spark):
    ok, err = _check("lower(_udf_param_0)", spark, samples=REAL_UA_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_unicode_version_skew_sample_rejects(spark):
    # U+A7D5 LATIN SMALL LETTER DOUBLE WYNN: Unicode-16 JDKs map it to
    # U+A7D4; Python 3.13 (Unicode 15.1) leaves it unchanged. Only testable
    # where the JVM and Python actually disagree.
    skewed = spark.sql("SELECT upper('ꟕ') AS u").collect()[0][0] != "ꟕ".upper()
    if not skewed:
        pytest.skip("JVM and Python Unicode versions agree on U+A7D5")
    ok, err = _check("upper(_udf_param_0)", spark, samples=[["ꟕ"]])
    assert not ok
    assert "mismatch" in (err or "")


def test_end_to_end_transpile_removes_python_worker(spark, sqlite_path):
    # No None row: the original JIRA UDF (useragent.upper()) raises on None,
    # which would fail the query -- verification allows python-raise/sql-NULL,
    # but actually executing the Python UDF on null data errors out.
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    df = spark.createDataFrame([(s[0],) for s in REAL_UA_SAMPLES], ["ua"])
    expected = [s[0].upper() for s in REAL_UA_SAMPLES]

    first = UserDefinedFunction(upper_useragent, StringType())
    assert not first.transpiled
    assert [r[0] for r in df.select(first("ua")).collect()] == expected

    import time

    deadline = time.time() + 60
    second = UserDefinedFunction(upper_useragent, StringType())
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(upper_useragent, StringType())
    assert second.transpiled, "fake backend never landed upper(_udf_param_0)"
    assert [r[0] for r in df.select(second("ua")).collect()] == expected
    row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
    assert row[0] == "success"
    assert row[1] == "upper(_udf_param_0)"
