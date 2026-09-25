# SPDX-License-Identifier: Apache-2.0
"""scatter_to_seconds: an array<string> return, and the silent-empty-list traps.

    def scatter_to_seconds(start, duration):
        ret = []
        try:
            start = str(start)
            duration = int(float(duration))
            if len(start) < 19:
                return ret
            start = start[:19]
            start_struct = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            for i in range(duration + 1):
                cur = (start_struct + datetime.timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
                ret.append(cur)
            return ret
        except:
            return ret

Every failure path -- null/short/unparseable start, non-numeric/NaN/inf/negative
duration -- returns [] via the bare except, so the faithful rewrite is a CASE
that yields array() in all those cases and otherwise a
transform(sequence(start, start + duration seconds, 1s), x -> date_format(x))
whose lambda closes over only its own argument (Spark's _udf_param_N
substitution does not descend into higher-order function lambda bodies).
The classic traps: off-by-one (range(duration + 1)), lenient timestamp
parsing (strptime rejects '2015-01-01'; Spark's default parse accepts it as
midnight), and NULL where python yields []. This is also the first supported
UDF with an array return type.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable, register_impl
from ai_udf_transpile.backends.fake import scatter_to_seconds
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.types import return_spark_type
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

# NOTE: _udf_param_N refs stay OUTSIDE the transform lambda on purpose --
# Spark's TranspiledPythonUDF placeholder substitution does not descend into
# higher-order function lambda bodies, so the lambda closes over only its own
# argument and literals.
FAITHFUL_SQL = (
    "CASE WHEN _udf_param_0 IS NULL THEN array() "
    "WHEN length(_udf_param_0) < 19 THEN array() "
    "WHEN try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss') IS NULL THEN array() "
    "WHEN try_cast(_udf_param_1 AS DOUBLE) IS NULL THEN array() "
    "WHEN isnan(try_cast(_udf_param_1 AS DOUBLE)) THEN array() "
    "WHEN abs(try_cast(_udf_param_1 AS DOUBLE)) = cast('inf' AS DOUBLE) THEN array() "
    "WHEN cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) < 0 THEN array() "
    "ELSE transform(sequence("
    "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss'), "
    "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT), "
    "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
    "interval 1 second), "
    "x -> date_format(x, 'yyyy-MM-dd HH:mm:ss')) END"
)
# The classic off-by-one: range(duration) instead of range(duration + 1),
# i.e. the sequence stops one second short.
OFF_BY_ONE_SQL = FAITHFUL_SQL.replace(
    "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT),",
    "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) - 1,",
)
# strptime with an explicit format rejects '2015-01-01'; Spark's default parse
# accepts it as midnight -- so dropping the length gate + format is observable.
LENIENT_SQL = FAITHFUL_SQL.replace(
    "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')",
    "try_to_timestamp(substr(_udf_param_0, 1, 19))",
).replace("WHEN length(_udf_param_0) < 19 THEN array() ", "")
# python returns [] for a null start, not NULL.
NULL_START_SQL = FAITHFUL_SQL.replace(
    "WHEN _udf_param_0 IS NULL THEN array()", "WHEN _udf_param_0 IS NULL THEN NULL"
)
# Value-equivalent (this exact shape passed the 20-case empirical battery) but
# the transform lambda closes over _udf_param_0 -- and TranspiledPythonUDF
# placeholder substitution does not descend into lambda bodies, so this takes
# the user's query down with UNRESOLVED_COLUMN at analysis. A real backend
# (coco) produced exactly this shape.
LAMBDA_CLOSING_SQL = (
    "CASE WHEN _udf_param_0 IS NULL THEN array() "
    "WHEN length(_udf_param_0) < 19 THEN array() "
    "WHEN try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss') IS NULL THEN array() "
    "WHEN try_cast(_udf_param_1 AS DOUBLE) IS NULL THEN array() "
    "WHEN isnan(try_cast(_udf_param_1 AS DOUBLE)) THEN array() "
    "WHEN abs(try_cast(_udf_param_1 AS DOUBLE)) = cast('inf' AS DOUBLE) THEN array() "
    "WHEN cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) < 0 THEN array() "
    "ELSE transform(sequence(0, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT)), "
    "i -> date_format(timestampadd(SECOND, i, "
    "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
    "'yyyy-MM-dd HH:mm:ss')) END"
)
# Missing the isnan/inf guards: 'nan'/'inf' parse as doubles, then int()
# misbehaves (NULL or clamped) where python's int(float(...)) raises -> [].
NO_NAN_INF_GUARDS_SQL = FAITHFUL_SQL.replace(
    "WHEN isnan(try_cast(_udf_param_1 AS DOUBLE)) THEN array() ", ""
).replace("WHEN abs(try_cast(_udf_param_1 AS DOUBLE)) = cast('inf' AS DOUBLE) THEN array() ", "")

SCATTER_SAMPLES = [
    ["2015-01-01 00:00:00", "2"],
    ["2015-01-01", "5"],  # short start: []
    [None, "2"],  # null start: []
    ["2015-01-01 00:00:00", "abc"],  # unparseable duration: []
    ["2015-12-31 23:59:59", "2"],  # year rollover
]


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(scatter_to_seconds),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string", "string"],
        return_type="array<string>",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_array_return_type_accepted():
    assert return_spark_type(ArrayType(StringType())) == "array<string>"
    assert return_spark_type("array<string>") == "array<string>"
    assert return_spark_type("array<struct<a:int>>") is None


def test_bug_faithful_rewrite_passes(spark):
    ok, err = _check(FAITHFUL_SQL, spark, samples=SCATTER_SAMPLES)
    assert ok, err


def test_off_by_one_rejected(spark):
    ok, err = _check(OFF_BY_ONE_SQL, spark, samples=SCATTER_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_lenient_parse_rejected(spark):
    ok, err = _check(LENIENT_SQL, spark, samples=SCATTER_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_null_instead_of_empty_rejected(spark):
    ok, err = _check(NULL_START_SQL, spark, samples=SCATTER_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_missing_nan_inf_guards_rejected_without_samples(spark):
    # Deterministic: the cross-combined built-ins pair a valid timestamp start
    # with 'nan'/'inf' durations. The one-param-at-a-time built-ins never
    # produce that pair (the other param is always NULL), which is how a
    # guard-less rewrite slips through value verification otherwise.
    ok, err = _check(NO_NAN_INF_GUARDS_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "") or "raised" in (err or "")


def test_lambda_closing_param_passes_values_but_fails_reconstruction(spark, sqlite_path):
    # Verify binds params as ordinary columns, where lambda outer references
    # resolve -- so this passes value verification...
    ok, err = _check(LAMBDA_CLOSING_SQL, spark, samples=SCATTER_SAMPLES)
    assert ok, err
    # ...but must be rejected by the reconstruction smoke test, because the
    # real TranspiledPythonUDF path leaves _udf_param_0 unresolved inside the
    # transform lambda and the query fails at analysis.
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError, match="reconstruction"):
        register_impl(
            spark,
            scatter_to_seconds,
            kind="catalyst",
            catalyst_sql=LAMBDA_CLOSING_SQL,
            return_type=ArrayType(StringType()),
        )


def _record_samples(spark, sqlite_path):
    """Run a sampling-wrapped UDF over the sample rows so register_impl verifies
    against real inputs (built-in examples vary one param at a time, so a valid
    (start, duration) pair never occurs without them)."""
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    df = spark.createDataFrame([(s[0], s[1]) for s in SCATTER_SAMPLES], ["start", "duration"])
    sampling_udf = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
    df.select(sampling_udf("start", "duration")).collect()


def test_register_impl_rejects_off_by_one(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            scatter_to_seconds,
            kind="catalyst",
            catalyst_sql=OFF_BY_ONE_SQL,
            return_type=ArrayType(StringType()),
        )


def test_register_impl_accepts_faithful(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    register_impl(
        spark,
        scatter_to_seconds,
        kind="catalyst",
        catalyst_sql=FAITHFUL_SQL,
        return_type=ArrayType(StringType()),
    )
    row = get_catalog()._conn.execute("SELECT status, origin, catalyst_sql FROM cache").fetchone()
    assert tuple(row) == ("success", "human", FAITHFUL_SQL)


class _StubBackend:
    """Hands back a fixed SQL string, like a CLI backend that answered."""

    name = "stub"

    def __init__(self, sql):
        self._sql = sql

    def run(self, job, sandbox):
        return TranspileResult(kind=KIND_CATALYST, sql=self._sql)


def _process_one_with_stub(spark, sqlite_path, sql):
    """Queue scatter_to_seconds for real (so the cache key matches what the
    UDF-construction hook recomputes), then process the row with a stub
    backend, skipping value verification to isolate the reconstruction gate."""
    from ai_udf_transpile.worker import process_row

    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))  # cache miss -> pending
    catalog = get_catalog()
    row = catalog.oldest_pending()
    assert row is not None, "UDF was never queued"
    assert catalog.claim(row.udf_key, "stub")
    process_row(
        catalog,
        catalog.get(row.udf_key),
        _StubBackend(sql),
        spark=spark,
        verify_fn=lambda **k: (True, None),  # values verified; isolate reconstruction
    )
    return catalog.get(row.udf_key)


def test_worker_marks_lambda_closing_sql_failed(spark, sqlite_path):
    # The coco scenario: value-correct SQL whose transform lambda closes over
    # _udf_param_0. Hypothesis passes it; the reconstruction smoke test must
    # flip the row to failed so it never reaches a user query.
    row = _process_one_with_stub(spark, sqlite_path, LAMBDA_CLOSING_SQL)
    assert row.status == "failed"
    assert "fails plan analysis" in (row.error or "")
    assert "_udf_param_0" in (row.error or "")


def test_worker_accepts_lambda_free_sql(spark, sqlite_path):
    # Positive control: the faithful SQL keeps params outside the lambda and
    # survives reconstruction through the worker path (which rebuilds the UDF
    # from source text, no live func).
    row = _process_one_with_stub(spark, sqlite_path, FAITHFUL_SQL)
    assert row.status == "success", row.error
    assert row.catalyst_sql == FAITHFUL_SQL


def test_end_to_end_transpile_with_null_and_short_rows(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    rows = [(s[0], s[1]) for s in SCATTER_SAMPLES]
    df = spark.createDataFrame(rows, ["start", "duration"])
    expected = [scatter_to_seconds(*s) for s in SCATTER_SAMPLES]
    assert expected[1] == [] and expected[2] == [] and expected[3] == []

    first = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
    assert not first.transpiled
    assert [r[0] for r in df.select(first("start", "duration")).collect()] == expected

    deadline = time.time() + 60
    second = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
    assert second.transpiled, "fake backend never landed the bug-faithful rewrite"
    assert [r[0] for r in df.select(second("start", "duration")).collect()] == expected
    row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
    assert row[0] == "success"
    assert row[1] == FAITHFUL_SQL
