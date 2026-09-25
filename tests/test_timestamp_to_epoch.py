# SPDX-License-Identifier: Apache-2.0
"""timestamp_to_epoch: a pandas_udf over NYC taxi timestamps, and the 'nan' trap.

    def timestamp_to_epoch(t):
        return t.dt.strftime("%s").apply(str)

    f = pandas_udf(timestamp_to_epoch, returnType=StringType())

Two things our pipeline does not handle as pasted: it is a *pandas* UDF
(Series API, and untyped -> never queued), and its input is a timestamp.
Modernized here to a typed scalar UDF preserving the observable behavior --
including the quirk that NaT goes through strftime as NaN and .apply(str)
turns it into the string 'nan'. That quirk is a semantic trap: the naive
rewrite cast(unix_timestamp(t) as string) returns NULL for NULL, which is
not 'nan', so it must be rejected; the bug-faithful rewrite coalesces to
'nan' and passes.
"""

from __future__ import annotations

import datetime
import json
import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, register_impl
from ai_udf_transpile.backends.fake import timestamp_to_epoch
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sampling import _encode, decode_args
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

NAIVE_SQL = "cast(unix_timestamp(_udf_param_0) as string)"  # NULL, not 'nan'
# date_trunc first: python's strftime('%s') drops microseconds *before* the
# epoch conversion, which differs from truncating a negative epoch afterward.
FAITHFUL_SQL = "coalesce(cast(unix_timestamp(date_trunc('SECOND', _udf_param_0)) as string), 'nan')"

TAXI_SAMPLES = [
    [datetime.datetime(2015, 1, 1, 0, 12, 0)],  # tpep_pickup_datetime style
    [datetime.datetime(1969, 12, 31, 23, 59, 59)],  # pre-1970: negative epoch
    [None],  # pandas path yields 'nan'
]


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(timestamp_to_epoch),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["timestamp"],
        return_type="string",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_default_gate_blocks_timestamp(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(timestamp_to_epoch, StringType())
    assert get_catalog().count() == 0, "timestamp-input UDF queued despite default gate"


def test_gate_with_timestamp_category_allows_queue(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map,timestamp")
    UserDefinedFunction(timestamp_to_epoch, StringType())
    assert get_catalog().count() == 1


def test_naive_unix_timestamp_rejected_nan_trap(spark):
    # None -> python 'nan', naive SQL NULL. Deterministic: the built-in
    # timestamp examples include None.
    ok, err = _check(NAIVE_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "")
    assert "nan" in (err or "")


def test_bug_faithful_rewrite_passes(spark):
    ok, err = _check(FAITHFUL_SQL, spark, samples=TAXI_SAMPLES)
    assert ok, err


def test_microsecond_truncation_trap_rejected(spark):
    # Null-safe but converts before dropping micros: for negative epochs
    # python floors (drops micros first), unix_timestamp truncates toward
    # zero. Found by Hypothesis on datetime(1949, 3, 5, 17, 42, 4, 56290).
    null_safe_only = "coalesce(cast(unix_timestamp(_udf_param_0) as string), 'nan')"
    ok, err = _check(null_safe_only, spark, samples=[[datetime.datetime(1949, 3, 5, 17, 42, 4, 56290)]])
    assert not ok
    assert "mismatch" in (err or "")
    assert "-657181076" in (err or "")


def test_register_impl_rejects_naive(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            timestamp_to_epoch,
            kind="catalyst",
            catalyst_sql=NAIVE_SQL,
            return_type=StringType(),
        )


def test_datetime_sample_roundtrip():
    args = [datetime.datetime(2015, 1, 1, 0, 12, 0), None]
    decoded = decode_args(json.dumps([_encode(v) for v in args]))
    assert decoded == args


def test_end_to_end_transpile_with_null_and_pre1970(spark, sqlite_path):
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map,timestamp")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    df = spark.createDataFrame([(s[0],) for s in TAXI_SAMPLES], ["tpep_pickup_datetime"])
    expected = [timestamp_to_epoch(s[0]) for s in TAXI_SAMPLES]
    assert expected[2] == "nan"  # the quirk being preserved

    first = UserDefinedFunction(timestamp_to_epoch, StringType())
    assert not first.transpiled
    assert [r[0] for r in df.select(first("tpep_pickup_datetime")).collect()] == expected

    deadline = time.time() + 60
    second = UserDefinedFunction(timestamp_to_epoch, StringType())
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(timestamp_to_epoch, StringType())
    assert second.transpiled, "fake backend never landed the bug-faithful rewrite"
    assert [r[0] for r in df.select(second("tpep_pickup_datetime")).collect()] == expected
    row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
    assert row[0] == "success"
    assert row[1] == FAITHFUL_SQL
