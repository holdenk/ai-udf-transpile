# SPDX-License-Identifier: Apache-2.0
"""flatlist: a one-level flatten UDF over array<array<string>>.

The contributed original is untyped (never queued by the plugin) and builds
the result into a local before returning it:

    def flatlist(s):
        fl = [item for sublist in s for item in sublist]
        return fl

    flatlist_function_udf = F.udf(flatlist, ArrayType(StringType()))

The fixture modernizes with annotations. Empirically (battery over nulls,
empties at both levels, duplicates, order, and null elements):

- flatten(_udf_param_0) is FAITHFUL: it preserves order, duplicates, and
  null elements ([['a', None], ['b']] -> ['a', None, 'b'] on both sides).
- A null outer or inner array raises TypeError in python (`for x in None`);
  flatten returns NULL there, which the verify policy allows (python raise).
- explode/collect_list-style rewrites DROP null elements, and
  array_distinct drops duplicates -- both must be rejected, so the nested
  array strategy generates null elements and the built-in nested examples
  include a duplicate-across-inners case.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, register_impl
from ai_udf_transpile.backends.fake import flatlist
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sampling import _encode, decode_args
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

ARRAY_OF_STRING = ArrayType(StringType())

FAITHFUL_SQL = "flatten(_udf_param_0)"
# Drops duplicates: [['x', 'y'], ['x']] flattens to ['x', 'y', 'x'].
DISTINCT_SQL = "array_distinct(flatten(_udf_param_0))"
# Drops null elements: python passes them through (and flatten preserves them).
FILTER_NULLS_SQL = "filter(flatten(_udf_param_0), x -> x IS NOT NULL)"
# Sorts: python preserves first-seen order.
SORT_SQL = "sort_array(flatten(_udf_param_0))"

FLAT_SAMPLES = [
    [["a", "b"], ["c"]],
    [[]],
    [["a", None], ["b"]],
    [["x", "y"], ["x"]],
    None,
]

ARRAY_CATEGORIES = "numeric,string,bool,binary,array"


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(flatlist),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["array<array<string>>"],
        return_type="array<string>",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


# SPARK-55206: _transpile_func returns before any registered pyTranspiler,
# including "ai", when the declared return type is not numeric, string,
# boolean, or binary. array<string> cannot be cast under ANSI rules, and a
# transpiled option is a child of TranspiledPythonUDF, so CheckAnalysis would
# fail the whole query instead of falling back to interpreted Python. The
# hook never runs, so these cannot tell our type gate from Spark's refusal.
# https://issues.apache.org/jira/browse/SPARK-55206
#
# def test_pasted_untyped_never_queued(spark, sqlite_path):
#     # The contributed original has no annotations -> the type gate declines.
#     def flatlist_pasted(s):
#         fl = [item for sublist in s for item in sublist]
#         return fl
#
#     enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
#     spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
#     conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
#     UserDefinedFunction(flatlist_pasted, ARRAY_OF_STRING)
#     assert get_catalog().count() == 0, "untyped UDF queued despite the type gate"
#
#
# def test_default_gate_blocks_array(spark, sqlite_path):
#     enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
#     spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
#     UserDefinedFunction(flatlist, ARRAY_OF_STRING)
#     assert get_catalog().count() == 0, "array-input UDF queued despite default gate"


# SPARK-55206: _transpile_func returns before any registered pyTranspiler,
# including "ai", when the declared return type is not numeric, string,
# boolean, or binary. array<string> cannot be cast under ANSI rules, and a
# transpiled option is a child of TranspiledPythonUDF, so CheckAnalysis would
# fail the whole query instead of falling back to interpreted Python. The
# hook never runs, so this cannot observe a queue.
# https://issues.apache.org/jira/browse/SPARK-55206
#
# def test_gate_with_array_category_allows_queue(spark, sqlite_path):
#     enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
#     spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
#     conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
#     UserDefinedFunction(flatlist, ARRAY_OF_STRING)
#     assert get_catalog().count() == 1


def test_flatten_faithful_passes(spark):
    ok, err = _check(FAITHFUL_SQL, spark, samples=FLAT_SAMPLES)
    assert ok, err


def test_array_distinct_rejected(spark):
    # Deterministic: the built-in nested examples include a duplicate across
    # inners ([['x', 'y'], ['x']]) where dedup drops the repeated 'x'.
    ok, err = _check(DISTINCT_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "")


def test_filter_nulls_rejected(spark):
    # Deterministic: built-in [['a', None], []] keeps the None in python.
    ok, err = _check(FILTER_NULLS_SQL, spark)
    assert not ok


def test_sort_array_rejected(spark):
    # Deterministic: ['x', 'y', 'x'] sorted is ['x', 'x', 'y'].
    ok, err = _check(SORT_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "")


def _record_samples(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
    df = spark.createDataFrame(
        [(s,) for s in FLAT_SAMPLES if s is not None],
        "s array<array<string>>",
    )
    sampling_udf = UserDefinedFunction(flatlist, ARRAY_OF_STRING)
    df.select(sampling_udf("s")).collect()


def test_register_impl_rejects_distinct(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            flatlist,
            kind="catalyst",
            catalyst_sql=DISTINCT_SQL,
            return_type=ARRAY_OF_STRING,
        )


def test_register_impl_accepts_flatten(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    register_impl(
        spark,
        flatlist,
        kind="catalyst",
        catalyst_sql=FAITHFUL_SQL,
        return_type=ARRAY_OF_STRING,
    )
    row = get_catalog()._conn.execute("SELECT status, origin FROM cache").fetchone()
    assert tuple(row) == ("success", "human")


def test_nested_sample_roundtrip():
    args = [[["a", None], []], None, [["x"]]]
    decoded = decode_args(json.dumps([_encode(v) for v in args]))
    assert decoded == args


# SPARK-55206: same return-type gate as test_gate_with_array_category_allows_queue.
# The hook never runs for array<string>, so a cache hit cannot land.
# https://issues.apache.org/jira/browse/SPARK-55206
#
# def test_end_to_end_f_udf_call_site(spark, sqlite_path):
#     # The pasted call site: F.udf(flatlist, ArrayType(StringType())).
#     conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
#     enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
#     spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
#     rows = [
#         ([["a", "b"], ["c"]],),
#         ([[]],),
#         ([["a", None], ["b"]],),
#         ([["x", "y"], ["x"]],),
#     ]
#     df = spark.createDataFrame(rows, "s array<array<string>>")
#     expected = [flatlist(r[0]) for r in rows]
#     assert expected == [["a", "b", "c"], [], ["a", None, "b"], ["x", "y", "x"]]
#
#     # F.udf(f, rtype) constructs a UserDefinedFunction internally (so the
#     # transpile hook fires and rows get sampled) but returns a plain wrapper
#     # function -- .transpiled is not observable on it.
#     first = F.udf(flatlist, ARRAY_OF_STRING)
#     got = [r[0] for r in df.select(first("s")).collect()]
#     assert got == expected
#
#     deadline = time.time() + 60
#     second = UserDefinedFunction(flatlist, ARRAY_OF_STRING)
#     while time.time() < deadline and not second.transpiled:
#         time.sleep(0.5)
#         second = UserDefinedFunction(flatlist, ARRAY_OF_STRING)
#     assert second.transpiled, "fake backend never landed the flatten rewrite"
#     got = [r[0] for r in df.select(second("s")).collect()]
#     assert got == expected
#     row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
#     assert row[0] == "success"
#     assert row[1] == FAITHFUL_SQL
