# SPDX-License-Identifier: Apache-2.0
"""BSPIn/EBRIn/BSPOut/EBROut: telecom LAC membership UDFs, and the `&` trap.

    lac_lst_bsp = ['14503','13403','11518','1452','1343','1518']
    lac_lst_mid = ['14506','1462']

    def BSPIn(old_lac, new_lac, lac_lst_bsp, lac_lst_mid):
        if old_lac in lac_lst_bsp & new_lac in lac_lst_mid:
            return 1
        else:
            return 0

    ...withColumn('BSPIn', BSPInUDF(col('OLD_LAC'), col('NEW_LAC'),
        F.array([F.lit(x) for x in lac_lst_bsp]), F.array([F.lit(x) for x in lac_lst_mid])))

The pasted body is broken Python: `&` binds tighter than `in` (and comparisons
chain), so `old_lac in lac_lst_bsp & new_lac in lac_lst_mid` parses as
`old_lac in (lac_lst_bsp & new_lac) in lac_lst_mid`, and `list & str` raises
TypeError on EVERY row. The intent is plainly `and`; the fixtures modernize
to it. These are also the first UDFs with array<string> inputs (the lists
arrive as array columns), hence the new "array" input category (gated by
default like map/timestamp).

Null semantics: `None in list` is False (-> 0), so a bare
`array_contains(...) AND ...` returns NULL where python returns 0 and must be
wrapped in a CASE. `x in None` raises TypeError, so a NULL list param is
allowed to return anything. Known limitation: python's `None in [None]` is
True while SQL array_contains(arr, NULL) is NULL -- null ELEMENTS do not
occur in these constant membership lists, and the array strategy excludes
them (a real sample containing one fails verification closed).
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, register_impl
from ai_udf_transpile.backends.fake import BSPIn
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sampling import _encode, decode_args
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

LAC_BSP = ["14503", "13403", "11518", "1452", "1343", "1518"]
LAC_MID = ["14506", "1462"]

FAITHFUL_SQL = (
    "CASE WHEN array_contains(_udf_param_2, _udf_param_0) "
    "AND array_contains(_udf_param_3, _udf_param_1) THEN 1 ELSE 0 END"
)
# The Out-UDF logic pasted into the In-UDF: lists swapped.
SWAPPED_SQL = (
    "CASE WHEN array_contains(_udf_param_3, _udf_param_0) "
    "AND array_contains(_udf_param_2, _udf_param_1) THEN 1 ELSE 0 END"
)
# OR instead of AND.
OR_SQL = FAITHFUL_SQL.replace(" AND ", " OR ")
# No CASE: NULL for a null scalar where python returns 0.
BARE_BOOL_SQL = "array_contains(_udf_param_2, _udf_param_0) AND array_contains(_udf_param_3, _udf_param_1)"

BSP_SAMPLES = [
    ["14503", "14506", LAC_BSP, LAC_MID],  # in: 1
    ["14506", "14503", LAC_BSP, LAC_MID],  # in: 0 (out: 1)
    ["14503", "99999", LAC_BSP, LAC_MID],  # one-sided: old in bsp, new nowhere
    [None, "14506", LAC_BSP, LAC_MID],  # null scalar -> 0
    ["99999", "99999", LAC_BSP, LAC_MID],  # nowhere -> 0
]

ARRAY_CATEGORIES = "numeric,string,bool,binary,array"


def _check(sql, spark, samples=None):
    return hypothesis_check(
        source_text=canonical_source_from_func(BSPIn),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string", "string", "array<string>", "array<string>"],
        return_type="int",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_pasted_code_raises_as_written():
    # `&` binds tighter than `in`: the pasted condition parses as
    # `old_lac in (lac_lst_bsp & new_lac) in lac_lst_mid` -> TypeError.
    def BSPIn_pasted(old_lac, new_lac, lac_lst_bsp, lac_lst_mid):
        if old_lac in lac_lst_bsp & new_lac in lac_lst_mid:
            return 1
        else:
            return 0

    with pytest.raises(TypeError):
        BSPIn_pasted("14503", "14506", LAC_BSP, LAC_MID)


def test_default_gate_blocks_array(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(BSPIn, IntegerType())
    assert get_catalog().count() == 0, "array-input UDF queued despite default gate"


def test_gate_with_array_category_allows_queue(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
    UserDefinedFunction(BSPIn, IntegerType())
    assert get_catalog().count() == 1


def test_faithful_rewrite_passes(spark):
    ok, err = _check(FAITHFUL_SQL, spark, samples=BSP_SAMPLES)
    assert ok, err


def test_swapped_lists_rejected(spark):
    ok, err = _check(SWAPPED_SQL, spark, samples=BSP_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def test_or_instead_of_and_rejected(spark):
    # Deterministic without samples: the cross-combined built-ins pair a
    # scalar with a singleton list containing it on one side only.
    ok, err = _check(OR_SQL, spark)
    assert not ok
    assert "mismatch" in (err or "")


def test_bare_boolean_rejected(spark):
    # array_contains(arr, NULL) is NULL; python's None in list is False -> 0.
    ok, err = _check(BARE_BOOL_SQL, spark, samples=BSP_SAMPLES)
    assert not ok
    assert "mismatch" in (err or "")


def _record_samples(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
    df = spark.createDataFrame(
        [(s[0], s[1], s[2], s[3]) for s in BSP_SAMPLES],
        ["OLD_LAC", "NEW_LAC", "lac_lst_bsp", "lac_lst_mid"],
    )
    sampling_udf = UserDefinedFunction(BSPIn, IntegerType())
    df.select(sampling_udf("OLD_LAC", "NEW_LAC", "lac_lst_bsp", "lac_lst_mid")).collect()


def test_register_impl_rejects_swapped(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            BSPIn,
            kind="catalyst",
            catalyst_sql=SWAPPED_SQL,
            return_type=IntegerType(),
        )


def test_register_impl_accepts_faithful(spark, sqlite_path):
    _record_samples(spark, sqlite_path)
    register_impl(
        spark,
        BSPIn,
        kind="catalyst",
        catalyst_sql=FAITHFUL_SQL,
        return_type=IntegerType(),
    )
    row = get_catalog()._conn.execute("SELECT status, origin FROM cache").fetchone()
    assert tuple(row) == ("success", "human")


def test_array_sample_roundtrip():
    args = ["14503", None, LAC_BSP, []]
    decoded = decode_args(json.dumps([_encode(v) for v in args]))
    assert decoded == args


def test_end_to_end_transpile_call_site_shape(spark, sqlite_path):
    # The pasted call site: scalar columns plus F.array(F.lit(...)) constants.
    conf.set_value(conf.INPUT_CATEGORIES, ARRAY_CATEGORIES)
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    rows = [("14503", "14506"), ("14506", "14503"), (None, "14506"), ("99999", "99999")]
    df = spark.createDataFrame(rows, ["OLD_LAC", "NEW_LAC"])
    bsp_arr = F.array(*[F.lit(x) for x in LAC_BSP])
    mid_arr = F.array(*[F.lit(x) for x in LAC_MID])
    expected = [BSPIn(o, n, LAC_BSP, LAC_MID) for o, n in rows]
    assert expected == [1, 0, 0, 0]

    first = UserDefinedFunction(BSPIn, IntegerType())
    assert not first.transpiled
    got = [r[0] for r in df.select(first("OLD_LAC", "NEW_LAC", bsp_arr, mid_arr)).collect()]
    assert got == expected

    deadline = time.time() + 60
    second = UserDefinedFunction(BSPIn, IntegerType())
    while time.time() < deadline and not second.transpiled:
        time.sleep(0.5)
        second = UserDefinedFunction(BSPIn, IntegerType())
    assert second.transpiled, "fake backend never landed the faithful rewrite"
    got = [r[0] for r in df.select(second("OLD_LAC", "NEW_LAC", bsp_arr, mid_arr)).collect()]
    assert got == expected
    row = get_catalog()._conn.execute("SELECT status, catalyst_sql FROM cache").fetchone()
    assert row[0] == "success"
    assert row[1] == FAITHFUL_SQL
