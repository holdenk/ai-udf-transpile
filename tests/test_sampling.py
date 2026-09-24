# SPDX-License-Identifier: Apache-2.0
"""Real-row sampling: captured values reach Hypothesis and catch vacuous passes."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.sampling import wrap_for_sampling
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

WIDGET = (
    "def widget_name(payload: str) -> str:\n"
    "    import json\n"
    "    if payload is None:\n"
    "        return None\n"
    "    return json.loads(payload).get('widget')\n"
)

# Wrong key: '$.gadget' never exists, so SQL returns NULL everywhere.
WRONG_KEY_SQL = "get_json_object(_udf_param_0, '$.gadget')"
RIGHT_KEY_SQL = "get_json_object(_udf_param_0, '$.widget')"

REAL_SAMPLE = [['{"widget": "real-value", "other": 1}']]


def _check(sql, samples, spark):
    return hypothesis_check(
        source_text=WIDGET,
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string"],
        return_type="string",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_wrong_key_passes_vacuously_without_samples(spark):
    # Random/built-in strings never contain "widget", so python raises or
    # returns None and the wrong SQL agrees. This is the vacuous pass.
    ok, err = _check(WRONG_KEY_SQL, [], spark)
    assert ok, err


def test_real_sample_exposes_wrong_key(spark):
    ok, err = _check(WRONG_KEY_SQL, REAL_SAMPLE, spark)
    assert not ok
    assert "mismatch" in (err or "")
    assert "real-value" in (err or "")


def test_right_key_passes_with_samples(spark):
    ok, err = _check(RIGHT_KEY_SQL, REAL_SAMPLE, spark)
    assert ok, err


def test_sampler_records_and_caps(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    seen = []

    def func(x):
        seen.append(x)
        return x

    wrapped = wrap_for_sampling(func, "key1", cat.path, 3)
    for value in ["a", None, "b", "c", "d", "e"]:
        wrapped(value)
    assert seen == ["a", None, "b", "c", "d", "e"]  # passthrough, all calls run
    samples = cat.samples_for("key1")
    assert samples == [["c"], ["b"], ["a"]]  # newest first; None-only call skipped; capped


def test_samples_captured_end_to_end(spark, sqlite_path):
    def greet(name: str) -> str:
        if name is None:
            return "hi ?"
        return "hi " + name

    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    udf = UserDefinedFunction(greet, StringType())  # cache miss -> func wrapped
    df = spark.createDataFrame([("bo",), ("holden",), (None,)], ["name"])
    rows = [r[0] for r in df.select(udf("name")).collect()]
    assert rows == ["hi bo", "hi holden", "hi ?"]
    catalog = get_catalog()
    raw = catalog._conn.execute("SELECT args_json FROM samples").fetchall()
    values = {r[0] for r in raw}
    assert any("bo" in v for v in values), f"no real samples captured: {values}"
