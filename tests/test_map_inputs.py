# SPDX-License-Identifier: Apache-2.0
"""Map-typed inputs, using a real-world UDF: pymap_to_json.

The original (PySpark 1.x era, Python 2) took a map[string,string] whose
values could be basic types, Python dicts/arrays as literals, or JSON, and
returned one JSON blob. Modernized to Python 3 with type annotations here.

It is the canonical "looks gucci but is not" rewrite target: to_json(inp)
type-checks and works on casual inspection, but is wrong -- it keeps values
as strings instead of parsing them, and Spark's compact JSON spacing never
matches json.dumps. The verifier must catch that.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable, register_impl
from ai_udf_transpile.sampling import decode_args
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark


def pymap_to_json(inp: dict[str, str]) -> str:
    import ast
    import json

    if inp is None:
        return None
    new = {}
    for k, v in inp.items():
        try:
            new[k] = json.loads(v)
        except Exception:
            try:
                new[k] = ast.literal_eval(v)
            except Exception:
                new[k] = v
    return json.dumps(new)


def value_for_a(inp: dict[str, str]) -> str:
    if inp is None:
        return None
    return inp.get("a")


def _source(func) -> str:
    import inspect
    import textwrap

    return textwrap.dedent(inspect.getsource(func))


def _check(func, sql, spark, samples=None):
    return hypothesis_check(
        source_text=_source(func),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["map<string,string>"],
        return_type="string",
        spark=spark,
        max_examples=20,
        samples=samples,
    )


def test_default_gate_blocks_map(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(pymap_to_json, StringType())
    assert get_catalog().count() == 0, "map-input UDF queued despite default gate"


def test_gate_with_map_category_allows_queue(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map")
    UserDefinedFunction(pymap_to_json, StringType())
    assert get_catalog().count() == 1


def test_verify_map_positive_control(spark):
    # Correct rewrite of a simple map UDF passes; built-in map examples
    # include {"a": ...} rows so the pass is not vacuous.
    ok, err = _check(value_for_a, "_udf_param_0['a']", spark)
    assert ok, err


def test_naive_to_json_rejected(spark):
    # to_json keeps values as strings and uses compact spacing: wrong on
    # built-in example {"a": "1"} (python -> {"a": 1}).
    ok, err = _check(pymap_to_json, "to_json(_udf_param_0)", spark)
    assert not ok
    assert "mismatch" in (err or "")


def test_naive_to_json_rejected_with_samples(spark):
    samples = [[{"a": "1", "b": "{'x': 2}", "c": "plain"}]]
    ok, err = _check(pymap_to_json, "to_json(_udf_param_0)", spark, samples=samples)
    assert not ok
    assert "mismatch" in (err or "")


def test_register_impl_rejects_naive_to_json(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError, match="Hypothesis"):
        register_impl(
            spark,
            pymap_to_json,
            kind="catalyst",
            catalyst_sql="to_json(_udf_param_0)",
            return_type=StringType(),
        )


def test_decode_args_map_roundtrip():
    args = [{"a": "1", "b": None}, "x", None]
    from ai_udf_transpile.sampling import _encode

    encoded = [_encode(v) for v in args]
    import json

    decoded = decode_args(json.dumps(encoded))
    assert decoded == args


def test_map_samples_captured_end_to_end(spark, sqlite_path):
    # Local def: cloudpickle ships it to executors by value (a module-level
    # test function pickles by reference and the executor cannot import it).
    def pymap(inp: dict[str, str]) -> str:
        import ast
        import json

        if inp is None:
            return None
        new = {}
        for k, v in inp.items():
            try:
                new[k] = json.loads(v)
            except Exception:
                try:
                    new[k] = ast.literal_eval(v)
                except Exception:
                    new[k] = v
        return json.dumps(new)

    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    udf = UserDefinedFunction(pymap, StringType())  # cache miss -> func wrapped
    df = spark.createDataFrame(
        [({"a": "1", "b": "plain"},), ({"c": "{'x': 2}"},), (None,)],
        ["inp"],
    )
    rows = [r[0] for r in df.select(udf("inp")).collect()]
    assert rows[0] == '{"a": 1, "b": "plain"}'
    assert rows[2] is None
    raw = get_catalog()._conn.execute("SELECT args_json FROM samples").fetchall()
    decoded = [decode_args(r[0]) for r in raw]
    decoded = [d for d in decoded if d]
    assert any(isinstance(d[0], dict) and d[0].get("a") == "1" for d in decoded), (
        f"no real map samples captured: {decoded}"
    )
