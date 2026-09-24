# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.functions import udf
from pyspark.sql.transpile import AbstractTranspiler, _get_transpilers
from pyspark.sql.types import LongType

from ai_udf_transpile import enable, shutdown
from ai_udf_transpile.transpiler import get_transpiler_class

pytestmark = pytest.mark.spark


def plus_one(x: int) -> int:
    return x + 1


def test_enable_registers_ai_after_catalyst(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    assert "ai" in AbstractTranspiler.varieties
    names = [t.variety for t in _get_transpilers(spark)]
    assert names[0] == "catalyst"
    assert "ai" in names
    py = spark.conf.get("spark.sql.experimental.optimizer.pyTranspilers")
    assert "ai" in py
    assert spark.conf.get("spark.sql.experimental.optimizer.transpilePyUDFs").lower() == "true"


def test_hook_is_invoked_and_python_still_works(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    cls = get_transpiler_class()
    called: list[bool] = []
    original = cls._transpile_from_ast

    def wrapped(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    cls._transpile_from_ast = wrapped
    try:
        f = udf(plus_one, LongType())
        df = spark.createDataFrame([(1,)], ["x"])
        assert df.select(f("x")).collect()[0][0] == 2
    finally:
        cls._transpile_from_ast = original
    assert called, "AITranspiler._transpile_from_ast was not invoked"
    shutdown()


def test_delta_without_delta_errors(spark, sqlite_path):
    with pytest.raises(RuntimeError, match="Delta"):
        enable(spark, catalog="delta", sqlite_path=sqlite_path, inline_worker=False)
