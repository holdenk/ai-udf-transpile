# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable, register_impl
from ai_udf_transpile.backends.fake import plus_one
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def test_register_impl_catalyst_hit(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    key = register_impl(
        spark,
        plus_one,
        kind="catalyst",
        catalyst_sql="_udf_param_0 + 1",
        return_type=LongType(),
        verify=True,
    )
    row = get_catalog().get(key)
    assert row is not None
    assert row.status == "success"
    assert row.origin == "human"
    constructed = UserDefinedFunction(plus_one, LongType())
    assert constructed.transpiled, "human catalyst impl was not reconstructed"
    df = spark.createDataFrame([(3,)], ["x"])
    assert df.select(constructed("x")).collect()[0][0] == 4


def test_register_impl_requires_types(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)

    def untyped(x):
        return x + 1

    with pytest.raises(ValueError):
        register_impl(
            spark,
            untyped,
            catalyst_sql="_udf_param_0 + 1",
            return_type=LongType(),
            verify=False,
        )
