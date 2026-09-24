# SPDX-License-Identifier: Apache-2.0
"""Input-category gate: conf restricts which UDFs are queued for transpilation."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable
from ai_udf_transpile.backends.fake import greet, plus_one
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def test_default_gate_allows_string(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(greet, StringType())
    assert get_catalog().count() == 1


def test_numeric_only_gate_blocks_string(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, "numeric")
    UserDefinedFunction(greet, StringType())
    assert get_catalog().count() == 0, "string-input UDF was queued despite numeric-only gate"
    UserDefinedFunction(plus_one, LongType())
    assert get_catalog().count() == 1


def test_numeric_string_gate_allows_both(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string")
    UserDefinedFunction(greet, StringType())
    UserDefinedFunction(plus_one, LongType())
    assert get_catalog().count() == 2
