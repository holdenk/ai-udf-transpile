# SPDX-License-Identifier: Apache-2.0
"""Java UDF target: compile (JDK or Janino), register, verify, reconstruct."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable
from ai_udf_transpile.backends.fake import BACKWARDS_JAVA, backwards
from ai_udf_transpile.javac import compile_java, extract_java_class, register_java_udf
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def test_compile_java_fixture(spark):
    compiled = compile_java(spark, BACKWARDS_JAVA)
    assert compiled.class_name == "ai_udf.Backwards"
    assert compiled.jar_bytes[:2] == b"PK"
    # Deterministic bytes so catalog/artifact re-registration is idempotent.
    assert compile_java(spark, BACKWARDS_JAVA).jar_bytes == compiled.jar_bytes


def test_extract_java_class():
    assert extract_java_class(BACKWARDS_JAVA) == "ai_udf.Backwards"
    assert extract_java_class("class NoPkg {}") == "NoPkg"
    assert extract_java_class("interface Nothing {}") is None


def test_register_and_call_java_udf(spark):
    compiled = compile_java(spark, BACKWARDS_JAVA)
    register_java_udf(
        spark,
        "backwards_probe",
        compiled.class_name,
        compiled.jar_bytes,
        StringType(),
        janino=compiled.janino,
    )
    df = spark.createDataFrame([("hello",), (None,)], ["x"])
    rows = df.selectExpr("backwards_probe(x) AS r").collect()
    assert [r[0] for r in rows] == ["olleh", None]


def test_java_udf_end_to_end(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    first = UserDefinedFunction(backwards, StringType())
    assert not first.transpiled
    catalog = get_catalog()
    deadline = time.time() + 90
    row = None
    while time.time() < deadline:
        rows = catalog._conn.execute(
            "SELECT udf_key, status, target_kind, impl_class, length(impl_binary) FROM cache"
        ).fetchall()
        if rows and rows[0][1] == "success":
            row = rows[0]
            break
        time.sleep(0.5)
    assert row is not None, "inline worker never produced a success row"
    assert row[2] == "java_udf"
    assert row[3] == "ai_udf.Backwards"
    assert row[4] and row[4] > 100, "compiled jar bytes were not stored"

    second = UserDefinedFunction(backwards, StringType())
    assert second.transpiled
    df = spark.createDataFrame([("hello",), (None,)], ["x"])
    assert [r[0] for r in df.select(second("x")).collect()] == ["olleh", None]
