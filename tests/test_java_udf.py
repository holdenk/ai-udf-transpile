# SPDX-License-Identifier: Apache-2.0
"""Java UDF target: compile (JDK or Janino), register, verify, reconstruct."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import conf, enable
from ai_udf_transpile.backends.fake import BACKWARDS_JAVA, backwards_java
from ai_udf_transpile.javac import compile_java, extract_java_class, register_java_udf, unique_class_name
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def test_compile_java_fixture(spark):
    compiled = compile_java(spark, BACKWARDS_JAVA)
    # Compiled under the declared name plus a hash of the source (one name,
    # one body -- see javac.unique_class_name).
    assert compiled.class_name == unique_class_name(BACKWARDS_JAVA)
    assert compiled.class_name.startswith("ai_udf.Backwards_")
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
    conf.set_value(conf.PARTY_ON_STRINGS, "true")
    first = UserDefinedFunction(backwards_java, StringType())
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
    assert row[3] == unique_class_name(BACKWARDS_JAVA)  # what was compiled is what is stored
    assert row[4] and row[4] > 100, "compiled jar bytes were not stored"

    second = UserDefinedFunction(backwards_java, StringType())
    assert second.transpiled
    df = spark.createDataFrame([("hello",), (None,)], ["x"])
    assert [r[0] for r in df.select(second("x")).collect()] == ["olleh", None]


@pytest.mark.spark
def test_same_class_name_different_code_does_not_collide(spark):
    """Every jar a session registers shares one classloader. Before the
    content-addressed rename the second rewrite here ran the first one's code
    -- in verification as well as serving."""
    from pyspark.sql.types import StringType

    from ai_udf_transpile.javac import compile_java, register_java_udf, verify_function_name

    src = (
        "package ai_udf;\nimport org.apache.spark.sql.api.java.UDF1;\n"
        "public class Collide implements UDF1<Object, Object> {\n"
        '    public Object call(Object s) { return "%s:" + s; }\n}\n'
    )
    for tag in ("first", "second"):
        compiled = compile_java(spark, src % tag, "ai_udf.Collide")
        assert compiled.class_name != "ai_udf.Collide"
        fname = verify_function_name("ai_udf", compiled.jar_bytes)
        register_java_udf(
            spark, fname, compiled.class_name, compiled.jar_bytes, StringType(), janino=compiled.janino
        )
        assert spark.sql(f"SELECT {fname}('x') AS v").first().v == f"{tag}:x"
