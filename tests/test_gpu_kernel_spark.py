# SPDX-License-Identifier: Apache-2.0
"""gpu_kernel target on a real (CPU) SparkSession.

There is no GPU here, so no kernel ever executes. What does run for real:
javac building the wrapper against a second jar, the merged jar, the sibling
session, registration, the wrapper's row path, the plan probe reading an
actual physical plan, and the gate around all of it. The RAPIDS API is three
stub classes with the method shapes the wrapper calls; the wrapper is
compiled against the real cudf sources in test_gpu_kernel.py.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import LongType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import canonical_source_from_func, conf, enable, rapids
from ai_udf_transpile.backends.fake import LOW_BITS_CUDA, LOW_BITS_JAVA, low_bits_set
from ai_udf_transpile.catalog import CacheRow
from ai_udf_transpile.javac import (
    JavaCompileError,
    _add_artifact,
    _compile_jdk,
    _stage_jar,
    compile_java,
    merge_jars,
    verify_function_name,
)
from ai_udf_transpile.targets import KIND_GPU_KERNEL, KIND_JAVA_UDF, TranspileResult
from ai_udf_transpile.transpiler import _pin_session, _reconstruct_column, get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark

ABI = rapids.stock_abi(["bigint"], "bigint")

_STUBS = [
    (
        "ai.rapids.cudf.DType",
        """package ai.rapids.cudf;
public final class DType {
    public static final DType BOOL8 = new DType();
    public static final DType INT32 = new DType();
    public static final DType INT64 = new DType();
}
""",
    ),
    (
        "ai.rapids.cudf.ColumnVector",
        """package ai.rapids.cudf;
public class ColumnVector implements AutoCloseable {
    public DType getType() { return DType.INT64; }
    public ColumnVector transform(String udf, boolean isPtx) { throw new UnsupportedOperationException(); }
    public ColumnVector castTo(DType type) { throw new UnsupportedOperationException(); }
    @Override public void close() {}
}
""",
    ),
    (
        "com.nvidia.spark.RapidsUDF",
        """package com.nvidia.spark;
public interface RapidsUDF {
    ai.rapids.cudf.ColumnVector evaluateColumnar(int numRows, ai.rapids.cudf.ColumnVector... args);
}
""",
    ),
]


@pytest.fixture(scope="module")
def stub_api(spark):
    """Path of a jar with the stub RAPIDS API, built with the code under test."""
    jars: list[bytes] = []
    paths: list[str] = []
    try:
        for name, source in _STUBS:
            jar = _compile_jdk(spark, source, name, extra_classpath=paths)
            jars.append(jar)
            paths.append(_stage_jar(verify_function_name("rapids_stub_part", jar), jar))
    except JavaCompileError as exc:
        pytest.skip(f"no JDK compiler on the driver: {exc}")
    merged = merge_jars(*jars)
    return _stage_jar(verify_function_name("rapids_stub", merged), merged)


@pytest.fixture
def stubbed(spark, stub_api, monkeypatch):
    """The stub API where RAPIDS would be: on javac's classpath and in every session."""
    real_session = rapids.gpu_session
    siblings = []

    def session_with_stubs(parent):
        sibling = real_session(parent)
        _add_artifact(sibling, "rapids_stub", stub_api)
        siblings.append(sibling)
        return sibling

    _add_artifact(spark, "rapids_stub", stub_api)
    monkeypatch.setattr(rapids, "rapids_classpath", lambda _spark: [stub_api])
    monkeypatch.setattr(rapids, "gpu_session", session_with_stubs)
    monkeypatch.setattr(rapids, "rapids_active", lambda _spark: True)
    conf.set_value(conf.GPU_KERNEL, "true")
    yield siblings
    _pin_session(spark)


def _candidate(spark) -> TranspileResult:
    compiled = compile_java(spark, LOW_BITS_JAVA)
    return TranspileResult(
        kind=KIND_GPU_KERNEL,
        java_source=LOW_BITS_JAVA,
        class_name=compiled.class_name,
        binary=compiled.jar_bytes,
        entry=LOW_BITS_CUDA,
    )


def _verify(spark, result):
    return rapids.verify_result(
        hypothesis_check,
        source_text=canonical_source_from_func(low_bits_set),
        captures={},
        result=result,
        input_types=["bigint"],
        return_type="bigint",
        spark=spark,
        max_examples=5,
    )


def test_wrapper_builds_against_the_cpu_jar_and_keeps_its_bytes(spark, stubbed):
    cpu = compile_java(spark, LOW_BITS_JAVA)
    jar, name = rapids.build_wrapper_jar(spark, cpu.jar_bytes, cpu.class_name, LOW_BITS_CUDA, ABI)
    assert name == "ai_udf.LowBits__Rapids"
    assert rapids.jar_has_class(jar, "ai_udf.LowBits__Rapids")
    # Deterministic, and the CPU class is the verified one, not a recompile.
    assert jar == rapids.build_wrapper_jar(spark, cpu.jar_bytes, cpu.class_name, LOW_BITS_CUDA, ABI)[0]
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(jar)) as merged, zipfile.ZipFile(io.BytesIO(cpu.jar_bytes)) as only_cpu:
        assert merged.read("ai_udf/LowBits.class") == only_cpu.read("ai_udf/LowBits.class")


def test_wrapper_needs_the_rapids_api_to_compile(spark, monkeypatch):
    # No stub on the classpath: the build fails, which the gate turns into a dropped kernel.
    monkeypatch.setattr(rapids, "rapids_classpath", lambda _spark: [])
    cpu = compile_java(spark, LOW_BITS_JAVA)
    with pytest.raises(JavaCompileError, match="RapidsUDF|com.nvidia"):
        rapids.build_wrapper_jar(spark, cpu.jar_bytes, cpu.class_name, LOW_BITS_CUDA, ABI)


def test_gpu_session_is_a_sibling_with_the_users_runtime_confs(spark):
    before = spark.conf.get("spark.sql.adaptive.enabled")
    spark.conf.set("spark.sql.session.timeZone", "Pacific/Auckland")
    try:
        sibling = rapids.gpu_session(spark)
        assert sibling is not spark
        assert sibling.conf.get("spark.sql.session.timeZone") == "Pacific/Auckland"
        assert sibling.conf.get("spark.sql.ansi.enabled") == spark.conf.get("spark.sql.ansi.enabled")
        assert sibling.conf.get("spark.sql.adaptive.enabled") == "false"
        assert sibling.conf.get("spark.rapids.sql.enabled") == "true"
        # The user's session did not move.
        assert spark.conf.get("spark.sql.adaptive.enabled") == before
        assert spark.conf.get("spark.rapids.sql.enabled", None) is None
    finally:
        spark.conf.unset("spark.sql.session.timeZone")


def test_cpu_plan_never_vouches_for_the_kernel(spark, stubbed):
    """Everything is in place except a GPU: the wrapper registers and its row
    path matches the Python, but the real plan is a CPU Project, so the
    kernel is dropped rather than called verified."""
    result = _candidate(spark)
    cpu_jar = result.binary
    assert _verify(spark, result) == (True, None)
    assert (result.kind, result.entry, result.binary) == (KIND_JAVA_UDF, None, cpu_jar)
    assert len(stubbed) == 1  # it got as far as the sibling session

    # The probe itself, against that session's real physical plan.
    wrapper_jar, name = rapids.build_wrapper_jar(spark, cpu_jar, "ai_udf.LowBits", LOW_BITS_CUDA, ABI)
    from ai_udf_transpile.verify import _java_udf_expr

    expr = _java_udf_expr(
        stubbed[0],
        TranspileResult(kind=KIND_JAVA_UDF, class_name=name, binary=wrapper_jar),
        ["bigint"],
        "bigint",
    )
    error = rapids.columnar_plan_error(stubbed[0], expr, ["bigint"])
    assert error is not None and "Project" in error and expr.split("(")[0] in error


def test_wrapper_row_path_delegates_to_the_cpu_class(spark, stubbed):
    cpu = compile_java(spark, LOW_BITS_JAVA)
    jar, name = rapids.build_wrapper_jar(spark, cpu.jar_bytes, cpu.class_name, LOW_BITS_CUDA, ABI)
    row = CacheRow(
        udf_key="k",
        status="success",
        target_kind=KIND_GPU_KERNEL,
        impl_class=cpu.class_name,
        impl_binary=jar,
        impl_entry=LOW_BITS_CUDA,
        param_names=["x"],
    )
    column = _reconstruct_column(row, LongType(), spark)
    df = spark.createDataFrame([(0,), (255,), (-1,), (None,)], "_udf_param_0 bigint")
    plan = df.select(column.alias("r"))
    assert [r[0] for r in plan.collect()] == [0, 8, 32, None]
    # The wrapper is what was registered (the GPU-serving name), not the CPU class.
    assert verify_function_name("ai_udf_gpu", jar) in plan._jdf.queryExecution().analyzed().toString()


def test_gate_passes_only_with_a_columnar_plan_then_serves_the_wrapper(spark, stubbed, monkeypatch):
    """With the plan probe satisfied (the one thing a CPU session cannot do
    honestly), the rest of the GPU leg is real: Hypothesis runs the wrapper in
    the sibling session and the result carries kernel plus merged jar."""
    real_plan = rapids._executed_plan
    monkeypatch.setattr(
        rapids,
        "_executed_plan",
        lambda session, expr, types: "GpuProject [gpu_" + real_plan(session, expr, types),
    )
    monkeypatch.setattr(rapids, "plan_runs_kernel", lambda plan, fname: plan.startswith("GpuProject [gpu_"))
    result = _candidate(spark)
    assert _verify(spark, result) == (True, None)
    assert (result.kind, result.entry, result.class_name) == (
        KIND_GPU_KERNEL,
        LOW_BITS_CUDA,
        "ai_udf.LowBits",
    )
    assert rapids.jar_has_class(result.binary, "ai_udf.LowBits__Rapids")


def test_gpu_kernel_row_serves_the_java_class_where_the_wrapper_cannot_load(spark, stub_api, monkeypatch):
    """A shared row read by a session without the RAPIDS classes: the wrapper
    fails to load and the row's verified java class is served instead."""
    monkeypatch.setattr(rapids, "rapids_classpath", lambda _spark: [stub_api])
    cpu = compile_java(spark, LOW_BITS_JAVA)
    jar, _ = rapids.build_wrapper_jar(spark, cpu.jar_bytes, cpu.class_name, LOW_BITS_CUDA, ABI)
    row = CacheRow(
        udf_key="k",
        status="success",
        target_kind=KIND_GPU_KERNEL,
        impl_class=cpu.class_name,
        impl_binary=jar,
        param_names=["x"],
    )
    bare = spark.newSession()  # no stub artifact here
    try:
        for claims_rapids in (False, True):
            monkeypatch.setattr(rapids, "rapids_active", lambda _spark, on=claims_rapids: on)
            conf.set_value(conf.GPU_KERNEL, "true")
            column = _reconstruct_column(row, LongType(), bare)
            df = bare.createDataFrame([(255,), (None,)], "_udf_param_0 bigint").select(column.alias("r"))
            assert [r[0] for r in df.collect()] == [8, None]
            analyzed = df._jdf.queryExecution().analyzed().toString()
            assert verify_function_name("ai_udf", jar) in analyzed
            assert verify_function_name("ai_udf_gpu", jar) not in analyzed
    finally:
        _pin_session(spark)


def _wait_for_row(catalog, seconds: int = 90):
    deadline = time.time() + seconds
    while time.time() < deadline:
        rows = catalog._conn.execute("SELECT udf_key, status FROM cache").fetchall()
        if rows and rows[0][1] in ("success", "failed"):
            return catalog.get(rows[0][0])
        time.sleep(0.5)
    return None


def _low_bits_via_udf(spark):
    udf = UserDefinedFunction(low_bits_set, LongType())
    df = spark.createDataFrame([(0,), (255,), (-1,)], "x bigint")
    return udf, [r[0] for r in df.select(udf("x")).collect()]


def test_end_to_end_without_rapids_serves_the_java_udf(spark, sqlite_path):
    """gpuKernel on, no RAPIDS: the backend still hands back a kernel, and the
    row that ships is the verified java_udf with no trace of it."""
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.GPU_KERNEL, "true")
    first, values = _low_bits_via_udf(spark)
    assert not first.transpiled and values == [0, 8, 32]
    row = _wait_for_row(get_catalog())
    assert row is not None and row.status == "success", getattr(row, "error", "no row")
    assert (row.target_kind, row.impl_entry, row.impl_class) == (KIND_JAVA_UDF, None, "ai_udf.LowBits")
    assert not rapids.jar_has_class(row.impl_binary, "ai_udf.LowBits__Rapids")
    second, values = _low_bits_via_udf(spark)
    assert second.transpiled and values == [0, 8, 32]


def test_end_to_end_gpu_kernel_row_is_stored_and_served_through_the_wrapper(
    spark, sqlite_path, stubbed, monkeypatch
):
    """The whole pipeline with the plan probe satisfied by hand: prompt offer,
    fake backend, both legs, smoke test, catalog row, and the transpiled UDF
    served through the wrapper class (its row path -- there is no GPU)."""
    monkeypatch.setattr(rapids, "plan_runs_kernel", lambda plan, fname: fname in plan)
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.GPU_KERNEL, "true")
    first, values = _low_bits_via_udf(spark)
    assert not first.transpiled and values == [0, 8, 32]
    row = _wait_for_row(get_catalog())
    assert row is not None and row.status == "success", getattr(row, "error", "no row")
    assert (row.target_kind, row.impl_entry, row.impl_class) == (
        KIND_GPU_KERNEL,
        LOW_BITS_CUDA,
        "ai_udf.LowBits",
    )
    assert rapids.jar_has_class(row.impl_binary, "ai_udf.LowBits__Rapids")
    second, values = _low_bits_via_udf(spark)
    assert values == [0, 8, 32]
    served = " ".join(str(sql) for sql in second.transpiled)
    assert verify_function_name("ai_udf_gpu", row.impl_binary) in served
    # Turning the conf off serves the same row as its java class.
    conf.set_value(conf.GPU_KERNEL, "false")
    third, values = _low_bits_via_udf(spark)
    assert values == [0, 8, 32]
    assert verify_function_name("ai_udf_gpu", row.impl_binary) not in " ".join(
        str(s) for s in third.transpiled
    )


def test_on_gpu_kernel_is_verified_and_served(spark, sqlite_path):
    """The acceptance test for the part nothing else here can reach: a real
    kernel, JIT-compiled by libcudf, planned by RAPIDS, matching the Python.

    Needs AI_UDF_TRANSPILE_TEST_GPU_JAR and a GPU. On failure the worker's
    "gpu kernel dropped" warning in the captured log says which step refused.
    """
    if not rapids.rapids_active(spark):
        pytest.skip("needs Spark RAPIDS: set AI_UDF_TRANSPILE_TEST_GPU_JAR on a GPU machine")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    conf.set_value(conf.GPU_KERNEL, "true")
    first, values = _low_bits_via_udf(spark)
    assert not first.transpiled and values == [0, 8, 32]
    row = _wait_for_row(get_catalog(), seconds=300)  # the first JIT compile is slow
    assert row is not None and row.status == "success", getattr(row, "error", "no row")
    assert row.target_kind == KIND_GPU_KERNEL, "the kernel was dropped; see the captured warning"
    second = UserDefinedFunction(low_bits_set, LongType())
    assert second.transpiled
    df = spark.createDataFrame([(0,), (255,), (-1,), (None,)], "x bigint").select(second("x"))
    assert [r[0] for r in df.collect()] == [0, 8, 32, None]
    fname = verify_function_name("ai_udf_gpu", row.impl_binary)
    assert rapids.plan_runs_kernel(df._jdf.queryExecution().executedPlan().toString(), fname)
