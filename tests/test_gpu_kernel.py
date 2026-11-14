# SPDX-License-Identifier: Apache-2.0
"""gpu_kernel target without a JVM: ABI, kernel scan, wrapper, parse, and the two-leg gate.

No test here touches a GPU. The gate is exercised with fakes standing in for
the RAPIDS session, which pins the one property that matters off-GPU: a
kernel that was not verified on a GPU is never what the row serves.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import types

import pytest

from ai_udf_transpile import conf, rapids
from ai_udf_transpile.backends.base import BackendError
from ai_udf_transpile.backends.cli import parse_sandbox
from ai_udf_transpile.backends.fake import LOW_BITS_CUDA, LOW_BITS_JAVA, FakeBackend, low_bits_set
from ai_udf_transpile.catalog import CacheRow
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.javac import _jar_entries, merge_jars
from ai_udf_transpile.sandbox import default_prompt, write_sandbox
from ai_udf_transpile.targets import (
    EXPERIMENTAL_KINDS,
    KIND_GPU_KERNEL,
    KIND_JAVA_UDF,
    RESERVED_KINDS,
    TranspileJob,
    TranspileResult,
)
from ai_udf_transpile.worker import process_row

ABI = rapids.stock_abi(["bigint"], "bigint")


class _FakeSpark:
    """Just the conf surface rapids_active reads (a class: the worker weakrefs its session)."""

    def __init__(self, plugins: str, enabled: str):
        self.sparkContext = types.SimpleNamespace(
            getConf=lambda: types.SimpleNamespace(get=lambda key, default="": plugins)
        )
        self.conf = types.SimpleNamespace(
            get=lambda key, default=None: enabled if key == "spark.rapids.sql.enabled" else default
        )


def _rapids_spark(plugins: str = rapids.RAPIDS_PLUGIN, enabled: str = "true"):
    return _FakeSpark(plugins, enabled)


def test_kind_is_experimental_not_reserved():
    assert KIND_GPU_KERNEL in EXPERIMENTAL_KINDS
    assert KIND_GPU_KERNEL not in RESERVED_KINDS


@pytest.mark.parametrize(
    ("inputs", "ret", "dtype", "ctype", "cast"),
    [
        (["bigint"], "bigint", "INT64", "int64_t", "INT64"),
        (["bigint"], "int", "INT64", "int64_t", None),
        (["double"], "boolean", "FLOAT64", "double", "BOOL8"),
        (["int"], "int", "INT32", "int32_t", None),
        ([" BIGINT "], "BigInt", "INT64", "int64_t", "INT64"),
    ],
)
def test_stock_abi_supported(inputs, ret, dtype, ctype, cast):
    abi = rapids.stock_abi(inputs, ret)
    assert (abi.input_dtype, abi.input_ctypes[0], abi.cast_to) == (dtype, ctype, cast)
    assert abi.signature("x") == f"__device__ inline void udf(int32_t* out, {ctype} x)"


@pytest.mark.parametrize(
    ("inputs", "ret"),
    [
        (["bigint", "bigint"], "bigint"),  # the Java binding is unary
        ([], "bigint"),
        (["string"], "bigint"),  # fixed-width inputs only
        (["bigint"], "double"),  # the output column is INT32
        (["bigint"], "string"),
        (["array<string>"], "int"),
    ],
)
def test_stock_abi_unsupported(inputs, ret):
    assert rapids.stock_abi(inputs, ret) is None


def test_kernel_scan_accepts_the_documented_shape():
    assert rapids.kernel_error(LOW_BITS_CUDA, ABI) is None
    assert rapids.kernel_error(LOW_BITS_CUDA.replace("\n", "\r\n"), ABI) is None
    # Reading the output back and a const input are both fine.
    ok = "__device__ void f(int* o, const int64_t a) { *o = 1; if (a > 0) { *o = *o + 1; } }"
    assert rapids.kernel_error(ok, ABI) is None


@pytest.mark.parametrize(
    ("cuda", "why"),
    [
        ("", "empty"),
        ("DECLINE", "exactly one function"),
        ("#include <cmath>\n" + LOW_BITS_CUDA, "preprocessor"),
        ("%:define X 1\n" + LOW_BITS_CUDA, "preprocessor"),
        (LOW_BITS_CUDA + "\n__device__ inline void g(int32_t* o, int64_t a) { *o = 2; }", "after its one"),
        ("int helper(int a) { return a; }\n" + LOW_BITS_CUDA, "exactly one function"),
        # An `int` parameter over an INT64 column would be silently narrowed.
        ("__device__ inline void udf(int32_t* out, int x) { *out = x; }", "kernel input must be"),
        ("__device__ inline void udf(int64_t* out, int64_t x) { *out = x; }", "kernel output must be"),
        ("__device__ inline void udf(int32_t* out, int64_t x, int64_t y) { *out = 1; }", "3 parameters"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { out[1] = 1; *out = 1; }", "only use"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *(out + 1) = 1; *out = 1; }", "only use"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out[1] = 1; *out = 1; }", "only use"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out++ = 0; *out = 1; }", "only use"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out-- = 0; *out = 1; }", "only use"),
        (
            "__device__ inline void udf(int32_t* out, int64_t x) { int* p = &*out; p[1] = 1; *out = 1; }",
            "only use",
        ),
        ("__device__ inline void udf(int32_t* out, int64_t x) { int y = x; }", "never assigns"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { if (*out == 1) { } }", "never assigns"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out = 1; ", "braces"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { goto a; a: *out = 1; }", "forbidden"),
        (
            "__device__ inline void udf(int32_t* out, int64_t x) "
            "{ *reinterpret_cast<int64_t*>(x) = 1; *out = 1; }",
            "forbidden",
        ),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out = 1; } // café", "ASCII"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out = 1; \\\n }", "splice"),
        ("__device__ inline void udf(int32_t* out, int64_t x) { *out = 1; } ??=", "trigraph"),
        ("x" * (rapids.MAX_KERNEL_BYTES + 1), "larger than"),
    ],
)
def test_kernel_scan_rejects(cuda, why):
    error = rapids.kernel_error(cuda, ABI)
    assert error is not None and why in error


def test_kernel_scan_ignores_comments_and_literals():
    # '#', a second "function", and `out[` appear only in comments and
    # literals; libcudf strips comments before it looks for the name too.
    cuda = (
        "/* #include <x>  void g(int a) */\n"
        "__device__ inline void udf(int32_t* out, int64_t x) {\n"
        "  // out[1] = 2;\n"
        "  char c = '#';\n"
        "  *out = (c == '#') ? 1 : 0;\n"
        "}\n"
    )
    assert rapids.kernel_error(cuda, ABI) is None


def test_wrapper_source_shape():
    src = rapids.wrapper_source("ai_udf.LowBits", LOW_BITS_CUDA, ABI)
    assert rapids.wrapper_class("ai_udf.LowBits") == "ai_udf.LowBits__Rapids"
    assert src.startswith("package ai_udf;\n")
    assert "public class LowBits__Rapids\n" in src
    assert "org.apache.spark.sql.api.java.UDF1<Object, Object>, com.nvidia.spark.RapidsUDF" in src
    assert "return cpu.call(arg0);" in src
    assert "ai.rapids.cudf.DType.INT64.equals(in.getType())" in src
    # bigint return: the INT32 the binding produced, widened and the temp closed.
    assert "try (ai.rapids.cudf.ColumnVector raw = in.transform(CUDA, false))" in src
    assert "raw.castTo(ai.rapids.cudf.DType.INT64)" in src
    assert '"  *out = n;\\n"' in src


def test_wrapper_source_int_return_has_no_cast_and_default_package():
    src = rapids.wrapper_source("Bare", LOW_BITS_CUDA, rapids.stock_abi(["int"], "int"))
    assert not src.startswith("package")
    assert "return in.transform(CUDA, false);" in src
    assert "castTo" not in src
    assert "DType.INT32.equals" in src


def test_wrapper_string_literal_escapes():
    literal = rapids._java_string('a "q" \\ b\tc\nnext')
    assert literal == '"a \\"q\\" \\\\ b\\tc\\n"\n            + "next\\n"'


@pytest.mark.skipif(
    not (shutil.which("javac") and os.environ.get("AI_UDF_TRANSPILE_TEST_RAPIDS_CLASSPATH")),
    reason="needs javac and AI_UDF_TRANSPILE_TEST_RAPIDS_CLASSPATH (RapidsUDF, cudf, Spark's UDF1)",
)
@pytest.mark.parametrize("ret", ["bigint", "int", "boolean"])
def test_wrapper_compiles_against_the_real_api(tmp_path, ret):
    src = tmp_path / "src" / "ai_udf"
    src.mkdir(parents=True)
    (src / "LowBits.java").write_text(LOW_BITS_JAVA)
    wrapper = rapids.wrapper_source("ai_udf.LowBits", LOW_BITS_CUDA, rapids.stock_abi(["bigint"], ret))
    (src / "LowBits__Rapids.java").write_text(wrapper)
    done = subprocess.run(
        [
            "javac",
            "-proc:none",
            "-Xlint:all,-serial",
            "-Werror",
            "-d",
            str(tmp_path / "out"),
            "-cp",
            os.environ["AI_UDF_TRANSPILE_TEST_RAPIDS_CLASSPATH"],
            str(src / "LowBits.java"),
            str(src / "LowBits__Rapids.java"),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out" / "ai_udf" / "LowBits__Rapids.class").exists()


def test_merge_jars_first_wins_and_jar_has_class():
    cpu = _jar_entries({"ai_udf/A.class": b"cpu", "ai_udf/A__Rapids.class": b"stale"})
    fresh = _jar_entries({"ai_udf/A__Rapids.class": b"fresh"})
    merged = merge_jars(fresh, cpu)
    assert merged == merge_jars(fresh, cpu)  # deterministic: content-addressed names depend on it
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(merged)) as jar:
        assert jar.read("ai_udf/A__Rapids.class") == b"fresh"
        assert jar.read("ai_udf/A.class") == b"cpu"
    assert rapids.jar_has_class(merged, "ai_udf.A__Rapids")
    assert not rapids.jar_has_class(merged, "ai_udf.B")
    assert not rapids.jar_has_class(None, "ai_udf.A")
    assert not rapids.jar_has_class(b"not a zip", "ai_udf.A")


def test_rapids_active_reads_plugin_and_switch():
    assert rapids.rapids_active(_rapids_spark())
    assert rapids.rapids_active(_rapids_spark(plugins=f"other.Plugin, {rapids.RAPIDS_PLUGIN}"))
    assert not rapids.rapids_active(_rapids_spark(plugins="org.apache.spark.CometPlugin"))
    assert not rapids.rapids_active(_rapids_spark(plugins=""))
    assert not rapids.rapids_active(_rapids_spark(enabled="FALSE"))
    assert not rapids.rapids_active(None)
    assert not rapids.rapids_active(types.SimpleNamespace())  # no sparkContext: not an error


def test_offered_abi_needs_conf_and_rapids():
    spark = _rapids_spark()
    assert rapids.offered_abi(spark, ["bigint"], "bigint") is None  # conf is off by default
    conf.set_value(conf.GPU_KERNEL, "true")
    assert rapids.offered_abi(spark, ["bigint"], "bigint") == ABI
    assert rapids.offered_abi(spark, ["bigint", "bigint"], "bigint") is None
    assert rapids.offered_abi(None, ["bigint"], "bigint") is None


def test_serving_class_needs_conf_rapids_and_the_wrapper_in_the_jar():
    jar = _jar_entries({"ai_udf/LowBits.class": b"c", "ai_udf/LowBits__Rapids.class": b"w"})
    cpu_only = _jar_entries({"ai_udf/LowBits.class": b"c"})
    spark = _rapids_spark()
    assert rapids.serving_class(spark, "ai_udf.LowBits", jar) is None  # conf off
    conf.set_value(conf.GPU_KERNEL, "true")
    assert rapids.serving_class(spark, "ai_udf.LowBits", jar) == "ai_udf.LowBits__Rapids"
    assert rapids.serving_class(spark, "ai_udf.LowBits", cpu_only) is None
    assert rapids.serving_class(_rapids_spark(plugins=""), "ai_udf.LowBits", jar) is None
    assert rapids.serving_class(spark, None, jar) is None


@pytest.mark.parametrize(
    ("plan", "runs"),
    [
        ("GpuColumnarToRow\n+- GpuProject [gpu_ai_udf_verify_ab12(_udf_param_0#1L) AS result#3L]", True),
        # A row UDF kept inside a GPU plan (rowBasedUDF) is not the kernel.
        ("GpuProject [gpu_row_based_ai_udf_verify_ab12(_udf_param_0#1L) AS result#3L]", False),
        ("Project [ai_udf_verify_ab12(_udf_param_0#1L) AS result#3L]\n+- Scan ExistingRDD", False),
        ("GpuProject [gpu_ai_udf_verify_ffff(_udf_param_0#1L) AS result#3L]", False),
    ],
)
def test_plan_runs_kernel(plan, runs):
    assert rapids.plan_runs_kernel(plan, "ai_udf_verify_ab12") is runs


def test_gpu_session_copies_runtime_confs_and_isolates_the_switches():
    class _Conf:
        def __init__(self):
            self.values = {}

        def set(self, key, value):
            if key == "spark.master":
                raise RuntimeError("CANNOT_MODIFY_CONFIG")
            self.values[key] = value

    sibling = types.SimpleNamespace(conf=_Conf())
    parent_conf = _Conf()
    spark = types.SimpleNamespace(
        conf=parent_conf,
        newSession=lambda: sibling,
        sql=lambda text: types.SimpleNamespace(
            collect=lambda: [
                ("spark.sql.ansi.enabled", "true"),
                ("spark.master", "local[1]"),
                ("spark.sql.adaptive.enabled", "true"),
                ("spark.rapids.sql.enabled", "false"),
            ]
        ),
    )
    assert rapids.gpu_session(spark) is sibling
    assert sibling.conf.values == {
        "spark.sql.ansi.enabled": "true",
        "spark.sql.adaptive.enabled": "false",
        "spark.rapids.sql.enabled": "true",
    }
    assert parent_conf.values == {}  # the user's session is never touched


def _sandbox(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def test_parse_sandbox_java_plus_kernel_is_gpu_kernel(tmp_path):
    result = parse_sandbox(_sandbox(tmp_path, **{"OUT.java": LOW_BITS_JAVA, "OUT.cu": LOW_BITS_CUDA}))
    assert result.kind == KIND_GPU_KERNEL
    assert result.class_name == "ai_udf.LowBits"
    assert result.entry == LOW_BITS_CUDA.strip()
    assert result.reconstructable()


@pytest.mark.parametrize("kernel", ["", "  \n", "DECLINE", "decline\n"])
def test_parse_sandbox_empty_or_declined_kernel_is_plain_java(tmp_path, kernel):
    result = parse_sandbox(_sandbox(tmp_path, **{"OUT.java": LOW_BITS_JAVA, "OUT.cu": kernel}))
    assert result.kind == KIND_JAVA_UDF and result.entry is None


def test_parse_sandbox_kernel_never_rides_on_sql_or_alone(tmp_path):
    sql = tmp_path / "sql"
    sql.mkdir()
    result = parse_sandbox(_sandbox(sql, **{"OUT.sql": "_udf_param_0 + 1", "OUT.cu": LOW_BITS_CUDA}))
    assert result.kind == "catalyst" and result.entry is None
    alone = tmp_path / "alone"
    alone.mkdir()
    with pytest.raises(BackendError):
        parse_sandbox(_sandbox(alone, **{"OUT.cu": LOW_BITS_CUDA}))


def _job():
    return TranspileJob(
        udf_key="k",
        source_text="def low_bits_set(x: int) -> int:\n    return bin(x & 0xFFFFFFFF).count('1')\n",
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
    )


def test_prompt_is_untouched_unless_a_kernel_is_offered(tmp_path):
    plain = (write_sandbox(tmp_path / "plain", _job()) / "PROMPT.md").read_text()
    assert "OUT.cu" not in plain
    assert "OUT.cu" not in default_prompt()
    offered = (write_sandbox(tmp_path / "gpu", _job(), gpu_kernel=ABI) / "PROMPT.md").read_text()
    assert "## Optional GPU kernel (`OUT.cu`)" in offered
    # The exact signature for this job, with the job's own parameter name.
    assert "__device__ inline void udf(int32_t* out, int64_t x) { ... }" in offered
    assert "fits in int32 for every input" in offered  # the bigint-return warning
    # Rules come before the job, which is data.
    assert offered.index("## Optional GPU kernel") < offered.index("## This job")
    # Nothing else moves: take the section out and the plain prompt is back.
    assert offered.replace("\n" + rapids.kernel_prompt(ABI, "x").rstrip() + "\n", "", 1) == plain


def test_kernel_prompt_per_return_type_and_safe_param_name():
    boolean = rapids.kernel_prompt(rapids.stock_abi(["double"], "boolean"), "out")
    assert "double a)" in boolean  # `out` would collide with the output parameter
    assert "`1` for True" in boolean and "fits in int32" not in boolean
    assert "int64_t a)" in rapids.kernel_prompt(ABI, "not an identifier")


def _gpu_result():
    return TranspileResult(
        kind=KIND_GPU_KERNEL,
        java_source=LOW_BITS_JAVA,
        class_name="ai_udf.LowBits",
        binary=_jar_entries({"ai_udf/LowBits.class": b"cpu"}),
        entry=LOW_BITS_CUDA,
        model="m",
    )


def _verify_args(**extra):
    args = dict(source_text="src", captures={}, input_types=["bigint"], return_type="bigint")
    args.update(extra)
    return args


def test_verify_result_passes_other_kinds_straight_through():
    seen = []
    result = TranspileResult(kind="catalyst", sql="_udf_param_0 + 1")

    def verify_fn(**kwargs):
        seen.append(kwargs)
        return False, "nope"

    assert rapids.verify_result(verify_fn, spark=None, result=result, **_verify_args()) == (False, "nope")
    assert len(seen) == 1 and seen[0]["result"] is result and result.kind == "catalyst"


def test_verify_result_without_rapids_ships_the_java_udf():
    kinds = []
    result = _gpu_result()

    def verify_fn(**kwargs):
        kinds.append((kwargs["result"].kind, kwargs["result"].entry))
        return True, None

    conf.set_value(conf.GPU_KERNEL, "true")
    assert rapids.verify_result(verify_fn, spark=None, result=result, **_verify_args()) == (True, None)
    # One leg ran, as a java_udf; the unverified kernel is gone from the result.
    assert kinds == [(KIND_JAVA_UDF, None)]
    assert (result.kind, result.entry) == (KIND_JAVA_UDF, None)


def test_verify_result_cpu_failure_fails_the_row_and_keeps_the_candidate():
    result = _gpu_result()
    out = rapids.verify_result(
        lambda **k: (False, "mismatch on (1,)"), spark=_rapids_spark(), result=result, **_verify_args()
    )
    assert out == (False, "mismatch on (1,)")
    assert (result.kind, result.entry) == (KIND_GPU_KERNEL, LOW_BITS_CUDA)


@pytest.fixture
def fake_gpu(monkeypatch):
    """A RAPIDS session where the wrapper builds, registers, and plans on the GPU."""
    import ai_udf_transpile.verify as verify_mod

    conf.set_value(conf.GPU_KERNEL, "true")
    state = types.SimpleNamespace(session=object(), plan="gpu_ai_udf_verify_x(_udf_param_0#1L)", built=[])

    def build(spark, cpu_jar, cpu_class, cuda, abi):
        state.built.append((cpu_class, cuda, abi))
        return merge_jars(
            _jar_entries({"ai_udf/LowBits__Rapids.class": b"w"}), cpu_jar
        ), cpu_class + "__Rapids"

    monkeypatch.setattr(rapids, "build_wrapper_jar", build)
    monkeypatch.setattr(rapids, "gpu_session", lambda spark: state.session)
    monkeypatch.setattr(rapids, "_executed_plan", lambda session, expr, input_types: state.plan)
    monkeypatch.setattr(verify_mod, "_java_udf_expr", lambda *a: "ai_udf_verify_x(_udf_param_0)")
    return state


def test_verify_result_gpu_leg_verifies_the_wrapper_in_the_sibling_session(fake_gpu):
    spark = _rapids_spark()
    result = _gpu_result()
    cpu_jar = result.binary
    legs = []

    def verify_fn(**kwargs):
        legs.append((kwargs["result"].class_name, kwargs["spark"], kwargs["max_examples"]))
        return True, None

    out = rapids.verify_result(verify_fn, spark=spark, result=result, **_verify_args(max_examples=7))
    assert out == (True, None)
    # CPU class in the user's session, then the wrapper in the sibling one, same settings.
    assert legs == [("ai_udf.LowBits", spark, 7), ("ai_udf.LowBits__Rapids", fake_gpu.session, 7)]
    assert fake_gpu.built == [("ai_udf.LowBits", LOW_BITS_CUDA, ABI)]
    assert (result.kind, result.entry, result.class_name) == (
        KIND_GPU_KERNEL,
        LOW_BITS_CUDA,
        "ai_udf.LowBits",
    )
    assert result.binary != cpu_jar and rapids.jar_has_class(result.binary, "ai_udf.LowBits__Rapids")


def test_verify_result_gpu_mismatch_drops_only_the_kernel(fake_gpu):
    result = _gpu_result()
    cpu_jar = result.binary

    def verify_fn(**kwargs):
        on_gpu = kwargs["result"].class_name.endswith("__Rapids")
        return (False, "mismatch on (2147483648,)") if on_gpu else (True, None)

    assert rapids.verify_result(verify_fn, spark=_rapids_spark(), result=result, **_verify_args()) == (
        True,
        None,
    )
    assert (result.kind, result.entry) == (KIND_JAVA_UDF, None)
    assert result.binary == cpu_jar  # the failed wrapper is not what gets stored


def test_verify_result_cpu_fallback_plan_never_counts_as_gpu_verified(fake_gpu):
    fake_gpu.plan = "Project [ai_udf_verify_x(_udf_param_0#1L) AS result#3L]"
    result = _gpu_result()
    legs = []

    def verify_fn(**kwargs):
        legs.append(kwargs["result"].class_name)
        return True, None

    assert rapids.verify_result(verify_fn, spark=_rapids_spark(), result=result, **_verify_args()) == (
        True,
        None,
    )
    # The wrapper's row path would have matched; it must not get the chance to vouch for the kernel.
    assert legs == ["ai_udf.LowBits"]
    assert (result.kind, result.entry) == (KIND_JAVA_UDF, None)


def _narrowing_kernel(result, args):
    result.entry = "__device__ inline void udf(int32_t* out, int x) { *out = x; }"


@pytest.mark.parametrize(
    "change",
    [
        _narrowing_kernel,
        lambda result, args: args.update(input_types=["bigint", "bigint"]),
        lambda result, args: args.update(return_type="double"),
        lambda result, args: conf.set_value(conf.GPU_KERNEL, "false"),
    ],
    ids=["kernel-fails-the-scan", "two-inputs", "double-return", "conf-off"],
)
def test_verify_result_never_builds_a_kernel_it_should_not_run(fake_gpu, change):
    result = _gpu_result()
    args = _verify_args()
    change(result, args)
    out = rapids.verify_result(lambda **k: (True, None), spark=_rapids_spark(), result=result, **args)
    assert out == (True, None)
    assert fake_gpu.built == []
    assert (result.kind, result.entry) == (KIND_JAVA_UDF, None)


def test_verify_result_gpu_leg_crash_is_a_dropped_kernel_not_a_failed_row(fake_gpu, monkeypatch):
    def boom(spark):
        raise RuntimeError("no GPU session for you")

    monkeypatch.setattr(rapids, "gpu_session", boom)
    result = _gpu_result()
    assert rapids.verify_result(
        lambda **k: (True, None), spark=_rapids_spark(), result=result, **_verify_args()
    ) == (True, None)
    assert (result.kind, result.entry) == (KIND_JAVA_UDF, None)


class _KernelBackend(FakeBackend):
    """The fake's low_bits_set fixture, remembering the prompt it was shown."""

    def run(self, job, sandbox):
        self.prompt = (sandbox / "PROMPT.md").read_text()
        return super().run(job, sandbox)


def test_fake_fixture_round_trips_through_the_sandbox_and_is_never_shared(tmp_path):
    from ai_udf_transpile import canonical_source_from_func
    from ai_udf_transpile.backends.fake import FIXTURES

    job = _job()
    job.source_text = canonical_source_from_func(low_bits_set)
    sandbox = write_sandbox(tmp_path, job)
    result = FakeBackend().run(job, sandbox)
    assert (result.kind, result.entry) == (KIND_GPU_KERNEL, LOW_BITS_CUDA)
    parsed = parse_sandbox(sandbox)
    assert (parsed.kind, parsed.class_name, parsed.entry) == (
        KIND_GPU_KERNEL,
        "ai_udf.LowBits",
        LOW_BITS_CUDA.strip(),
    )
    assert rapids.kernel_error(parsed.entry, ABI) is None
    # The worker demotes results in place; the fixture table must not notice.
    result.kind, result.entry = KIND_JAVA_UDF, None
    assert FIXTURES[job.source_text].kind == KIND_GPU_KERNEL
    assert FakeBackend().run(job, sandbox).entry == LOW_BITS_CUDA


def _pending(tmp_path):
    from ai_udf_transpile import canonical_source_from_func

    cat = SqliteCatalog(tmp_path / "c.sqlite")
    cat.insert_pending(
        udf_key="k",
        source_text=canonical_source_from_func(low_bits_set),
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "fake")
    return cat


def test_process_row_without_rapids_stores_a_java_udf_and_never_offers_the_kernel(tmp_path):
    cat = _pending(tmp_path)
    backend = _KernelBackend()
    process_row(cat, cat.get("k"), backend, spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert (row.target_kind, row.impl_entry, row.impl_class) == (KIND_JAVA_UDF, None, "ai_udf.LowBits")
    assert "OUT.cu" not in backend.prompt


def test_process_row_stores_a_gpu_verified_kernel_and_offers_it_in_the_prompt(
    tmp_path, fake_gpu, monkeypatch
):
    import ai_udf_transpile.worker as worker_mod

    cat = _pending(tmp_path)
    backend = _KernelBackend()
    spark = _rapids_spark()
    # Compile and the reconstruction smoke test need a JVM; the gate is what is under test.
    monkeypatch.setattr(worker_mod, "smoke_test_reconstruction", lambda *a, **k: None)
    monkeypatch.setattr(
        "ai_udf_transpile.javac.compile_java",
        lambda spark, source, class_name=None: types.SimpleNamespace(
            jar_bytes=_jar_entries({"ai_udf/LowBits.class": b"cpu"}),
            class_name="ai_udf.LowBits",
            janino=False,
        ),
    )
    process_row(cat, cat.get("k"), backend, spark=spark, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert (row.target_kind, row.impl_entry, row.impl_class) == (
        KIND_GPU_KERNEL,
        LOW_BITS_CUDA,
        "ai_udf.LowBits",
    )
    assert rapids.jar_has_class(row.impl_binary, "ai_udf.LowBits__Rapids")
    assert row.reconstructable()
    assert "__device__ inline void udf(int32_t* out, int64_t x) { ... }" in backend.prompt


def test_cache_row_gpu_kernel_is_reconstructable_like_its_java_class():
    row = CacheRow(udf_key="k", status="success", target_kind=KIND_GPU_KERNEL, impl_class="ai_udf.LowBits")
    assert row.reconstructable()
    assert not CacheRow(udf_key="k", status="success", target_kind=KIND_GPU_KERNEL).reconstructable()


def test_reconstruct_gpu_kernel_row_without_a_session_returns_none():
    pytest.importorskip("pyspark")
    from ai_udf_transpile import transpiler as transpiler_mod

    row = CacheRow(
        udf_key="k",
        status="success",
        target_kind=KIND_GPU_KERNEL,
        impl_class="ai_udf.LowBits",
        impl_binary=b"jar",
        param_names=["x"],
    )
    assert transpiler_mod._reconstruct_column(row, "bigint", None) is None


def test_rapids_test_hook_is_env_gated(monkeypatch, tmp_path):
    from conftest import _rapids_configs

    monkeypatch.delenv("AI_UDF_TRANSPILE_TEST_GPU_JAR", raising=False)
    assert _rapids_configs() == []
    monkeypatch.setenv("AI_UDF_TRANSPILE_TEST_GPU_JAR", str(tmp_path / "nope.jar"))
    assert _rapids_configs() == []
    jar = tmp_path / "rapids.jar"
    jar.write_bytes(b"PK")
    monkeypatch.setenv("AI_UDF_TRANSPILE_TEST_GPU_JAR", str(jar))
    configs = dict(_rapids_configs())
    assert configs["spark.jars"] == str(jar)
    # The same plugin name rapids_active looks for.
    assert configs["spark.plugins"] == rapids.RAPIDS_PLUGIN
