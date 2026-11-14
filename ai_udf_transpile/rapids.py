# SPDX-License-Identifier: Apache-2.0
"""gpu_kernel target: a CUDA row kernel behind a Spark RAPIDS ``RapidsUDF``.

A gpu_kernel rewrite is a java_udf plus one CUDA device function. The Java
class is the CPU row path and is verified exactly like any java_udf. The
kernel is wrapped in a generated class that also implements
``com.nvidia.spark.RapidsUDF``; Spark RAPIDS calls its ``evaluateColumnar``
from ``GpuProject`` and the wrapper hands the kernel source to libcudf's JIT
(``ColumnView.transform``), so nothing native is built here.

The kernel is trusted only after it has matched the Python on a GPU. With no
RAPIDS session, the conf off, an unsupported signature, a kernel that does not
compile, or a mismatch, the kernel is dropped and the row is an ordinary
java_udf.

The ABI is the one cudf's Java binding has today: one fixed-width input
column, an INT32 output column, NULL in gives NULL out. libcudf's C++
``cudf::transform`` takes many columns, any fixed-width or string output, and
null-aware kernels; ``stock_abi`` is the only place that would change when
the Java binding catches up.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.targets import KIND_GPU_KERNEL, KIND_JAVA_UDF, TranspileResult

logger = logging.getLogger(__name__)

RAPIDS_PLUGIN = "com.nvidia.spark.SQLPlugin"
# Compile-time dependencies of the generated wrapper.
RAPIDS_CLASSES = ("com.nvidia.spark.RapidsUDF", "ai.rapids.cudf.ColumnVector")
WRAPPER_SUFFIX = "__Rapids"
# A Java string constant holds 65535 UTF-8 bytes; a row kernel needs far less.
MAX_KERNEL_BYTES = 16 * 1024

# Spark simpleString -> (ai.rapids.cudf.DType constant, C spellings; the first is canonical).
_FIXED_WIDTH: dict[str, tuple[str, tuple[str, ...]]] = {
    "boolean": ("BOOL8", ("bool",)),
    "tinyint": ("INT8", ("int8_t",)),
    "smallint": ("INT16", ("int16_t",)),
    "int": ("INT32", ("int32_t", "int")),
    "bigint": ("INT64", ("int64_t",)),
    "float": ("FLOAT32", ("float",)),
    "double": ("FLOAT64", ("double",)),
}
# The binding's output column is always INT32. Returns it can carry, and the
# DType the wrapper casts it to (None: already the declared type).
_INT32_RETURNS: dict[str, Optional[str]] = {"int": None, "bigint": "INT64", "boolean": "BOOL8"}
_OUT_CTYPES = ("int32_t", "int")


@dataclass(frozen=True)
class KernelAbi:
    """What one kernel must look like for the binding that will run it."""

    input_type: str  # Spark simpleString of the one input column
    input_dtype: str  # ai.rapids.cudf.DType constant
    input_ctypes: tuple[str, ...]
    return_type: str  # Spark simpleString of the declared UDF return
    cast_to: Optional[str]  # DType the INT32 result is cast to

    def signature(self, arg: str = "a") -> str:
        return f"__device__ inline void udf({_OUT_CTYPES[0]}* out, {self.input_ctypes[0]} {arg})"


def stock_abi(input_types: list[str], return_type: str) -> Optional[KernelAbi]:
    """The ABI for this signature under cudf's Java ``transform``, or None."""
    if len(input_types) != 1:
        return None
    simple = input_types[0].strip().lower()
    declared = (return_type or "").strip().lower()
    if simple not in _FIXED_WIDTH or declared not in _INT32_RETURNS:
        return None
    dtype, ctypes = _FIXED_WIDTH[simple]
    return KernelAbi(simple, dtype, ctypes, declared, _INT32_RETURNS[declared])


# Comments and literals, blanked before the scan (libcudf strips comments
# before it looks for the function name, so the scan sees what it sees).
_C_SKIP = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'",
    re.DOTALL,
)
_KERNEL_HEADER = re.compile(r"\s*__device__\s+(?:inline\s+)?void\s+([A-Za-z_]\w*)\s*\(([^()]*)\)\s*\{")
# Not a sandbox: a kernel is native code on the GPU. These are the constructs
# a row function never needs and that turn a wrong kernel into a wild write.
_FORBIDDEN_C = re.compile(
    r"\b(?:asm|__asm__|__asm|extern|goto|new|delete|malloc|free|reinterpret_cast|const_cast"
    r"|__host__|namespace)\b"
)


def _blank(match: re.Match) -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def kernel_error(cuda: str, abi: KernelAbi) -> Optional[str]:
    """Why this kernel must not be compiled or run, or None.

    A tripwire, not a boundary: it pins the one shape libcudf's parser and the
    INT32 binding accept, so a kernel that would be silently coerced (an
    ``int`` parameter over an INT64 column) or that writes outside its row is
    refused before it reaches a GPU.
    """
    text = (cuda or "").replace("\r\n", "\n")
    if not text.strip():
        return "kernel is empty"
    if len(text.encode("utf-8", "replace")) > MAX_KERNEL_BYTES:
        return f"kernel is larger than {MAX_KERNEL_BYTES} bytes"
    if any(not (32 <= ord(ch) < 127 or ch in "\n\t") for ch in text):
        return "kernel must be printable ASCII"
    # A spliced line or a trigraph changes what the lexer sees after this scan.
    if "\\\n" in text or "??" in text:
        return "kernel uses a line splice or a trigraph"
    scan = _C_SKIP.sub(_blank, text)
    if "#" in scan or "%:" in scan:
        return "kernel uses a preprocessor directive"
    header = _KERNEL_HEADER.match(scan)
    if header is None:
        return f"kernel must be exactly one function: {abi.signature()} {{ ... }}"
    params = [part.strip() for part in header.group(2).split(",")]
    if len(params) != 2:
        return f"kernel takes (out, input), not {len(params)} parameters"
    out = re.fullmatch(rf"(?:{'|'.join(_OUT_CTYPES)})\s*\*\s*([A-Za-z_]\w*)", params[0])
    if out is None:
        return f"kernel output must be `{_OUT_CTYPES[0]}* out`, not `{params[0]}`"
    arg = re.fullmatch(rf"(?:const\s+)?(?:{'|'.join(abi.input_ctypes)})\s+([A-Za-z_]\w*)", params[1])
    if arg is None:
        return (
            f"kernel input must be `{abi.input_ctypes[0]}` for a {abi.input_type} column, not `{params[1]}`"
        )
    depth = 1
    end = header.end()
    while end < len(scan) and depth:
        depth += {"{": 1, "}": -1}.get(scan[end], 0)
        end += 1
    if depth:
        return "kernel braces do not balance"
    if scan[end:].strip():
        return "kernel has code after its one function"
    body = scan[header.end() : end - 1]
    forbidden = _FORBIDDEN_C.search(body)
    if forbidden:
        return f"kernel uses a forbidden construct: {forbidden.group(0)}"
    out_name = out.group(1)
    for use in re.finditer(rf"\b{re.escape(out_name)}\b", body):
        before = body[: use.start()].rstrip()
        after = body[use.end() :].lstrip()
        # `*out++` moves the pointer, `&*out` aliases it for later indexing.
        deref = before.endswith("*") and not before[:-1].rstrip().endswith("&")
        if not deref or after.startswith(("[", "++", "--")):
            return f"kernel may only use the output as `*{out_name}`"
    if not re.search(rf"\*\s*{re.escape(out_name)}\s*=(?!=)", body):
        return f"kernel never assigns `*{out_name}`"
    return None


def wrapper_class(cpu_class: str) -> str:
    return f"{cpu_class}{WRAPPER_SUFFIX}"


def _java_string(text: str) -> str:
    """``text`` as a Java string expression, one literal per source line."""
    lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
    parts = []
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t")
        parts.append(f'"{escaped}\\n"')
    return "\n            + ".join(parts)


def wrapper_source(cpu_class: str, cuda: str, abi: KernelAbi) -> str:
    """Java source of the RapidsUDF wrapper around ``cpu_class`` and ``cuda``.

    Fully qualified names throughout, so the file needs no imports. ``call``
    delegates to the verified CPU class; ``evaluateColumnar`` is the GPU path.
    """
    package, _, simple = wrapper_class(cpu_class).rpartition(".")
    cudf = "ai.rapids.cudf"
    if abi.cast_to:
        compute = (
            f"        try ({cudf}.ColumnVector raw = in.transform(CUDA, false)) {{\n"
            f"            return raw.castTo({cudf}.DType.{abi.cast_to});\n"
            "        }\n"
        )
    else:
        compute = "        return in.transform(CUDA, false);\n"
    return (
        (f"package {package};\n\n" if package else "")
        + "// Generated by ai-udf-transpile. Do not edit.\n"
        + f"public class {simple}\n"
        + "        implements org.apache.spark.sql.api.java.UDF1<Object, Object>,"
        + " com.nvidia.spark.RapidsUDF {\n"
        + f"    private static final String CUDA =\n            {_java_string(cuda)};\n\n"
        + f"    private final {cpu_class} cpu = new {cpu_class}();\n\n"
        + "    @Override\n"
        + "    public Object call(Object arg0) throws Exception {\n"
        + "        return cpu.call(arg0);\n"
        + "    }\n\n"
        + "    @Override\n"
        + f"    public {cudf}.ColumnVector evaluateColumnar(int numRows, {cudf}.ColumnVector... args) {{\n"
        + "        if (args.length != 1) {\n"
        + '            throw new IllegalArgumentException("expected 1 column, got " + args.length);\n'
        + "        }\n"
        + f"        {cudf}.ColumnVector in = args[0];\n"
        + f"        if (!{cudf}.DType.{abi.input_dtype}.equals(in.getType())) {{\n"
        + "            throw new IllegalArgumentException(\n"
        + f'                    "expected {abi.input_dtype} input, got " + in.getType());\n'
        + "        }\n"
        + "        // cudf's Java transform: INT32 out, the input's null mask kept.\n"
        + compute
        + "    }\n"
        + "}\n"
    )


def jar_has_class(jar_bytes: Optional[bytes], class_name: str) -> bool:
    if not jar_bytes:
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(jar_bytes)) as jar:
            return class_name.replace(".", "/") + ".class" in jar.namelist()
    except zipfile.BadZipFile:
        return False


def kernel_enabled(spark: Any = None) -> bool:
    return conf.get_bool(conf.GPU_KERNEL, spark, False)


def rapids_active(spark: Any) -> bool:
    """The RAPIDS SQL plugin is loaded in this application and not switched off."""
    if spark is None:
        return False
    try:
        plugins = spark.sparkContext.getConf().get("spark.plugins", "") or ""
        if RAPIDS_PLUGIN not in [name.strip() for name in plugins.split(",")]:
            return False
        return str(spark.conf.get("spark.rapids.sql.enabled", "true")).strip().lower() != "false"
    except Exception:
        logger.debug("could not read the RAPIDS confs", exc_info=True)
        return False


def offered_abi(spark: Any, input_types: list[str], return_type: str) -> Optional[KernelAbi]:
    """The ABI to describe in the prompt, or None when a kernel could not ship."""
    if not kernel_enabled(spark) or not rapids_active(spark):
        return None
    return stock_abi(input_types, return_type)


def kernel_prompt(abi: KernelAbi, arg: str = "a") -> str:
    """The prompt section that offers ``OUT.cu`` for one job."""
    arg = arg if re.fullmatch(r"[A-Za-z_]\w*", arg or "") and arg != "out" else "a"
    lines = [
        "## Optional GPU kernel (`OUT.cu`)",
        "",
        "This session runs Spark RAPIDS. A Catalyst rewrite (`OUT.sql`) already runs",
        "on the GPU, so it needs nothing more. If you write `OUT.java`, you may also",
        "write `OUT.cu`: the same function as one CUDA device function, JIT-compiled",
        "by libcudf and run once per row on the GPU. `OUT.java` stays the CPU path",
        "and must be complete on its own.",
        "",
        "Write exactly this signature (the types come from `types.json`):",
        "",
        "```",
        f"{abi.signature(arg)} {{ ... }}",
        "```",
        "",
        "- One function and nothing else: no `#include`, no `#define`, no helper",
        "  functions, no globals. Lambdas and local arrays inside the body are fine.",
        "- Rows whose input is NULL never reach the kernel; their output is NULL.",
        "  The kernel cannot return NULL for a non-NULL input. If the Python returns",
        "  a value for `None`, or `None` for some real input, do not write `OUT.cu`.",
        "- Assign `*out` on every path. Never index or offset `out`.",
    ]
    if abi.return_type == "boolean":
        lines.append("- `*out` is an int32: write `1` for True and `0` for False.")
    elif abi.return_type == "bigint":
        lines += [
            "- `*out` is an int32, widened to the declared bigint afterwards. Write a",
            "  kernel only if the result fits in int32 for every input (`x + 1` does",
            "  not: it is wrong for 2147483647).",
        ]
    lines += [
        "- C integers wrap silently where Python ints grow; C `/` and `%` truncate",
        "  toward zero where Python floors.",
        "- The kernel is tested against the Python on a GPU. A kernel that fails is",
        "  dropped and the Java UDF still ships, so if you are unsure, skip `OUT.cu`.",
        "",
        "### Example (kernel)",
        "",
        "`udf.py`:",
        "```python",
        "def low_bits_set(x: int) -> int:",
        '    return bin(x & 0xFFFFFFFF).count("1")',
        "```",
        "`OUT.cu` (next to an `OUT.java` that does the same on the CPU):",
        "```",
        "__device__ inline void udf(int32_t* out, int64_t x) {",
        "  uint64_t v = static_cast<uint64_t>(x) & 0xFFFFFFFFull;",
        "  int32_t n = 0;",
        "  while (v) { n += static_cast<int32_t>(v & 1); v >>= 1; }",
        "  *out = n;",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"


def rapids_classpath(spark: Any) -> list[str]:
    """Local jar paths that hold ``RapidsUDF`` and the cudf Java classes.

    ``--jars`` puts the RAPIDS jar on Spark's classloader, not on
    ``java.class.path``, so javac has to be told where it is.
    """
    jvm = spark._jvm
    loader = jvm.org.apache.spark.util.Utils.getContextOrSparkClassLoader()
    paths: list[str] = []
    for name in RAPIDS_CLASSES:
        location = loader.loadClass(name).getProtectionDomain().getCodeSource().getLocation()
        path = str(jvm.java.nio.file.Paths.get(location.toURI()).toString())
        if path not in paths:
            paths.append(path)
    return paths


def build_wrapper_jar(
    spark: Any, cpu_jar: bytes, cpu_class: str, cuda: str, abi: KernelAbi
) -> tuple[bytes, str]:
    """Compile the wrapper against ``cpu_jar``; return (cpu + wrapper jar, wrapper class).

    The CPU class is not recompiled: its bytes in the merged jar are the ones
    the CPU leg verified.
    """
    from ai_udf_transpile.javac import _compile_jdk, _stage_jar, merge_jars, verify_function_name

    name = wrapper_class(cpu_class)
    cpu_path = _stage_jar(verify_function_name("ai_udf_cpu", cpu_jar), cpu_jar)
    wrapper_jar = _compile_jdk(
        spark,
        wrapper_source(cpu_class, cuda, abi),
        name,
        extra_classpath=[*rapids_classpath(spark), cpu_path],
    )
    # Wrapper first: a re-verify of a jar that already holds a wrapper keeps the new one.
    return merge_jars(wrapper_jar, cpu_jar), name


def gpu_session(spark: Any) -> Any:
    """A sibling session for the GPU leg: its confs never touch the user's queries.

    ``newSession`` starts from the application conf, so runtime settings (ANSI,
    time zone, ``spark.rapids.*``) are copied over; the ones Spark refuses to
    set at runtime are the static ones it already shares.
    """
    session = spark.newSession()
    for row in spark.sql("SET").collect():
        try:
            session.conf.set(row[0], row[1])
        except Exception:
            continue
    session.conf.set("spark.rapids.sql.enabled", "true")
    # Final plan up front: AQE would show the CPU plan until the query ran.
    session.conf.set("spark.sql.adaptive.enabled", "false")
    return session


def plan_runs_kernel(plan: str, fname: str) -> bool:
    """Whether a physical plan evaluates ``fname`` through ``evaluateColumnar``.

    RAPIDS names a RapidsUDF expression ``gpu_<name>`` and a row UDF kept
    inside a GPU plan ``gpu_row_based_<name>``; a CPU fallback keeps the bare
    name. Only the first proves the kernel is what runs.
    """
    return f"gpu_{fname}(" in plan


def _executed_plan(session: Any, expr: str, input_types: list[str]) -> str:
    from pyspark.sql.types import StructType

    from ai_udf_transpile.verify import _param_fields

    df = session.createDataFrame([tuple(None for _ in input_types)], StructType(_param_fields(input_types)))
    return str(df.selectExpr(f"({expr}) AS result")._jdf.queryExecution().executedPlan().toString())


def columnar_plan_error(session: Any, expr: str, input_types: list[str]) -> Optional[str]:
    """None when RAPIDS plans ``expr`` (``fname(args)``) on the GPU through the wrapper.

    Without this, a CPU fallback would run the wrapper's row path, match the
    Python, and vouch for a kernel that never executed.
    """
    plan = _executed_plan(session, expr, input_types)
    if plan_runs_kernel(plan, expr.split("(", 1)[0]):
        return None
    return f"RAPIDS did not plan the kernel on the GPU: {plan.strip()[:400]}"


def _verify_kernel(
    verify_fn: Callable[..., tuple[bool, Optional[str]]],
    *,
    spark: Any,
    result: TranspileResult,
    cuda: Optional[str],
    input_types: list[str],
    return_type: str,
    **kwargs: Any,
) -> Optional[str]:
    """Run the GPU leg. None means the kernel is verified and ``result`` carries it."""
    if not cuda or not cuda.strip():
        return "no kernel source"
    if not kernel_enabled(spark):
        return f"{conf.GPU_KERNEL} is off"
    if not rapids_active(spark):
        return "Spark RAPIDS is not active in this session"
    abi = stock_abi(input_types, return_type)
    if abi is None:
        return (
            f"({', '.join(input_types)}) -> {return_type} does not fit cudf's Java transform "
            "(one fixed-width input; int, bigint, or boolean return)"
        )
    shape = kernel_error(cuda, abi)
    if shape:
        return shape
    if not (result.binary and result.class_name):
        return "no compiled java class to wrap"
    from ai_udf_transpile.verify import _java_udf_expr

    jar, name = build_wrapper_jar(spark, result.binary, result.class_name, cuda, abi)
    candidate = TranspileResult(kind=KIND_JAVA_UDF, class_name=name, binary=jar, model=result.model)
    session = gpu_session(spark)
    expr = _java_udf_expr(session, candidate, input_types, return_type)
    if expr is None:
        return "wrapper could not be registered"
    plan_error = columnar_plan_error(session, expr, input_types)
    if plan_error:
        return plan_error
    ok, err = verify_fn(
        result=candidate, spark=session, input_types=input_types, return_type=return_type, **kwargs
    )
    if not ok:
        return f"kernel failed verification on the GPU: {err}"
    result.kind, result.entry, result.binary = KIND_GPU_KERNEL, cuda, jar
    return None


def verify_result(
    verify_fn: Callable[..., tuple[bool, Optional[str]]],
    *,
    spark: Any,
    result: TranspileResult,
    input_types: list[str],
    return_type: str,
    **kwargs: Any,
) -> tuple[bool, Optional[str]]:
    """``verify_fn`` for every kind; a gpu_kernel result gets a CPU leg and a GPU leg.

    The Java class is verified first, as the java_udf it is. The result stays
    gpu_kernel only when the kernel then matched the Python on a GPU; anything
    else leaves a verified java_udf with the kernel dropped.
    """
    if getattr(result, "kind", None) != KIND_GPU_KERNEL:
        return verify_fn(
            result=result, spark=spark, input_types=input_types, return_type=return_type, **kwargs
        )
    cuda = result.entry
    result.kind, result.entry = KIND_JAVA_UDF, None
    ok, err = verify_fn(
        result=result, spark=spark, input_types=input_types, return_type=return_type, **kwargs
    )
    if not ok:
        # The failed row keeps what was rejected.
        result.kind, result.entry = KIND_GPU_KERNEL, cuda
        return ok, err
    try:
        reason = _verify_kernel(
            verify_fn,
            spark=spark,
            result=result,
            cuda=cuda,
            input_types=input_types,
            return_type=return_type,
            **kwargs,
        )
    except Exception as exc:
        logger.debug("gpu leg raised", exc_info=True)
        reason = f"{type(exc).__name__}: {exc}"
    if reason:
        logger.warning("gpu kernel dropped, serving the java UDF: %s", reason)
    return True, None


def serving_class(spark: Any, cpu_class: Optional[str], jar_bytes: Optional[bytes]) -> Optional[str]:
    """The wrapper class to serve from a gpu_kernel row, or None for the CPU class."""
    if not cpu_class or not kernel_enabled(spark) or not rapids_active(spark):
        return None
    name = wrapper_class(cpu_class)
    return name if jar_has_class(jar_bytes, name) else None
