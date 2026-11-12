# SPDX-License-Identifier: Apache-2.0
"""Compile Java UDF source into jar bytes and register it with a classic SparkSession.

Primary path is the in-process JDK compiler (``javax.tools``); any standard
Java 17+ runtime includes it. Fallback is Janino, which Spark already bundles
for Catalyst codegen. Janino emits no generic ``Signature`` attributes and no
bridge methods, so Janino-compiled classes cannot go through
``registerJavaFunction`` (Spark reflects on ``UDFN`` type arguments); they are
instantiated reflectively and wrapped with ``functions.udf(UDFN, DataType)``
under the ``spark.sql.legacy.allowUntypedScalaUDF`` escape hatch. Prompts
therefore ask backends for erased signatures (``UDF1<Object, Object>``), which
work under both compilers.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_][\w.]*)")
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_UDF_IFACE_PREFIX = "org.apache.spark.sql.api.java.UDF"

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class JavaCompileError(Exception):
    """Java source could not be compiled into a UDF class."""


# Comments and literals, where a class name is text rather than a reference.
# (verify._JAVA_SKIP is the same lexer; it is duplicated here so javac does
# not import the verifier.)
_JAVA_TEXT = re.compile(
    r"//[^\n\r]*"
    r"|/\*.*?\*/"
    r'|"""(?:[^"\\]|\\.|"(?!""))*"""'
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'",
    re.DOTALL,
)


def _code_only(source: str) -> str:
    """The source with comments and literals blanked (newlines kept), so a
    `// This class implements ...` comment is not read as a declaration."""
    return _JAVA_TEXT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), source)


def _declares(source: str, simple: str) -> bool:
    return re.search(rf"\bclass\s+{re.escape(simple)}(?![\w$])", _code_only(source)) is not None


def extract_java_class(source: str) -> Optional[str]:
    code = _code_only(source)
    pkg_match = _PACKAGE_RE.search(code)
    class_match = _CLASS_RE.search(code)
    if not class_match:
        return None
    simple = class_match.group(1)
    if pkg_match:
        return f"{pkg_match.group(1)}.{simple}"
    return simple


@dataclass
class CompiledJava:
    jar_bytes: bytes
    class_name: str
    janino: bool


def _jar_entries(entries: dict[str, bytes]) -> bytes:
    """Deterministic jar: sorted names, fixed timestamps (content-addressed)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as jar:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            jar.writestr(info, entries[name])
    return buf.getvalue()


def _compile_jdk(spark: Any, java_source: str, class_name: str) -> bytes:
    jvm = spark._jvm
    compiler = jvm.javax.tools.ToolProvider.getSystemJavaCompiler()
    if compiler is None:
        raise JavaCompileError("no system Java compiler (driver is not a JDK)")
    pkg, _, simple = class_name.rpartition(".")
    tmpdir = tempfile.mkdtemp(prefix="ai_udf_javac_")
    try:
        src_dir = os.path.join(tmpdir, "src", *pkg.split(".")) if pkg else os.path.join(tmpdir, "src")
        out_dir = os.path.join(tmpdir, "classes")
        os.makedirs(src_dir)
        os.makedirs(out_dir)
        src_file = os.path.join(src_dir, f"{simple}.java")
        with open(src_file, "w", encoding="utf-8") as handle:
            handle.write(java_source)
        file_manager = compiler.getStandardFileManager(None, None, None)
        units = file_manager.getJavaFileObjectsFromStrings(jvm.java.util.Collections.singletonList(src_file))
        classpath = jvm.java.lang.System.getProperty("java.class.path")
        options = jvm.java.util.ArrayList()
        for opt in ("-classpath", classpath, "-d", out_dir, "-proc:none", "-encoding", "UTF-8"):
            options.add(opt)
        diagnostics = jvm.javax.tools.DiagnosticCollector()
        ok = bool(compiler.getTask(None, file_manager, diagnostics, options, None, units).call())
        if not ok:
            messages = "; ".join(str(d) for d in diagnostics.getDiagnostics().toArray())
            raise JavaCompileError(f"javac failed for {class_name}: {messages or 'no diagnostics'}")
        entries: dict[str, bytes] = {}
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, out_dir).replace(os.sep, "/")
                with open(full, "rb") as handle:
                    entries[arc] = handle.read()
        if not entries:
            raise JavaCompileError(f"javac produced no classes for {class_name}")
        return _jar_entries(entries)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _compile_janino(spark: Any, java_source: str) -> bytes:
    jvm = spark._jvm
    try:
        compiler = jvm.org.codehaus.janino.SimpleCompiler()
    except Exception as exc:
        raise JavaCompileError(f"janino not on the driver classpath: {exc}") from exc
    compiler.setParentClassLoader(jvm.org.apache.spark.util.Utils.getContextOrSparkClassLoader())
    try:
        compiler.cook(java_source)
    except Exception as exc:
        raise JavaCompileError(f"janino cook failed: {exc}") from exc
    bytecodes = compiler.getBytecodes()
    entries = {}
    for entry in bytecodes.entrySet().toArray():
        name = str(entry.getKey()).replace(".", "/") + ".class"
        entries[name] = bytes(entry.getValue())
    if not entries:
        raise JavaCompileError("janino produced no classes")
    return _jar_entries(entries)


def _resolve_class(java_source: str, class_name: Optional[str]) -> Optional[tuple[str, str, str]]:
    """(package, declared simple name, content-addressed simple name).

    An explicitly supplied name wins when the source declares it (bare, or
    already carrying this source's suffix); otherwise the first declaration
    in the code (comments and literals excluded) is the class.
    """
    suffix = "_" + hashlib.sha256(java_source.encode("utf-8")).hexdigest()[:16]
    declared = extract_java_class(java_source)
    for candidate in (class_name, declared):
        if not candidate:
            continue
        pkg, _, simple = candidate.rpartition(".")
        base = simple[: -len(suffix)] if simple.endswith(suffix) else simple
        if candidate is declared or _declares(java_source, base):
            if not pkg and declared and candidate is not declared:
                pkg = declared.rpartition(".")[0]
            return pkg, base, base + suffix
    return None


def unique_class_name(java_source: str, class_name: Optional[str] = None) -> Optional[str]:
    """The name a source is compiled under: its class, suffixed with a hash of
    the source text.

    Every jar a session registers lands in one classloader, and a class name
    resolves to whichever jar loaded it first. Two different rewrites that
    happen to pick the same name (retry attempts, a re-verify after the UDF
    changed, two functions both called `clean`) would otherwise all run the
    first one's code -- in verification as well as in serving. Hashing the
    source makes the name a function of the code, so one name, one body.
    """
    resolved = _resolve_class(java_source, class_name)
    if resolved is None:
        return None
    pkg, _base, unique = resolved
    return f"{pkg}.{unique}" if pkg else unique


def _rename_class(java_source: str, old_simple: str, new_simple: str) -> str:
    """Replace the identifier `old_simple` in code (not in comments or literals)."""
    word = re.compile(rf"(?<![\w$]){re.escape(old_simple)}(?![\w$])")
    out: list[str] = []
    last = 0
    for text in _JAVA_TEXT.finditer(java_source):
        out.append(word.sub(new_simple, java_source[last : text.start()]))
        out.append(text.group(0))
        last = text.end()
    out.append(word.sub(new_simple, java_source[last:]))
    return "".join(out)


def compile_java(spark: Any, java_source: str, class_name: Optional[str] = None) -> CompiledJava:
    """Compile Java UDF source to jar bytes. JDK compiler first, Janino fallback.

    The class is compiled under `unique_class_name` (the declared name plus a
    hash of the source) and that is the name returned: callers register and
    store `CompiledJava.class_name`, never the name the source declares.
    Passing either name back in for the same source yields the same result.
    """
    if spark is None:
        raise JavaCompileError("compile_java requires a SparkSession")
    resolved = _resolve_class(java_source, class_name)
    if resolved is None:
        raise JavaCompileError("no class declaration found in Java source")
    pkg, base, simple = resolved
    unique = f"{pkg}.{simple}" if pkg else simple
    source = _rename_class(java_source, base, simple)
    try:
        return CompiledJava(_compile_jdk(spark, source, unique), unique, janino=False)
    except JavaCompileError as exc:
        logger.info("JDK compile unavailable (%s); falling back to janino", exc)
    return CompiledJava(_compile_janino(spark, source), unique, janino=True)


def _stage_jar(name: str, jar_bytes: bytes) -> str:
    root = os.path.join(tempfile.gettempdir(), "ai_udf_transpile_jars")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{name}.jar")
    existing = None
    if os.path.exists(path):
        with open(path, "rb") as handle:
            existing = handle.read()
    if existing != jar_bytes:
        with open(path, "wb") as handle:
            handle.write(jar_bytes)
    return path


def _add_artifact(spark: Any, name: str, jar_path: str) -> None:
    jvm = spark._jvm
    artifact_manager = spark._jsparkSession.artifactManager()
    try:
        artifact_manager.addArtifact(
            jvm.java.io.File(f"jars/{name}.jar").toPath(),
            jvm.java.io.File(jar_path).toPath(),
            jvm.scala.Option.empty(),
            False,
        )
    except Exception as exc:
        if "ARTIFACT_ALREADY_EXISTS" not in str(exc):
            raise


def _register_raw(spark: Any, name: str, class_name: str, return_type: Any) -> None:
    """Register a Janino-compiled (raw UDFN) class, bypassing the generic check."""
    jvm = spark._jvm
    gateway = spark.sparkContext._gateway
    # Session-wide conf; restore it so the registration does not leak the
    # untyped-UDF escape hatch to the rest of the session.
    prior = spark.conf.get("spark.sql.legacy.allowUntypedScalaUDF", None)
    spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    try:
        classloader = spark._jsparkSession.artifactManager().classloader()
        clazz = classloader.loadClass(class_name)
        instance = clazz.getConstructor(gateway.new_array(jvm.java.lang.Class, 0)).newInstance(
            gateway.new_array(jvm.java.lang.Object, 0)
        )
        iface = None
        for candidate in clazz.getInterfaces():
            if str(candidate.getName()).startswith(_UDF_IFACE_PREFIX):
                iface = candidate
                break
        if iface is None:
            raise JavaCompileError(f"{class_name} does not implement a Spark UDFN interface")
        jdt = spark._jsparkSession.parseDataType(return_type.json())
        param_types = gateway.new_array(jvm.java.lang.Class, 2)
        param_types[0] = iface
        param_types[1] = jvm.java.lang.Class.forName("org.apache.spark.sql.types.DataType")
        functions_cls = jvm.java.lang.Class.forName("org.apache.spark.sql.functions")
        method = functions_cls.getMethod("udf", param_types)
        args = gateway.new_array(jvm.java.lang.Object, 2)
        args[0] = instance
        args[1] = jdt
        udf = method.invoke(None, args)
        spark._jsparkSession.udf().register(name, udf)
    finally:
        if prior is None:
            spark.conf.unset("spark.sql.legacy.allowUntypedScalaUDF")
        else:
            spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", prior)


def register_java_udf(
    spark: Any,
    name: str,
    class_name: str,
    jar_bytes: bytes,
    return_type: Any,
    *,
    janino: Optional[bool] = None,
) -> str:
    """Stage jar bytes as a session artifact and register ``class_name`` as ``name``.

    ``janino=None`` auto-detects: standard ``registerJavaFunction`` first, raw
    UDFN bypass when the class carries no generic signature (Janino output).
    """
    jar_path = _stage_jar(name, jar_bytes)
    _add_artifact(spark, name, jar_path)
    if janino is not True:
        try:
            spark.udf.registerJavaFunction(name, class_name, return_type)
            return name
        except Exception:
            logger.debug("registerJavaFunction failed for %s; using raw UDF bypass", class_name)
    _register_raw(spark, name, class_name, return_type)
    return name


def verify_function_name(prefix: str, jar_bytes: bytes) -> str:
    return f"{prefix}_{hashlib.sha256(jar_bytes).hexdigest()[:16]}"
