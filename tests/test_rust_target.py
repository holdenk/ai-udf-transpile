# SPDX-License-Identifier: Apache-2.0
"""The rust target without a JVM: contract, glue, gates, parsing, catalog.

The cargo-backed tests at the bottom build real libraries and skip cleanly
when there is no cargo or no comet-udf-sdk ($COMET_HOME) to build against.
Running the result needs Comet: that is tests/test_rust_udf_comet.py.
"""

from __future__ import annotations

import json
import os
import shutil
import types
from pathlib import Path

import pytest

from ai_udf_transpile import conf, rustc
from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.cli import parse_sandbox
from ai_udf_transpile.backends.fake import (
    BACKWARDS_RS,
    COLLATZ_STEPS_RS,
    PLUS_ONE_RS,
    FakeBackend,
    collatz_steps,
)
from ai_udf_transpile.catalog import CacheRow
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox, write_sandbox
from ai_udf_transpile.targets import (
    KIND_RUST,
    OPT_IN_KINDS,
    RESERVED_KINDS,
    SERVABLE_KINDS,
    TranspileJob,
    TranspileResult,
)
from ai_udf_transpile.transpiler import _reconstruct_rust
from ai_udf_transpile.verify import hypothesis_check, rejected_rewrite
from ai_udf_transpile.worker import process_row

_COMET_PLUGIN = "org.apache.spark.CometPlugin"


def _job(func=collatz_steps, *, targets=None, input_types=None, return_type="bigint") -> TranspileJob:
    params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    job = TranspileJob(
        udf_key="k",
        source_text=canonical_source_from_func(func),
        param_names=params,
        input_types=input_types or ["bigint"] * len(params),
        input_categories=["numeric"] * len(params),
        return_type=return_type,
        captures={},
    )
    if targets is not None:
        job.targets = targets
    return job


class _Conf:
    def __init__(self, values: dict):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _scala_object(**members):
    holder = types.SimpleNamespace()
    setattr(holder, "MODULE$", types.SimpleNamespace(**members))
    return holder


def _comet_jvm(*, loaded=True, version="1.2.0", revision="abc123", native_udf=True):
    """Just enough of the driver JVM for the probes: Comet's Scala objects by name."""
    comet = types.SimpleNamespace(udf=types.SimpleNamespace())
    if native_udf:
        setattr(comet.udf, "CometNativeUDF$", _scala_object())
    setattr(comet, "CometSparkSessionExtensions$", _scala_object(isCometLoaded=lambda _conf: loaded))
    setattr(comet, "package$", _scala_object(COMET_VERSION=lambda: version, COMET_REVISION=lambda: revision))
    return types.SimpleNamespace(org=types.SimpleNamespace(apache=types.SimpleNamespace(comet=comet)))


class _FakeSpark:
    """A session as the gates see it: confs, plus the Comet classes when present."""

    def __init__(self, values=None, jvm=None):
        self.conf = _Conf(dict(values or {}))
        self._jvm = jvm
        state = types.SimpleNamespace(conf=lambda: None)
        self._jsparkSession = types.SimpleNamespace(sessionState=lambda: state)


def _comet_session(**overrides):
    values = {
        "spark.plugins": _COMET_PLUGIN,
        "spark.comet.enabled": "true",
        "spark.master": "local[2]",
        conf.TARGETS: "catalyst,java_udf,rust",
    }
    jvm_kwargs = overrides.pop("jvm", {})
    values.update(overrides)
    return _FakeSpark(values, _comet_jvm(**jvm_kwargs))


# -- the kind itself ----------------------------------------------------------


def test_rust_is_an_opt_in_kind_not_a_reserved_one():
    assert KIND_RUST in OPT_IN_KINDS
    assert KIND_RUST in SERVABLE_KINDS
    assert KIND_RUST not in RESERVED_KINDS
    assert KIND_RUST not in conf.get_csv(conf.TARGETS, None, conf.DEFAULTS[conf.TARGETS])
    assert KIND_RUST not in TranspileJob("k", "", [], [], [], "bigint").targets


def test_result_payload_is_the_row_function():
    result = TranspileResult(kind=KIND_RUST, rust_source=PLUS_ONE_RS)
    assert result.reconstructable()
    assert result.impl_source == PLUS_ONE_RS
    assert not TranspileResult(kind=KIND_RUST, rust_source="  ").reconstructable()
    # A java source on a rust result is not the payload, and vice versa.
    assert TranspileResult(kind=KIND_RUST, java_source="class A {}").impl_source is None
    assert TranspileResult(kind="java_udf", java_source="class A {}", rust_source="x").impl_source == (
        "class A {}"
    )


def test_cache_row_round_trips_by_kind():
    rust = CacheRow(udf_key="k", status="success", target_kind="rust", impl_source=PLUS_ONE_RS)
    assert rust.reconstructable()
    assert rust.as_result().rust_source == PLUS_ONE_RS
    assert rust.as_result().java_source is None
    java = CacheRow(udf_key="k", status="success", target_kind="java_udf", impl_source="class A {}")
    assert java.as_result().java_source == "class A {}"
    assert java.as_result().rust_source is None
    assert not CacheRow(udf_key="k", status="success", target_kind="rust").reconstructable()


def test_sqlite_stores_the_rust_source(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    cat.insert_pending(
        udf_key="k",
        source_text="def f(x: int) -> int:\n    return x + 1\n",
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    cat.mark_success(
        "k", TranspileResult(kind=KIND_RUST, rust_source=PLUS_ONE_RS, entry="ai_udf_rs_x"), "fake"
    )
    row = cat.get("k")
    assert (row.target_kind, row.impl_source, row.impl_entry) == ("rust", PLUS_ONE_RS, "ai_udf_rs_x")
    assert row.impl_binary is None  # a .so is platform- and SDK-specific: never stored
    assert row.reconstructable()


# -- the row-function contract ------------------------------------------------


def test_signature_follows_the_declared_types():
    assert rustc.rust_signature(["bigint", "string"], "string") == (
        "pub fn udf(arg0: Option<i64>, arg1: Option<&str>) -> Result<Option<String>, String>"
    )
    assert rustc.rust_signature([], "boolean") == "pub fn udf() -> Result<Option<bool>, String>"
    assert rustc.rust_signature(["binary", "double", "int"], "binary") == (
        "pub fn udf(arg0: Option<&[u8]>, arg1: Option<f64>, arg2: Option<i32>) "
        "-> Result<Option<Vec<u8>>, String>"
    )


@pytest.mark.parametrize(
    "input_types,return_type",
    [
        (["array<string>"], "bigint"),
        (["map<string,string>"], "string"),
        (["timestamp"], "string"),
        (["bigint"], "array<string>"),
        (["bigint"], "decimal(10,2)"),
        (["bigint"], "LongType"),
    ],
)
def test_only_atomics_are_supported(input_types, return_type):
    assert not rustc.supports_types(input_types, return_type)
    with pytest.raises(rustc.RustCompileError):
        rustc.rust_signature(input_types, return_type)


def test_call_expr_casts_every_argument():
    # Comet's analyzer check inserts no casts: an int column into a bigint
    # parameter fails analysis unless the call casts.
    assert rustc.call_expr("f", ["bigint", "string"]) == (
        "f(CAST(_udf_param_0 AS BIGINT), CAST(_udf_param_1 AS STRING))"
    )
    assert rustc.call_expr("f", []) == "f()"


def test_glue_wraps_the_row_function():
    lib = rustc.render_lib("ai_udf_rs_abc", ["bigint", "string"], "string")
    assert "#[forbid(unsafe_code)]\nmod udf;" in lib
    assert 'const NAME: &str = "ai_udf_rs_abc";' in lib
    assert "const ARITY: usize = 2;" in lib
    assert "downcast_ref::<Int64Array>()" in lib and "downcast_ref::<StringArray>()" in lib
    assert "Field::new(NAME, DataType::Utf8, true)" in lib
    assert "StringBuilder::new()" in lib
    assert "udf::udf(v0, v1)" in lib
    assert "comet_c_udf_export!(AiUdf);" in lib
    # Sized builders take the row count; a zero-arg UDF still loops n_rows.
    zero = rustc.render_lib("ai_udf_rs_abc", [], "bigint")
    assert "Int64Builder::with_capacity(n_rows)" in zero and "udf::udf()" in zero


def test_manifest_pins_the_sdk_and_its_arrow():
    manifest = rustc.render_manifest("ai_udf_rs_abc", '{ path = "/comet/native/comet-udf-sdk" }', "59.2.0")
    assert 'name = "ai_udf_rs_abc"' in manifest
    assert 'crate-type = ["cdylib"]' in manifest
    assert 'arrow = "59.2.0"' in manifest
    assert 'comet-udf-sdk = { path = "/comet/native/comet-udf-sdk" }' in manifest
    # Overflow checks are per-package so the shared dependency build is untouched.
    assert '[profile.release.package."ai_udf_rs_abc"]\noverflow-checks = true' in manifest
    assert manifest.rstrip().endswith("[workspace]")
    assert "abort" not in manifest  # the SDK's panic guards need unwinding


def test_name_is_content_addressed():
    base = rustc.udf_name(PLUS_ONE_RS, ["bigint"], "bigint", "sdk-a")
    assert base == rustc.udf_name(PLUS_ONE_RS, ["BIGINT "], "bigint", "sdk-a")
    assert base.startswith("ai_udf_rs_") and len(base) == len("ai_udf_rs_") + 16
    assert base != rustc.udf_name(PLUS_ONE_RS + "// v2\n", ["bigint"], "bigint", "sdk-a")
    assert base != rustc.udf_name(PLUS_ONE_RS, ["int"], "bigint", "sdk-a")
    assert base != rustc.udf_name(PLUS_ONE_RS, ["bigint"], "int", "sdk-a")
    # Libraries are rebuilt per Comet SDK, never shared across them.
    assert base != rustc.udf_name(PLUS_ONE_RS, ["bigint"], "bigint", "sdk-b")


# -- the forbidden scan -------------------------------------------------------


@pytest.mark.parametrize("source", [PLUS_ONE_RS, BACKWARDS_RS, COLLATZ_STEPS_RS])
def test_fixtures_pass_the_scan(source):
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


@pytest.mark.parametrize(
    "body",
    [
        "unsafe { std::hint::unreachable_unchecked() }",
        'extern "C" { fn getpid() -> i32; }',
        'let _ = std::fs::read_to_string("/etc/passwd");',
        "use std::fs;",
        "use std::{collections::HashMap, fs};",
        "use std :: process :: Command;",
        'let _ = process::Command::new("sh");',
        'let _ = std::env::var("HOME");',
        'let _ = std::net::TcpStream::connect("example.com:80");',
        'let _ = std::path::Path::new("/etc/passwd").exists();',
        'let _ = include_str!("/etc/passwd");',
        'let _ = include_bytes ! ("/etc/passwd");',
        'let _ = env!("HOME");',
        '#[path = "/etc/passwd"] mod leak;',
        '#[cfg_attr(all(), path = "/etc/passwd")] mod leak;',
        '#[link(name = "c")] extern {}',
        "#[no_mangle] pub fn comet_udf_abi_version() -> u32 { 0 }",
        "let _ = arrow::ffi::FFI_ArrowArray::empty();",
        "let _ = std::thread::spawn(|| ());",
        "let _ = std::time::SystemTime::now();",
        # Raw identifiers name the same module.
        'let _ = std::r#fs::read("/x");',
        'use std::r#fs as quiet; let _ = quiet::read("/x");',
        # A macro can paste the module name where no token scan sees it.
        'macro_rules! m { ($a:ident) => { std::$a::read("/x") } } let _ = m!(fs);',
        "use std::time::Instant; let _ = Instant::now();",
        # A raw string is complete at its closing quote: no escapes inside.
        'let _a = r"\\"; unsafe { } let _b = "";',
        'let _a = r#"x"#; let _ = std::fs::read("/x"); let _b = r#"y"#;',
        # A char literal holding a quote does not open a string.
        "let _q = '\"'; unsafe { } let _r = '\"';",
        # Nested block comment: the tail after the first */ is still code to us.
        "/* /* */ */ unsafe { }",
        # ...and a quote inside the nested comment must not open a string
        # that hides the code after it (reported against the regex scan).
        '/* /* */ " */ std::process::exit(0); // "',
        '/* /* /* */ */ " */ let _ = std::fs::read("/x"); // "',
        # A lifetime is not a char literal: the quote does not open anything.
        "fn f<'a>(s: &'a str) -> &'a str { s } let _ = std::fs::read(\"/x\"); let _c = 'x';",
        # A raw string ends at its own closing hashes, not at the first quote.
        'let _a = r##"a "# b"##; std::process::exit(0); let _b = "";',
        # A renamed or globbed crate root hides the module from `fs::`.
        'use std as s; use s::fs as q; let _ = q::metadata("/").is_ok();',
        'use std::*; use fs as q; let _ = q::metadata("/").is_ok();',
        "use ::core as c; let _ = c::mem::size_of::<i64>();",
        "use std::{*}; let _ = 1;",
        'use std::{fs as quiet}; let _ = quiet::read("/x");',
    ],
)
def test_scan_rejects(body):
    source = f"pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {{\n    {body}\n    Ok(x)\n}}\n"
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected and "rust rewrite uses a forbidden construct" in rejected


@pytest.mark.parametrize(
    "body",
    [
        "// unsafe would be wrong here; std::fs is not needed",
        "/* extern, include_str!, env! */",
        'let _doc = "unsafe std::fs::read include_str!(x) #[path]";',
        'let _raw = r#"unsafe { std::process::exit(1) }"#;',
        "let _fsync = 1; let _envelope = 2; let _iota = 3; let _pathological = 4;",
        "use std::collections::HashMap; let _m: HashMap<i64, i64> = HashMap::new();",
        "use std::fmt::Write as _; let _io = 3 as i64; let _path = 2;",
        'let _s = format!("{}", 1); let _v: Vec<char> = "ab".chars().rev().collect();',
        "fn helper<'a>(s: &'a str) -> &'a str { s } let _ = helper(\"x\");",
        "/* outer /* inner std::fs */ still a comment: unsafe { std::process::exit(0) } */",
        'let _a = r##"unsafe "# std::fs::read"##;',
        "let _q = '\\''; let _u = '\\u{1F600}'; let _b = b'\\x7f';",
    ],
)
def test_scan_allows(body):
    source = f"pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {{\n    {body}\n    Ok(x)\n}}\n"
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


def test_hypothesis_check_rejects_before_any_build():
    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source="pub fn udf() { unsafe {} }"),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert not ok and "forbidden construct" in err


def test_hypothesis_check_needs_spark_for_rust():
    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source=PLUS_ONE_RS),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert (ok, err) == (False, "rust verify requires Spark")


def test_hypothesis_check_refuses_rust_without_comet():
    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x\n",
        captures={},
        result=TranspileResult(kind=KIND_RUST, rust_source=PLUS_ONE_RS),
        input_types=["bigint"],
        return_type="bigint",
        spark=_FakeSpark({"spark.master": "local[1]"}),
    )
    assert not ok
    assert "rust verify setup failed" in err and "can only be verified under Comet" in err


# -- parsing a backend's answer -----------------------------------------------


def test_out_rs_is_a_rust_result(tmp_path):
    (tmp_path / "OUT.rs").write_text(PLUS_ONE_RS, encoding="utf-8")
    result = parse_sandbox(tmp_path)
    assert result.kind == KIND_RUST
    assert result.rust_source == PLUS_ONE_RS.strip()
    assert result.java_source is None and result.sql is None


def test_sql_wins_over_rust_wins_over_java(tmp_path):
    (tmp_path / "OUT.rs").write_text(PLUS_ONE_RS, encoding="utf-8")
    (tmp_path / "OUT.java").write_text("package ai_udf; public class A {}", encoding="utf-8")
    assert parse_sandbox(tmp_path).kind == KIND_RUST
    (tmp_path / "OUT.sql").write_text("_udf_param_0 + 1", encoding="utf-8")
    assert parse_sandbox(tmp_path).kind == "catalyst"


def test_out_rs_decline_is_final(tmp_path):
    (tmp_path / "OUT.rs").write_text("DECLINE\n", encoding="utf-8")
    (tmp_path / "OUT.java").write_text("package ai_udf; public class A {}", encoding="utf-8")
    with pytest.raises(BackendDecline, match="OUT.rs"):
        parse_sandbox(tmp_path)


def test_rust_on_stdout(tmp_path):
    assert parse_sandbox(tmp_path, PLUS_ONE_RS).kind == KIND_RUST
    named = parse_sandbox(tmp_path, "OUT.rs\n" + PLUS_ONE_RS)
    assert named.kind == KIND_RUST and named.rust_source.startswith("pub fn udf(")
    # Prose that merely mentions a function is still SQL-or-nothing.
    assert parse_sandbox(tmp_path, "_udf_param_0 + 1").kind == "catalyst"
    # ...and so is SQL whose string literal happens to spell the signature.
    assert parse_sandbox(tmp_path, "concat(_udf_param_0, 'pub fn udf(')").kind == "catalyst"


# -- what the model is shown --------------------------------------------------


def test_prompt_offers_rust_only_when_the_job_accepts_it(tmp_path):
    plain = write_sandbox(tmp_path / "plain", _job())
    text = (plain / "PROMPT.md").read_text(encoding="utf-8")
    assert "OUT.rs" not in text and "Rust" not in text
    assert "rust_signature" not in (plain / "types.json").read_text(encoding="utf-8")

    offered = write_sandbox(tmp_path / "rust", _job(targets=["catalyst", "java_udf", "rust"]))
    text = (offered / "PROMPT.md").read_text(encoding="utf-8")
    signature = "pub fn udf(arg0: Option<i64>) -> Result<Option<i64>, String>"
    assert "## Rust target (`OUT.rs`)" in text
    assert signature in text and "@@SIGNATURE@@" not in text
    # The Rust rules come before the job, and SQL is still the first choice.
    assert text.index("## Rust target") < text.index("## This job")
    assert "**Preferred:** Spark SQL in `OUT.sql`" in text
    assert json.loads((offered / "types.json").read_text(encoding="utf-8"))["rust_signature"] == signature


def test_prompt_never_offers_rust_for_unsupported_types(tmp_path):
    job = _job(targets=["catalyst", "java_udf", "rust"], input_types=["array<string>"])
    root = write_sandbox(tmp_path, job)
    assert "OUT.rs" not in (root / "PROMPT.md").read_text(encoding="utf-8")


def test_rust_prompt_covers_the_lookalikes():
    from ai_udf_transpile.sandbox import rust_prompt

    prompt = rust_prompt("pub fn udf() -> Result<Option<i64>, String>")
    for needle in (
        "chars().count()",
        "round_ties_even",
        "ZeroDivisionError",
        "i128",
        "overflow",
        "unsafe",
        "pure function",
        "U+001C",
        "DECLINE",
    ):
        assert needle in prompt


def test_fake_declines_a_rust_fixture_the_job_does_not_accept():
    job = _job()
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline, match="does not accept"):
            FakeBackend().run(job, sandbox)


def test_fake_serves_a_rust_fixture_when_offered():
    job = _job(targets=["catalyst", "java_udf", "rust"])
    with make_sandbox(job) as sandbox:
        result = FakeBackend().run(job, sandbox)
        assert (Path(sandbox) / "OUT.rs").read_text(encoding="utf-8") == COLLATZ_STEPS_RS
        assert parse_sandbox(Path(sandbox)).kind == KIND_RUST
    assert (result.kind, result.rust_source) == (KIND_RUST, COLLATZ_STEPS_RS)
    # Verification stamps build outputs on the result: never on the shared fixture.
    result.entry = "stamped"
    with make_sandbox(job) as sandbox:
        assert FakeBackend().run(job, sandbox).entry is None


# -- can this session use it? -------------------------------------------------


def test_unservable_until_opted_in():
    spark = _comet_session(**{conf.TARGETS: "catalyst,java_udf"})
    assert rustc.rust_unservable(spark) == f"rust is not in {conf.TARGETS}"
    assert rustc.rust_unservable(_comet_session()) is None


def test_unservable_without_a_session():
    conf.set_value(conf.TARGETS, "rust")
    assert rustc.rust_unservable(None) == "no SparkSession"


def test_unservable_without_comet():
    spark = _FakeSpark({conf.TARGETS: "rust", "spark.master": "local[1]"})
    assert "not installed" in rustc.rust_unservable(spark)
    # The jar must also be on the driver and carry the native UDF entry point.
    spark = _FakeSpark({conf.TARGETS: "rust", "spark.plugins": _COMET_PLUGIN}, jvm=None)
    assert "no native UDF support" in rustc.rust_unservable(spark)
    assert "no native UDF support" in rustc.rust_unservable(_comet_session(jvm={"native_udf": False}))


def test_unservable_when_the_extension_turned_itself_off():
    # The plugin is installed but e.g. spark.shuffle.manager disabled the extension.
    assert "extension is disabled" in rustc.rust_unservable(_comet_session(jvm={"loaded": False}))


@pytest.mark.parametrize("key", ["spark.comet.exec.enabled", "spark.comet.nativeUdf.enabled"])
def test_unservable_when_comet_exec_or_native_udfs_are_off(key):
    assert rustc.rust_unservable(_comet_session(**{key: "false"})) == f"{key} is off"


def test_extension_class_counts_as_installed():
    spark = _comet_session(
        **{"spark.plugins": "", "spark.sql.extensions": "org.apache.comet.CometSparkSessionExtensions"}
    )
    assert rustc.rust_unservable(spark) is None


def test_cluster_masters_need_the_shared_build_dir_promise():
    spark = _comet_session(**{"spark.master": "yarn"})
    assert "executors could not load the library" in rustc.rust_unservable(spark)
    spark = _comet_session(**{"spark.master": "yarn", conf.RUST_SHARED_BUILD_DIR: "true"})
    assert rustc.rust_unservable(spark) is None


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """rust_offered's toolchain probe, with cargo replaced by a recorder."""
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        monkeypatch.delenv(key, raising=False)
    conf.set_value(conf.RUST_BUILD_DIR, str(tmp_path / "build"))
    monkeypatch.setattr(rustc, "_UNBUILDABLE", {})
    monkeypatch.setattr(rustc, "_SDK_STATES", {})
    monkeypatch.setattr(rustc.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    calls = []
    outcome = {"returncode": 0}

    def fake_cargo(spark, args, cwd, target):
        calls.append(args[0])
        metadata = {
            "packages": [{"name": "comet-udf-sdk", "dependencies": [{"name": "arrow", "req": "^59.2.0"}]}]
        }
        return types.SimpleNamespace(
            returncode=outcome["returncode"], stdout=json.dumps(metadata), stderr="error: no network"
        )

    monkeypatch.setattr(rustc, "_run_cargo", fake_cargo)
    return types.SimpleNamespace(calls=calls, outcome=outcome, build=tmp_path / "build")


def test_offered_needs_supported_types_and_the_opt_in(probe):
    assert not rustc.rust_offered(_comet_session(), ["array<string>"], "bigint")
    assert not rustc.rust_offered(_comet_session(**{conf.TARGETS: "catalyst"}), ["bigint"], "bigint")
    assert probe.calls == []  # neither is worth a cargo call


def test_offered_once_cargo_resolves_the_sdk(probe):
    spark = _comet_session()  # no checkout: the SDK is the jar's release tag, from git
    assert rustc.rust_offered(spark, ["bigint"], "bigint")
    assert probe.calls == ["metadata"]
    # The arrow requirement came from what cargo resolved for the SDK.
    assert next((probe.build / "sdk").glob("*/arrow_req")).read_text().strip() == "59.2.0"
    assert rustc.rust_offered(spark, ["string"], "string")
    assert probe.calls == ["metadata"]  # one probe per SDK, not per row


def test_not_offered_without_cargo_or_an_sdk(probe, monkeypatch):
    assert not rustc.rust_offered(_comet_session(jvm={"version": "", "revision": ""}), ["bigint"], "bigint")
    monkeypatch.setattr(rustc, "_UNBUILDABLE", {})
    monkeypatch.setattr(rustc.shutil, "which", lambda binary: None)
    assert not rustc.rust_offered(_comet_session(), ["bigint"], "bigint")
    assert probe.calls == []


def test_a_broken_toolchain_is_probed_once_then_remembered(probe, monkeypatch):
    # e.g. a git SDK with no network: answering in Rust would only burn the
    # row's cooldown, so Rust is not offered -- and cargo is not re-run per row.
    spark = _comet_session()
    probe.outcome["returncode"] = 101
    assert not rustc.rust_offered(spark, ["bigint"], "bigint")
    assert not rustc.rust_offered(spark, ["bigint"], "bigint")
    assert probe.calls == ["metadata"]
    # Fixed toolchain, and the memory has expired: offered again.
    probe.outcome["returncode"] = 0
    monkeypatch.setattr(rustc, "_UNBUILDABLE_TTL", 0.0)
    assert rustc.rust_offered(spark, ["bigint"], "bigint")
    assert probe.calls == ["metadata", "metadata"]


# -- where the SDK comes from -------------------------------------------------


def _fake_checkout(root: Path, arrow: str = "59.2.0") -> Path:
    sdk = root / "native" / "comet-udf-sdk"
    (sdk / "src").mkdir(parents=True)
    (root / "native" / "Cargo.toml").write_text(
        f'[workspace]\nmembers = ["comet-udf-sdk"]\n\n[workspace.dependencies]\n'
        f'arrow = {{ version = "{arrow}", features = ["ffi"] }}\n',
        encoding="utf-8",
    )
    (sdk / "Cargo.toml").write_text(
        '[package]\nname = "comet-udf-sdk"\n\n[dependencies]\narrow = { workspace = true }\n',
        encoding="utf-8",
    )
    (sdk / "src" / "lib.rs").write_text("pub const COMET_UDF_ABI_VERSION: u32 = 1;\n", encoding="utf-8")
    return sdk


def test_sdk_from_conf_checkout_or_crate(tmp_path, monkeypatch):
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        monkeypatch.delenv(key, raising=False)
    sdk = _fake_checkout(tmp_path / "comet")
    conf.set_value(conf.RUST_COMET_SDK, str(tmp_path / "comet"))
    from_checkout = rustc.resolve_sdk()
    assert from_checkout.path == sdk.resolve()
    assert from_checkout.dependency == f'{{ path = "{sdk.resolve()}" }}'
    conf.set_value(conf.RUST_COMET_SDK, str(sdk))
    assert rustc.resolve_sdk() == from_checkout
    # Moving the checkout to another ref changes the SDK bytes, and so every name.
    (sdk / "src" / "lib.rs").write_text("pub const COMET_UDF_ABI_VERSION: u32 = 2;\n", encoding="utf-8")
    assert rustc.resolve_sdk().ident != from_checkout.ident


def test_sdk_conf_that_is_not_a_checkout_is_an_error(tmp_path):
    conf.set_value(conf.RUST_COMET_SDK, str(tmp_path))
    with pytest.raises(rustc.RustUnavailable, match="holds no comet-udf-sdk crate"):
        rustc.resolve_sdk()


def test_sdk_from_raw_toml():
    table = '{ git = "https://example.invalid/comet", rev = "deadbeef" }'
    conf.set_value(conf.RUST_COMET_SDK, table)
    spec = rustc.resolve_sdk()
    assert spec.dependency == table and spec.path is None


def test_sdk_from_env_then_jar(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_UDF_TRANSPILE_COMET_HOME", raising=False)
    sdk = _fake_checkout(tmp_path / "comet")
    monkeypatch.setenv("COMET_HOME", str(tmp_path / "comet"))
    assert rustc.resolve_sdk(_comet_session()).path == sdk.resolve()
    monkeypatch.delenv("COMET_HOME")
    release = rustc.resolve_sdk(_comet_session())
    assert release.dependency == '{ git = "https://github.com/apache/datafusion-comet", tag = "1.2.0" }'
    snapshot = rustc.resolve_sdk(_comet_session(jvm={"version": "1.3.0-SNAPSHOT", "revision": "330b29c"}))
    assert snapshot.dependency == ('{ git = "https://github.com/apache/datafusion-comet", rev = "330b29c" }')
    assert release.ident != snapshot.ident
    with pytest.raises(rustc.RustUnavailable, match="no comet-udf-sdk source"):
        rustc.resolve_sdk()


def test_arrow_requirement_comes_from_the_sdk_workspace(tmp_path):
    sdk = _fake_checkout(tmp_path / "comet", arrow="59.2.0")
    assert rustc._arrow_from_manifest(sdk) == "59.2.0"
    (sdk / "Cargo.toml").write_text(
        '[package]\nname = "comet-udf-sdk"\n\n[dependencies]\narrow = "60"\n', encoding="utf-8"
    )
    assert rustc._arrow_from_manifest(sdk) == "60"
    assert rustc._arrow_from_manifest(tmp_path / "nope") is None


def test_build_dir_default_and_override(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert rustc.build_dir() == tmp_path / "cache" / "ai_udf_transpile" / "rust"
    conf.set_value(conf.RUST_BUILD_DIR, str(tmp_path / "elsewhere"))
    assert rustc.build_dir() == tmp_path / "elsewhere"


def test_compiler_errors_drop_the_progress_chatter():
    stderr = (
        "   Compiling arrow v59.3.2\n"
        "warning: unused variable: `y`\n --> src/udf.rs:2:9\n"
        "error[E0425]: cannot find value `nope` in this scope\n --> src/udf.rs:1:78\n"
        "error: could not compile `ai_udf_rs_x` (lib) due to 1 previous error\n"
    )
    text = rustc._compiler_errors(stderr)
    assert text.startswith("error[E0425]") and "src/udf.rs:1:78" in text
    assert "Compiling" not in text and "unused variable" not in text
    assert rustc._compiler_errors("plain failure") == "plain failure"


def test_no_cargo_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(rustc.shutil, "which", lambda binary: None)
    with pytest.raises(rustc.RustUnavailable, match="cargo not found"):
        rustc._cargo()


# -- the worker and the lookup fail open --------------------------------------


class _RustBackend:
    name = "stub"

    def run(self, job, sandbox):
        return TranspileResult(kind=KIND_RUST, rust_source=PLUS_ONE_RS, model="stub-1")


def _pending(tmp_path, input_types=None) -> SqliteCatalog:
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    cat.insert_pending(
        udf_key="k",
        source_text="def f(x: int) -> int:\n    return x + 1\n",
        param_names=["x"],
        input_types=input_types or ["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "stub")
    return cat


def test_worker_refuses_rust_nobody_asked_for(tmp_path):
    cat = _pending(tmp_path)
    verified = []
    process_row(cat, cat.get("k"), _RustBackend(), spark=None, verify_fn=lambda **k: verified.append(1))
    row = cat.get("k")
    assert row.status == "failed" and not verified
    assert "rust rewrite cannot run in this session" in row.error
    assert row.impl_source == PLUS_ONE_RS  # the candidate survives for inspection


def test_worker_refuses_rust_for_types_the_contract_cannot_carry(tmp_path):
    cat = _pending(tmp_path, input_types=["array<string>"])
    process_row(cat, cat.get("k"), _RustBackend(), spark=None, verify_fn=lambda **k: (True, None))
    assert "rust target does not support (array<string>) -> bigint" in cat.get("k").error


def test_worker_never_offers_rust_without_a_session(tmp_path):
    seen = []

    class _Recorder:
        name = "stub"

        def run(self, job, sandbox):
            seen.append(list(job.targets))
            raise BackendDecline("no")

    cat = _pending(tmp_path)
    conf.set_value(conf.TARGETS, "catalyst,java_udf,rust")
    process_row(cat, cat.get("k"), _Recorder(), spark=None)
    assert seen == [["catalyst", "java_udf"]]


def _success_row() -> CacheRow:
    return CacheRow(
        udf_key="k" * 16,
        status="success",
        target_kind="rust",
        impl_source=PLUS_ONE_RS,
        input_types=["bigint"],
        return_type="bigint",
        param_names=["x"],
    )


def test_lookup_keeps_python_wherever_rust_could_fail_the_query():
    row = _success_row()
    # No session, target off, no Comet, extension off, cluster master: all Python.
    assert _reconstruct_rust(row, "bigint", None) is None
    assert _reconstruct_rust(row, "bigint", _comet_session(**{conf.TARGETS: "catalyst"})) is None
    assert _reconstruct_rust(row, "bigint", _FakeSpark({conf.TARGETS: "rust"})) is None
    assert _reconstruct_rust(row, "bigint", _comet_session(jvm={"loaded": False})) is None
    assert _reconstruct_rust(row, "bigint", _comet_session(**{"spark.master": "k8s://x"})) is None


def test_lookup_never_waits_on_cargo(tmp_path, monkeypatch):
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        monkeypatch.delenv(key, raising=False)
    conf.set_value(conf.RUST_BUILD_DIR, str(tmp_path / "build"))
    started = []
    monkeypatch.setattr(rustc, "compile_in_background", lambda *args: started.append(args[1:]))
    monkeypatch.setattr(rustc, "compile_rust", lambda *a, **k: pytest.fail("blocked on cargo"))
    # Servable session, but the library is not in this machine's cache (a row
    # from the shared table): Python now, a background build for next time.
    assert _reconstruct_rust(_success_row(), "bigint", _comet_session()) is None
    assert started == [(PLUS_ONE_RS, ["bigint"], "bigint")]


def test_lookup_rescans_a_row_before_it_becomes_native_code(tmp_path, monkeypatch):
    # A row from the shared write-back table was verified elsewhere, if at
    # all: it must pass this process's scan before cargo sees it.
    conf.set_value(conf.RUST_BUILD_DIR, str(tmp_path / "build"))
    monkeypatch.setattr(rustc, "compile_in_background", lambda *a: pytest.fail("built a rejected row"))
    row = _success_row()
    row.impl_source = (
        'pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { std::fs::read("/x"); Ok(x) }'
    )
    assert _reconstruct_rust(row, "bigint", _comet_session()) is None


def test_lookup_ignores_rows_the_contract_cannot_carry():
    row = _success_row()
    row.input_types = ["array<string>"]
    assert _reconstruct_rust(row, "bigint", _comet_session()) is None
    row = _success_row()
    row.impl_source = None
    assert _reconstruct_rust(row, "bigint", _comet_session()) is None


# -- real builds (cargo + a Comet checkout) -----------------------------------


def _sdk_checkout() -> str:
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        if rustc._sdk_dir(os.environ.get(key, "")) is not None:
            return os.environ[key]
    return ""


needs_cargo = pytest.mark.skipif(
    not (shutil.which("cargo") and _sdk_checkout()),
    reason="needs cargo and a Comet checkout ($COMET_HOME) to build against",
)


@pytest.fixture
def rust_build_dir():
    """The real cache: arrow compiles once per machine, not once per test."""
    override = os.environ.get("AI_UDF_TRANSPILE_TEST_RUST_BUILD_DIR")
    if override:
        conf.set_value(conf.RUST_BUILD_DIR, override)
    return rustc.build_dir()


@needs_cargo
def test_builds_a_library_and_reuses_it(rust_build_dir, monkeypatch):
    compiled = rustc.compile_rust(None, COLLATZ_STEPS_RS, ["bigint"], "bigint")
    assert compiled.library.is_file() and compiled.library.parent == rust_build_dir / "lib"
    assert compiled.name in compiled.library.name
    assert rustc.cached_library(None, COLLATZ_STEPS_RS, ["bigint"], "bigint") == compiled
    # A second request is a cache hit: cargo is not run again.
    monkeypatch.setattr(rustc, "_run_cargo", lambda *a, **k: pytest.fail("rebuilt a cached library"))
    assert rustc.compile_rust(None, COLLATZ_STEPS_RS, ["bigint"], "bigint") == compiled
    # The crate and its intermediates are gone; only the library is kept.
    assert not (rust_build_dir / "crates" / compiled.name).exists()


@needs_cargo
def test_same_source_under_other_types_is_another_library(rust_build_dir):
    source = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x) }\n"
    planned = rustc.cached_library(None, source, ["bigint"], "int")
    assert planned is None or planned.name != rustc.compile_rust(None, source, ["bigint"], "bigint").name


@needs_cargo
def test_compile_error_carries_rustc_output(rust_build_dir):
    source = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(|v| v + nope)) }\n"
    with pytest.raises(rustc.RustCompileError, match="cannot find value `nope`"):
        rustc.compile_rust(None, source, ["bigint"], "bigint")
    assert rustc.cached_library(None, source, ["bigint"], "bigint") is None


@needs_cargo
def test_wrong_signature_does_not_compile(rust_build_dir):
    # The glue calls udf::udf with the declared types; a rewrite that guessed
    # i32 for a bigint column is a compile error, not a silent truncation.
    source = "pub fn udf(x: Option<i32>) -> Result<Option<i64>, String> { Ok(x.map(|v| v as i64)) }\n"
    with pytest.raises(rustc.RustCompileError, match="mismatched types"):
        rustc.compile_rust(None, source, ["bigint"], "bigint")


@needs_cargo
def test_unsafe_is_a_compile_error_even_past_the_scan(rust_build_dir):
    # The scan would reject this first; the lint is the second lock.
    source = "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { unsafe { Ok(x) } }\n"
    with pytest.raises(rustc.RustCompileError, match="unsafe"):
        rustc.compile_rust(None, source, ["bigint"], "bigint")


@needs_cargo
def test_empty_source_is_rejected(rust_build_dir):
    with pytest.raises(rustc.RustCompileError, match="empty Rust source"):
        rustc.compile_rust(None, "  \n", ["bigint"], "bigint")


@pytest.mark.skipif(rustc.fcntl is None, reason="flock is POSIX-only")
def test_build_lock_excludes_other_processes(tmp_path):
    # Two drivers sharing a build directory must not build one crate at once.
    import subprocess
    import sys

    probe = (
        "import fcntl, sys\n"
        "with open(sys.argv[1], 'a+b') as h:\n"
        "    try:\n"
        "        fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    except BlockingIOError:\n"
        "        print('held')\n"
        "    else:\n"
        "        print('free')\n"
    )
    lock = tmp_path / "locks" / "ai_udf_rs_x.lock"

    def other_process_sees() -> str:
        done = subprocess.run([sys.executable, "-c", probe, str(lock)], capture_output=True, text=True)
        return done.stdout.strip()

    with rustc._locked("ai_udf_rs_x", tmp_path):
        assert other_process_sees() == "held"
    assert other_process_sees() == "free"


# -- review fixes ---------------------------------------------------------------


def _rust(body: str) -> str:
    return f"pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {{\n    {body}\n    Ok(x)\n}}\n"


@pytest.mark.parametrize(
    "body",
    [
        # A C raw string is raw: the backslash does not escape its closing quote.
        'let _a = cr"\\"; std::process::exit(0); let _b = "";',
        'let _a = cr#"\\"#; std::process::exit(0); let _b = "";',
    ],
)
def test_scan_sees_past_c_raw_strings(body):
    assert "forbidden construct" in rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=_rust(body)))


def test_scan_allows_a_variable_named_like_a_macro():
    for body in ("let env = 3; if env != 0 { }", "let asm = 1; let _ = asm!= 2;"):
        assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=_rust(body))) is None


def test_scan_rejects_recursion():
    # A Rust stack overflow aborts the process: the driver, in local mode.
    source = (
        "fn depth(n: i64) -> i64 { if n <= 0 { 0 } else { 1 + depth(n - 1) } }\n"
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(depth)) }\n"
    )
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected and "recursive (depth)" in rejected
    recursive_udf = _rust("let _ = udf(None);")
    assert "recursive (udf)" in rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=recursive_udf))


def test_scan_allows_helpers_and_same_named_methods():
    source = (
        "fn trim(s: &str) -> &str { s.trim() }\n"  # a method of the same name is not recursion
        "fn len(s: &str) -> usize { str::len(s) }\n"  # nor a path call
        "fn twice(n: i64) -> i64 { n * 2 }\n"
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {\n"
        '    let _ = (trim(" a "), len("b"));\n'
        "    Ok(x.map(twice))\n"
        "}\n"
    )
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


def test_stray_out_rs_never_shadows_java_in_a_job_without_rust(tmp_path):
    (tmp_path / "OUT.rs").write_text(PLUS_ONE_RS, encoding="utf-8")
    (tmp_path / "OUT.java").write_text("package ai_udf; public class A {}", encoding="utf-8")
    assert parse_sandbox(tmp_path, targets=["catalyst", "java_udf"]).kind == "java_udf"
    assert parse_sandbox(tmp_path, targets=["catalyst", "java_udf", "rust"]).kind == KIND_RUST
    (tmp_path / "OUT.java").unlink()
    with pytest.raises(BackendDecline, match="does not accept Rust"):
        parse_sandbox(tmp_path, targets=["catalyst", "java_udf"])
    (tmp_path / "OUT.rs").unlink()
    # Rust on stdout is SQL-or-nothing for a job without Rust (and fails verification as SQL).
    assert parse_sandbox(tmp_path, PLUS_ONE_RS, targets=["catalyst"]).kind == "catalyst"


def test_cli_backend_passes_the_job_targets(tmp_path):
    from ai_udf_transpile.backends.cli import CLAUDE_SPEC, CliBackend

    def runner(argv, cwd, **kwargs):
        (Path(cwd) / "OUT.rs").write_text(PLUS_ONE_RS, encoding="utf-8")
        (Path(cwd) / "OUT.java").write_text("package ai_udf; public class A {}", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    backend = CliBackend(CLAUDE_SPEC, runner=runner)
    (tmp_path / "PROMPT.md").write_text("p", encoding="utf-8")
    assert backend.run(_job(), tmp_path).kind == "java_udf"
    assert backend.run(_job(targets=["catalyst", "java_udf", "rust"]), tmp_path).kind == KIND_RUST


def test_unbuildable_memo_is_per_toolchain(probe, monkeypatch):
    spark = _comet_session()
    probe.outcome["returncode"] = 101
    assert not rustc.rust_offered(spark, ["bigint"], "bigint")
    assert probe.calls == ["metadata"]
    # A different SDK source is a different toolchain: probed again at once.
    probe.outcome["returncode"] = 0
    conf.set_value(conf.RUST_COMET_SDK, '{ git = "https://example.invalid/comet", tag = "1.2.0" }')
    assert rustc.rust_offered(spark, ["bigint"], "bigint")
    assert probe.calls == ["metadata", "metadata"]


def test_native_rows_switch_is_refcounted():
    from ai_udf_transpile.verify import _comet_native_rows

    class _RuntimeConf:
        def __init__(self):
            self.values = {}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def set(self, key, value):
            self.values[key] = value

        def unset(self, key):
            self.values.pop(key, None)

    spark = types.SimpleNamespace(conf=_RuntimeConf())
    key = "spark.comet.convert.rdd.enabled"
    outer = _comet_native_rows(spark, True)
    inner = _comet_native_rows(spark, True)
    outer.__enter__()
    inner.__enter__()
    assert spark.conf.get(key) == "true"
    outer.__exit__(None, None, None)  # the first one out must not restore yet
    assert spark.conf.get(key) == "true"
    inner.__exit__(None, None, None)
    assert key not in spark.conf.values  # the value from before the first one: unset
    spark.conf.set(key, "false")
    with _comet_native_rows(spark, True):
        assert spark.conf.get(key) == "true"
    assert spark.conf.get(key) == "false"


def test_no_rust_twin_rows_are_never_offered_or_accept_rust(tmp_path):
    from ai_udf_transpile.targets import NO_RUST_SUFFIX

    cat = SqliteCatalog(tmp_path / "c.sqlite")
    cat.insert_pending(
        udf_key="k",
        source_text="def f(x: int) -> int:\n    return x + 1\n",
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="4.3.0" + NO_RUST_SUFFIX,
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "stub")
    process_row(cat, cat.get("k"), _RustBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "failed" and "cannot run Rust" in row.error


# -- second CodeRabbit round ----------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        # Recursion through a function value: no bare `udf(` call in the body.
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {\n"
        "    let f = udf;\n    match x { Some(v) if v > 0 => f(Some(v - 1)), _ => Ok(x) }\n}\n",
        # A cycle of helpers.
        "fn ping(n: i64) -> i64 { if n <= 0 { 0 } else { pong(n - 1) } }\n"
        "fn pong(n: i64) -> i64 { ping(n) }\n"
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(ping)) }\n",
        # Through a function-pointer table.
        "fn step(n: i64) -> i64 {\n"
        "    let t: [fn(i64) -> i64; 1] = [step];\n"
        "    if n > 0 { t[0](n - 1) } else { 0 }\n"
        "}\n"
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(step)) }\n",
    ],
)
def test_scan_rejects_indirect_recursion(source):
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected and "recursive" in rejected


def test_scan_counts_a_local_that_shadows_its_function():
    # By name, not by scope: a scope-aware exception for shadowing locals let
    # a function value taken before an inner shadowing block through.
    source = (
        "fn count(s: &str) -> i64 { let mut count = 0; for _c in s.chars() { count += 1; } count }\n"
        "pub fn udf(x: Option<&str>) -> Result<Option<i64>, String> { Ok(x.map(count)) }\n"
    )
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert "recursive (count)" in rejected and "rename it" in rejected
    renamed = source.replace("count = 0", "n = 0").replace("{ count += 1; } count }", "{ n += 1; } n }")
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=renamed)) is None


# -- third CodeRabbit round -------------------------------------------------------


def _udf(body: str) -> str:
    return f"pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> {{ {body} }}\n"


@pytest.mark.parametrize(
    "source, reason",
    [
        # The function value is taken before an inner block shadows the name.
        (_udf("let f = udf; { let udf = 0; let _ = udf; } f(x)"), "recursive (udf)"),
        # A method calling itself through `self.` / `Self::`.
        (
            "struct S; impl S { fn go(&self, n: i64) -> i64 { if n > 0 { self.go(n - 1) } else { 0 } } }\n"
            + _udf("Ok(x.map(|v| S.go(v)))"),
            "recursive (go)",
        ),
        (
            "struct S; impl S { fn go(n: i64) -> i64 { if n > 0 { Self::go(n - 1) } else { 0 } } }\n"
            + _udf("Ok(x.map(S::go))"),
            "recursive (go)",
        ),
        # A path through the crate root.
        (_udf("if x == Some(0) { crate::udf(None) } else { Ok(x) }"), "recursive (udf)"),
        (_udf("use self::udf as again; if x == Some(0) { again(None) } else { Ok(x) }"), "recursive (udf)"),
        # A function value in a static, called from the body.
        (
            "static F: fn(Option<i64>) -> Result<Option<i64>, String> = udf;\n" + _udf("F(x)"),
            "static or const holding a function",
        ),
        # Trait dispatch names no function: `{}` calls Display::fmt on self.
        (
            "struct W(i64);\n"
            "impl std::fmt::Display for W {\n"
            '    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self) }\n'
            "}\n" + _udf("Ok(x)"),
            "implements a trait",
        ),
        # An erased closure, Any, a type holding a function: closure fixpoints.
        (_udf("let f: &dyn Fn(i64) -> i64 = &|n| n; Ok(x.map(f))"), "`dyn`"),
        (_udf("let _a: Option<&std::any::Any> = None; Ok(x)"), "std::any"),
        ("struct S(Option<fn(&S, i64) -> i64>);\n" + _udf("Ok(x)"), "type holding a function (S)"),
        ("struct S([u8; 4], fn(i64) -> i64);\n" + _udf("Ok(x)"), "type holding a function (S)"),
    ],
)
def test_scan_rejects_recursion_that_names_no_bare_call(source, reason):
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected and reason in rejected


@pytest.mark.parametrize(
    "source",
    [
        "fn n(v: i64) -> i64 { v + 1 }\n" + _udf("Ok(x.map(n))"),  # `n` is also in `fn`
        "fn a<'a>(s: &'a str) -> &'a str { s }\n" + _udf('let _ = a("x"); Ok(x)'),  # lifetime `'a`
        "fn h(v: [u8; 4]) -> u8 { v[0] }\n" + _udf("let _ = h([1; 4]); Ok(x)"),
        "struct P([u8; 4], i64);\n" + _udf("let _ = P([0; 4], 1); Ok(x)"),
        "fn digits(n: i64) -> impl Iterator<Item = u32> {\n"
        "    n.to_string().chars().filter_map(|c| c.to_digit(10)).collect::<Vec<_>>().into_iter()\n"
        "}\n" + _udf("Ok(x.map(|v| digits(v).count() as i64))"),
        "struct Acc { n: i64 }\n"
        "impl Acc { fn add(&mut self, v: i64) { self.n += v; } fn total(&self) -> i64 { self.n } }\n"
        + _udf("let mut a = Acc { n: 0 }; a.add(1); Ok(x.map(|v| v + a.total()))"),
        "fn apply(f: impl Fn(i64) -> i64, v: i64) -> i64 { f(v) }\n"
        + _udf("Ok(x.map(|v| apply(|n| n * 2, v)))"),
    ],
)
def test_scan_allows_plain_functions_methods_and_closures(source):
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


def test_scan_allows_a_call_chain_without_a_cycle():
    source = (
        "fn a(n: i64) -> i64 { b(n) + 1 }\n"
        "fn b(n: i64) -> i64 { c(n) * 2 }\n"
        "fn c(n: i64) -> i64 { n.abs().max(1) }\n"  # `.max` is a method, not a fn named max
        "pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x.map(a)) }\n"
    )
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


def test_local_sdk_identity_includes_the_workspace_manifest(tmp_path, monkeypatch):
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rustc, "_PATH_SDKS", {})
    _fake_checkout(tmp_path / "comet", arrow="59.2.0")
    conf.set_value(conf.RUST_COMET_SDK, str(tmp_path / "comet"))
    before = rustc.resolve_sdk().ident
    # The SDK crate is untouched; only the workspace's arrow pin moves.
    workspace = tmp_path / "comet" / "native" / "Cargo.toml"
    workspace.write_text(workspace.read_text().replace("59.2.0", "60.0.0"), encoding="utf-8")
    assert rustc.resolve_sdk().ident != before


def test_local_sdk_identity_includes_its_build_script(tmp_path, monkeypatch):
    """A build.rs change (nothing in src/ or the manifests) is a new SDK, not a cache hit."""
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rustc, "_PATH_SDKS", {})
    _fake_checkout(tmp_path / "comet", arrow="59.2.0")
    conf.set_value(conf.RUST_COMET_SDK, str(tmp_path / "comet"))
    crate = rustc.resolve_sdk().path
    (crate / "build.rs").write_text("fn main() {}\n", encoding="utf-8")
    before = rustc.resolve_sdk().ident
    (crate / "build.rs").write_text('fn main() { println!("cargo:rustc-cfg=changed"); }\n', encoding="utf-8")
    assert rustc.resolve_sdk().ident != before
    # Build output under target/ is not part of the SDK's identity.
    (crate / "target" / "debug").mkdir(parents=True)
    after = rustc.resolve_sdk().ident
    (crate / "target" / "debug" / "out.rs").write_text("// generated\n", encoding="utf-8")
    assert rustc.resolve_sdk().ident == after


def test_native_rows_state_is_per_session():
    from ai_udf_transpile.verify import _comet_native_rows

    class _RuntimeConf:
        def __init__(self, prior=None):
            self.values = {} if prior is None else {"spark.comet.convert.rdd.enabled": prior}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def set(self, key, value):
            self.values[key] = value

        def unset(self, key):
            self.values.pop(key, None)

    key = "spark.comet.convert.rdd.enabled"
    one = types.SimpleNamespace(conf=_RuntimeConf())
    two = types.SimpleNamespace(conf=_RuntimeConf(prior="false"))
    a, b = _comet_native_rows(one, True), _comet_native_rows(two, True)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    assert key not in one.conf.values  # session one restored on its own exit
    assert two.conf.get(key) == "true"  # session two still inside its scope
    b.__exit__(None, None, None)
    assert two.conf.get(key) == "false"  # its own prior, not session one's


# -- adversarial round on the by-name recursion check ------------------------------
# Each bypass below compiled with rustc and overflowed its stack before the fix;
# each false positive compiled, was not recursive, and was refused.


@pytest.mark.parametrize(
    "source, reason",
    [
        # A trait impl whose header holds a `;` (`.into()` reaches `from`).
        (
            "pub struct D(pub i64);\n"
            "impl From<[i64; 2]> for D {\n"
            "    fn from(a: [i64; 2]) -> Self {\n"
            "        if a[0] <= 0 { D(a[1]) } else { let d: D = [a[0] - 1, a[1]].into(); d }\n"
            "    }\n"
            "}\n" + _udf("Ok(x.map(|n| D::from([n, 0]).0))"),
            "implements a trait",
        ),
        # A call right after `..` (a range or slice bound) or `:` (a field init).
        (
            "fn depth(v: i64) -> i64 { if v <= 0 { 0 } else { let r = 0..depth(v - 1); r.end } }\n"
            + _udf("Ok(x.map(depth))"),
            "recursive (depth)",
        ),
        (
            "pub struct H<F> { f: F }\n"
            + _udf("let h = H { f:udf }; if x == Some(0) { (h.f)(None) } else { Ok(x) }"),
            "recursive (udf)",
        ),
        # A path through a `use .. as` alias of the crate.
        ("use crate as k;\n" + _udf("if x == Some(0) { k::udf(None) } else { Ok(x) }"), "recursive (udf)"),
        # A const-generic block in an impl header hid the impl, so `walk` was no method.
        (
            "pub struct W<const N: usize>;\n"
            "impl W<{ 1 + 1 }> {\n"
            "    fn walk(&self, n: i64) -> i64 { if n <= 0 { 0 } else { self.walk(n - 1) } }\n"
            "}\n" + _udf("Ok(x.map(|n| W::<2>.walk(n)))"),
            "recursive (walk)",
        ),
        # A function stored at run time in a static, called back from itself.
        (
            "use std::sync::OnceLock;\n"
            "static STEP: OnceLock<fn(i64) -> i64> = OnceLock::new();\n"
            "fn step(n: i64) -> i64 { if n <= 0 { 0 } else { STEP.get().unwrap()(n - 1) } }\n"
            + _udf("let _ = STEP.set(step); Ok(x.map(step))"),
            "static or const holding a function",
        ),
        # A non-ASCII name, which the ASCII name patterns would never see.
        (
            "fn ñ(n: i64) -> i64 { if n <= 0 { 0 } else { ñ(n - 1) } }\n" + _udf("Ok(x.map(ñ))"),
            "non-ASCII identifier",
        ),
        # A generic struct field holding the function, called through the method.
        (
            "struct W<F> { step: F }\n"
            "impl<F: Fn(u64) -> u64> W<F> { fn step(&self, n: u64) -> u64 { (self.step)(n) } }\n"
            "fn helper(n: u64) -> u64 { if n == 0 { 0 } else { W { step: helper }.step(n - 1) } }\n"
            + _udf("Ok(x.map(|v| helper(v as u64) as i64))"),
            "recursive (helper)",
        ),
        # A std-typed field does not hide a user type that a chain argument brings in.
        (
            "struct S { o: Option<i64> }\n"
            "struct T;\n"
            "impl T {\n"
            "    fn go(&self, n: i64) -> i64 {\n"
            "        if n <= 0 { 0 } else { S { o: None }.o.and(Some(T)).unwrap().go(n - 1) }\n"
            "    }\n"
            "}\n" + _udf("Ok(x.map(|n| T.go(n)))"),
            "recursive (go)",
        ),
        # Second round. A local type named like a std one is not std.
        (
            "struct Range { lo: i64 }\nstruct Span { r: Range }\n"
            "impl Span { fn width(&self, n: i64) -> i64 { if n <= 0 { 0 } else { self.r.shrink(n - 1) } } }\n"
            "impl Range { fn shrink(&self, n: i64) -> i64 { Span { r: Range { lo: self.lo } }.width(n) } }\n"
            + _udf("Ok(x.map(|v| Span { r: Range { lo: 0 } }.width(v)))"),
            "recursive (",
        ),
        # `1 << 2` in an array length is a shift, not two generic openers.
        (
            "fn walk(n: i64, pad: [u8; 1 << 2]) -> i64 { if n <= 0 { 0 } else { walk(n - 1, pad) } }\n"
            + _udf("Ok(x.map(|v| walk(v, [0; 4])))"),
            "recursive (walk)",
        ),
        (
            "struct Buf<T>(T);\n"
            "impl Buf<[u8; 1 << 4]> {\n"
            "    fn fill(&self, n: i64) -> i64 { if n <= 0 { 0 } else { self.fill(n - 1) } }\n"
            "}\n" + _udf("Ok(x.map(|v| Buf([0u8; 16]).fill(v)))"),
            "recursive (fill)",
        ),
        # No function recurses, but dropping a long chain of a recursive type does.
        (
            "struct Node { next: Option<Box<Node>> }\n"
            + _udf(
                "let mut h: Option<Box<Node>> = None;\n"
                "for _ in 0..x.unwrap_or(0) { h = Some(Box::new(Node { next: h })); } Ok(x)"
            ),
            "recursive type (Node)",
        ),
        (
            "enum A { Leaf, More(Vec<B>) }\nstruct B { a: A }\n" + _udf("Ok(x)"),
            "recursive type",
        ),
        # Third round: a recursive type named through `use .. as` or `Self`.
        (
            "use self::Node as Link;\nstruct Node { next: Option<Box<Link>> }\n" + _udf("Ok(x)"),
            "recursive type (Node)",
        ),
        (
            "use self::A as A1;\nuse self::A1 as Far;\n"
            "struct A { b: Option<Box<B>> }\nstruct B { a: Option<Box<Far>> }\n" + _udf("Ok(x)"),
            "recursive type",
        ),
        ("struct Node { next: Option<Box<Self>> }\n" + _udf("Ok(x)"), "recursive type (Node)"),
        ("enum List { Cons(i64, Box<Self>), Nil }\n" + _udf("Ok(x)"), "recursive type (List)"),
        # No recursion at all: one big array on the stack.
        (_udf("let mut counts = [0u32; 1 << 20]; counts[0] = 1; Ok(x)"), "KB on the stack"),
        (_udf("let v = vec![[0u8; 30_000_000]; 1]; Ok(x.map(|n| n + v[0][0] as i64))"), "KB on the stack"),
        (
            "const N: usize = 1 << 22;\n" + _udf("let a = [0u8; N]; Ok(x.map(|n| n + a[0] as i64))"),
            "KB on the stack",
        ),
        (
            "struct Big { a: [i64; 16_384] }\n"
            + _udf("let p = [Big { a: [0; 16_384] }, Big { a: [0; 16_384] }]; Ok(x.map(|n| n + p[1].a[0]))"),
            "KB on the stack",
        ),
        (_udf("let a = [0u8; x.unwrap_or(0) as usize]; Ok(x)"), "cannot size"),
        ("pub fn udf(x: Option<i64>) -> Result<Option<i64>, String> { Ok(x) }\n" + " " * 20_001, "too long"),
    ],
)
def test_scan_rejects_adversarial_recursion(source, reason):
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected and reason in rejected


@pytest.mark.parametrize(
    "source",
    [
        # A constructor calling std constructors of the same name.
        "struct P { out: Vec<char>, s: String }\n"
        "impl P { fn new() -> Self { P { out: Vec::new(), s: String::new() } } }\n"
        + _udf("let p = P::new(); Ok(x.map(|v| v + p.out.len() as i64 + p.s.len() as i64))"),
        # Wrappers that forward to the std method of the same name on a std field,
        # in a struct with only a lifetime parameter, through a no-argument chain.
        "struct Stack { items: Vec<char> }\n"
        "impl Stack {\n"
        "    fn push(&mut self, c: char) { self.items.push(c); }\n"
        "    fn pop(&mut self) -> Option<char> { self.items.pop() }\n"
        "    fn len(&self) -> usize { self.items.len() }\n"
        "    fn max(&self) -> Option<char> { self.items.iter().copied().max() }\n"
        "}\n"
        "struct Cursor<'a> { chars: std::iter::Peekable<std::str::Chars<'a>> }\n"
        "impl<'a> Cursor<'a> { fn peek(&mut self) -> Option<char> { self.chars.peek().copied() } }\n"
        + _udf(
            "let mut s = Stack { items: Vec::new() }; s.push('a'); let _ = (s.pop(), s.max());\n"
            'let mut c = Cursor { chars: "ab".chars().peekable() }; let _ = c.peek();\n'
            "Ok(x.map(|v| v + s.len() as i64))"
        ),
        # A getter named like its field; a field named like a free function.
        "struct Tally { vowels: i64 }\nimpl Tally { fn vowels(&self) -> i64 { self.vowels } }\n"
        "struct Acc { total: i64 }\nfn total(xs: &[i64]) -> i64 { xs.iter().sum() }\n"
        + _udf(
            "let t = Tally { vowels: 1 }; let a = Acc { total: total(&[1, 2]) };\n"
            "Ok(x.map(|v| v + t.vowels() + a.total))"
        ),
        # A const fn evaluated in a const initializer (a compile-time table).
        "const fn table() -> [u32; 4] {\n"
        "    let mut t = [0u32; 4]; let mut i = 0; while i < 4 { t[i] = i as u32 * 3; i += 1; } t\n"
        "}\n"
        "const TABLE: [u32; 4] = table();\n" + _udf("Ok(x.map(|v| v + TABLE[(v & 3) as usize] as i64))"),
        # Heap buffers and small tables are fine; only the stack is bounded.
        "const TABLE_LEN: usize = 256;\nstatic CRC: [u32; TABLE_LEN] = [0; TABLE_LEN];\n"
        + _udf(
            "let v = vec![0u8; 30_000_000]; let small = [[0i64; 64]; 64]; let bytes = [0u8; 40_000];\n"
            "Ok(x.map(|n| n + v[0] as i64 + small[0][0] + bytes[0] as i64 + CRC[0] as i64))"
        ),
        # An enum variant named `Any`; `use std::cmp::max` beside a method `max`.
        "enum Tok { Any, Lit(char) }\n"
        "use std::cmp::max;\n"
        "struct R;\nimpl R { fn max(a: i64, b: i64) -> i64 { if a > b { a } else { b } } }\n"
        + _udf(
            "let t = Tok::Any; let n = match t { Tok::Any => 1, Tok::Lit(_) => 2 };\n"
            "Ok(x.map(|v| max(R::max(v, n), 0)))"
        ),
    ],
)
def test_scan_allows_std_named_methods_and_compile_time_calls(source):
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None


def test_scan_corpus_of_realistic_rewrites_compiles_and_passes(tmp_path):
    """Every accepted source above is real Rust: compile them when rustc is here."""
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc not on PATH")
    import subprocess

    params = test_scan_allows_std_named_methods_and_compile_time_calls.pytestmark[0].args[1]
    for i, source in enumerate(params):
        path = tmp_path / f"c{i}.rs"
        path.write_text("#![allow(dead_code, unused)]\n" + source)
        out = subprocess.run(
            [
                rustc,
                "--edition",
                "2021",
                "--crate-type",
                "lib",
                "-o",
                str(tmp_path / f"c{i}.rlib"),
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, (source, out.stderr)


# -- verification deadline -------------------------------------------------------


class _FakeSc:
    def __init__(self):
        self.props = {"spark.jobGroup.id": "outer"}
        self.cancelled = []

    def getLocalProperty(self, key):
        return self.props.get(key)

    def setLocalProperty(self, key, value):
        if value is None:
            self.props.pop(key, None)
        else:
            self.props[key] = value

    def setJobGroup(self, group, description, interruptOnCancel=False):
        self.props.update(
            {"spark.jobGroup.id": group, "spark.job.interruptOnCancel": str(interruptOnCancel).lower()}
        )

    def cancelJobGroup(self, group):
        self.cancelled.append(group)


def test_query_deadline_cancels_a_query_that_never_ends(monkeypatch):
    """A candidate whose loop never terminates fails verification instead of hanging it."""
    import time

    from ai_udf_transpile import verify

    monkeypatch.setitem(conf._RUNTIME, conf.VERIFY_QUERY_TIMEOUT, "0.2")
    sc = _FakeSc()
    spark = types.SimpleNamespace(sparkContext=sc)
    start = time.monotonic()
    with pytest.raises(verify.VerifyQueryTimeout, match="longer than 0.2s"):
        with verify._query_deadline(spark):
            group = sc.props["spark.jobGroup.id"]
            while not sc.cancelled:  # the "collect()" of a spinning candidate
                time.sleep(0.01)
            raise RuntimeError("Job group cancelled")
    assert time.monotonic() - start < 5
    assert sc.cancelled == [group]
    assert sc.props == {"spark.jobGroup.id": "outer"}  # the caller's job group is back


def test_query_deadline_leaves_a_fast_query_and_its_errors_alone(monkeypatch):
    from ai_udf_transpile import verify

    monkeypatch.setitem(conf._RUNTIME, conf.VERIFY_QUERY_TIMEOUT, "30")
    sc = _FakeSc()
    spark = types.SimpleNamespace(sparkContext=sc)
    with verify._query_deadline(spark):
        pass
    with pytest.raises(ZeroDivisionError):
        with verify._query_deadline(spark):
            raise ZeroDivisionError
    assert sc.cancelled == [] and sc.props == {"spark.jobGroup.id": "outer"}


def test_build_dir_is_absolute(monkeypatch, tmp_path):
    """cargo resolves a relative CARGO_TARGET_DIR inside the generated crate."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(conf._RUNTIME, conf.RUST_BUILD_DIR, "cache/rust")
    assert rustc.build_dir() == tmp_path / "cache" / "rust"


@pytest.mark.parametrize(
    "length, size",
    [("1 << 20", 1 << 20), ("2 * (3 + 4)", 14), ("1_000", 1000), ("N * 2", 32), ("(((1)))", 1)],
)
def test_array_lengths_are_sized_without_eval(length, size):
    from ai_udf_transpile.verify import _rust_const_value

    assert _rust_const_value(length, {"N": "16"}) == size


@pytest.mark.parametrize(
    "length", ["9**9**9", "1 << 99999999", "10 / 0", "-1", "1e9", "1 << 64", "M", "f(3)"]
)
def test_unsizable_array_lengths_are_refused_at_once(length):
    """No eval: `9**9**9` or a huge shift would stall the worker."""
    import time

    from ai_udf_transpile.verify import _rust_const_value

    start = time.monotonic()
    assert _rust_const_value(length, {}) == float("inf")
    assert time.monotonic() - start < 1


# -- review of the requireApproval x Rust integration ------------------------------


@pytest.mark.spark  # the transpiler hook imports pyspark (no JVM started)
def test_a_session_that_cannot_run_rust_never_parks_the_rust_row(tmp_path):
    """requireApproval: lookup parks a gated row. A session that never serves the Rust row
    must route to its twin before lookup, not park the row and wait on a review."""
    import ast

    from ai_udf_transpile import transpiler as T
    from ai_udf_transpile.backends.fake import BACKWARDS_RS
    from ai_udf_transpile.catalog.sqlite import SqliteCatalog

    cat = SqliteCatalog(tmp_path / "c.sqlite")
    prior_catalog, prior_session = T.get_catalog(), T._SESSION_REF
    T._SESSION_REF = None  # spark=None: this session cannot run Rust
    T.set_catalog(cat)
    try:
        tree = ast.parse("def backwards(name: str) -> str:\n    return name[::-1]\n")

        def call():
            return T.get_transpiler_class()()._transpile_from_ast(
                None, tree, tree.body[0], ["name"], "string"
            )

        call()  # MISS: the base row is queued
        (base,) = [r[0] for r in cat._conn.execute("SELECT udf_key FROM cache")]
        assert cat.claim(base, "fake")
        cat.mark_success(base, TranspileResult(kind=KIND_RUST, rust_source=BACKWARDS_RS), "fake")
        conf.set_value(conf.REQUIRE_APPROVAL, "all")
        call()
        assert cat.get(base).status == "success"  # untouched, not parked
        rows = {r[0]: r[1] for r in cat._conn.execute("SELECT spark_version, status FROM cache")}
        assert any(v.endswith("+no-rust") and s == "pending" for v, s in rows.items())  # the twin is queued
        assert cat.awaiting_approval() == []
    finally:
        conf.set_value(conf.REQUIRE_APPROVAL, conf.DEFAULTS.get(conf.REQUIRE_APPROVAL, "none"))
        T.set_catalog(prior_catalog)
        T._SESSION_REF = prior_session


@pytest.mark.spark  # the transpiler hook imports pyspark (no JVM started)
def test_no_rust_routing_forces_the_twin_key(tmp_path):
    """The worker's smoke test of a twin row must reconstruct that row, even where Rust runs."""
    import ast

    from ai_udf_transpile import transpiler as T
    from ai_udf_transpile.catalog.sqlite import SqliteCatalog

    cat = SqliteCatalog(tmp_path / "c.sqlite")
    prior_catalog, prior_session = T.get_catalog(), T._SESSION_REF
    T._SESSION_REF = None
    T.set_catalog(cat)
    try:
        tree = ast.parse("def backwards(name: str) -> str:\n    return name[::-1]\n")

        def call():
            return T.get_transpiler_class()()._transpile_from_ast(
                None, tree, tree.body[0], ["name"], "string"
            )

        with T.no_rust_routing():
            call()  # MISS under the twin key, with no base row at all
        (version,) = [r[0] for r in cat._conn.execute("SELECT spark_version FROM cache")]
        assert version.endswith("+no-rust")
    finally:
        T.set_catalog(prior_catalog)
        T._SESSION_REF = prior_session


@pytest.mark.parametrize(
    "body, reason",
    [
        # An unsizable element times zero is NaN, which compared false and hid the next array.
        ("let a = [[0u8; M]; 0]; let b = [0u8; 30_000_000]; Ok(x)", "cannot size"),
        ("let a = [0u8; " + "(" * 5000 + "1" + ")" * 5000 + "]; Ok(x)", "cannot size"),
    ],
)
def test_stack_budget_fails_closed(body, reason):
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=_udf(body)))
    assert rejected and reason in rejected


def test_const_lengths_are_evaluated_once():
    """`const C8 = C7 + C7 + ...` chains: exponential without memoizing, now instant and exact."""
    import time

    lines = ["const C0: usize = 1;"] + [
        f"const C{i}: usize = " + "+".join([f"C{i - 1}"] * 4) + ";" for i in range(1, 9)
    ]
    source = "\n".join(lines) + "\n" + _udf("let a = [0u8; C8]; Ok(x.map(|v| v + a.len() as i64))")
    start = time.monotonic()
    assert rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source)) is None  # 4**8 bytes fits
    big = source.replace("let a = [0u8; C8];", "let a = [0u64; C8];")  # 512 KB does not
    assert "KB on the stack" in rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=big))
    assert time.monotonic() - start < 2


def test_deeply_nested_source_gets_a_verdict_not_an_exception():
    source = _udf("Ok(x)").replace("Ok(x)", "let a = " + "vec![" * 3000 + "0" + "]" * 3000 + "; Ok(x)")
    rejected = rejected_rewrite(TranspileResult(kind=KIND_RUST, rust_source=source))
    assert rejected is None or isinstance(rejected, str)
