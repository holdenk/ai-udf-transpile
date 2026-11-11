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
        'let _s = format!("{}", 1); let _v: Vec<char> = "ab".chars().rev().collect();',
        "fn helper<'a>(s: &'a str) -> &'a str { s } let _ = helper(\"x\");",
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
