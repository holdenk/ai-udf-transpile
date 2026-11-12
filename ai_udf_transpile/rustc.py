# SPDX-License-Identifier: Apache-2.0
"""Build a Rust row function into a Comet native UDF and register it.

The ``rust`` target runs inside Apache DataFusion Comet's native execution:
Comet loads a ``cdylib`` built against its ``comet-udf-sdk`` and calls it on
Arrow arrays, with no JVM or Python hop per row (Comet 1.2+,
``org.apache.comet.udf.CometNativeUDF``; experimental upstream).

A rewrite is NOT a whole Comet UDF. It is one row function::

    pub fn udf(arg0: Option<i64>, arg1: Option<&str>) -> Result<Option<String>, String>

``None`` is NULL, ``Err`` fails the query the way a Python raise would. The
Arrow glue around it (downcasts, the row loop, the output builder, the
exported name) is generated here from the catalog's declared types, so a
model never touches the arrow-rs API and cannot get the types wrong.

Two properties shape everything else in this module:

* **Spark cannot evaluate the call.** Where Comet does not take the operator
  holding it, the query fails (``CometUdfNotEvaluatedException``) instead of
  falling back. So the target is opt-in (``targets`` conf), is only offered
  and served while Comet is live in the session, and verification runs the
  candidate through Comet for real.
* **The library must exist at the same path on every executor.** Builds land
  in a content-addressed cache directory; they are served on ``local``
  masters, or anywhere once ``rust.sharedBuildDir`` says that directory is
  cluster-visible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.targets import KIND_RUST

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

logger = logging.getLogger(__name__)

# Bump when the generated glue changes: the name (and so the cache) keys on it.
_WRAPPER_VERSION = "1"
_NAME_PREFIX = "ai_udf_rs_"
_COMET_GIT = "https://github.com/apache/datafusion-comet"
_SDK_CRATE = "comet-udf-sdk"
_COMET_PLUGIN = "org.apache.spark.CometPlugin"
_COMET_EXTENSION = "org.apache.comet.CometSparkSessionExtensions"


class RustCompileError(Exception):
    """The Rust row function could not be built into a Comet UDF library."""


class RustUnavailable(Exception):
    """This session cannot build or run a Rust UDF (no cargo, no SDK, no Comet)."""


@dataclass(frozen=True)
class _RustType:
    arg: str  # what the row function receives, inside Option<>
    ret: str  # what the row function returns, inside Option<>
    arrow: str  # arrow DataType variant
    sql: str  # Spark SQL type for the argument CAST
    json: str  # Spark's JSON name for the type (what parseDataType reads)
    sized: bool = True  # the builder takes a single capacity

    @property
    def datatype(self) -> str:
        return f"DataType::{self.arrow}"

    @property
    def array(self) -> str:
        return f"{_ARRAY_STEMS.get(self.arrow, self.arrow)}Array"

    @property
    def builder(self) -> str:
        return f"{_ARRAY_STEMS.get(self.arrow, self.arrow)}Builder"


# arrow-rs names the string array after the Rust type, not the Arrow one.
_ARRAY_STEMS = {"Utf8": "String"}

_I64 = _RustType("i64", "i64", "Int64", "BIGINT", "long")
_I32 = _RustType("i32", "i32", "Int32", "INT", "integer")
_I16 = _RustType("i16", "i16", "Int16", "SMALLINT", "short")
_I8 = _RustType("i8", "i8", "Int8", "TINYINT", "byte")
_BOOL = _RustType("bool", "bool", "Boolean", "BOOLEAN", "boolean")

# Spark simple type -> the row function's view of it. Atomics only: the
# types Comet's native UDF boundary documents and the verifier can generate.
RUST_TYPES: dict[str, _RustType] = {
    "bigint": _I64,
    "long": _I64,
    "int": _I32,
    "integer": _I32,
    "smallint": _I16,
    "short": _I16,
    "tinyint": _I8,
    "byte": _I8,
    "double": _RustType("f64", "f64", "Float64", "DOUBLE", "double"),
    "float": _RustType("f32", "f32", "Float32", "FLOAT", "float"),
    "boolean": _BOOL,
    "bool": _BOOL,
    "string": _RustType("&str", "String", "Utf8", "STRING", "string", sized=False),
    "binary": _RustType("&[u8]", "Vec<u8>", "Binary", "BINARY", "binary", sized=False),
}


def _spec(simple: str) -> Optional[_RustType]:
    return RUST_TYPES.get((simple or "").strip().lower())


def supports_types(input_types: list[str], return_type: str) -> bool:
    """Whether the row-function contract can carry this signature."""
    return _spec(return_type) is not None and all(_spec(t) is not None for t in input_types)


def rust_signature(input_types: list[str], return_type: str) -> str:
    """The exact ``pub fn udf`` signature a rewrite must define."""
    if not supports_types(input_types, return_type):
        raise RustCompileError(f"rust target does not support ({', '.join(input_types)}) -> {return_type}")
    args = ", ".join(f"arg{i}: Option<{_spec(t).arg}>" for i, t in enumerate(input_types))
    return f"pub fn udf({args}) -> Result<Option<{_spec(return_type).ret}>, String>"


def call_expr(name: str, input_types: list[str]) -> str:
    """SQL calling the registered UDF on the ``_udf_param_N`` placeholders.

    Comet's analyzer check inserts no casts: an ``int`` column passed to a
    ``bigint`` parameter fails analysis. The CAST makes the call total over
    whatever the caller passes (a no-op the optimizer drops when it matches).
    """
    args = ", ".join(f"CAST(_udf_param_{i} AS {_spec(t).sql})" for i, t in enumerate(input_types))
    return f"{name}({args})"


# -- the generated crate ------------------------------------------------------


def render_lib(name: str, input_types: list[str], return_type: str) -> str:
    """``src/lib.rs``: the Arrow glue around ``udf::udf``."""
    specs = [_spec(t) for t in input_types]
    out = _spec(return_type)
    if out is None or any(s is None for s in specs):
        raise RustCompileError(f"rust target does not support ({', '.join(input_types)}) -> {return_type}")
    arrays = sorted({"Array", "ArrayRef", out.builder, *(s.array for s in specs)})
    lines = [
        f"// Generated by ai_udf_transpile (wrapper v{_WRAPPER_VERSION}). The row function is udf.rs.",
        "use std::sync::Arc;",
        "",
        f"use arrow::array::{{{', '.join(arrays)}}};",
        "use arrow::datatypes::{DataType, Field};",
        "use comet_udf_sdk::c_abi::CometCScalarUdf;",
        "use comet_udf_sdk::comet_c_udf_export;",
        "",
        "#[forbid(unsafe_code)]",
        "mod udf;",
        "",
        f'const NAME: &str = "{name}";',
        f"const ARITY: usize = {len(specs)};",
        "",
        "#[derive(Default)]",
        "pub struct AiUdf;",
        "",
        "#[allow(dead_code)]",
        "fn typed(args: &[ArrayRef], i: usize, want: &DataType) -> Result<ArrayRef, String> {",
        "    let a = &args[i];",
        "    if a.data_type() == want {",
        "        return Ok(Arc::clone(a));",
        "    }",
        '    arrow::compute::cast(a, want).map_err(|e| format!("{NAME}: argument {i}: {e}"))',
        "}",
        "",
        "// A length-1 input is a broadcast scalar.",
        "#[allow(dead_code)]",
        "#[inline(always)]",
        "fn at(len: usize, row: usize) -> usize {",
        "    if len == 1 {",
        "        0",
        "    } else {",
        "        row",
        "    }",
        "}",
        "",
        "impl CometCScalarUdf for AiUdf {",
        "    fn name(&self) -> &str {",
        "        NAME",
        "    }",
        "",
        "    fn return_field(&self, args: &[Field]) -> Result<Field, String> {",
        "        if args.len() != ARITY {",
        '            return Err(format!("{NAME} expects {ARITY} argument(s), got {}", args.len()));',
        "        }",
        f"        Ok(Field::new(NAME, {out.datatype}, true))",
        "    }",
        "",
        "    fn invoke(&self, args: &[ArrayRef], n_rows: usize) -> Result<ArrayRef, String> {",
        "        if args.len() != ARITY {",
        '            return Err(format!("{NAME} expects {ARITY} argument(s), got {}", args.len()));',
        "        }",
    ]
    for i, s in enumerate(specs):
        lines += [
            f"        let c{i} = typed(args, {i}, &{s.datatype})?;",
            f"        let c{i} = c{i}",
            "            .as_any()",
            f"            .downcast_ref::<{s.array}>()",
            f'            .ok_or_else(|| format!("{{NAME}}: argument {i} is not {s.array}"))?;',
        ]
    builder = f"{out.builder}::with_capacity(n_rows)" if out.sized else f"{out.builder}::new()"
    lines += [f"        let mut out = {builder};", "        for row in 0..n_rows {"]
    for i in range(len(specs)):
        lines += [
            f"            let i{i} = at(c{i}.len(), row);",
            f"            let v{i} = if c{i}.is_null(i{i}) {{ None }} else {{ Some(c{i}.value(i{i})) }};",
        ]
    call = ", ".join(f"v{i}" for i in range(len(specs)))
    lines += [
        f'            match udf::udf({call}).map_err(|e| format!("{{NAME}}: {{e}}"))? {{',
        "                Some(v) => out.append_value(v),",
        "                None => out.append_null(),",
        "            }",
        "        }",
        "        Ok(Arc::new(out.finish()))",
        "    }",
        "}",
        "",
        "comet_c_udf_export!(AiUdf);",
        "",
    ]
    return "\n".join(lines)


def render_manifest(name: str, sdk_dependency: str, arrow_req: str) -> str:
    """``Cargo.toml`` for one UDF crate.

    Every crate shares one target directory, so the profile must be the same
    for all of them or arrow would rebuild per UDF: the overflow checks are a
    per-package override, which leaves the dependencies' flags alone.
    """
    return "\n".join(
        [
            "# Generated by ai_udf_transpile.",
            "[package]",
            f'name = "{name}"',
            'version = "0.0.0"',
            'edition = "2021"',
            "publish = false",
            "",
            "[lib]",
            'crate-type = ["cdylib"]',
            "",
            "[dependencies]",
            f'arrow = "{arrow_req}"',
            f"{_SDK_CRATE} = {sdk_dependency}",
            "",
            "# panic = unwind (the default) is what the SDK's panic guards need.",
            "[profile.release]",
            "opt-level = 3",
            "debug = false",
            "",
            "# Python ints do not wrap: an overflow is a query error, never a wrong value.",
            f'[profile.release.package."{name}"]',
            "overflow-checks = true",
            "",
            "# Not a member of any enclosing workspace.",
            "[workspace]",
            "",
        ]
    )


# -- locating the SDK ---------------------------------------------------------


@dataclass(frozen=True)
class SdkSpec:
    dependency: str  # TOML inline table for the comet-udf-sdk dependency
    ident: str  # stable identity: part of every UDF name and cache key
    path: Optional[Path] = None  # local crate directory, when not a git source


def _toml_str(value: str) -> str:
    return json.dumps(value)


def _sdk_dir(candidate: str) -> Optional[Path]:
    """The comet-udf-sdk crate under ``candidate`` (a checkout, or the crate itself)."""
    if not candidate:
        return None
    root = Path(candidate).expanduser()
    for crate in (root / "native" / _SDK_CRATE, root):
        manifest = crate / "Cargo.toml"
        if not manifest.is_file():
            continue
        try:
            package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
        except Exception:
            continue
        if package.get("name") == _SDK_CRATE:
            return crate.resolve()
    return None


_PATH_SDKS: dict[tuple, SdkSpec] = {}


def _workspace_manifest(crate: Path) -> Optional[Path]:
    """The nearest ancestor Cargo.toml with a [workspace] table, if any."""
    for parent in crate.parents:
        candidate = parent / "Cargo.toml"
        if not candidate.is_file():
            continue
        try:
            if "workspace" in tomllib.loads(candidate.read_text(encoding="utf-8")):
                return candidate
        except Exception:
            return None
    return None


def _path_sdk(crate: Path) -> SdkSpec:
    """A local SDK crate, identified by its bytes (a checkout can change ref).

    Every lookup hit resolves the SDK, so the hash is memoized on the files'
    stat signature instead of re-read per UDF definition.
    """

    # Every file of the crate, not only src/**/*.rs: a build script (build.rs,
    # or the path `build = ".."` names) and whatever it reads there change what
    # the SDK compiles to. Build output (target/) and hidden directories are
    # not the SDK. Inputs outside the crate (other files, environment) are not
    # tracked: after changing those, clear rust.buildDir.
    def _sdk_file(path: Path) -> bool:
        parts = path.relative_to(crate).parts
        return path.is_file() and "target" not in parts and not any(p.startswith(".") for p in parts)

    files_ = sorted(p for p in crate.rglob("*") if _sdk_file(p))
    if crate / "Cargo.toml" not in files_:
        files_.insert(0, crate / "Cargo.toml")
    # The SDK takes its arrow version from the workspace manifest
    # (`arrow = { workspace = true }`): an in-place bump there is a new SDK.
    workspace = _workspace_manifest(crate)
    if workspace is not None:
        files_.append(workspace)
    stamp = (str(crate), *((str(item), item.stat().st_mtime_ns, item.stat().st_size) for item in files_))
    cached = _PATH_SDKS.get(stamp)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    for item in files_:
        digest.update(str(item).encode())  # the crate's Cargo.toml and the workspace's differ by path
        digest.update(item.read_bytes())
    spec = SdkSpec(
        dependency=f"{{ path = {_toml_str(str(crate))} }}",
        ident=f"path:{crate}:{digest.hexdigest()[:16]}",
        path=crate,
    )
    _PATH_SDKS[stamp] = spec
    return spec


def _jar_version(spark: Any) -> tuple[str, str]:
    """(version, revision) of the Comet jar on the driver, or ("", "")."""
    try:
        pkg = getattr(getattr(spark._jvm.org.apache.comet, "package$"), "MODULE$")
        return str(pkg.COMET_VERSION() or ""), str(pkg.COMET_REVISION() or "")
    except Exception:
        logger.debug("could not read the Comet version from the driver", exc_info=True)
        return "", ""


def resolve_sdk(spark: Any = None) -> SdkSpec:
    """Where ``comet-udf-sdk`` comes from, most explicit source first.

    1. ``rust.cometSdk``: a Comet checkout, the SDK crate directory, or a raw
       TOML inline table (``{ git = "...", tag = "1.2.0" }``).
    2. ``$AI_UDF_TRANSPILE_COMET_HOME`` / ``$COMET_HOME``: a Comet checkout
       (what ``examples/setup_comet.sh`` builds the jar from).
    3. The Comet jar on the driver: its release tag, or its exact commit for
       a snapshot build, fetched by cargo from GitHub.

    The SDK's ABI is experimental and checked at load time: upstream's rule
    is to rebuild against the SDK of the Comet you run. So the identity is
    part of every UDF name: a new SDK means new builds.
    """
    explicit = conf.get_value(conf.RUST_COMET_SDK, spark, "").strip()
    if explicit.startswith("{"):
        return SdkSpec(dependency=explicit, ident=f"toml:{explicit}")
    if explicit:
        crate = _sdk_dir(explicit)
        if crate is None:
            raise RustUnavailable(f"{conf.RUST_COMET_SDK}={explicit!r} holds no {_SDK_CRATE} crate")
        return _path_sdk(crate)
    for key in ("AI_UDF_TRANSPILE_COMET_HOME", "COMET_HOME"):
        crate = _sdk_dir(os.environ.get(key, ""))
        if crate is not None:
            return _path_sdk(crate)
    if spark is not None:
        version, revision = _jar_version(spark)
        if version and not version.endswith("-SNAPSHOT"):
            return SdkSpec(
                dependency=f"{{ git = {_toml_str(_COMET_GIT)}, tag = {_toml_str(version)} }}",
                ident=f"git:tag:{version}",
            )
        if revision:
            return SdkSpec(
                dependency=f"{{ git = {_toml_str(_COMET_GIT)}, rev = {_toml_str(revision)} }}",
                ident=f"git:rev:{revision}",
            )
    raise RustUnavailable(
        f"no {_SDK_CRATE} source: set {conf.RUST_COMET_SDK} or $COMET_HOME to a Comet checkout"
    )


# -- building -----------------------------------------------------------------


def build_dir(spark: Any = None) -> Path:
    configured = conf.get_value(conf.RUST_BUILD_DIR, spark, "").strip()
    # Absolute: cargo runs inside the generated crate and resolves a relative
    # CARGO_TARGET_DIR against that, not against the driver's directory.
    if configured:
        return Path(configured).expanduser().absolute()
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return (Path(cache) / "ai_udf_transpile" / "rust").absolute()


def _cargo(spark: Any = None) -> str:
    binary = conf.get_value(conf.RUST_CARGO, spark, conf.DEFAULTS[conf.RUST_CARGO])
    found = shutil.which(binary)
    if not found:
        raise RustUnavailable(f"cargo not found ({binary!r}); install Rust or set {conf.RUST_CARGO}")
    return found


def _lib_filename(name: str) -> str:
    if sys.platform == "darwin":
        return f"lib{name}.dylib"
    if sys.platform.startswith("win"):
        return f"{name}.dll"
    return f"lib{name}.so"


def _run_cargo(spark: Any, args: list[str], cwd: Path, target: Path) -> subprocess.CompletedProcess:
    timeout = conf.get_int(conf.RUST_BUILD_TIMEOUT, spark, int(conf.DEFAULTS[conf.RUST_BUILD_TIMEOUT]))
    env = os.environ.copy()
    env["CARGO_TERM_COLOR"] = "never"
    env["CARGO_TARGET_DIR"] = str(target)
    try:
        return subprocess.run(
            [_cargo(spark), *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RustCompileError(f"cargo {args[0]} timed out after {timeout}s") from exc


def _compiler_errors(stderr: str, limit: int = 1500) -> str:
    """The error blocks from cargo's stderr, without the progress chatter."""
    kept: list[str] = []
    keeping = False
    for line in (stderr or "").splitlines():
        if line.startswith("error"):
            keeping = True
        elif line.startswith(("warning", "   Compiling", "    Finished", "    Updating", "  Downloaded")):
            keeping = False
        if keeping:
            kept.append(line)
    text = "\n".join(kept).strip() or (stderr or "").strip()
    return text[:limit]


def _arrow_from_manifest(crate: Path) -> Optional[str]:
    """The SDK's own ``arrow`` requirement, read from its manifest (or its workspace's)."""
    try:
        manifest = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
        dep = manifest.get("dependencies", {}).get("arrow")
        if isinstance(dep, dict) and dep.get("workspace"):
            dep = None
            for parent in crate.parents:
                candidate = parent / "Cargo.toml"
                if not candidate.is_file():
                    continue
                workspace = tomllib.loads(candidate.read_text(encoding="utf-8")).get("workspace")
                if workspace:
                    dep = workspace.get("dependencies", {}).get("arrow")
                    break
        version = dep.get("version") if isinstance(dep, dict) else dep
        return str(version) if version else None
    except Exception:
        logger.debug("could not read the arrow requirement from %s", crate, exc_info=True)
        return None


@dataclass(frozen=True)
class _SdkState:
    spec: SdkSpec
    arrow_req: str
    lock: Optional[Path]  # Cargo.lock every UDF crate starts from


_SDK_STATES: dict[tuple[str, str], _SdkState] = {}
_SDK_LOCK = threading.Lock()


def _sdk_state(spark: Any, spec: SdkSpec) -> _SdkState:
    """The ``arrow`` requirement to build against, plus a shared lock file.

    A UDF crate names ``arrow`` directly, and it has to be the SDK's arrow:
    two majors in one crate graph fail to compile (the trait's signatures use
    the SDK's types). The SDK is not on crates.io, so ask cargo what the SDK
    resolved to. One ``cargo metadata`` per SDK; its lock file then seeds
    every UDF crate so they all reuse the same compiled dependencies.
    """
    root = build_dir(spark)
    key = (str(root), spec.ident)
    sdk_name = "sdk-" + hashlib.sha256(spec.ident.encode()).hexdigest()[:16]
    with _SDK_LOCK, _locked(sdk_name, root):
        cached = _SDK_STATES.get(key)
        if cached is not None:
            return cached
        home = root / "sdk" / hashlib.sha256(spec.ident.encode()).hexdigest()[:16]
        req_file = home / "arrow_req"
        lock_file = home / "Cargo.lock"
        if req_file.is_file():
            state = _SdkState(
                spec, req_file.read_text(encoding="utf-8").strip(), lock_file if lock_file.is_file() else None
            )
            _SDK_STATES[key] = state
            return state
        arrow_req = _arrow_from_manifest(spec.path) if spec.path is not None else None
        probe = home / "probe"
        (probe / "src").mkdir(parents=True, exist_ok=True)
        (probe / "src" / "lib.rs").write_text("", encoding="utf-8")
        (probe / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "ai_udf_rs_probe"',
                    'version = "0.0.0"',
                    'edition = "2021"',
                    "publish = false",
                    "",
                    "[dependencies]",
                    f"{_SDK_CRATE} = {spec.dependency}",
                    *([f'arrow = "{arrow_req}"'] if arrow_req else []),
                    "",
                    "[workspace]",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        done = _run_cargo(spark, ["metadata", "--format-version", "1"], probe, root / "target")
        if done.returncode != 0:
            detail = _compiler_errors(done.stderr)
            raise RustUnavailable(f"cannot resolve {_SDK_CRATE} ({spec.ident}): {detail}")
        if not arrow_req:
            try:
                packages = json.loads(done.stdout).get("packages", [])
                sdk = next(p for p in packages if p.get("name") == _SDK_CRATE)
                arrow = next(d for d in sdk.get("dependencies", []) if d.get("name") == "arrow")
                arrow_req = str(arrow.get("req") or "").lstrip("^").strip()
            except Exception as exc:
                raise RustUnavailable(f"{_SDK_CRATE} ({spec.ident}) declares no arrow dependency") from exc
            if not arrow_req:
                raise RustUnavailable(f"{_SDK_CRATE} ({spec.ident}) declares no arrow dependency")
        if (probe / "Cargo.lock").is_file():
            shutil.copyfile(probe / "Cargo.lock", lock_file)
        req_file.write_text(arrow_req + "\n", encoding="utf-8")
        state = _SdkState(spec, arrow_req, lock_file if lock_file.is_file() else None)
        _SDK_STATES[key] = state
        return state


def udf_name(rust_source: str, input_types: list[str], return_type: str, sdk_ident: str) -> str:
    """Content-addressed: the SQL function, the crate, and the library file.

    A loaded library stays loaded for the life of the process and must never
    be overwritten, so anything that changes the bytes changes the name.
    """
    types = [t.strip().lower() for t in (*input_types, "->", return_type)]
    digest = hashlib.sha256()
    for part in (_WRAPPER_VERSION, sdk_ident, *types, rust_source):
        digest.update(part.encode())
        digest.update(b"\0")
    return _NAME_PREFIX + digest.hexdigest()[:16]


@dataclass(frozen=True)
class CompiledRust:
    name: str  # registered SQL function == the kernel's name()
    library: Path


_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()
_BACKGROUND: dict[str, threading.Thread] = {}


def _build_lock(name: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(name, threading.Lock())


@contextmanager
def _locked(name: str, root: Path):
    """Serialize work on one UDF crate across threads AND processes.

    Two drivers sharing a build directory (the per-user default, or
    rust.sharedBuildDir) would otherwise build the same crate in the same
    directory and delete each other's sources mid-build. The thread lock
    comes first: flock is per open file, so threads of one process need
    their own exclusion anyway. No fcntl (Windows): threads only.
    """
    with _build_lock(name):
        locks = root / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        with open(locks / f"{name}.lock", "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _planned(spark: Any, rust_source: str, input_types: list[str], return_type: str) -> CompiledRust:
    spec = resolve_sdk(spark)
    name = udf_name(rust_source, input_types, return_type, spec.ident)
    return CompiledRust(name=name, library=build_dir(spark) / "lib" / _lib_filename(name))


def cached_library(
    spark: Any, rust_source: str, input_types: list[str], return_type: str
) -> Optional[CompiledRust]:
    """The already-built library for this rewrite, or None. Never builds."""
    try:
        planned = _planned(spark, rust_source, input_types, return_type)
    except RustUnavailable:
        return None
    return planned if planned.library.is_file() else None


def compile_rust(spark: Any, rust_source: str, input_types: list[str], return_type: str) -> CompiledRust:
    """Build (or reuse) the cdylib for one row function. Blocks on cargo.

    The first build against an SDK compiles arrow (minutes); after that only
    the UDF's own crate compiles (seconds).
    """
    if not (rust_source or "").strip():
        raise RustCompileError("empty Rust source")
    planned = _planned(spark, rust_source, input_types, return_type)
    root = build_dir(spark)
    with _locked(planned.name, root):
        if planned.library.is_file():
            return planned  # built by another thread or process while we waited
        state = _sdk_state(spark, resolve_sdk(spark))
        crate = root / "crates" / planned.name
        # Fresh directory: nothing stale (a build.rs, an extra module) rides along.
        shutil.rmtree(crate, ignore_errors=True)
        (crate / "src").mkdir(parents=True)
        (crate / "Cargo.toml").write_text(
            render_manifest(planned.name, state.spec.dependency, state.arrow_req), encoding="utf-8"
        )
        (crate / "src" / "lib.rs").write_text(
            render_lib(planned.name, input_types, return_type), encoding="utf-8"
        )
        (crate / "src" / "udf.rs").write_text(rust_source.rstrip() + "\n", encoding="utf-8")
        if state.lock is not None:
            shutil.copyfile(state.lock, crate / "Cargo.lock")
        target = root / "target"
        try:
            done = _run_cargo(spark, ["build", "--release", "--lib"], crate, target)
            if done.returncode != 0:
                raise RustCompileError(f"cargo build failed: {_compiler_errors(done.stderr)}")
            built = target / "release" / _lib_filename(planned.name)
            if not built.is_file():
                raise RustCompileError(f"cargo produced no library at {built}")
        except RustCompileError:
            # The source is on the failed catalog row; nothing here is worth keeping.
            _prune(target, planned.name)
            shutil.rmtree(crate, ignore_errors=True)
            raise
        planned.library.parent.mkdir(parents=True, exist_ok=True)
        # Rename into place: a reader never sees a half-written library, and
        # an existing (possibly loaded) file is never rewritten in place.
        staging = planned.library.with_name(f".{planned.library.name}.{os.getpid()}.{threading.get_ident()}")
        shutil.copyfile(built, staging)
        os.replace(staging, planned.library)
        _prune(target, planned.name)
        shutil.rmtree(crate, ignore_errors=True)
        return planned


def _prune(target: Path, name: str) -> None:
    """Drop one UDF's intermediates; the shared dependency artifacts stay."""
    release = target / "release"
    for pattern in (f"*{name}*", f"deps/*{name}*", f".fingerprint/{name}-*", f"incremental/{name}-*"):
        for item in release.glob(pattern):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
            except OSError:
                logger.debug("could not prune %s", item, exc_info=True)


def compile_in_background(spark: Any, rust_source: str, input_types: list[str], return_type: str) -> None:
    """Start a build unless one is already running. UDF definition never waits on cargo."""
    try:
        name = _planned(spark, rust_source, input_types, return_type).name
    except RustUnavailable:
        return

    def _build() -> None:
        try:
            compile_rust(spark, rust_source, input_types, return_type)
        except Exception:
            logger.info("background rust build of %s failed", name, exc_info=True)

    with _BUILD_LOCKS_GUARD:
        running = _BACKGROUND.get(name)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(target=_build, name=f"ai-udf-rustc-{name[-8:]}", daemon=True)
        _BACKGROUND[name] = thread
        thread.start()


def register_rust_udf(spark: Any, compiled: CompiledRust, input_types: list[str], return_type: str) -> str:
    """Register the library's UDF as a temporary function of ``spark``'s session.

    Comet loads the library on the driver and confirms the function exists;
    executors load it themselves from the same path on first use.
    """
    jvm = spark._jvm
    jsession = spark._jsparkSession

    def _jtype(simple: str) -> Any:
        return jsession.parseDataType(json.dumps(_spec(simple).json))

    native = getattr(getattr(jvm.org.apache.comet.udf, "CometNativeUDF$"), "MODULE$")
    native.register(
        jsession,
        compiled.name,
        str(compiled.library),
        jvm.PythonUtils.toSeq([_jtype(t) for t in input_types]),
        _jtype(return_type),
        True,
    )
    return compiled.name


# -- can this session use it? -------------------------------------------------


def target_enabled(spark: Any = None) -> bool:
    return KIND_RUST in conf.get_csv(conf.TARGETS, spark, conf.DEFAULTS[conf.TARGETS])


def _session_conf(spark: Any, key: str, default: str = "") -> str:
    try:
        value = spark.conf.get(key, default)
        return default if value is None else str(value)
    except Exception:
        return default


def comet_unavailable(spark: Any) -> Optional[str]:
    """Why a Comet native UDF cannot run in this session, or None when it can."""
    if spark is None:
        return "no SparkSession"
    wired = _COMET_PLUGIN in _session_conf(spark, "spark.plugins") or _COMET_EXTENSION in _session_conf(
        spark, "spark.sql.extensions"
    )
    if not wired:
        return "Comet is not installed in this session (spark.plugins)"
    try:
        jvm = spark._jvm
        getattr(getattr(jvm.org.apache.comet.udf, "CometNativeUDF$"), "MODULE$")
    except Exception:
        return "this Comet has no native UDF support (needs Comet 1.2+ on the driver classpath)"
    try:
        extensions = getattr(getattr(jvm.org.apache.comet, "CometSparkSessionExtensions$"), "MODULE$")
        loaded = bool(extensions.isCometLoaded(spark._jsparkSession.sessionState().conf()))
    except Exception:
        logger.debug("isCometLoaded probe failed; falling back to spark.comet.enabled", exc_info=True)
        loaded = _session_conf(spark, "spark.comet.enabled", "false").lower() == "true"
    if not loaded:
        return "the Comet extension is disabled in this session (see the driver log for the reason)"
    for key in ("spark.comet.exec.enabled", "spark.comet.nativeUdf.enabled"):
        if _session_conf(spark, key, "true").lower() != "true":
            return f"{key} is off"
    return None


def rust_unservable(spark: Any) -> Optional[str]:
    """Why a Rust rewrite must not be attached to a UDF here, or None when it may.

    Everything that would turn "Spark keeps the Python UDF" into "the query
    fails" is checked before the rewrite is offered.
    """
    if not target_enabled(spark):
        return f"rust is not in {conf.TARGETS}"
    reason = comet_unavailable(spark)
    if reason:
        return reason
    master = _session_conf(spark, "spark.master")
    if not master.startswith("local") and not conf.get_bool(conf.RUST_SHARED_BUILD_DIR, spark, False):
        return (
            f"master {master!r} is not local and {conf.RUST_SHARED_BUILD_DIR} is off: "
            "executors could not load the library"
        )
    return None


# How long a failed toolchain probe is believed before cargo is asked again.
_UNBUILDABLE_TTL = 300.0
_UNBUILDABLE: dict[str, tuple[float, str]] = {}


def rust_offered(spark: Any, input_types: list[str], return_type: str) -> bool:
    """Whether to ask a backend for Rust: servable here, buildable, and typed.

    "Buildable" is checked for real -- cargo exists and resolves the SDK (one
    ``cargo metadata`` per SDK, then cached) -- because an answer in Rust that
    cannot be built is a failed row and a 24h cooldown for a UDF that SQL or
    Java might have served. A failed probe is remembered for a few minutes so
    a broken toolchain costs one cargo call, not one per pending row.
    """
    if not supports_types(input_types, return_type) or rust_unservable(spark) is not None:
        return False
    # Everything that decides buildability: changing any of it re-probes.
    root = "\0".join(
        [
            str(build_dir(spark)),
            conf.get_value(conf.RUST_COMET_SDK, spark, ""),
            conf.get_value(conf.RUST_CARGO, spark, conf.DEFAULTS[conf.RUST_CARGO]),
            os.environ.get("AI_UDF_TRANSPILE_COMET_HOME", ""),
            os.environ.get("COMET_HOME", ""),
        ]
    )
    failed = _UNBUILDABLE.get(root)
    if failed is not None and time.monotonic() - failed[0] < _UNBUILDABLE_TTL:
        return False
    try:
        _cargo(spark)
        _sdk_state(spark, resolve_sdk(spark))
    except (RustUnavailable, RustCompileError) as exc:
        _UNBUILDABLE[root] = (time.monotonic(), str(exc))
        logger.warning("rust target is enabled but cannot build here; not offering rust: %s", exc)
        return False
    _UNBUILDABLE.pop(root, None)
    return True
