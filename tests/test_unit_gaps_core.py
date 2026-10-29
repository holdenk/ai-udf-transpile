# SPDX-License-Identifier: Apache-2.0
"""Coverage gaps in conf/types/keys/targets/backends: branches the other unit
tests never hit, picked off the baseline coverage report's Missing column."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile.backends import BackendError, get_backend, pick_auto
from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode import BytecodeBackend
from ai_udf_transpile.backends.claude import ClaudeBackend
from ai_udf_transpile.backends.cli import (
    CliBackend,
    CliSpec,
    _parse_claude_json,
    _parse_coco_json,
    parse_sandbox,
)
from ai_udf_transpile.backends.coco import CocoBackend
from ai_udf_transpile.backends.codex import CodexBackend
from ai_udf_transpile.backends.cursor import CursorBackend
from ai_udf_transpile.backends.ensemble import EnsembleBackend
from ai_udf_transpile.backends.pi import PiBackend
from ai_udf_transpile.keys import (
    _resolve_name,
    closure_fingerprint,
    extract_captures,
    free_names,
    func_from_spark_stack,
)
from ai_udf_transpile.targets import KIND_JAVA_UDF, KIND_RUST, TranspileResult
from ai_udf_transpile.types import (
    annotation_name,
    categories_for_spark_types,
    input_categories,
    return_spark_type,
)


@pytest.fixture(autouse=True)
def _reset_conf_runtime():
    conf.reset_runtime()
    yield
    conf.reset_runtime()


def _ast(func) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(__import__("inspect").getsource(func))).body[0]


def _parse(src: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(src).strip()).body[0]


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------- conf.py


def test_default_backend_env_wins(monkeypatch):
    monkeypatch.setenv("AI_UDF_TRANSPILE_BACKEND", "codex")
    assert conf.default_backend() == "codex"


def test_default_backend_outside_pytest_is_auto(monkeypatch):
    monkeypatch.delenv("AI_UDF_TRANSPILE_BACKEND", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert conf.default_backend() == "auto"


def test_default_auto_backend_priority_env(monkeypatch):
    monkeypatch.setenv("AI_UDF_TRANSPILE_BACKEND_PRIORITY", "codex,coco")
    assert conf.default_auto_backend_priority() == "codex,coco"


def test_default_max_examples_env(monkeypatch):
    monkeypatch.setenv("AI_UDF_TRANSPILE_MAX_EXAMPLES", "42")
    assert conf.default_max_examples() == "42"


def test_default_max_examples_github_actions(monkeypatch):
    monkeypatch.delenv("AI_UDF_TRANSPILE_MAX_EXAMPLES", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert conf.default_max_examples() == "5"


class _ExplodingConf:
    def set(self, key, value):
        raise RuntimeError("boom")

    def get(self, key, default):
        raise RuntimeError("boom")


class _ExplodingSpark:
    conf = _ExplodingConf()


def test_set_value_swallows_spark_conf_errors():
    conf.set_value(conf.BACKEND, "bytecode", spark=_ExplodingSpark())
    assert conf.get_value(conf.BACKEND) == "bytecode"


def test_get_value_swallows_spark_conf_errors():
    assert conf.get_value(conf.CATALOG, spark=_ExplodingSpark()) == "sqlite"


def test_get_int_falls_back_on_bad_value():
    conf.set_value(conf.MAX_RETRIES, "notanumber")
    assert conf.get_int(conf.MAX_RETRIES, default=7) == 7


def test_get_float_falls_back_on_bad_value():
    conf.set_value(conf.TOLERANCE, "notafloat")
    assert conf.get_float(conf.TOLERANCE, default=1.5) == 1.5


# ----------------------------------------------------------- backends/__init__


def test_pick_auto_skips_empty_and_duplicate_priority_entries(monkeypatch):
    conf.set_value(conf.AUTO_BACKEND_PRIORITY, "coco,,coco,pi")
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: "/tmp/pi" if binary == "pi" else None,
    )
    assert pick_auto() == "pi"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("bytecode", BytecodeBackend),
        ("coco", CocoBackend),
        ("pi", PiBackend),
        ("cursor", CursorBackend),
        ("claude", ClaudeBackend),
        ("codex", CodexBackend),
        ("ensemble", EnsembleBackend),
    ],
)
def test_get_backend_dispatches_by_name(name, cls):
    assert isinstance(get_backend(name), cls)


def test_get_backend_auto_resolves_via_pick_auto(monkeypatch):
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: "/tmp/cortex" if binary == "cortex" else None,
    )
    assert isinstance(get_backend("auto"), CocoBackend)


def test_get_backend_unknown_name_errors():
    with pytest.raises(BackendError, match="unknown backend"):
        get_backend("not-a-backend")


# -------------------------------------------------------------- ensemble.py


def test_ensemble_default_backends_from_conf():
    ensemble = EnsembleBackend()
    assert ensemble._backend_names == conf.DEFAULTS[conf.ENSEMBLE_BACKENDS].split(",")


# --------------------------------------------------------------------- cli.py


def test_parse_sandbox_java_decline():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        sandbox = Path(d)
        (sandbox / "OUT.java").write_text("DECLINE\n")
        with pytest.raises(BackendDecline):
            parse_sandbox(sandbox)


def test_parse_stdout_java_without_files(tmp_path):
    result = parse_sandbox(tmp_path, stdout="public class Plus {\n  int x;\n}\n")
    assert result.kind == KIND_JAVA_UDF
    assert result.class_name == "Plus"


def test_parse_stdout_sql_prefix_without_files(tmp_path):
    result = parse_sandbox(tmp_path, stdout="SQL\n_udf_param_0 + 1")
    assert result.sql == "_udf_param_0 + 1"


def test_parse_stdout_plain_text_is_sql(tmp_path):
    result = parse_sandbox(tmp_path, stdout="_udf_param_0 + 1")
    assert result.sql == "_udf_param_0 + 1"


def test_parse_claude_json_invalid_json_passthrough():
    model, text = _parse_claude_json("{not valid")
    assert model == ""
    assert text == "{not valid"


def test_parse_claude_json_non_dict_passthrough():
    model, text = _parse_claude_json("[1, 2, 3]")
    assert model == ""
    assert text == "[1, 2, 3]"


def test_parse_claude_json_falls_back_to_top_level_model():
    payload = json.dumps({"model": "claude-direct", "result": "txt"})
    model, text = _parse_claude_json(payload)
    assert model == "claude-direct"
    assert text == "txt"


def test_parse_coco_json_skips_malformed_and_non_dict_lines():
    stdout = "\n".join(
        [
            "{not valid json",
            "[1,2,3]",
            json.dumps({"type": "assistant", "message": {"model": "m1"}}),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )
    model, text = _parse_coco_json(stdout)
    assert model == "m1"
    assert text == "done"


def test_model_returns_empty_without_model_conf_key():
    spec = CliSpec(
        name="x",
        default_binary="x",
        binary_conf_key=conf.BINARY_COCO,
        argv=lambda binary, sandbox, prompt: [binary],
    )
    assert CliBackend(spec).model() == ""


def test_reported_model_default_is_name_unknown():
    assert CursorBackend().reported_model() == "cursor-unknown"


def test_reported_model_uses_configured_model():
    conf.set_value(conf.MODEL_CURSOR, "gpt-x-1")
    assert CursorBackend().reported_model() == "gpt-x-1"


def test_run_wraps_timeout(tmp_path):
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

    with pytest.raises(BackendError, match="timed out"):
        CursorBackend(runner=runner).run(None, tmp_path)


def test_run_wraps_missing_binary(tmp_path):
    def runner(argv, **kwargs):
        raise FileNotFoundError()

    with pytest.raises(BackendError, match="binary not found"):
        CursorBackend(runner=runner).run(None, tmp_path)


def test_run_nonzero_returncode_still_parses_sandbox(tmp_path):
    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed(returncode=1)

    result = CursorBackend(runner=runner).run(None, tmp_path)
    assert result.sql == "_udf_param_0 + 1"


# ------------------------------------------------------------- targets.py


def test_reconstructable_java_udf():
    assert TranspileResult(kind=KIND_JAVA_UDF, class_name="com.x.Y").reconstructable() is True
    assert TranspileResult(kind=KIND_JAVA_UDF).reconstructable() is False


def test_reconstructable_unknown_kind_is_false():
    assert TranspileResult(kind=KIND_RUST).reconstructable() is False


# ---------------------------------------------------------------- types.py


def test_categories_for_spark_types_known_map_array_and_unknown():
    assert categories_for_spark_types(["bigint", "string", "boolean", "binary"]) == [
        "numeric",
        "string",
        "bool",
        "binary",
    ]
    assert categories_for_spark_types(["map<string,string>"]) == ["map"]
    assert categories_for_spark_types(["array<string>"]) == ["array"]
    assert categories_for_spark_types(["struct<a:int>"]) == ["numeric"]


def test_annotation_name_from_attribute():
    fn = _parse("def f(x: dt.datetime) -> int:\n    return 1")
    assert annotation_name(fn.args.args[0].annotation) == "datetime"


def test_input_categories_unknown_annotation_declines():
    fn = _parse("def f(x: SomeWeirdType) -> int:\n    return 1")
    assert input_categories(fn, ["x"]) is None


def test_return_spark_type_simplestring_raises_falls_back_to_type_name():
    class StringType:
        def simpleString(self):
            raise RuntimeError("boom")

    assert return_spark_type(StringType()) == "StringType"


def test_return_spark_type_unknown_object_is_none():
    class Mystery:
        pass

    assert return_spark_type(Mystery()) is None


# ---------------------------------------------------------------- keys.py


def test_closure_fingerprint_stabilizes_tuple_bytes_module():
    fp1 = closure_fingerprint({"t": (1, 2), "b": b"xx", "m": sys})
    fp2 = closure_fingerprint({"t": (1, 2), "b": b"xx", "m": sys})
    assert fp1 == fp2
    assert fp1 != ""


def test_extract_captures_allows_tuple_literal():
    def make(offset):
        def add_offset(x: int) -> int:
            return x + offset

        return add_offset

    f = make((1, 2))
    assert extract_captures(f, _ast(f), ["x"]) == {"offset": (1, 2)}


def test_free_names_binds_posonly_kwonly_vararg_kwarg():
    tree = _parse("def f(x, /, y, *args, z, **kwargs):\n    return x + y + z + len(args) + len(kwargs)")
    assert free_names(tree, ["x", "y", "z"]) == {"len"}


def test_free_names_visits_decorators():
    tree = _parse("@some_decorator\ndef f(x: int) -> int:\n    return x")
    assert "some_decorator" in free_names(tree, ["x"])


def test_free_names_binds_nested_def_class_import_from():
    tree = _parse(
        """
        def f(x: int) -> int:
            from os import path as p
            def helper():
                return 1
            async def ahelper():
                return 1
            class C:
                pass
            return x + len(p.sep) + helper() + len(C.__name__)
        """
    )
    names = free_names(tree, ["x"])
    assert not {"p", "helper", "ahelper", "C"} & names


def test_free_names_binds_except_and_match_captures():
    tree = _parse(
        """
        def f(x):
            try:
                y = 1
            except Exception as e:
                y = str(e)
            match [x]:
                case [a, *rest]:
                    y = len(rest) + a
                case {"k": v, **rest2}:
                    y = v
            return y
        """
    )
    names = free_names(tree, ["x"])
    assert not {"e", "a", "rest", "v", "rest2"} & names


_GLOBAL_OFFSET = 7


def _uses_global_offset(x: int) -> int:
    return x + _GLOBAL_OFFSET


def test_extract_captures_resolves_module_global():
    assert extract_captures(_uses_global_offset, _ast(_uses_global_offset), ["x"]) == {"_GLOBAL_OFFSET": 7}


def test_extract_captures_func_none_builtins_only():
    fn = _parse("def f(x: int) -> int:\n    return len(str(x))")
    assert extract_captures(None, fn, ["x"]) == {}


def test_extract_captures_func_none_with_unknown_name_declines():
    fn = _parse("def f(x: int) -> int:\n    return x + UNKNOWN_THING")
    assert extract_captures(None, fn, ["x"]) is None


def _f_refs_undefined(x: int) -> int:
    return x + TOTALLY_UNDEFINED_NAME  # noqa: F821


def test_extract_captures_unresolvable_name_declines():
    assert extract_captures(_f_refs_undefined, _ast(_f_refs_undefined), ["x"]) is None


abs = abs  # rebind the builtin to itself: a global lookup that still resolves to it


def _f_uses_rebound_abs(x: int) -> int:
    return abs(x)


def test_extract_captures_global_rebound_builtin_is_not_a_capture():
    assert extract_captures(_f_uses_rebound_abs, _ast(_f_uses_rebound_abs), ["x"]) == {}


def test_resolve_name_without_func_raises_keyerror():
    # Dead via extract_captures (it never calls _resolve_name when func is
    # None), reachable only by calling this private helper directly.
    with pytest.raises(KeyError):
        _resolve_name(None, "x")


def test_func_from_spark_stack_returns_none_outside_worker():
    assert func_from_spark_stack() is None
