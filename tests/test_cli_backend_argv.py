# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from ai_udf_transpile import conf
from ai_udf_transpile.backends.claude import ClaudeBackend
from ai_udf_transpile.backends.cli import _parse_claude_json, parse_sandbox
from ai_udf_transpile.backends.coco import CocoBackend
from ai_udf_transpile.backends.cursor import CursorBackend
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_coco_argv(tmp_path):
    seen = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed()

    backend = CocoBackend(runner=runner)
    prompt = tmp_path / "PROMPT.md"
    prompt.write_text("hello")
    argv = backend.build_argv(tmp_path, "hello")
    assert argv[0] == "cortex"
    assert "exec" in argv
    assert "--file" in argv
    assert "--workdir" in argv
    assert "--bypass" in argv
    result = backend.run(None, tmp_path)  # job unused
    assert result.sql == "_udf_param_0 + 1"
    assert seen["cwd"] == str(tmp_path)


def test_cursor_argv(tmp_path):
    backend = CursorBackend()
    argv = backend.build_argv(tmp_path, "PROMPT TEXT")
    assert argv[0] == "agent"
    assert "-p" in argv
    assert "PROMPT TEXT" in argv
    assert "--trust" in argv
    assert "--force" in argv
    assert "--output-format" in argv


def test_claude_argv(tmp_path):
    backend = ClaudeBackend()
    argv = backend.build_argv(tmp_path, "PROMPT TEXT")
    assert argv[0] == "claude"
    assert "-p" in argv
    assert "--bare" not in argv  # bare mode skips apiKeyHelper auth
    assert "--permission-mode" in argv
    assert "dontAsk" in argv
    assert "--allowedTools" in argv
    assert "Read,Write" in argv
    assert "--max-turns" in argv
    assert "--output-format" in argv
    assert "json" in argv
    assert "PROMPT TEXT" in argv


def test_model_flag_appended_when_conf_set(tmp_path):
    conf.set_value(conf.MODEL_CURSOR, "gpt-x-1")
    argv = CursorBackend().build_argv(tmp_path, "PROMPT TEXT")
    assert argv[argv.index("--model") + 1] == "gpt-x-1"
    assert "PROMPT TEXT" in argv


def test_model_flag_absent_by_default(tmp_path):
    argv = CursorBackend().build_argv(tmp_path, "PROMPT TEXT")
    assert "--model" not in argv


def test_coco_model_flag(tmp_path):
    conf.set_value(conf.MODEL_COCO, "snow-intelligence-x")
    argv = CocoBackend().build_argv(tmp_path, "PROMPT TEXT")
    assert "-m" in argv
    assert argv[argv.index("-m") + 1] == "snow-intelligence-x"


def test_parse_claude_json_extracts_model_and_result():
    payload = json.dumps(
        {
            "type": "result",
            "result": "_udf_param_0 + 1",
            "modelUsage": {"claude-test-model": {"inputTokens": 3}},
        }
    )
    model, text = _parse_claude_json(payload)
    assert model == "claude-test-model"
    assert text == "_udf_param_0 + 1"


def test_parse_claude_json_passthrough_on_non_json():
    model, text = _parse_claude_json("plain text output")
    assert model == ""
    assert text == "plain text output"


def test_claude_run_records_detected_model(tmp_path):
    payload = json.dumps({"result": "ignored", "modelUsage": {"claude-x": {}}})

    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed(stdout=payload)

    backend = ClaudeBackend(runner=runner)
    result = backend.run(None, tmp_path)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "claude-x"


def _sql_runner(argv, **kwargs):
    (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
    return _Completed()


def test_cursor_run_records_unknown_model(tmp_path):
    result = CursorBackend(runner=_sql_runner).run(None, tmp_path)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "cursor-unknown"


def test_coco_run_records_unknown_model(tmp_path):
    result = CocoBackend(runner=_sql_runner).run(None, tmp_path)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "coco-unknown"


def test_claude_run_records_unknown_model_without_json(tmp_path):
    result = ClaudeBackend(runner=_sql_runner).run(None, tmp_path)
    assert result.model == "claude-unknown"


def test_run_records_configured_model(tmp_path):
    conf.set_value(conf.MODEL_CURSOR, "gpt-x-1")
    result = CursorBackend(runner=_sql_runner).run(None, tmp_path)
    assert result.model == "gpt-x-1"


def test_parse_sql_wins(tmp_path):
    (tmp_path / "OUT.sql").write_text("_udf_param_0 + 1\n")
    (tmp_path / "OUT.java").write_text("class X {}\n")
    result = parse_sandbox(tmp_path, stdout="ignored")
    assert result.kind == KIND_CATALYST
    assert result.sql == "_udf_param_0 + 1"


def test_parse_java_when_no_sql(tmp_path):
    (tmp_path / "OUT.java").write_text("package com.ex;\nclass PlusOne {}\n")
    result = parse_sandbox(tmp_path)
    assert result.kind == KIND_JAVA_UDF
    assert result.class_name == "com.ex.PlusOne"
