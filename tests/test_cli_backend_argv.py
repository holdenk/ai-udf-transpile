# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile.backends.base import BackendDecline, BackendError
from ai_udf_transpile.backends.claude import ClaudeBackend
from ai_udf_transpile.backends.cli import _parse_claude_json, _parse_coco_json, parse_sandbox
from ai_udf_transpile.backends.coco import CocoBackend
from ai_udf_transpile.backends.codex import CodexBackend
from ai_udf_transpile.backends.cursor import CursorBackend
from ai_udf_transpile.backends.pi import PiBackend
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
    assert argv[argv.index("--format") + 1] == "json"
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


def test_pi_argv(tmp_path):
    backend = PiBackend()
    argv = backend.build_argv(tmp_path, "PROMPT TEXT")
    assert argv[0] == "pi"
    assert "-p" in argv
    assert "PROMPT TEXT" in argv
    assert "--trust" in argv
    assert "--force" in argv
    assert "--output-format" in argv


def test_codex_argv(tmp_path):
    backend = CodexBackend()
    argv = backend.build_argv(tmp_path, "PROMPT TEXT")
    assert argv[0] == "codex"
    assert "exec" in argv
    assert "--skip-git-repo-check" in argv
    assert "--sandbox" in argv
    assert "workspace-write" in argv
    assert "--approve-for-me" in argv
    assert "PROMPT TEXT" in argv


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


def test_pi_model_flag(tmp_path):
    conf.set_value(conf.MODEL_PI, "pi-foo")
    argv = PiBackend().build_argv(tmp_path, "PROMPT TEXT")
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "pi-foo"


def test_codex_model_flag(tmp_path):
    conf.set_value(conf.MODEL_CODEX, "codex-foo")
    argv = CodexBackend().build_argv(tmp_path, "PROMPT TEXT")
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "codex-foo"


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


def test_parse_coco_json_extracts_model_and_result():
    stdout = "\n".join(
        [
            "\x1b[32m✓ Auto-apply [_CORTEX_CODE_ENGINEER]: 5 skill repos\x1b[0m",
            json.dumps({"type": "system", "subtype": "init", "model": "auto"}),
            json.dumps({"type": "assistant", "message": {"model": "auto", "role": "assistant"}}),
            json.dumps({"type": "assistant", "message": {"model": "claude-sonnet-5"}}),
            json.dumps({"type": "result", "subtype": "success", "result": "OUT.sql written"}),
        ]
    )
    model, text = _parse_coco_json(stdout)
    assert model == "claude-sonnet-5"  # last assistant record wins
    assert text == "OUT.sql written"


def test_parse_coco_json_passthrough_on_non_json():
    model, text = _parse_coco_json("plain text output")
    assert model == ""
    assert text == "plain text output"


def test_coco_run_records_detected_model(tmp_path):
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "model": "auto"}),
            json.dumps({"type": "assistant", "message": {"model": "glm-5.3"}}),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
        ]
    )

    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed(stdout=stdout)

    result = CocoBackend(runner=runner).run(None, tmp_path)
    assert result.sql == "_udf_param_0 + 1"  # OUT.sql beats the JSON result text
    assert result.model == "glm-5.3"


def test_coco_run_records_auto_when_routing(tmp_path):
    # Under auto routing the CLI only ever says "auto" (modelUsage stays
    # empty too); recording that beats an invented "coco-unknown".
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "model": "auto"}),
            json.dumps({"type": "assistant", "message": {"model": "auto"}}),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
        ]
    )

    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed(stdout=stdout)

    result = CocoBackend(runner=runner).run(None, tmp_path)
    assert result.model == "auto"


def test_coco_decline_carries_detected_model(tmp_path):
    stdout = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"model": "glm-5.3"}}),
            json.dumps({"type": "result", "subtype": "success", "result": "DECLINE"}),
        ]
    )

    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("DECLINE\n")
        return _Completed(stdout=stdout)

    with pytest.raises(BackendDecline) as excinfo:
        CocoBackend(runner=runner).run(None, tmp_path)
    assert excinfo.value.model == "glm-5.3"


def test_claude_decline_carries_detected_model(tmp_path):
    payload = json.dumps({"result": "DECLINE", "modelUsage": {"claude-x": {}}})

    def runner(argv, **kwargs):
        return _Completed(stdout=payload)

    with pytest.raises(BackendDecline) as excinfo:
        ClaudeBackend(runner=runner).run(None, tmp_path)
    assert excinfo.value.model == "claude-x"


def test_coco_configured_model_beats_detected(tmp_path):
    conf.set_value(conf.MODEL_COCO, "snow-intelligence-x")
    stdout = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"model": "glm-5.3"}}),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
        ]
    )

    def runner(argv, **kwargs):
        (Path(kwargs["cwd"]) / "OUT.sql").write_text("_udf_param_0 + 1\n")
        return _Completed(stdout=stdout)

    result = CocoBackend(runner=runner).run(None, tmp_path)
    assert result.model == "snow-intelligence-x"


def test_coco_no_result_record_is_an_error_not_a_decline(tmp_path):
    # A stream with no result record is a crashed/killed CLI run, not a
    # decision: BackendError retries, a decline would cool down for 24h.
    stdout = json.dumps({"type": "assistant", "message": {"model": "glm-5.3"}})

    def runner(argv, **kwargs):
        return _Completed(stdout=stdout)

    with pytest.raises(BackendError):
        CocoBackend(runner=runner).run(None, tmp_path)


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


def test_parse_empty_output_is_an_error_not_a_decline(tmp_path):
    # No files, no stdout: the CLI crashed or was rate-limited; it did not
    # decide anything. An error retries; a decline cools down.
    with pytest.raises(BackendError):
        parse_sandbox(tmp_path, stdout="")


def test_parse_decline_stdout(tmp_path):
    with pytest.raises(BackendDecline):
        parse_sandbox(tmp_path, stdout="DECLINE\narray return is not expressible")
