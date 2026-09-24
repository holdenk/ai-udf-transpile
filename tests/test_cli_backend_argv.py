# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from ai_udf_transpile.backends.claude import ClaudeBackend
from ai_udf_transpile.backends.cli import parse_sandbox
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
    assert "PROMPT TEXT" in argv


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
