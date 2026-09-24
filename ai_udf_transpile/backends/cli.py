# SPDX-License-Identifier: Apache-2.0
"""Shared CLI backend: write sandbox, exec argv, parse OUT.sql / OUT.java / stdout."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.backends.base import BackendDecline, BackendError
from ai_udf_transpile.javac import extract_java_class
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileJob, TranspileResult

logger = logging.getLogger(__name__)


def parse_sandbox(sandbox: Path, stdout: str = "") -> TranspileResult:
    sql_path = sandbox / "OUT.sql"
    java_path = sandbox / "OUT.java"
    if sql_path.exists():
        sql = sql_path.read_text(encoding="utf-8").strip()
        if sql and sql.upper() != "DECLINE":
            return TranspileResult(kind=KIND_CATALYST, sql=sql)
    if java_path.exists():
        source = java_path.read_text(encoding="utf-8").strip()
        if source and source.upper() != "DECLINE":
            return TranspileResult(
                kind=KIND_JAVA_UDF,
                java_source=source,
                class_name=extract_java_class(source),
            )
    text = (stdout or "").strip()
    if not text or text.upper().startswith("DECLINE"):
        raise BackendDecline(text or "empty CLI output")
    first, _, rest = text.partition("\n")
    if first.strip().lower().endswith(".java") or "class " in text:
        return TranspileResult(
            kind=KIND_JAVA_UDF,
            java_source=text,
            class_name=extract_java_class(text),
        )
    sql = rest.strip() if first.strip().upper() == "SQL" else text
    return TranspileResult(kind=KIND_CATALYST, sql=sql)


ArgvBuilder = Callable[[str, Path, str], list[str]]


def _parse_claude_json(stdout: str) -> tuple[str, str]:
    """Extract (model, result_text) from claude's ``--output-format json`` stdout.

    Falls back to ("", original stdout) when the output is not a JSON object.
    """
    text = (stdout or "").strip()
    if not text.startswith("{"):
        return "", stdout
    try:
        data = json.loads(text)
    except Exception:
        return "", stdout
    if not isinstance(data, dict):
        return "", stdout
    model = ""
    usage = data.get("modelUsage")
    if isinstance(usage, dict) and usage:
        model = str(next(iter(usage.keys())))
    if not model and data.get("model"):
        model = str(data["model"])
    result_text = data.get("result")
    return model, (result_text if isinstance(result_text, str) else "")


@dataclass(frozen=True)
class CliSpec:
    name: str
    default_binary: str
    binary_conf_key: str
    argv: ArgvBuilder
    extra_env: tuple[str, ...] = ()
    model_conf_key: Optional[str] = None
    model_flag: Optional[str] = None


def coco_argv(binary: str, sandbox: Path, prompt_text: str) -> list[str]:
    del prompt_text
    return [
        binary,
        "exec",
        "--file",
        str(sandbox / "PROMPT.md"),
        "--workdir",
        str(sandbox),
        "--bypass",
    ]


def cursor_argv(binary: str, sandbox: Path, prompt_text: str) -> list[str]:
    del sandbox
    return [binary, "-p", prompt_text, "--trust", "--force", "--output-format", "text"]


def claude_argv(binary: str, sandbox: Path, prompt_text: str) -> list[str]:
    del sandbox
    return [
        binary,
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read,Write",
        "--max-turns",
        "8",
        prompt_text,
    ]


COCO_SPEC = CliSpec(
    name="coco",
    default_binary=conf.DEFAULTS[conf.BINARY_COCO],
    binary_conf_key=conf.BINARY_COCO,
    argv=coco_argv,
    model_conf_key=conf.MODEL_COCO,
    model_flag="-m",
)
CURSOR_SPEC = CliSpec(
    name="cursor",
    default_binary=conf.DEFAULTS[conf.BINARY_CURSOR],
    binary_conf_key=conf.BINARY_CURSOR,
    argv=cursor_argv,
    extra_env=("CURSOR_API_KEY",),
    model_conf_key=conf.MODEL_CURSOR,
    model_flag="--model",
)
CLAUDE_SPEC = CliSpec(
    name="claude",
    default_binary=conf.DEFAULTS[conf.BINARY_CLAUDE],
    binary_conf_key=conf.BINARY_CLAUDE,
    argv=claude_argv,
    extra_env=("ANTHROPIC_API_KEY",),
    model_conf_key=conf.MODEL_CLAUDE,
    model_flag="--model",
)


class CliBackend:
    def __init__(self, spec: CliSpec, spark: Any = None, *, runner: Any = None):
        self.spec = spec
        self.name = spec.name
        self.spark = spark
        self._runner = runner or subprocess.run

    def binary(self) -> str:
        return conf.get_value(self.spec.binary_conf_key, self.spark, self.spec.default_binary)

    def model(self) -> str:
        if not self.spec.model_conf_key:
            return ""
        return conf.get_value(self.spec.model_conf_key, self.spark, "").strip()

    def reported_model(self) -> str:
        """Model to record in the catalog: the configured model or ``<name>-unknown``.

        We do our own dispatch, so a row should never have a NULL model just
        because the CLI did not print which model served the request.
        """
        return self.model() or f"{self.name}-unknown"

    def build_argv(self, sandbox: Path, prompt_text: str) -> list[str]:
        argv = self.spec.argv(self.binary(), sandbox, prompt_text)
        model = self.model()
        if model and self.spec.model_flag:
            # Right after the binary: safe regardless of where the prompt sits.
            argv = [argv[0], self.spec.model_flag, model] + argv[1:]
        return argv

    def run(self, job: TranspileJob, sandbox: Path) -> TranspileResult:
        del job
        prompt_path = sandbox / "PROMPT.md"
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        argv = self.build_argv(sandbox, prompt_text)
        timeout = conf.get_int(conf.CLI_TIMEOUT, self.spark, int(conf.DEFAULTS[conf.CLI_TIMEOUT]))
        env = os.environ.copy()
        logger.debug("running CLI backend %s: %s", self.name, argv)
        try:
            completed = self._runner(
                argv,
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.name} timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise BackendError(f"{self.name} binary not found: {argv[0]}") from exc
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode not in (0, None) and completed.returncode != 0:
            # Still try to parse sandbox outputs; some CLIs exit non-zero after writing files.
            logger.debug("%s exit %s stderr=%s", self.name, completed.returncode, stderr[:500])
        model = self.model()
        if self.spec.name == "claude":
            detected, stdout = _parse_claude_json(stdout)
            model = model or detected
        try:
            result = parse_sandbox(sandbox, stdout)
        except BackendDecline:
            raise
        except Exception as exc:
            raise BackendError(f"{self.name} produced no rewrite: {exc}; stderr={stderr[:300]}") from exc
        if not result.model:
            result.model = model or f"{self.name}-unknown"
        return result
