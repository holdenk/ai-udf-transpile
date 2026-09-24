# SPDX-License-Identifier: Apache-2.0
"""Temporary sandbox for CLI backends."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from ai_udf_transpile.targets import TranspileJob


def default_prompt() -> str:
    return files("ai_udf_transpile.resources").joinpath("prompt.md").read_text(encoding="utf-8")


def write_sandbox(root: Path, job: TranspileJob, prompt: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "PROMPT.md").write_text(prompt if prompt is not None else default_prompt(), encoding="utf-8")
    (root / "udf.py").write_text(job.source_text.rstrip() + "\n", encoding="utf-8")
    (root / "captures.json").write_text(
        json.dumps(job.captures, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (root / "types.json").write_text(
        json.dumps(
            {
                "param_names": job.param_names,
                "input_types": job.input_types,
                "input_categories": job.input_categories,
                "return_type": job.return_type,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


@contextmanager
def make_sandbox(job: TranspileJob, prompt: str | None = None) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="ai-udf-transpile-") as tmp:
        yield write_sandbox(Path(tmp), job, prompt)
