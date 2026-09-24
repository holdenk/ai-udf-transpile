# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ai_udf_transpile.backends.cli import CLAUDE_SPEC, CliBackend


class ClaudeBackend(CliBackend):
    def __init__(self, spark: Any = None, *, runner: Any = None):
        super().__init__(CLAUDE_SPEC, spark, runner=runner)
