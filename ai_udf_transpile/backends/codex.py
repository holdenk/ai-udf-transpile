# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ai_udf_transpile.backends.cli import CODEX_SPEC, CliBackend


class CodexBackend(CliBackend):
    def __init__(self, spark: Any = None, *, runner: Any = None):
        super().__init__(CODEX_SPEC, spark, runner=runner)
