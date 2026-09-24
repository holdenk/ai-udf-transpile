# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ai_udf_transpile.backends.cli import CURSOR_SPEC, CliBackend


class CursorBackend(CliBackend):
    def __init__(self, spark: Any = None, *, runner: Any = None):
        super().__init__(CURSOR_SPEC, spark, runner=runner)
