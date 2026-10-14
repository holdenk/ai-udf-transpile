# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ai_udf_transpile.backends.cli import PI_SPEC, CliBackend


class PiBackend(CliBackend):
    def __init__(self, spark: Any = None, *, runner: Any = None):
        super().__init__(PI_SPEC, spark, runner=runner)
