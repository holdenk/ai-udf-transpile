# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from ai_udf_transpile.backends.cli import COCO_SPEC, CliBackend


class CocoBackend(CliBackend):
    def __init__(self, spark: Any = None, *, runner: Any = None):
        super().__init__(COCO_SPEC, spark, runner=runner)
