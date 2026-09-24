# SPDX-License-Identifier: Apache-2.0
"""Backend protocol and factory."""

from __future__ import annotations

import shutil
from typing import Any, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.backends.base import Backend, BackendDecline, BackendError

__all__ = ["Backend", "BackendDecline", "BackendError", "get_backend", "pick_auto"]


def _binary(spark: Any, key: str, default: str) -> str:
    return conf.get_value(key, spark, default)


def pick_auto(spark: Any = None) -> str:
    candidates = [
        ("coco", _binary(spark, conf.BINARY_COCO, conf.DEFAULTS[conf.BINARY_COCO])),
        ("cursor", _binary(spark, conf.BINARY_CURSOR, conf.DEFAULTS[conf.BINARY_CURSOR])),
        ("claude", _binary(spark, conf.BINARY_CLAUDE, conf.DEFAULTS[conf.BINARY_CLAUDE])),
    ]
    for name, binary in candidates:
        if shutil.which(binary):
            return name
    raise BackendError("auto backend: none of cortex, agent, claude found on PATH")


def get_backend(name: Optional[str] = None, spark: Any = None) -> Backend:
    chosen = (name or conf.get_value(conf.BACKEND, spark, conf.default_backend())).strip().lower()
    if chosen == "auto":
        chosen = pick_auto(spark)
    if chosen == "fake":
        from ai_udf_transpile.backends.fake import FakeBackend

        return FakeBackend()
    if chosen == "coco":
        from ai_udf_transpile.backends.coco import CocoBackend

        return CocoBackend(spark)
    if chosen == "cursor":
        from ai_udf_transpile.backends.cursor import CursorBackend

        return CursorBackend(spark)
    if chosen == "claude":
        from ai_udf_transpile.backends.claude import ClaudeBackend

        return ClaudeBackend(spark)
    raise ValueError(f"unknown backend {chosen!r}")
