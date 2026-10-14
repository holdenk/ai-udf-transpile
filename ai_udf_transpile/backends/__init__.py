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


def _auto_priority(spark: Any = None) -> list[str]:
    raw = conf.get_value(
        conf.AUTO_BACKEND_PRIORITY,
        spark,
        conf.default_auto_backend_priority(),
    )
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        name = part.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def pick_auto(spark: Any = None) -> str:
    binaries = {
        "coco": _binary(spark, conf.BINARY_COCO, conf.DEFAULTS[conf.BINARY_COCO]),
        "pi": _binary(spark, conf.BINARY_PI, conf.DEFAULTS[conf.BINARY_PI]),
        "claude": _binary(spark, conf.BINARY_CLAUDE, conf.DEFAULTS[conf.BINARY_CLAUDE]),
        "codex": _binary(spark, conf.BINARY_CODEX, conf.DEFAULTS[conf.BINARY_CODEX]),
        "cursor": _binary(spark, conf.BINARY_CURSOR, conf.DEFAULTS[conf.BINARY_CURSOR]),
    }
    order = _auto_priority(spark)
    candidates = [(name, binaries[name]) for name in order if name in binaries]
    for name, binary in candidates:
        if shutil.which(binary):
            return name
    unknown = [name for name in order if name not in binaries]
    tried = ", ".join(f"{name} ({binary})" for name, binary in candidates) or "<empty>"
    msg = [f"auto backend: none of these are on PATH: {tried}"]
    if unknown:
        msg.append(f"unknown names in {conf.AUTO_BACKEND_PRIORITY}: {', '.join(unknown)}")
    raise BackendError("; ".join(msg))


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
    if chosen == "pi":
        from ai_udf_transpile.backends.pi import PiBackend

        return PiBackend(spark)
    if chosen == "cursor":
        from ai_udf_transpile.backends.cursor import CursorBackend

        return CursorBackend(spark)
    if chosen == "claude":
        from ai_udf_transpile.backends.claude import ClaudeBackend

        return ClaudeBackend(spark)
    if chosen == "codex":
        from ai_udf_transpile.backends.codex import CodexBackend

        return CodexBackend(spark)
    raise BackendError(f"unknown backend {chosen!r}; expected auto, fake, coco, pi, claude, codex, or cursor")
