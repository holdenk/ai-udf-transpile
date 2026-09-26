# SPDX-License-Identifier: Apache-2.0
"""Configuration keys under spark.sql.experimental.aiUdfTranspile.*"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

PREFIX = "spark.sql.experimental.aiUdfTranspile."

CATALOG = PREFIX + "catalog"
SQLITE_PATH = PREFIX + "sqlitePath"
TABLE = PREFIX + "table"
BACKEND = PREFIX + "backend"
INLINE_WORKER = PREFIX + "inlineWorker"
POLL_INTERVAL = PREFIX + "pollIntervalSeconds"
CLAIM_TIMEOUT = PREFIX + "claimTimeoutSeconds"
CLI_TIMEOUT = PREFIX + "cliTimeoutSeconds"
BINARY_COCO = PREFIX + "binary.coco"
BINARY_CURSOR = PREFIX + "binary.cursor"
BINARY_CLAUDE = PREFIX + "binary.claude"
MAX_EXAMPLES = PREFIX + "maxExamples"
MAX_RETRIES = PREFIX + "maxRetries"
FAIL_COOLDOWN = PREFIX + "failCooldownSeconds"
MODEL_COCO = PREFIX + "model.coco"
MODEL_CURSOR = PREFIX + "model.cursor"
MODEL_CLAUDE = PREFIX + "model.claude"
INPUT_CATEGORIES = PREFIX + "inputCategories"
SAMPLING = PREFIX + "sampling"
MAX_SAMPLES = PREFIX + "maxSamples"
WRITEBACK_TABLE = PREFIX + "writebackTable"
WRITEBACK_FORMAT = PREFIX + "writebackFormat"
WRITEBACK_THRESHOLD = PREFIX + "writebackThreshold"

DEFAULTS: dict[str, str] = {
    CATALOG: "sqlite",
    TABLE: "default.ai_udf_transpile_cache",
    INLINE_WORKER: "true",
    POLL_INTERVAL: "2",
    CLAIM_TIMEOUT: "600",
    CLI_TIMEOUT: "300",
    BINARY_COCO: "cortex",
    BINARY_CURSOR: "agent",
    BINARY_CLAUDE: "claude",
    MAX_EXAMPLES: "20",
    MAX_RETRIES: "3",
    FAIL_COOLDOWN: "86400",
    INPUT_CATEGORIES: "numeric,string,bool,binary",
    SAMPLING: "true",
    MAX_SAMPLES: "8",
    # Empty = local SQLite only. When set, verified success rows are appended
    # to this parquet/iceberg table once writebackThreshold staged rows
    # accumulate, and lookups fall through to it.
    WRITEBACK_TABLE: "",
    WRITEBACK_FORMAT: "parquet",
    WRITEBACK_THRESHOLD: "32",
}

# Side channel so tests and the worker thread can read values even if the JVM
# rejected an unregistered SQLConf key.
_RUNTIME: dict[str, str] = {}

_TRUE = {"1", "true", "yes", "on"}


def reset_runtime() -> None:
    _RUNTIME.clear()


def default_backend() -> str:
    env = os.environ.get("AI_UDF_TRANSPILE_BACKEND")
    if env:
        return env
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "fake"
    return "auto"


def default_max_examples() -> str:
    env = os.environ.get("AI_UDF_TRANSPILE_MAX_EXAMPLES")
    if env:
        return env
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "5"
    return DEFAULTS[MAX_EXAMPLES]


def set_value(key: str, value: Any, spark: Any = None) -> None:
    text = str(value)
    _RUNTIME[key] = text
    if spark is None:
        return
    try:
        spark.conf.set(key, text)
    except Exception:
        logger.debug("spark.conf.set(%s) failed; using runtime overlay", key, exc_info=True)


def get_value(key: str, spark: Any = None, default: Optional[str] = None) -> str:
    if key in _RUNTIME:
        return _RUNTIME[key]
    fallback = default if default is not None else DEFAULTS.get(key, "")
    if spark is None:
        return fallback
    try:
        got = spark.conf.get(key, fallback)
        return fallback if got is None else str(got)
    except Exception:
        logger.debug("spark.conf.get(%s) failed", key, exc_info=True)
        return fallback


def get_bool(key: str, spark: Any = None, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return get_value(key, spark, fallback).strip().lower() in _TRUE


def get_int(key: str, spark: Any = None, default: int = 0) -> int:
    return int(get_value(key, spark, str(default)))


def get_float(key: str, spark: Any = None, default: float = 0.0) -> float:
    return float(get_value(key, spark, str(default)))


def get_csv(key: str, spark: Any = None, default: Optional[str] = None) -> set[str]:
    """Comma-separated conf value as a lowercase set."""
    raw = get_value(key, spark, default)
    return {part.strip().lower() for part in raw.split(",") if part.strip()}
