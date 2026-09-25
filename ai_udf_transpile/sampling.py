# SPDX-License-Identifier: Apache-2.0
"""Best-effort capture of real input values seen by the Python UDF.

On a cache miss the transpiler wraps the UDF's ``func`` (before PySpark
pickles it) with a sampler that records a few argument tuples per executor
process into the catalog's ``samples`` table. Purely random strings can make
a wrong rewrite look fine -- a JSON-parser UDF raises on random text and so
does the SQL side, a vacuous match -- so the worker feeds these real rows to
Hypothesis as explicit examples. Sampling is a no-op when the catalog is not
reachable from the executor (e.g. cluster mode with a driver-local sqlite
path); verification then falls back to the built-in interesting strings.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Deterministic strings always tried for string-typed params, samples or not.
BUILTIN_STRING_EXAMPLES: tuple[str, ...] = (
    "",
    " ",
    "0",
    "-1",
    "1e5",
    "null",
    "true",
    "{}",
    "[]",
    '{"a": 1}',
    "not json",
    "é",
    "ß",  # full case mapping (SS): catches ASCII-only upper rewrites
)

# Deterministic maps always tried for map<string,string>-typed params: values
# that are valid JSON, valid Python literals, both, or neither.
BUILTIN_MAP_EXAMPLES: tuple[dict, ...] = (
    {},
    {"a": "1"},
    {"a": "plain"},
    {"k": '{"x": 1}'},
    {"k": "[1, 2]"},
    {"k": "{'x': 1}"},
    {"k": "None"},
    {"k": "True"},
    {"k": "null"},
)


def _builtin_timestamps() -> tuple:
    import datetime as dt

    return (
        None,  # NULL handling is where timestamp rewrites diverge
        dt.datetime(1970, 1, 1),  # epoch zero
        dt.datetime(1960, 6, 1),  # pre-1970: negative epoch
        dt.datetime(2038, 1, 19, 3, 14, 7),  # 32-bit time_t rollover
        dt.datetime(2016, 2, 29),  # leap day
        dt.datetime(2015, 1, 1),
    )


# Deterministic timestamps always tried for timestamp-typed params.
BUILTIN_TIMESTAMP_EXAMPLES: tuple = _builtin_timestamps()

_MAX_STORED_SAMPLES = 64

_process_lock = threading.Lock()
_process_seen: dict[str, int] = {}


def _encode(value: Any) -> Any:
    import datetime as _dt

    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, _dt.datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, dict):
        # map<string,string> args are real values, tagged so they cannot be
        # confused with the __bytes__ / __repr__ marker dicts.
        return {"__map__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    return {"__repr__": repr(value)[:200]}


def decode_args(text: str) -> Optional[list]:
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if isinstance(item, dict) and "__bytes__" in item:
            try:
                out.append(bytes.fromhex(item["__bytes__"]))
            except Exception:
                return None
        elif isinstance(item, dict) and "__map__" in item:
            try:
                out.append({k: v for k, v in item["__map__"]})
            except Exception:
                return None
        elif isinstance(item, dict) and "__datetime__" in item:
            import datetime as _dt

            try:
                out.append(_dt.datetime.fromisoformat(item["__datetime__"]))
            except Exception:
                return None
        elif isinstance(item, dict):
            return None  # __repr__ placeholders are not real values
        else:
            out.append(item)
    return out


def record_samples(sqlite_path: str, udf_key: str, samples: list[list]) -> None:
    """Insert arg tuples into the samples table. Never raises."""
    if not sqlite_path or not samples:
        return
    try:
        conn = sqlite3.connect(sqlite_path, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.executemany(
                "INSERT INTO samples (udf_key, args_json, created_at) "
                "VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%S','now'))",
                [(udf_key, json.dumps([_encode(v) for v in args])) for args in samples],
            )
            conn.execute(
                "DELETE FROM samples WHERE udf_key = ? AND rowid NOT IN ("
                "SELECT rowid FROM samples WHERE udf_key = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                (udf_key, udf_key, _MAX_STORED_SAMPLES),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.debug("record_samples failed", exc_info=True)


def maybe_record(sqlite_path: str, udf_key: str, args: list, max_samples: int) -> None:
    """Record one arg tuple, throttled to ``max_samples`` per process per key.

    Module-level so the sampling wrapper can reference it by reference under
    cloudpickle; keeps the lock/dict out of the pickled closure.
    """
    try:
        if not any(v is not None for v in args):
            return
        with _process_lock:
            seen = _process_seen.get(udf_key, 0)
            if seen >= max_samples:
                return
            _process_seen[udf_key] = seen + 1
        record_samples(sqlite_path, udf_key, [args])
    except Exception:
        pass


def wrap_for_sampling(
    func: Callable[..., Any],
    udf_key: str,
    sqlite_path: str,
    max_samples: int,
) -> Callable[..., Any]:
    """Wrap ``func`` so the first ``max_samples`` non-all-None calls per process
    are recorded. The wrapper is cloudpickle-safe (references only module-level
    functions and plain data) and never changes the return value or raises."""

    def _sampling_wrapper(*args: Any) -> Any:
        maybe_record(sqlite_path, udf_key, list(args), max_samples)
        return func(*args)

    return _sampling_wrapper


def udf_instance_from_spark_stack() -> Any:
    """Best-effort: the UserDefinedFunction being constructed holds ``self``
    in a parent frame of the transpiler call."""
    import inspect as _inspect

    try:
        from pyspark.sql.udf import UserDefinedFunction
    except Exception:
        return None
    frame = _inspect.currentframe()
    try:
        frame = frame.f_back
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if isinstance(candidate, UserDefinedFunction):
                return candidate
            frame = frame.f_back
    finally:
        del frame
    return None
