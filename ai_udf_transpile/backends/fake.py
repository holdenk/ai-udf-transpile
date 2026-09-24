# SPDX-License-Identifier: Apache-2.0
"""Deterministic backend for CI: exact canonical source → TranspileResult."""

from __future__ import annotations

from pathlib import Path

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.keys import canonical_source_from_func, canonical_source_text
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob, TranspileResult


def plus_one(x: int) -> int:
    return x + 1


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def greet(name: str) -> str:
    return "hi " + name


def always_decline(x: int) -> int:
    import os

    return len(os.getcwd())


def _result(sql: str) -> TranspileResult:
    return TranspileResult(kind=KIND_CATALYST, sql=sql)


FIXTURES: dict[str, TranspileResult] = {
    canonical_source_from_func(plus_one): _result("_udf_param_0 + 1"),
    canonical_source_from_func(is_none_branch): _result(
        "CASE WHEN _udf_param_0 IS NULL THEN -1 ELSE _udf_param_0 END"
    ),
    canonical_source_from_func(both_positive): _result("_udf_param_0 > 0 AND _udf_param_1 > 0"),
    canonical_source_from_func(greet): _result("concat('hi ', _udf_param_0)"),
}


class FakeBackend:
    name = "fake"

    def run(self, job: TranspileJob, sandbox: Path) -> TranspileResult:
        try:
            key = canonical_source_text(job.source_text)
        except Exception:
            key = (job.source_text or "").strip()
        result = FIXTURES.get(key)
        if result is None:
            raise BackendDecline(f"no fake fixture for source {key!r}")
        if result.sql:
            (sandbox / "OUT.sql").write_text(result.sql + "\n", encoding="utf-8")
        return result
