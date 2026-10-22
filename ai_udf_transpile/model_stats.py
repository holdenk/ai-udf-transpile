# SPDX-License-Identifier: Apache-2.0
"""Per-model success/decline/fail rates from the catalog.

Queries the SQLite catalog's ``cache`` table for per-(backend, model, status)
counts so a user can see which models to prioritize.  Works in demo mode and
as a general table -- call it on any catalog with rows in it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ModelStat:
    backend: str
    model: str
    success: int = 0
    failed: int = 0
    other: int = 0
    total: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0


def collect_stats(catalog: Any) -> list[ModelStat]:
    """Group catalog rows by (backend, model) and count statuses."""
    conn = getattr(catalog, "_conn", None)
    if conn is None:
        return []
    cur = conn.execute(
        "SELECT backend, model, status, COUNT(*) FROM cache "
        "WHERE backend IS NOT NULL GROUP BY backend, model, status "
        "ORDER BY backend, model"
    )
    bucket: dict[tuple[str, str], ModelStat] = {}
    for backend, model, status, count in cur.fetchall():
        key = (backend or "?", model or "?")
        stat = bucket.get(key, ModelStat(backend=key[0], model=key[1]))
        count = int(count)
        if status == "success":
            bucket[key] = replace(stat, success=stat.success + count, total=stat.total + count)
        elif status == "failed":
            bucket[key] = replace(stat, failed=stat.failed + count, total=stat.total + count)
        else:
            # pending, running, provisional -- everything in-flight
            bucket[key] = replace(stat, other=stat.other + count, total=stat.total + count)
    return sorted(bucket.values(), key=lambda s: s.success_rate, reverse=True)


def format_stats_table(stats: list[ModelStat]) -> str:
    """Render stats as a fixed-width text table."""
    header = f"{'backend':<12} {'model':<24} {'succ':>5} {'fail':>5} {'other':>5} {'total':>5} {'rate':>6}"
    lines = [header, "-" * len(header)]
    for s in stats:
        lines.append(
            f"{s.backend:<12} {s.model:<24} {s.success:>5} {s.failed:>5} "
            f"{s.other:>5} {s.total:>5} {s.success_rate:>5.1%}"
        )
    return "\n".join(lines)
