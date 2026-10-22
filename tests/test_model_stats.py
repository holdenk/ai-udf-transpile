# SPDX-License-Identifier: Apache-2.0
"""model_stats: collect_stats and format_stats_table."""

from __future__ import annotations

import sqlite3

from ai_udf_transpile.model_stats import ModelStat, collect_stats, format_stats_table


class _CatalogShim:
    def __init__(self, conn):
        self._conn = conn


def _make_catalog(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE cache (backend TEXT, model TEXT, status TEXT)")
    conn.executemany("INSERT INTO cache VALUES (?,?,?)", rows)
    return _CatalogShim(conn)


def test_collect_stats_groups_by_backend_model():
    catalog = _make_catalog(
        [
            ("claude", "sonnet", "success"),
            ("claude", "sonnet", "success"),
            ("claude", "sonnet", "failed"),
            ("codex", "codex-1", "success"),
            ("codex", "codex-1", "failed"),
            ("codex", "codex-1", "failed"),
        ]
    )
    stats = collect_stats(catalog)
    assert len(stats) == 2
    # Sorted by success_rate descending: claude (2/3) > codex (1/3)
    assert stats[0].backend == "claude"
    assert stats[0].success == 2
    assert stats[0].failed == 1
    assert stats[0].total == 3
    assert abs(stats[0].success_rate - 2 / 3) < 1e-9
    assert stats[1].backend == "codex"
    assert stats[1].success == 1
    assert stats[1].failed == 2
    assert stats[1].total == 3


def test_collect_stats_counts_other_statuses():
    catalog = _make_catalog(
        [
            ("claude", "sonnet", "success"),
            ("claude", "sonnet", "pending"),
            ("claude", "sonnet", "running"),
            ("claude", "sonnet", "provisional"),
        ]
    )
    stats = collect_stats(catalog)
    assert len(stats) == 1
    assert stats[0].success == 1
    assert stats[0].failed == 0
    assert stats[0].other == 3  # pending + running + provisional
    assert stats[0].total == 4


def test_collect_stats_empty():
    catalog = _make_catalog([])
    assert collect_stats(catalog) == []


def test_collect_stats_no_conn():
    assert collect_stats(object()) == []


def test_format_stats_table_renders():
    stats = [
        ModelStat(backend="claude", model="sonnet", success=3, failed=1, total=4),
        ModelStat(backend="codex", model="codex-1", success=1, failed=2, total=3),
    ]
    table = format_stats_table(stats)
    assert "backend" in table
    assert "claude" in table
    assert "codex" in table
    assert "75.0%" in table
    lines = table.strip().split("\n")
    # header + separator + 2 data rows
    assert len(lines) == 4
