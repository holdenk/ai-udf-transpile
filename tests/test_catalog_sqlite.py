# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile.catalog import HIT, MISS, WAIT
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.targets import TranspileResult


def _pending(cat: SqliteCatalog, key: str = "k1") -> None:
    cat.insert_pending(
        udf_key=key,
        source_text="def f(x: int) -> int:\n    return x + 1",
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )


def test_insert_pending_rejects_missing_types(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    with pytest.raises(ValueError):
        cat.insert_pending(
            udf_key="k",
            source_text="def f(x): return x",
            param_names=["x"],
            input_types=[],
            input_categories=[],
            return_type="",
            spark_version="t",
            closure_fingerprint="",
            captures={},
        )


def test_cas_two_threads_one_winner(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    barrier = threading.Barrier(2)
    winners: list[bool] = []
    lock = threading.Lock()

    def claim() -> None:
        barrier.wait()
        won = cat.claim("k1", "fake")
        with lock:
            winners.append(won)

    t1 = threading.Thread(target=claim)
    t2 = threading.Thread(target=claim)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert winners.count(True) == 1
    assert winners.count(False) == 1
    row = cat.get("k1")
    assert row is not None
    assert row.status == "running"
    assert row.attempt_count == 1


def test_failed_cooldown_does_not_requeue(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_failed("k1", "nope")
    kind, row = cat.lookup("k1")
    assert kind == WAIT
    assert row is not None
    assert row.status == "failed"
    # Second lookup still wait (no extra pending insert from lookup)
    kind2, _ = cat.lookup("k1")
    assert kind2 == WAIT


def test_failed_after_cooldown_requeues_when_retries_remain(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_failed("k1", "nope")
    # Pretend the failure was yesterday and we still have retry budget.
    old = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    cat._conn.execute("UPDATE cache SET failed_at = ?, attempt_count = 1 WHERE udf_key = 'k1'", (old,))
    kind, row = cat.lookup("k1")
    assert kind == WAIT
    assert row is not None
    assert row.status == "pending"


def test_max_retries_exhausted_stays_failed(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_failed("k1", "nope")
    old = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    max_retries = int(conf.DEFAULTS[conf.MAX_RETRIES])
    cat._conn.execute(
        "UPDATE cache SET failed_at = ?, attempt_count = ? WHERE udf_key = 'k1'",
        (old, max_retries),
    )
    kind, row = cat.lookup("k1")
    assert kind == WAIT
    assert row is not None
    assert row.status == "failed"


def test_success_is_a_hit(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_success(
        "k1",
        TranspileResult(kind="catalyst", sql="_udf_param_0 + 1"),
        origin="fake",
    )
    kind, row = cat.lookup("k1")
    assert kind == HIT
    assert row is not None
    assert row.catalyst_sql == "_udf_param_0 + 1"


def test_miss_on_empty(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    kind, row = cat.lookup("missing")
    assert kind == MISS
    assert row is None


def test_model_recorded_on_success(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "claude")
    cat.mark_success(
        "k1",
        TranspileResult(kind="catalyst", sql="_udf_param_0 + 1", model="claude-x-1"),
        origin="claude",
    )
    row = cat.get("k1")
    assert row is not None
    assert row.model == "claude-x-1"
    assert row.origin == "claude"
    assert row.backend == "claude"


def test_failure_records_origin_and_model(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "coco")
    cat.mark_failed("k1", "hypothesis failed: mismatch", origin="coco", model="snow-x")
    row = cat.get("k1")
    assert row is not None
    assert row.status == "failed"
    assert row.error == "hypothesis failed: mismatch"
    assert row.origin == "coco"
    assert row.model == "snow-x"
    assert row.failed_at is not None


def test_mark_failed_without_origin_keeps_existing(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "cursor")
    cat.mark_failed("k1", "first", origin="cursor", model="m1")
    cat.mark_failed("k1", "second")
    row = cat.get("k1")
    assert row is not None
    assert row.error == "second"
    assert row.origin == "cursor"  # COALESCE: not overwritten by NULL
    assert row.model == "m1"


def test_model_column_added_to_existing_db(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE cache (udf_key TEXT PRIMARY KEY, status TEXT NOT NULL, "
        "return_type TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO cache (udf_key, status, return_type) VALUES ('old', 'success', 'bigint')")
    conn.commit()
    conn.close()
    cat = SqliteCatalog(path)  # migration adds model
    cols = {r[1] for r in cat._conn.execute("PRAGMA table_info(cache)")}
    assert "model" in cols
    assert "publish_ready" in cols
    ready = cat._conn.execute("SELECT publish_ready FROM cache WHERE udf_key = 'old'").fetchone()
    assert ready is not None and ready[0] == 1
    tol = cat._conn.execute("SELECT tolerance FROM cache WHERE udf_key = 'old'").fetchone()
    assert tol is not None and tol[0] is None


def test_missing_tolerance_is_an_exact_check(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_success(
        "k1",
        TranspileResult(kind="catalyst", sql="_udf_param_0 + 1"),
        origin="fake",
    )
    assert cat.get("k1").tolerance is None
    assert cat.lookup("k1")[0] == HIT


def test_looser_than_configured_tolerance_is_rechecked(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_success(
        "k1",
        TranspileResult(kind="catalyst", sql="_udf_param_0 + 1"),
        origin="fake",
        tolerance=1.0,
    )
    assert cat.get("k1").tolerance == 1.0
    conf.set_value(conf.TOLERANCE, "1")
    assert cat.lookup("k1")[0] == HIT
    conf.set_value(conf.TOLERANCE, "0.5")
    kind, row = cat.lookup("k1")
    assert kind == WAIT
    assert row is not None and row.status == "pending"


def test_provisional_is_hidden_until_promoted(tmp_path):
    import threading

    from ai_udf_transpile.catalog import expose_provisional

    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k1", "fake")
    cat.mark_success(
        "k1",
        TranspileResult(kind="catalyst", sql="_udf_param_0 + 1"),
        origin="fake",
        visible=False,
    )
    kind, row = cat.lookup("k1")
    assert kind == WAIT
    assert row is not None and row.status == "provisional"
    seen: dict[str, str] = {}

    def other() -> None:
        seen["kind"] = cat.lookup("k1")[0]

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert seen["kind"] == WAIT
    with expose_provisional():
        assert cat.lookup("k1")[0] == HIT
    assert cat.staged_successes() == []
    cat.promote_success("k1")
    assert cat.lookup("k1")[0] == HIT
    assert cat.staged_successes() == []
    cat.mark_publish_ready("k1")
    assert [r.udf_key for r in cat.staged_successes()] == ["k1"]
