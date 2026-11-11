# SPDX-License-Identifier: Apache-2.0
"""requireApproval: AI rewrites park in awaiting_approval until a person signs off."""

from __future__ import annotations

import io

import pytest

from ai_udf_transpile import approve, conf
from ai_udf_transpile.approval import (
    approval_reasons,
    network_modules_used,
    parse_triggers,
)
from ai_udf_transpile.backends.fake import FakeBackend, greet, kms_encrypt, plus_one
from ai_udf_transpile.catalog import AWAITING_APPROVAL, HIT, WAIT
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.catalog.writeback import WritebackCatalog
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.worker import process_row


@pytest.fixture(autouse=True)
def _reset_conf():
    conf.reset_runtime()
    yield
    conf.reset_runtime()


def _pending(cat, key, func, input_types, return_type):
    cat.insert_pending(
        udf_key=key,
        source_text=canonical_source_from_func(func),
        param_names=[f"p{i}" for i in range(len(input_types))],
        input_types=input_types,
        input_categories=["string" if t == "string" else "numeric" for t in input_types],
        return_type=return_type,
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )


def _run(cat, key, backend=None):
    assert cat.claim(key, "fake")
    process_row(cat, cat.get(key), backend or FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    return cat.get(key)


# -- policy ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", set()),
        ("none", set()),
        ("false", set()),
        ("all", {"all"}),
        ("TRUE", {"all"}),
        ("string, network", {"string", "network"}),
        ("hard", {"string", "struct", "map", "array", "network"}),
        ("string,bogus", {"string"}),
    ],
)
def test_parse_triggers(raw, expected):
    assert parse_triggers(raw) == expected


def test_network_detection():
    assert network_modules_used(canonical_source_from_func(kms_encrypt)) == ["boto3"]
    assert network_modules_used(canonical_source_from_func(plus_one)) == []
    src = "def f(u: str) -> str:\n    from urllib.request import urlopen\n    return urlopen(u).read()\n"
    assert network_modules_used(src) == ["urllib.request", "urlopen"]
    assert network_modules_used("def f(x: str) -> str:\n    return requests.get(x).text\n") == ["requests"]
    # Captured module (keys.extract_captures stores stdlib modules this way).
    assert network_modules_used("def f(x): return x", {"s": {"__module__": "socket"}}) == ["socket"]


def test_approval_reasons_by_type_and_origin():
    hard = parse_triggers("hard")
    assert approval_reasons(input_types=["bigint"], return_type="bigint", triggers=hard) == []
    assert approval_reasons(input_types=["string"], return_type="bigint", triggers=hard) == ["string type"]
    assert approval_reasons(input_types=["bigint"], return_type="struct<a:int>", triggers=hard) == [
        "struct type"
    ]
    assert approval_reasons(
        input_types=["map<string,string>"], return_type="bigint", triggers=parse_triggers("map")
    ) == ["map type"]
    # Not AI output: never gated.
    assert approval_reasons(input_types=["string"], return_type="string", origin="human", triggers=hard) == []
    assert (
        approval_reasons(
            input_types=["bigint"], return_type="bigint", origin="bytecode", triggers=parse_triggers("all")
        )
        == []
    )
    reasons = approval_reasons(
        input_types=["string", "string"],
        return_type="string",
        source_text=canonical_source_from_func(kms_encrypt),
        triggers=parse_triggers("network"),
    )
    assert reasons == ["network call (boto3)"]


def test_conf_default_is_off():
    assert approval_reasons(input_types=["string"], return_type="string") == []


# -- worker + catalog ------------------------------------------------------


def test_worker_parks_string_udf_and_approve_serves_it(tmp_path):
    conf.set_value(conf.REQUIRE_APPROVAL, "string")
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", greet, ["string"], "string")
    row = _run(cat, "k")
    assert row.status == AWAITING_APPROVAL
    assert row.catalyst_sql  # the rewrite is kept for the reviewer
    assert cat.lookup("k")[0] == WAIT
    assert cat.staged_successes() == []  # never published while parked
    # Not reclaimed as a stale claim.
    assert cat.reclaim_stale(0, 3) == 0
    assert cat.get("k").status == AWAITING_APPROVAL

    assert [r.udf_key for r in cat.awaiting_approval()] == ["k"]
    assert cat.approve("k", "alice")
    kind, row = cat.lookup("k")
    assert kind == HIT
    assert (row.reviewed_by, row.status) == ("alice", "success")
    assert row.reviewed_at
    assert [r.udf_key for r in cat.staged_successes()] == ["k"]
    assert not cat.approve("k", "alice"), "second approve is a no-op"


def test_worker_does_not_park_ungated_types(tmp_path):
    conf.set_value(conf.REQUIRE_APPROVAL, "string,network")
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", plus_one, ["bigint"], "bigint")
    assert _run(cat, "k").status == "success"


def test_reject_fails_with_reason_and_keeps_candidate(tmp_path):
    conf.set_value(conf.REQUIRE_APPROVAL, "all")
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", plus_one, ["bigint"], "bigint")
    _run(cat, "k")
    assert cat.reject("k", "bob", "off by one on negatives")
    row = cat.get("k")
    assert row.status == "failed"
    assert row.error == "rejected by reviewer bob: off by one on negatives"
    assert row.catalyst_sql == "_udf_param_0 + 1"
    assert row.reviewed_by == "bob"
    assert cat.lookup("k")[0] == WAIT  # cooldown, like any failure


def test_new_payload_clears_review(tmp_path):
    conf.set_value(conf.REQUIRE_APPROVAL, "all")
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", plus_one, ["bigint"], "bigint")
    _run(cat, "k")
    cat.approve("k", "alice")
    cat.mark_success("k", TranspileResult(kind="catalyst", sql="1 + _udf_param_0"), "fake")
    row = cat.get("k")
    assert row.reviewed_at is None and row.reviewed_by is None
    assert cat.lookup("k")[0] == WAIT, "a fresh payload needs a fresh review"
    assert cat.get("k").status == AWAITING_APPROVAL


def test_turning_approval_on_parks_existing_unreviewed_success(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", greet, ["string"], "string")
    assert _run(cat, "k").status == "success"
    assert cat.lookup("k")[0] == HIT
    conf.set_value(conf.REQUIRE_APPROVAL, "string")
    assert cat.lookup("k")[0] == WAIT
    assert cat.get("k").status == AWAITING_APPROVAL
    cat.approve("k", "alice")
    assert cat.lookup("k")[0] == HIT, "an approved row stays served"


def test_human_rows_are_never_parked(tmp_path):
    conf.set_value(conf.REQUIRE_APPROVAL, "all")
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, "k", greet, ["string"], "string")
    cat.mark_success("k", TranspileResult(kind="catalyst", sql="concat('hi ', _udf_param_0)"), "human")
    assert cat.lookup("k")[0] == HIT


def test_schema_migration_adds_review_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE cache (udf_key TEXT PRIMARY KEY, return_type TEXT NOT NULL, "
        "status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.commit()
    conn.close()
    SqliteCatalog(path).close()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(cache)")}
    assert {"reviewed_by", "reviewed_at"} <= cols


# -- write-back: an unreviewed shared row is not served ----------------------


def _bare_writeback(staging, remote):
    wb = WritebackCatalog.__new__(WritebackCatalog)
    wb._staging = staging
    wb._available = True
    wb._remote_success = lambda key: remote if remote and remote.udf_key == key else None
    wb._maybe_flush = lambda: None
    return wb


def test_writeback_unreviewed_remote_is_parked_locally(tmp_path):
    writer = SqliteCatalog(tmp_path / "w.sqlite")
    _pending(writer, "k", greet, ["string"], "string")
    remote = _run(writer, "k")  # writer runs without requireApproval
    assert remote.status == "success" and remote.reviewed_at is None

    conf.set_value(conf.REQUIRE_APPROVAL, "string")
    reader = _bare_writeback(SqliteCatalog(tmp_path / "r.sqlite"), remote)
    kind, row = reader.lookup("k")
    assert kind == WAIT
    assert row.status == AWAITING_APPROVAL
    assert reader.approve("k", "alice")
    local = reader._staging.get("k")
    assert local.status == "success" and local.written_back_at is None, "approval is re-published"
    assert reader.lookup("k")[0] == HIT


def test_writeback_reviewed_remote_is_served(tmp_path):
    writer = SqliteCatalog(tmp_path / "w.sqlite")
    conf.set_value(conf.REQUIRE_APPROVAL, "string")
    _pending(writer, "k", greet, ["string"], "string")
    _run(writer, "k")
    writer.approve("k", "alice")
    remote = writer.get("k")
    reader = _bare_writeback(SqliteCatalog(tmp_path / "r.sqlite"), remote)
    kind, row = reader.lookup("k")
    assert kind == HIT
    assert reader._staging.get("k").reviewed_by == "alice"


# -- the review script -------------------------------------------------------


def _parked(tmp_path, n=2):
    conf.set_value(conf.REQUIRE_APPROVAL, "all")
    path = tmp_path / "c.sqlite"
    cat = SqliteCatalog(path)
    keys = []
    for i in range(n):
        key = f"{i}abc{i}"
        _pending(cat, key, greet, ["string"], "string")
        _run(cat, key)
        keys.append(key)
    cat.close()
    return str(path), keys


def test_cli_list(tmp_path):
    path, keys = _parked(tmp_path)
    out = io.StringIO()
    assert approve.main(["--sqlite-path", path, "--list"], out=out) == 0
    text = out.getvalue()
    for key in keys:
        assert key in text
    assert "string type" in text
    assert "concat" in text.lower() or "_udf_param_0" in text


def test_cli_approve_and_reject_by_prefix(tmp_path):
    path, keys = _parked(tmp_path)
    out = io.StringIO()
    assert approve.main(["--sqlite-path", path, "--approve", "0abc", "--reviewer", "alice"], out=out) == 0
    assert (
        approve.main(
            ["--sqlite-path", path, "--reject", "1a", "--reason", "nope", "--reviewer", "bob"], out=out
        )
        == 0
    )
    assert approve.main(["--sqlite-path", path, "--approve", "zzz"], out=out) == 1
    cat = SqliteCatalog(path)
    assert cat.get(keys[0]).reviewed_by == "alice" and cat.get(keys[0]).status == "success"
    assert cat.get(keys[1]).status == "failed" and "nope" in cat.get(keys[1]).error


def test_interactive_review(tmp_path):
    path, keys = _parked(tmp_path, n=3)
    cat = SqliteCatalog(path)
    answers = iter(["?", "a", "r", "looks wrong", "s"])
    out = io.StringIO()
    counts = approve.review(cat, reviewer="carol", input_fn=lambda _: next(answers), out=out)
    assert counts == {"approved": 1, "rejected": 1, "skipped": 1}
    assert "please answer" in out.getvalue()
    assert cat.get(keys[0]).status == "success"
    assert cat.get(keys[1]).error == "rejected by reviewer carol: looks wrong"
    assert cat.get(keys[2]).status == AWAITING_APPROVAL


def test_interactive_quit_and_eof(tmp_path):
    path, keys = _parked(tmp_path, n=2)
    cat = SqliteCatalog(path)

    def eof(_):
        raise EOFError

    counts = approve.review(cat, reviewer="d", input_fn=eof, out=io.StringIO())
    assert counts == {"approved": 0, "rejected": 0, "skipped": 2}
    assert len(cat.awaiting_approval()) == 2


def test_resolve_ambiguous_prefix(tmp_path):
    path, _ = _parked(tmp_path)
    rows = SqliteCatalog(path).awaiting_approval()
    with pytest.raises(KeyError, match="ambiguous"):
        approve.resolve(rows, "")


def test_browse_is_read_only(tmp_path):
    path, keys = _parked(tmp_path, n=3)
    cat = SqliteCatalog(path)
    answers = iter(["n", "p", "p", "huh", "3", "q"])
    out = io.StringIO()
    shown = approve.browse(cat, input_fn=lambda _: next(answers), out=out)
    text = out.getvalue()
    assert shown == 3
    assert "read-only" in text
    assert "already at the first row" in text
    assert "please answer n, p" in text
    assert "[3/3]" in text
    assert [r.udf_key for r in cat.awaiting_approval()] == keys  # nothing changed
    assert all(cat.get(k).reviewed_at is None for k in keys)


def test_browse_ends_at_last_row_and_cli(tmp_path, monkeypatch):
    path, _ = _parked(tmp_path, n=2)
    cat = SqliteCatalog(path)
    answers = iter(["", ""])
    out = io.StringIO()
    assert approve.browse(cat, input_fn=lambda _: next(answers), out=out) == 2
    assert "end of list" in out.getvalue()
    cat.close()

    monkeypatch.setattr("sys.stdin", io.StringIO("2\nq\n"))
    out = io.StringIO()
    assert approve.main(["--sqlite-path", path, "--browse"], out=out) == 0
    assert "[1/2]" in out.getvalue() and "[2/2]" in out.getvalue()
    assert len(SqliteCatalog(path).awaiting_approval()) == 2


def test_browse_empty(tmp_path):
    out = io.StringIO()
    assert approve.browse(SqliteCatalog(tmp_path / "e.sqlite"), input_fn=lambda _: "q", out=out) == 0
    assert "nothing awaiting approval" in out.getvalue()
