# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from ai_udf_transpile import conf, enable, shutdown
from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.fake import FakeBackend, always_decline, plus_one
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.worker import inline_thread_alive, poll_once, process_row


def test_process_row_fake_success(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(plus_one)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "fake")
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert row.catalyst_sql == "_udf_param_0 + 1"
    assert row.origin == "fake"


def test_process_row_hypothesis_failure_keeps_candidate(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(plus_one)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "fake")
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (False, "mismatch"))
    row = cat.get("k")
    assert row.status == "failed"
    assert "mismatch" in row.error
    assert row.catalyst_sql == "_udf_param_0 + 1"  # the rejected rewrite survives


def test_rejected_human_candidate_retries_on_backend(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(plus_one)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "human")
    cat.mark_failed(
        "k", "mismatch", origin="human", result=TranspileResult(kind="catalyst", sql="_udf_param_0 + 2")
    )
    # cooldown expires -> the row re-pends; the backend gets a fresh shot,
    # not another round of verifying the payload that already failed
    cat._conn.execute("UPDATE cache SET status = 'pending' WHERE udf_key = 'k'")
    assert cat.claim("k", "fake")
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert row.origin == "fake"
    assert row.catalyst_sql == "_udf_param_0 + 1"


def test_process_row_records_the_tolerance_it_checked(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(plus_one)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "fake")
    conf.set_value(conf.TOLERANCE, "0.01")
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert row.tolerance == 0.01


def test_process_row_fake_decline_marks_failed(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(always_decline)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "fake")
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "failed"
    assert "declined" in (row.error or "")


class _DecliningReportedBackend:
    name = "coco"

    def reported_model(self):
        return "coco-unknown"

    def run(self, job, sandbox):
        raise BackendDecline("nope")


class _DecliningDetectedModelBackend:
    name = "coco"

    def run(self, job, sandbox):
        exc = BackendDecline("nope")
        exc.model = "glm-5.3"
        raise exc


def _run_decline(backend, tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(always_decline)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert cat.claim("k", "coco")
    process_row(
        cat, cat.get("k"), backend, spark=None, verify_fn=lambda **k: (True, None)
    )
    return cat.get("k")


def test_decline_records_reported_model(tmp_path):
    row = _run_decline(_DecliningReportedBackend(), tmp_path)
    assert row.status == "failed"
    assert "declined" in (row.error or "")
    assert row.model == "coco-unknown"


def test_decline_records_detected_model(tmp_path):
    row = _run_decline(_DecliningDetectedModelBackend(), tmp_path)
    assert row.status == "failed"
    assert "declined" in (row.error or "")
    assert row.model == "glm-5.3"


def test_poll_once_claims_one(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    src = canonical_source_from_func(plus_one)
    cat.insert_pending(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )
    assert poll_once(cat, FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    assert cat.get("k").status == "success"
    assert poll_once(cat, FakeBackend(), spark=None, verify_fn=lambda **k: (True, None)) is False


def test_worker_module_help():
    from ai_udf_transpile.worker import main

    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0


@pytest.mark.spark
def test_inline_worker_false_starts_no_thread(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    assert inline_thread_alive() is False
    shutdown()
