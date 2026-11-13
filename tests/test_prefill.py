# SPDX-License-Identifier: Apache-2.0
"""Prefill: directory walk, per-UDF catalog flow, and the path+hash file tracker.

No JVM: SQLite catalog, the fake backend, and a stub verifier. The Spark
round trip (prefill, then the job's own UDF construction is a HIT) lives in
test_prefill_spark.py.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile import extract as extract_mod
from ai_udf_transpile.backends.base import BackendError
from ai_udf_transpile.backends.fake import FakeBackend
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.prefill import FileTracker, iter_python_files, main, prefill_paths

VERSION = "4.3.0-test"

# plus_one is a fake-backend fixture (exact canonical source); mystery is not.
JOB = textwrap.dedent(
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import LongType


    @udf(returnType=LongType())
    def plus_one(x: int) -> int:
        return x + 1


    def mystery(x: int) -> int:
        return x * 7 - 3


    mystery_udf = udf(mystery, "bigint")
    shout = udf(lambda s: s.upper())
    """
)


def _ok(**_kwargs):
    return True, None


class _CountingBackend:
    def __init__(self, inner=None):
        self.inner = inner or FakeBackend()
        self.name = self.inner.name
        self.calls = 0

    def run(self, job, sandbox):
        self.calls += 1
        return self.inner.run(job, sandbox)


class _FlakyBackend:
    """A CLI that is down: every run is a transient BackendError."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def run(self, job, sandbox):
        self.calls += 1
        raise BackendError("rate limited")


@pytest.fixture
def catalog(sqlite_path):
    cat = SqliteCatalog(sqlite_path)
    yield cat
    cat.close()


@pytest.fixture
def tracker(sqlite_path):
    t = FileTracker(sqlite_path)
    yield t
    t.close()


def _run(paths, catalog, tracker, backend, **kwargs):
    return prefill_paths(
        [str(p) for p in paths],
        catalog=catalog,
        backend=backend,
        spark=None,
        spark_version=VERSION,
        tracker=tracker,
        verify_fn=_ok,
        log=lambda _line: None,
        **kwargs,
    )


def _jobs(tmp_path) -> Path:
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "job.py").write_text(JOB)
    return root


def test_first_run_transpiles_and_records(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend)
    assert report.files_scanned == 1
    assert report.udfs_found == 2 and report.udfs_skipped == 1  # the lambda
    by_name = {o.name: o for o in report.outcomes}
    assert by_name["plus_one"].outcome == "transpiled"
    assert by_name["mystery"].outcome == "failed"  # fake declines: settled, not retried
    assert backend.calls == 2
    row = catalog.get(by_name["plus_one"].udf_key)
    assert row.status == "success" and row.catalyst_sql == "_udf_param_0 + 1"
    assert "@" not in row.source_text


def test_unchanged_file_is_not_reread(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend())
    backend = _CountingBackend()
    with mock.patch("ai_udf_transpile.prefill.extract_file", wraps=extract_mod.extract_file) as spy:
        report = _run([root], catalog, tracker, backend)
    assert report.files_unchanged == 1 and report.files_scanned == 0
    assert report.outcomes == []
    assert backend.calls == 0
    assert spy.call_count == 1  # hashed, not re-processed


def test_same_content_new_path_is_deduped(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend())
    copy = tmp_path / "elsewhere" / "copy_of_job.py"
    copy.parent.mkdir()
    shutil.copy(root / "job.py", copy)
    backend = _CountingBackend()
    report = _run([copy], catalog, tracker, backend)
    assert report.files_unchanged == 1 and backend.calls == 0
    # ...and the new path is now tracked in its own right.
    assert tracker.done(str(copy.resolve()), report_sha(copy), VERSION)["path"] == str(copy.resolve())


def report_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_edited_file_rescans_but_known_udfs_are_hits(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend())
    (root / "job.py").write_text(JOB + "\n# a comment changes the hash, not the keys\n")
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend)
    assert report.files_scanned == 1
    outcomes = {o.name: o.outcome for o in report.outcomes}
    assert outcomes == {"plus_one": "cached", "mystery": "previously-failed"}
    assert backend.calls == 0


def test_transient_backend_errors_leave_the_file_incomplete(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    flaky = _FlakyBackend()
    report = _run([root], catalog, tracker, flaky)
    assert {o.outcome for o in report.outcomes} == {"failed"}
    assert flaky.calls == 2
    sha = report_sha(root / "job.py")
    assert tracker.done(str((root / "job.py").resolve()), sha, VERSION) is None
    # Next run (backend back up) retries: infra failures skip the cooldown.
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend)
    assert {o.name: o.outcome for o in report.outcomes}["plus_one"] == "transpiled"
    assert backend.calls == 2
    assert tracker.done(str((root / "job.py").resolve()), sha, VERSION) is not None


def test_queue_only_leaves_pending_rows_for_a_worker(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend, queue_only=True)
    assert {o.outcome for o in report.outcomes} == {"queued"}
    assert backend.calls == 0
    assert catalog.oldest_pending() is not None
    # Not complete until a worker settles them: the next run looks again.
    assert _run([root], catalog, tracker, backend, queue_only=True).files_scanned == 1


def test_rescan_ignores_the_tracker(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend())
    report = _run([root], catalog, tracker, _CountingBackend(), rescan=True)
    assert report.files_scanned == 1
    assert {o.name: o.outcome for o in report.outcomes}["plus_one"] == "cached"


def test_spark_version_is_part_of_the_file_identity(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend())
    report = prefill_paths(
        [str(root)],
        catalog=catalog,
        backend=_CountingBackend(),
        spark=None,
        spark_version="9.9.9",
        tracker=tracker,
        verify_fn=_ok,
        log=lambda _line: None,
    )
    assert report.files_scanned == 1


def test_input_category_gate_matches_runtime(tmp_path, catalog, tracker):
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "ts.py").write_text(
        textwrap.dedent(
            """
            from datetime import datetime
            from pyspark.sql.functions import udf

            @udf("string")
            def to_iso_date(ts: datetime) -> str:
                return ts.strftime("%Y-%m-%d")
            """
        )
    )
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend)
    assert report.udfs_gated == 1 and report.outcomes == [] and backend.calls == 0
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,timestamp")
    # No --rescan: a gated UDF keeps the file incomplete, so opting in is enough.
    report = _run([root], catalog, tracker, backend)
    assert [o.outcome for o in report.outcomes] == ["transpiled"]  # a fake fixture


def test_unparseable_file_is_recorded_once(tmp_path, catalog, tracker):
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "py2.py").write_text("print 'hello'\n")
    report = _run([root], catalog, tracker, _CountingBackend())
    assert report.files_errored == 1
    report = _run([root], catalog, tracker, _CountingBackend())
    assert report.files_errored == 0 and report.files_unchanged == 1


def test_walk_skips_hidden_and_venv_dirs(tmp_path):
    root = tmp_path / "repo"
    for rel in ["a.py", "pkg/b.py", ".git/c.py", "venv/d.py", "pkg/__pycache__/e.py", "notes.txt"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    found = sorted(p.relative_to(root.resolve()).as_posix() for p in iter_python_files([str(root)]))
    assert found == ["a.py", "pkg/b.py"]
    found = [p.name for p in iter_python_files([str(root)], exclude=["*/pkg/*"])]
    assert found == ["a.py"]


def test_main_dry_run(tmp_path, capsys):
    root = _jobs(tmp_path)
    assert main(["--dry-run", "--spark-version", VERSION, str(root)]) == 0
    out = capsys.readouterr().out
    assert "plus_one(x: bigint) -> bigint" in out
    assert "skip <lambda>: lambda UDFs carry no type annotations" in out
    assert "2 UDF(s) extracted, 1 skipped" in out


def test_main_requires_sqlite_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_UDF_TRANSPILE_SQLITE", raising=False)
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


def test_interrupted_claim_is_reclaimed_and_resumed(tmp_path, catalog, tracker):
    root = _jobs(tmp_path)
    _run([root], catalog, tracker, _CountingBackend(), queue_only=True)
    key = catalog.oldest_pending().udf_key
    # A previous run claimed this row and was killed before finishing it.
    assert catalog.claim(key, "fake")
    catalog._conn.execute("UPDATE cache SET claimed_at = '2000-01-01T00:00:00' WHERE udf_key = ?", (key,))
    report = _run([root], catalog, tracker, _CountingBackend())
    outcomes = {o.udf_key: o.outcome for o in report.outcomes}
    assert outcomes[key] in {"transpiled", "failed"}, outcomes  # processed, not left in-flight
    assert catalog.get(key).status in {"success", "failed"}


def test_main_queue_only_needs_no_backend(tmp_path, sqlite_path, monkeypatch, capsys):
    import ai_udf_transpile
    import ai_udf_transpile.backends
    from ai_udf_transpile import prefill

    class _Spark:
        version = VERSION

    def _no_backend(*_a, **_k):
        raise BackendError("auto backend: none of these are on PATH")

    monkeypatch.setattr(prefill, "_spark_session", lambda master, confs: _Spark())
    monkeypatch.setattr(ai_udf_transpile, "enable", lambda spark, **kw: SqliteCatalog(kw["sqlite_path"]))
    monkeypatch.setattr(ai_udf_transpile.backends, "get_backend", _no_backend)
    root = _jobs(tmp_path)
    assert main([str(root), "--sqlite-path", sqlite_path, "--queue-only"]) == 0
    assert "2 queued" in capsys.readouterr().out


def test_rust_row_this_session_cannot_serve_routes_to_the_twin(tmp_path, catalog, tracker, monkeypatch):
    """Same routing as the hook: a served Rust row a non-Comet session cannot
    attach sends the UDF to its +no-rust twin, which gets its own rewrite."""
    from ai_udf_transpile.extract import extract_file
    from ai_udf_transpile.targets import NO_RUST_SUFFIX, TranspileResult

    root = _jobs(tmp_path)
    plus_one = next(u for u in extract_file(root / "job.py").udfs if u.name == "plus_one")
    primary = plus_one.key(VERSION)
    catalog.upsert_success(
        udf_key=primary,
        source_text=plus_one.source_text,
        param_names=plus_one.param_names,
        input_types=plus_one.input_types,
        input_categories=plus_one.input_categories,
        return_type=plus_one.return_type,
        spark_version=VERSION,
        closure_fingerprint=plus_one.closure_fingerprint,
        captures=plus_one.captures,
        result=TranspileResult(kind="rust", rust_source="fn f(x: i64) -> i64 { x + 1 }"),
        origin="fake",
        hypothesis_passed=True,
    )
    monkeypatch.setattr("ai_udf_transpile.rustc.rust_unservable", lambda spark: "no Comet here")
    report = _run([root], catalog, tracker, _CountingBackend())
    result = next(o for o in report.outcomes if o.name == "plus_one")
    twin = plus_one.key(VERSION + NO_RUST_SUFFIX)
    assert result.udf_key == twin and result.outcome == "transpiled"
    assert catalog.get(twin).catalyst_sql == "_udf_param_0 + 1"
    assert catalog.get(primary).target_kind == "rust"  # untouched


def test_finished_file_is_revisited_when_its_rows_no_longer_serve(tmp_path, catalog, tracker):
    """A stricter tolerance makes lookup re-queue loosely verified rows."""
    root = _jobs(tmp_path)
    conf.set_value(conf.TOLERANCE, "0.1")
    _run([root], catalog, tracker, _CountingBackend())
    conf.set_value(conf.TOLERANCE, "0")
    backend = _CountingBackend()
    report = _run([root], catalog, tracker, backend)
    assert report.files_unchanged == 0 and report.files_scanned == 1
    assert {o.name: o.outcome for o in report.outcomes}["plus_one"] == "transpiled"
    assert catalog.get(next(o.udf_key for o in report.outcomes if o.name == "plus_one")).tolerance == 0
