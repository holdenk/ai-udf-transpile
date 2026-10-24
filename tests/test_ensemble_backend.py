# SPDX-License-Identifier: Apache-2.0
"""EnsembleBackend: a fallback chain, first sub-backend to answer wins."""

from __future__ import annotations

import pytest

from ai_udf_transpile.backends.base import BackendDecline, BackendError
from ai_udf_transpile.backends.ensemble import EnsembleBackend
from ai_udf_transpile.backends.fake import plus_one
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import KIND_CATALYST, TranspileJob, TranspileResult


def _job_for(func, **kwargs) -> TranspileJob:
    src = canonical_source_from_func(func)
    params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
    return TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=params,
        input_types=kwargs.get("input_types", ["bigint"] * len(params)),
        input_categories=kwargs.get("input_categories", ["numeric"] * len(params)),
        return_type=kwargs.get("return_type", "bigint"),
        captures={},
    )


class _FakeBackendA:
    name = "fake"

    def run(self, job, sandbox):
        return TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 1", model="model-a")

    def reported_model(self):
        return "model-a"


class _FakeBackendB:
    name = "fake"

    def run(self, job, sandbox):
        return TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 2", model="model-b")

    def reported_model(self):
        return "model-b"


class _NoModelBackend:
    name = "nomodel"

    def run(self, job, sandbox):
        return TranspileResult(kind=KIND_CATALYST, sql="_udf_param_0 + 3")


class _DeclineBackend:
    name = "decline"

    def run(self, job, sandbox):
        raise BackendDecline("nope")

    def reported_model(self):
        return None


def test_ensemble_first_candidate_wins_and_stops():
    calls = []
    b = _FakeBackendB()
    inner = b.run

    def counting_run(job, sandbox):
        calls.append("b")
        return inner(job, sandbox)

    b.run = counting_run
    ensemble = EnsembleBackend(backends=["a", "b"])
    ensemble._sub_backends = lambda: [("a", _FakeBackendA()), ("b", b)]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "a/model-a"  # stamped with the sub-backend that served it
    assert calls == []  # fallback chain: b never ran once a answered


def test_ensemble_stamp_fills_in_missing_model():
    ensemble = EnsembleBackend(backends=["nomodel"])
    ensemble._sub_backends = lambda: [("nomodel", _NoModelBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.model == "nomodel/unknown"


def test_ensemble_skips_decliners():
    ensemble = EnsembleBackend(backends=["decline", "a"])
    ensemble._sub_backends = lambda: [("decline", _DeclineBackend()), ("a", _FakeBackendA())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "a/model-a"


def test_ensemble_falls_back_when_first_backend_errors():
    # CoCo down (auth, rate limit): the next sub-backend answers the row.
    ensemble = EnsembleBackend(backends=["e1", "a"])
    ensemble._sub_backends = lambda: [("e1", _ErrorBackend()), ("a", _FakeBackendA())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.sql == "_udf_param_0 + 1"
    assert result.model == "a/model-a"


def test_ensemble_raises_when_all_decline():
    ensemble = EnsembleBackend(backends=["d1", "d2"])
    ensemble._sub_backends = lambda: [("d1", _DeclineBackend()), ("d2", _DeclineBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline):
            ensemble.run(job, sandbox)


class _ErrorBackend:
    name = "error"

    def run(self, job, sandbox):
        raise BackendError("rate limited")


def test_ensemble_all_errored_is_a_retryable_error_not_a_decline():
    # A transient failure on every sub-backend must not burn a 24h decline
    # cooldown on the row.
    ensemble = EnsembleBackend(backends=["e1", "e2"])
    ensemble._sub_backends = lambda: [("e1", _ErrorBackend()), ("e2", _ErrorBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendError, match="rate limited"):
            ensemble.run(job, sandbox)


def test_ensemble_decline_and_error_still_raises_error():
    # All declined + one errored: retry, because the errored sub-backend
    # never got its turn to speak -- the decline alone is not the story.
    ensemble = EnsembleBackend(backends=["d1", "e1"])
    ensemble._sub_backends = lambda: [("d1", _DeclineBackend()), ("e1", _ErrorBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendError):
            ensemble.run(job, sandbox)


def test_ensemble_unbuildable_leg_is_an_error_not_a_decline():
    # A sub-backend that cannot even be constructed never got its turn to
    # say no: that is a retryable error, not a 24h decline.
    ensemble = EnsembleBackend(backends=["not-a-backend"])
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendError, match="unavailable"):
            ensemble.run(job, sandbox)


def test_ensemble_reported_model_is_none():
    ensemble = EnsembleBackend(backends=["a"])
    assert ensemble.reported_model() is None
