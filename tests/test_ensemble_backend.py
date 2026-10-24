# SPDX-License-Identifier: Apache-2.0
"""EnsembleBackend: runs multiple sub-backends, collects candidates."""

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


class _DeclineBackend:
    name = "decline"

    def run(self, job, sandbox):
        raise BackendDecline("nope")

    def reported_model(self):
        return None


def test_ensemble_collects_candidates_and_returns_first():
    ensemble = EnsembleBackend(backends=["a", "b"])
    # Monkey-patch _sub_backends to return our fakes.
    ensemble._sub_backends = lambda: [("a", _FakeBackendA()), ("b", _FakeBackendB())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.sql == "_udf_param_0 + 1"
    assert len(ensemble.candidates) == 2
    names = [c[0] for c in ensemble.candidates]
    models = [c[1] for c in ensemble.candidates]
    assert names == ["a", "b"]
    assert models == ["model-a", "model-b"]


def test_ensemble_skips_decliners():
    ensemble = EnsembleBackend(backends=["decline", "a"])
    ensemble._sub_backends = lambda: [("decline", _DeclineBackend()), ("a", _FakeBackendA())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = ensemble.run(job, sandbox)
    assert result.sql == "_udf_param_0 + 1"
    assert len(ensemble.candidates) == 1
    assert ensemble.candidates[0][0] == "a"


def test_ensemble_raises_when_all_decline():
    ensemble = EnsembleBackend(backends=["d1", "d2"])
    ensemble._sub_backends = lambda: [("d1", _DeclineBackend()), ("d2", _DeclineBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline):
            ensemble.run(job, sandbox)
    assert ensemble.candidates == []


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
    assert ensemble.candidates == []


def test_ensemble_decline_and_error_still_raises_error():
    # All declined + one errored: retry, because the errored sub-backend
    # never got its turn to speak -- the decline alone is not the story.
    ensemble = EnsembleBackend(backends=["d1", "e1"])
    ensemble._sub_backends = lambda: [("d1", _DeclineBackend()), ("e1", _ErrorBackend())]
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendError):
            ensemble.run(job, sandbox)


def test_ensemble_reported_model_is_none():
    ensemble = EnsembleBackend(backends=["a"])
    assert ensemble.reported_model() is None
