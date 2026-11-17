# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.fake import (
    FIXTURES,
    FakeBackend,
    always_decline,
    both_positive,
    greet,
    is_none_branch,
    plus_one,
)
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import make_sandbox
from ai_udf_transpile.targets import TranspileJob


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


def test_fixtures_cover_annotated_defs():
    assert canonical_source_from_func(plus_one) in FIXTURES
    assert canonical_source_from_func(is_none_branch) in FIXTURES
    assert canonical_source_from_func(both_positive) in FIXTURES
    assert canonical_source_from_func(greet) in FIXTURES
    assert canonical_source_from_func(always_decline) not in FIXTURES


def test_plus_one_sql():
    backend = FakeBackend()
    job = _job_for(plus_one)
    with make_sandbox(job) as sandbox:
        result = backend.run(job, sandbox)
        assert (Path(sandbox) / "OUT.sql").read_text().strip() == "_udf_param_0 + 1"
    assert result.kind == "catalyst"
    assert result.sql == "_udf_param_0 + 1"


def test_greet_and_branch_and_bool():
    backend = FakeBackend()
    with make_sandbox(_job_for(greet, input_types=["string"], return_type="string")) as sandbox:
        assert backend.run(_job_for(greet, input_types=["string"], return_type="string"), sandbox).sql == (
            "concat('hi ', _udf_param_0)"
        )
    with make_sandbox(_job_for(is_none_branch)) as sandbox:
        assert "IS NULL" in backend.run(_job_for(is_none_branch), sandbox).sql
    with make_sandbox(
        _job_for(both_positive, input_types=["bigint", "bigint"], return_type="boolean")
    ) as sandbox:
        result = backend.run(
            _job_for(both_positive, input_types=["bigint", "bigint"], return_type="boolean"),
            sandbox,
        )
        assert "AND" in result.sql


def test_always_decline():
    backend = FakeBackend()
    job = _job_for(always_decline)
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline):
            backend.run(job, sandbox)


def test_unknown_source_declines():
    backend = FakeBackend()
    job = TranspileJob(
        udf_key="x",
        source_text="def mystery(x: int) -> int:\n    return x * 3",
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
    )
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline):
            backend.run(job, sandbox)


def _job_with_targets(func, targets, **kwargs) -> TranspileJob:
    job = _job_for(func, **kwargs)
    job.targets = list(targets)
    return job


@pytest.mark.parametrize(
    ("targets", "kind"),
    [
        (["catalyst", "java_udf"], "catalyst"),
        (["catalyst", "java_udf", "rust"], "catalyst"),
        (["java_udf", "rust"], "rust"),
        (["java_udf"], "java_udf"),
        (["rust"], "rust"),
    ],
)
def test_one_udf_answers_every_pinned_kind(targets, kind):
    """No renamed copies: pinning targets picks which rewrite of the same UDF comes back."""
    job = _job_with_targets(plus_one, targets)
    with make_sandbox(job) as sandbox:
        assert FakeBackend().run(job, sandbox).kind == kind


def test_a_kind_with_no_fixture_is_a_decline():
    from ai_udf_transpile.backends.fake import zfill3

    job = _job_with_targets(zfill3, ["rust"], input_types=["string"], return_type="string")
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline, match="does not accept"):
            FakeBackend().run(job, sandbox)


def test_fixtures_for_orders_by_preference():
    from ai_udf_transpile.backends.fake import fixtures_for

    kinds = [r.kind for r in fixtures_for(canonical_source_from_func(plus_one))]
    assert kinds == ["catalyst", "rust", "java_udf"]
