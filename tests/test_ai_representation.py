# SPDX-License-Identifier: Apache-2.0
"""aiUdfTranspile.aiRepresentation: what the AI backends see. No JVM needed."""

from __future__ import annotations

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.fake import FakeBackend, plus_one
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.sandbox import disassembly_text, make_sandbox, representation, write_sandbox
from ai_udf_transpile.targets import TranspileJob


def _job() -> TranspileJob:
    src = canonical_source_from_func(plus_one)
    return TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        captures={},
    )


@pytest.fixture(autouse=True)
def _reset_representation():
    conf.reset_runtime()
    yield
    conf.reset_runtime()


def test_invalid_mode_falls_back_to_source():
    assert representation("nonsense") == "source"
    assert representation(None) == "source"
    assert representation("bytecode") == "bytecode"
    assert representation("source+bytecode") == "source+bytecode"


def test_source_mode_is_the_default():
    assert representation() == "source"


def test_disassembly_text_is_real_dis_output():
    text = disassembly_text(canonical_source_from_func(plus_one))
    assert "LOAD_FAST" in text and "BINARY_OP" in text and "(x)" in text


def test_bytecode_mode_hides_source():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = write_sandbox(Path(tmp), _job(), inputs="bytecode")
        assert not (root / "udf.py").exists()
        dis_text = (root / "disassembly.txt").read_text(encoding="utf-8")
        assert "LOAD_FAST" in dis_text
        prompt = (root / "PROMPT.md").read_text(encoding="utf-8")
        assert "return x + 1" not in prompt
        assert "not source" in prompt


def test_source_plus_bytecode_shows_both():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = write_sandbox(Path(tmp), _job(), inputs="source+bytecode")
        assert (root / "udf.py").exists()
        assert "LOAD_FAST" in (root / "disassembly.txt").read_text(encoding="utf-8")
        prompt = (root / "PROMPT.md").read_text(encoding="utf-8")
        assert "return x + 1" in prompt
        assert "LOAD_FAST" in prompt


def test_conf_drives_the_default():
    conf.set_value(conf.AI_REPRESENTATION, "bytecode")
    job = _job()
    with make_sandbox(job) as sandbox:
        assert not (sandbox / "udf.py").exists()
        assert (sandbox / "disassembly.txt").exists()


def test_fake_backend_declines_without_source():
    conf.set_value(conf.AI_REPRESENTATION, "bytecode")
    job = _job()
    with make_sandbox(job) as sandbox:
        with pytest.raises(BackendDecline, match="aiRepresentation"):
            FakeBackend().run(job, sandbox)


def test_fake_backend_still_works_in_source_mode():
    job = _job()
    with make_sandbox(job) as sandbox:
        result = FakeBackend().run(job, sandbox)
        assert result.sql.strip() == "_udf_param_0 + 1"
