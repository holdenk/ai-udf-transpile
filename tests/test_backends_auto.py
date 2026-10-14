# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from ai_udf_transpile import conf
from ai_udf_transpile.backends import BackendError, pick_auto


@pytest.fixture(autouse=True)
def _reset_conf_runtime():
    conf.reset_runtime()
    yield
    conf.reset_runtime()


def test_pick_auto_default_priority_prefers_coco(monkeypatch):
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: f"/tmp/{binary}" if binary in {"cortex", "agent"} else None,
    )
    assert pick_auto() == "coco"


def test_pick_auto_default_priority_uses_pi_before_claude_and_cursor(monkeypatch):
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: f"/tmp/{binary}" if binary in {"pi", "claude", "agent"} else None,
    )
    assert pick_auto() == "pi"


def test_pick_auto_default_priority_uses_codex_before_cursor(monkeypatch):
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: f"/tmp/{binary}" if binary in {"codex", "agent"} else None,
    )
    assert pick_auto() == "codex"


def test_pick_auto_respects_configured_priority(monkeypatch):
    conf.set_value(conf.AUTO_BACKEND_PRIORITY, "cursor,coco,claude")
    monkeypatch.setattr(
        "ai_udf_transpile.backends.shutil.which",
        lambda binary: f"/tmp/{binary}" if binary in {"cortex", "agent"} else None,
    )
    assert pick_auto() == "cursor"


def test_pick_auto_reports_unknown_priority_names(monkeypatch):
    conf.set_value(conf.AUTO_BACKEND_PRIORITY, "wut,nope")
    monkeypatch.setattr("ai_udf_transpile.backends.shutil.which", lambda _binary: None)
    with pytest.raises(BackendError) as err:
        pick_auto()
    text = str(err.value)
    assert "unknown names" in text
    assert "wut" in text and "nope" in text
