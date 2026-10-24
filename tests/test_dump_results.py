# SPDX-License-Identifier: Apache-2.0
"""dump_results.py writes the what-transpiles-to-what markdown."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.spark

DUMP = Path(__file__).resolve().parents[1] / "examples" / "dump_results.py"


def test_dump_writes_all_three_paths(spark, tmp_path):
    out = tmp_path / "dump.md"
    proc = subprocess.run(
        [sys.executable, str(DUMP), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    text = out.read_text(encoding="utf-8")
    # One section per fixture, with every path's answer -- conversions AND
    # declines (a decline is a result).
    assert "## plus_one" in text
    assert "(_udf_param_0 + 1)" in text  # bytecode -> SQL
    assert "CAST(+(_udf_param_0, 1) AS BIGINT)" in text  # built-in static
    assert "declined: unsupported bytecode: CONTAINS_OP" in text  # BSPIn
    assert "java UDF" in text  # the java fixtures
    assert "AI (fake fixture)" in text
