# SPDX-License-Identifier: Apache-2.0
"""run_demos.sh / regen_examples_doc.sh: the shell wiring, without running a bench."""

from __future__ import annotations

import subprocess
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_scripts_parse():
    for script in ("run_demos.sh", "regen_examples_doc.sh"):
        subprocess.run(["bash", "-n", str(EXAMPLES / script)], check=True)


def test_all_benches_flag_is_wired():
    demos = (EXAMPLES / "run_demos.sh").read_text()
    for bench in (
        "bench_micro.py",
        "bench_bytecode_vs_static.py",
        "bench_tpcds_udfs.py",
        "bench_catalog.py",
        "bench_graph_aggregate.py",
    ):
        assert bench in demos
    # One CSV per bench: micro and bytecode must not share a path.
    assert "BYTECODE_CSV" in demos
    assert "--all-benches" in (EXAMPLES / "regen_examples_doc.sh").read_text()
