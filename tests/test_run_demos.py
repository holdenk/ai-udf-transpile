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
        "bench_fhi_udfs.py",
        "bench_catalog.py",
        "bench_graph_aggregate.py",
    ):
        assert bench in demos
    # --all-benches itself turns the fhi bench on, and its fills land in the
    # cache before the coverage bench reads it.
    start = demos.index("--all-benches)")
    assert "BENCH_FHI=1" in demos[start : demos.index(";;", start)]
    assert demos.index("examples/bench_fhi_udfs.py") < demos.index("examples/bench_catalog.py")
    # ...which only happens with the live arm on: the pinned arms fill nothing.
    assert "examples/bench_fhi_udfs.py --ai" in demos
    # One CSV per bench: micro and bytecode must not share a path.
    assert "BYTECODE_CSV" in demos
    regen = (EXAMPLES / "regen_examples_doc.sh").read_text()
    assert "--all-benches" in regen
    # Pass 1 is the fill: a bench left out of it only gets cache misses in pass 2's doc.
    assert "--bench-fhi" in regen
