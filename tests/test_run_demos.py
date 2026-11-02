# SPDX-License-Identifier: Apache-2.0
"""run_demos.sh / regen_examples_doc.sh: the shell wiring, without running a bench."""

from __future__ import annotations

import os
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


def test_all_benches_preserves_plain_and_comet_micro_results(tmp_path):
    spark_home = tmp_path / "spark"
    (spark_home / "python" / "pyspark" / "sql").mkdir(parents=True)
    (spark_home / "python" / "pyspark" / "sql" / "transpile.py").write_text("")
    (spark_home / "python" / "lib").mkdir()
    (spark_home / "python" / "lib" / "py4j-test-src.zip").write_bytes(b"")
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$RUN_LOG"\n'
        "csv=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = \'--csv\' ]; then csv="$2"; shift; fi\n'
        "  shift\n"
        "done\n"
        'if [ -n "$csv" ]; then\n'
        '  mkdir -p "$(dirname "$csv")"\n'
        "  printf 'function\\n' > \"$csv\"\n"
        "fi\n"
    )
    python.chmod(0o755)
    log = tmp_path / "calls.log"
    comet_jar = tmp_path / "comet.jar"
    comet_jar.write_bytes(b"PK")
    micro_csv = tmp_path / "bench.csv"
    proc = subprocess.run(
        [str(EXAMPLES / "run_demos.sh"), "--all-benches", "--csv", str(micro_csv)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "SPARK_HOME": str(spark_home),
            "PYTHON": str(python),
            "RUN_LOG": str(log),
            "AI_UDF_TRANSPILE_SQLITE": str(tmp_path / "catalog.sqlite"),
            "COMET_JAR": str(comet_jar),
            "AI_UDF_TRANSPILE_COMET_JAR": str(comet_jar),
        },
    )
    assert proc.returncode == 0, proc.stderr
    comet_csv = tmp_path / "bench_comet.csv"
    assert micro_csv.exists()
    assert comet_csv.exists()
    calls = log.read_text().splitlines()
    micro = [call for call in calls if "examples/bench_micro.py" in call]
    assert any(f"--csv {micro_csv}" in call and "--rust-accel" not in call for call in micro)
    assert any(f"--csv {comet_csv}" in call and "--rust-accel" in call for call in micro)
    assert any(f"bench_graph.py {comet_csv}" in call for call in calls)
