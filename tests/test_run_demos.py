# SPDX-License-Identifier: Apache-2.0
"""run_demos.sh / regen_examples_doc.sh: the shell wiring, without running a bench."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_scripts_parse():
    for script in ("run_demos.sh", "regen_examples_doc.sh", "setup_spark.sh", "setup_comet.sh"):
        subprocess.run(["bash", "-n", str(EXAMPLES / script)], check=True)


def _fake_release(cache, version="4.3.0"):
    """An unpacked binary release, as setup_spark.sh leaves it in its cache."""
    home = cache / f"spark-{version}-bin-hadoop3"
    (home / "python" / "pyspark" / "sql").mkdir(parents=True)
    (home / "python" / "pyspark" / "sql" / "transpile.py").write_text("")
    (home / "python" / "lib").mkdir()
    (home / "python" / "lib" / "py4j-test-src.zip").write_bytes(b"")
    (home / "jars").mkdir()
    (home / "jars" / f"spark-core_2.13-{version}.jar").write_bytes(b"PK")
    (home / "RELEASE").write_text(f"Spark {version} (git revision abc) built for Hadoop 3\n")
    return home


def _logging_python(tmp_path):
    python = tmp_path / "python"
    python.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$RUN_LOG"\n')
    python.chmod(0o755)
    return python


def test_setup_spark_reuses_a_cached_release(tmp_path):
    home = _fake_release(tmp_path)
    env_file = tmp_path / "spark.env"
    proc = subprocess.run(
        [
            str(EXAMPLES / "setup_spark.sh"),
            "--spark",
            "4.3.0",
            "--cache-dir",
            str(tmp_path),
            "--write-env-file",
            str(env_file),
            "--print-home",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(home)
    exports = env_file.read_text()
    assert f'SPARK_HOME="{home}"' in exports
    assert "py4j-test-src.zip" in exports


def test_setup_spark_refuses_a_release_without_the_hook(tmp_path):
    home = _fake_release(tmp_path, "4.2.1")
    (home / "python" / "pyspark" / "sql" / "transpile.py").unlink()
    proc = subprocess.run(
        [str(EXAMPLES / "setup_spark.sh"), "--spark", "4.2.1", "--cache-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "transpile.py" in proc.stderr


def test_run_demos_spark_flag_picks_the_release_and_records_it(tmp_path):
    cache = tmp_path / "cache"
    home = _fake_release(cache)
    log = tmp_path / "calls.log"
    out = tmp_path / "bench" / "spark-4.3"
    graphs = tmp_path / "graphs"
    drop = ("SPARK_HOME", "COMET_JAR", "AI_UDF_TRANSPILE_COMET_JAR")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    proc = subprocess.run(
        [
            str(EXAMPLES / "run_demos.sh"),
            "--spark",
            "4.3.0",
            "--all-benches",
            "--csv",
            str(out / "micro.csv"),
            "--graph-dir",
            str(graphs),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **env,
            "AI_UDF_TRANSPILE_SPARK_CACHE": str(cache),
            "PYTHON": str(_logging_python(tmp_path)),
            "RUN_LOG": str(log),
            "AI_UDF_TRANSPILE_SQLITE": str(tmp_path / "catalog.sqlite"),
        },
    )
    # The fake python writes no CSVs, so the graph steps that need one are
    # skipped; what matters is which Spark ran and where outputs would land.
    assert f"Spark: Spark 4.3.0 (git revision abc) built for Hadoop 3 [{home}]" in proc.stdout
    assert (out / "spark.txt").read_text().startswith("Spark 4.3.0")
    calls = log.read_text().splitlines()
    assert any("bench_micro.py" in c for c in calls)
    assert graphs.is_dir()


def test_setup_comet_reads_version_from_release_jars(tmp_path):
    for cmd in ("git", "make", "java"):
        if subprocess.run(["bash", "-c", f"command -v {cmd}"], capture_output=True).returncode:
            pytest.skip(f"{cmd} not installed")
    home = _fake_release(tmp_path)
    comet = tmp_path / "comet"
    subprocess.run(["git", "init", "-q", str(comet)], check=True)
    proc = subprocess.run(
        [
            str(EXAMPLES / "setup_comet.sh"),
            "--spark-home",
            str(home),
            "--comet-home",
            str(comet),
            "--skip-build",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Spark detected: 4.3 (scala 2.13)" in proc.stdout


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
    # One CSV per bench: micro and bytecode must not share a path, and the
    # bytecode arms must reach the generated examples document.
    assert "BYTECODE_CSV" in demos
    assert 'DUMP_ARGS+=(--perf-csv "$BYTECODE_CSV")' in demos
    regen = (EXAMPLES / "regen_examples_doc.sh").read_text()
    assert "--all-benches" in regen
    # Pass 1 is the fill: a bench left out of it only gets cache misses in pass 2's doc.
    assert "--bench-fhi" in regen


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


def test_comet_output_path_collision_is_rejected(tmp_path):
    spark_home = tmp_path / "spark"
    (spark_home / "python" / "pyspark" / "sql").mkdir(parents=True)
    (spark_home / "python" / "pyspark" / "sql" / "transpile.py").write_text("")
    (spark_home / "python" / "lib").mkdir()
    (spark_home / "python" / "lib" / "py4j-test-src.zip").write_bytes(b"")
    proc = subprocess.run(
        [
            str(EXAMPLES / "run_demos.sh"),
            "--all-benches",
            "--rust-accel",
            "--csv",
            str(tmp_path / "out_comet.csv"),
            "--transpilation-timing-csv",
            str(tmp_path / "out.csv"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "SPARK_HOME": str(spark_home), "PYTHON": sys.executable},
    )
    assert proc.returncode == 2
    assert "collision" in proc.stderr
