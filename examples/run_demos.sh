#!/usr/bin/env bash
# Run the demos end to end against a packaged Spark master.
#
#   examples/run_demos.sh                        # demos on $AI_UDF_TRANSPILE_BACKEND, else auto (CoCo first)
#   examples/run_demos.sh --live                 # + every real CLI backend (cortex/pi/agent/claude/codex on PATH)
#   examples/run_demos.sh --bench                 # + local micro benchmarks (~10M rows)
#   examples/run_demos.sh --bench --csv out.csv   # + also write the wide per-function CSV
#   examples/run_demos.sh --bench --transpilation-timing-csv out_transpilation_timing.csv
#   examples/run_demos.sh --bench --rust-accel [--rust-accel-conf key=value]
#   examples/run_demos.sh --bench-bytecode        # python vs static (AST) vs bytecode->SQL vs Java bytecode eval
#   examples/run_demos.sh --bench-tpcds           # TPC-DS-inspired UDF patterns vs vanilla SQL
#   examples/run_demos.sh --all-benches --csv out.csv  # every bench: micro (+comet if COMET_JAR), bytecode, tpcds, catalog; then the graphs
#   examples/run_demos.sh --dump                  # + dump $AI_UDF_TRANSPILE_SQLITE to docs/EXAMPLES_TRANSPILED.md
#   examples/setup_comet.sh --write-env-file .comet-env && source .comet-env
#
# Never the fake backend: that is a test fixture. examples/regen_examples_doc.sh
# is the two-pass CoCo run that regenerates the doc.
#
# Env: SPARK_HOME (default ~/spark), PYTHON (default repo .venv).
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPARK_HOME="${SPARK_HOME:-$HOME/spark}"
if [ ! -f "$SPARK_HOME/python/pyspark/sql/transpile.py" ]; then
  echo "error: $SPARK_HOME has no transpile surface; set SPARK_HOME to a packaged Spark master" >&2
  exit 1
fi
export SPARK_HOME

# py4j ships as a versioned zip; glob it instead of pinning a version.
PY4J_ZIP="$(ls "$SPARK_HOME"/python/lib/py4j-*-src.zip 2>/dev/null | head -n1)"
if [ -z "$PY4J_ZIP" ]; then
  echo "error: no py4j-*-src.zip under $SPARK_HOME/python/lib (need a package build)" >&2
  exit 1
fi
export PYTHONPATH="$SPARK_HOME/python:$PY4J_ZIP${PYTHONPATH:+:$PYTHONPATH}"

# Workers must run the driver's interpreter (cloudpickle shares bytecode).
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "error: $PYTHON not executable; set PYTHON to an interpreter with the dev deps" >&2
  exit 1
fi
export PYSPARK_PYTHON="$PYTHON"
export PYSPARK_DRIVER_PYTHON="$PYTHON"

# This box exports AWS creds; their mere presence makes Spark resolve the
# local hostname, which fails here. Genuinely unset -- empty is not enough.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export SPARK_LOCAL_IP=127.0.0.1 SPARK_LOCAL_HOSTNAME=localhost

LIVE=0
BENCH=0
BENCH_BYTECODE=0
BENCH_TPCDS=0
BENCH_ALL=0
ALL_BENCHES=0
DUMP=0
CSV=""
TPCDS_CSV=""
CATALOG_CSV=""
BYTECODE_CSV=""
TRANSPILE_TIMING_CSV=""
RUST_ACCEL=0
RUST_PRESET="comet"
RUST_CONF=()
while [ $# -gt 0 ]; do
  case "$1" in
    --live) LIVE=1 ;;
    --bench) BENCH=1 ;;
    --bench-bytecode) BENCH_BYTECODE=1 ;;
    --bench-tpcds) BENCH_TPCDS=1 ;;
    --bench-all) BENCH_ALL=1 ;;
    # Every bench and every transpiled arm (static, bytecode, ai, comet).
    --all-benches)
      ALL_BENCHES=1
      BENCH=1
      BENCH_BYTECODE=1
      BENCH_TPCDS=1
      BENCH_ALL=1
      ;;
    --dump) DUMP=1 ;;
    # Passed through to bench_micro.py; implies --bench (only the bench reads it).
    --csv)
      [ $# -ge 2 ] || { echo "error: --csv needs a path" >&2; exit 2; }
      CSV="$2"
      TPCDS_CSV="${CSV%.csv}_tpcds.csv"
      CATALOG_CSV="${CSV%.csv}_catalog.csv"
      BYTECODE_CSV="${CSV%.csv}_bytecode.csv"
      BENCH=1
      shift
      ;;
    --transpilation-timing-csv|--timing-csv)
      [ $# -ge 2 ] || { echo "error: $1 needs a path" >&2; exit 2; }
      TRANSPILE_TIMING_CSV="$2"
      BENCH=1
      shift
      ;;
    --rust-accel)
      RUST_ACCEL=1
      BENCH=1
      ;;
    --rust-accel-preset)
      [ $# -ge 2 ] || { echo "error: --rust-accel-preset needs a value" >&2; exit 2; }
      RUST_PRESET="$2"
      RUST_ACCEL=1
      BENCH=1
      shift
      ;;
    --rust-accel-conf)
      [ $# -ge 2 ] || { echo "error: --rust-accel-conf needs key=value" >&2; exit 2; }
      RUST_CONF+=("$2")
      RUST_ACCEL=1
      BENCH=1
      shift
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# The comet arm of the micro bench rides along whenever a jar is configured.
if [ "$ALL_BENCHES" -eq 1 ] && [ "$RUST_ACCEL" -eq 0 ] \
   && { [ -n "${AI_UDF_TRANSPILE_COMET_JAR:-}" ] || [ -n "${COMET_JAR:-}" ]; }; then
  RUST_ACCEL=1
fi

NAMES=()
STATUS=()
SECS=()

run_demo() {
  local name="$1"; shift
  echo "===== $name ====="
  local start rc
  start=$SECONDS
  "$PYTHON" "$@"
  rc=$?
  NAMES+=("$name")
  SECS+=("$((SECONDS - start))s")
  if [ "$rc" -eq 0 ]; then STATUS+=("ok"); else STATUS+=("FAIL($rc)"); fi
  echo
}

run_demo "quickstart" examples/demo_quickstart.py
run_demo "full demo" examples/demo.py
run_demo "inspect" examples/demo_transpile_inspect.py
run_demo "multi-model" examples/demo_multi_model.py
[ "$LIVE" -eq 1 ] && run_demo "live backends" examples/demo_live_backends.py
if [ "$BENCH" -eq 1 ]; then
  if [ "$RUST_ACCEL" -eq 1 ] && [ "$RUST_PRESET" = "comet" ]; then
    if [ -z "${AI_UDF_TRANSPILE_COMET_JAR:-}" ] && [ -z "${COMET_JAR:-}" ]; then
      echo "warning: rust preset 'comet' usually needs COMET_JAR/AI_UDF_TRANSPILE_COMET_JAR set" >&2
    fi
  fi
  BENCH_ARGS=(examples/bench_micro.py)
  [ -n "$CSV" ] && BENCH_ARGS+=(--csv "$CSV")
  [ -n "$TRANSPILE_TIMING_CSV" ] && BENCH_ARGS+=(--transpilation-timing-csv "$TRANSPILE_TIMING_CSV")
  if [ "$RUST_ACCEL" -eq 1 ]; then
    BENCH_ARGS+=(--rust-accel --rust-accel-preset "$RUST_PRESET")
    for kv in "${RUST_CONF[@]}"; do
      BENCH_ARGS+=(--rust-accel-conf "$kv")
    done
  fi
  run_demo "micro bench" "${BENCH_ARGS[@]}"
fi

if [ "$BENCH_BYTECODE" -eq 1 ]; then
  BC_ARGS=(examples/bench_bytecode_vs_static.py)
  [ -n "$CSV" ] && BC_ARGS+=(--csv "$BYTECODE_CSV")
  run_demo "bytecode vs static bench" "${BC_ARGS[@]}"
fi

if [ "$BENCH_TPCDS" -eq 1 ]; then
  TPCDS_ARGS=(examples/bench_tpcds_udfs.py)
  if [ -n "$CSV" ]; then
    TPCDS_ARGS+=(--csv "$TPCDS_CSV")
  fi
  run_demo "tpcds udf bench" "${TPCDS_ARGS[@]}"
fi

if [ "$BENCH_ALL" -eq 1 ]; then
  [ -n "${AI_UDF_TRANSPILE_SQLITE:-}" ] || { echo "error: --bench-all needs AI_UDF_TRANSPILE_SQLITE" >&2; exit 2; }
  BCAT_ARGS=(examples/bench_catalog.py --sqlite "$AI_UDF_TRANSPILE_SQLITE")
  if [ -n "$CSV" ]; then
    BCAT_ARGS+=(--csv "$CATALOG_CSV")
  fi
  run_demo "catalog bench" "${BCAT_ARGS[@]}"
fi

if [ "$ALL_BENCHES" -eq 1 ] && [ -n "$CSV" ]; then
  [ -f "$CSV" ] && run_demo "graph: micro" examples/bench_graph.py "$CSV" docs/bench_perf.png
  [ -f "$BYTECODE_CSV" ] && run_demo "graph: bytecode" examples/bench_graph.py "$BYTECODE_CSV" docs/bench_bytecode.png
  run_demo "graph: aggregate" examples/bench_graph_aggregate.py "$(dirname "$CSV")" --png "$(dirname "$CSV")/bench_aggregate.png"
fi

if [ "$DUMP" -eq 1 ]; then
  [ -n "${AI_UDF_TRANSPILE_SQLITE:-}" ] || { echo "error: --dump needs AI_UDF_TRANSPILE_SQLITE" >&2; exit 2; }
  DUMP_ARGS=(examples/dump_results.py --sqlite "$AI_UDF_TRANSPILE_SQLITE" --out docs/EXAMPLES_TRANSPILED.md)
  if [ -n "$CSV" ]; then
    # Catalog CSV first (widest coverage); micro/tpcds take over per function
    # with their deeper arms. Later --perf-csv files win.
    [ -f "$CATALOG_CSV" ] && DUMP_ARGS+=(--perf-csv "$CATALOG_CSV")
    DUMP_ARGS+=(--perf-csv "$CSV")
    [ -f "$TPCDS_CSV" ] && DUMP_ARGS+=(--perf-csv "$TPCDS_CSV")
  fi
  run_demo "dump" "${DUMP_ARGS[@]}"
fi

echo "===== SUMMARY ====="
for i in "${!NAMES[@]}"; do
  printf '%-14s %-8s %s\n' "${NAMES[$i]}" "${STATUS[$i]}" "${SECS[$i]}"
done
for s in "${STATUS[@]}"; do
  [ "$s" = "ok" ] || exit 1
done
