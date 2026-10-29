#!/usr/bin/env bash
# Regenerate docs/EXAMPLES_TRANSPILED.md from real AI backends (never the
# fake one). Default backend is ensemble: CoCo answers, and when CoCo
# declines or errors (auth, rate limit), cursor gets the row. Every example
# runs twice against one fresh SQLite cache: pass one is the cache miss that
# waits on the backend, pass two is the cache hit where things actually
# party on. Then dump the cache.
#
# The benches run in both passes and write
# questionable_benchmark_results/examples_transpiled_bench{,_bytecode,_tpcds,_catalog}.csv;
# the dump joins them into the doc: interpreted python rows/s at the top of
# each section, each transpiled arm's rows/s at the end of its own line.
# Pass 2 is --all-benches: every bench (micro, bytecode, tpcds, catalog) and the
# graphs; pass 1 only needs numbers good enough to smoke-test the fill. bench_catalog.py is the coverage bench -- every cached function gets
# numbers or a loud skip; micro/tpcds add their deeper arms.
#
#   examples/regen_examples_doc.sh                                     # CoCo, cursor fallback
#   AI_UDF_TRANSPILE_BACKEND=claude examples/regen_examples_doc.sh
#   examples/regen_examples_doc.sh --rust-accel                     # extra run_demos.sh flags
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A .comet-env (examples/setup_comet.sh --write-env-file) adds the comet arm
# to bench_catalog's numbers; without it the doc just has no comet bullets.
[ -f "$REPO_ROOT/.comet-env" ] && source "$REPO_ROOT/.comet-env"

export AI_UDF_TRANSPILE_BACKEND="${AI_UDF_TRANSPILE_BACKEND:-ensemble}"
export AI_UDF_TRANSPILE_SQLITE="${AI_UDF_TRANSPILE_SQLITE:-$REPO_ROOT/target/examples_transpiled.sqlite}"

mkdir -p "$(dirname "$AI_UDF_TRANSPILE_SQLITE")" "$REPO_ROOT/questionable_benchmark_results"
rm -f "$AI_UDF_TRANSPILE_SQLITE"{,-wal,-shm}

CSV="$REPO_ROOT/questionable_benchmark_results/examples_transpiled_bench.csv"
echo "##### pass 1/2: filling $AI_UDF_TRANSPILE_SQLITE from $AI_UDF_TRANSPILE_BACKEND #####"
"$REPO_ROOT/examples/run_demos.sh" --bench --bench-tpcds --bench-all --csv "$CSV" "$@" || echo "pass 1 had failures; pass 2 decides"
echo "##### pass 2/2: cache hits, then the dump #####"
"$REPO_ROOT/examples/run_demos.sh" --all-benches --csv "$CSV" "$@" --dump
