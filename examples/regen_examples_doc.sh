#!/usr/bin/env bash
# Regenerate docs/EXAMPLES_TRANSPILED.md from a real AI backend (CoCo by
# default; never the fake one). Every example runs twice against one fresh
# SQLite cache: pass one is the cache miss that waits on the backend, pass
# two is the cache hit where things actually party on. Then dump the cache.
#
#   examples/regen_examples_doc.sh                                  # CoCo
#   AI_UDF_TRANSPILE_BACKEND=claude examples/regen_examples_doc.sh
#   examples/regen_examples_doc.sh --rust-accel                     # extra run_demos.sh flags
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AI_UDF_TRANSPILE_BACKEND="${AI_UDF_TRANSPILE_BACKEND:-coco}"
export AI_UDF_TRANSPILE_SQLITE="${AI_UDF_TRANSPILE_SQLITE:-$REPO_ROOT/target/examples_transpiled.sqlite}"

mkdir -p "$(dirname "$AI_UDF_TRANSPILE_SQLITE")"
rm -f "$AI_UDF_TRANSPILE_SQLITE"{,-wal,-shm}

ARGS=(--bench --bench-tpcds "$@")
echo "##### pass 1/2: filling $AI_UDF_TRANSPILE_SQLITE from $AI_UDF_TRANSPILE_BACKEND #####"
"$REPO_ROOT/examples/run_demos.sh" "${ARGS[@]}" || echo "pass 1 had failures; pass 2 decides"
echo "##### pass 2/2: cache hits, then the dump #####"
"$REPO_ROOT/examples/run_demos.sh" "${ARGS[@]}" --dump
