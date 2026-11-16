#!/usr/bin/env bash
# Regenerate docs/EXAMPLES_TRANSPILED.md with every UDF's SQL, Java and Rust
# rewrites side by side (rejections and perf included).
#
# Needs target/examples_transpiled.sqlite (examples/regen_examples_doc.sh, or
# any catalog of sources) and a .comet-env for the Rust kind. One kind and one
# bench at a time: fill_targets.py asks the backend (default ensemble: CoCo,
# Claude fallback) for that kind only, bench_catalog.py times what was
# accepted, then the next kind.
#
#   examples/regen_targets_doc.sh [--only name]...     # --only: smoke a few UDFs
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f .comet-env ] && source .comet-env

SPARK_HOME="${SPARK_HOME:-$HOME/spark}"
PY4J_ZIP="$(ls "$SPARK_HOME"/python/lib/py4j-*-src.zip 2>/dev/null | head -n1)"
[ -n "$PY4J_ZIP" ] || { echo "error: no py4j zip under $SPARK_HOME/python/lib" >&2; exit 1; }
export SPARK_HOME PYTHONPATH="$SPARK_HOME/python:$PY4J_ZIP${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export SPARK_LOCAL_IP=127.0.0.1 SPARK_LOCAL_HOSTNAME=localhost
export AI_UDF_TRANSPILE_BACKEND="${AI_UDF_TRANSPILE_BACKEND:-ensemble}"

FAILED=0
DIR="${TARGETS_DIR:-$REPO_ROOT/target/targets}"
mkdir -p "$DIR"
for kind in catalyst java_udf rust; do
  echo "##### $kind: fill #####"
  "$PYTHON" -u examples/fill_targets.py --target "$kind" --out "$DIR/$kind.sqlite" "$@" || { echo "$kind fill failed"; FAILED=1; }
  echo "##### $kind: bench #####"
  rm -f "$DIR/${kind}_bench.csv"
  "$PYTHON" -u examples/bench_catalog.py --strict-comet \
    --sqlite "$DIR/$kind.sqlite" --csv "$DIR/${kind}_bench.csv" \
    || { echo "$kind bench failed"; FAILED=1; }
done
# A partial (--only) or failed run must not replace the committed doc.
OUT=docs/EXAMPLES_TRANSPILED.md
if [ $# -gt 0 ] || [ "$FAILED" = 1 ]; then OUT="$DIR/EXAMPLES_TRANSPILED.partial.md"; fi
"$PYTHON" -u examples/dump_results.py --targets-dir "$DIR" --out "$OUT" || FAILED=1
exit "$FAILED"
