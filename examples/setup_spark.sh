#!/usr/bin/env bash
# Fetch a Spark to benchmark against and print the SPARK_HOME for it.
#
# Two flavors:
#   - a release ("4.3", "4.3.0"): the Apache binary tarball, unpacked once
#     into the cache dir and reused. "4.3" resolves to the newest 4.3.x.
#   - a git ref ("branch-4.x", "master", a tag or a SHA): a shallow clone of
#     apache/spark built the way CI builds it (package, -DskipTests). A
#     rebuild only happens when the ref moved.
#
# Typical use:
#   examples/setup_spark.sh --spark 4.3 --write-env-file .spark-env && source .spark-env
#   examples/setup_spark.sh --spark branch-4.x --write-env-file .spark-env && source .spark-env
#   examples/run_demos.sh --spark branch-4.x --all-benches --csv questionable_benchmark_results/branch-4.x/micro.csv
#
# Progress and build logs go to stderr, so
# `SPARK_HOME="$(examples/setup_spark.sh --spark 4.3 --print-home)"` works.

set -euo pipefail
# fetch_* run inside $(...); without this a failure there is ignored.
shopt -s inherit_errexit 2>/dev/null || true  # bash >= 4.4; the steps that matter also check explicitly

usage() {
  cat <<'EOF'
Usage: examples/setup_spark.sh --spark SPEC [options]

SPEC:
  4.3 | 4.3.0              Apache release (binary tarball, hadoop3). X.Y picks the newest X.Y.z.
  branch-4.x | master | REF  apache/spark git ref, built from source with -DskipTests.

Options:
  --cache-dir PATH         Where releases/checkouts live (default: $AI_UDF_TRANSPILE_SPARK_CACHE
                           or ~/.cache/ai_udf_transpile/spark)
  --repo URL               Spark git remote for refs (default: https://github.com/apache/spark.git)
  --write-env-file PATH    Write SPARK_HOME / PYTHONPATH exports to PATH for `source PATH`
  --print-home             Print only the resolved SPARK_HOME on stdout (for scripts)
  --skip-build             Ref only: reuse the existing checkout/build as is (no fetch, no build)
  --rebuild                Ref only: rebuild even if the checkout is already built at this SHA
  --dry-run                Show what would be fetched/built, then stop
  -h, --help               Show this help
EOF
}

error() { echo "error: $*" >&2; exit 2; }
log() { echo "$*" >&2; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || error "missing required command: $1"; }

SPEC=""
CACHE_DIR="${AI_UDF_TRANSPILE_SPARK_CACHE:-$HOME/.cache/ai_udf_transpile/spark}"
REPO_URL="https://github.com/apache/spark.git"
WRITE_ENV_FILE=""
PRINT_HOME=0
SKIP_BUILD=0
REBUILD=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --spark) [ $# -ge 2 ] || error "--spark needs a value"; SPEC="$2"; shift ;;
    --cache-dir) [ $# -ge 2 ] || error "--cache-dir needs a value"; CACHE_DIR="$2"; shift ;;
    --repo) [ $# -ge 2 ] || error "--repo needs a value"; REPO_URL="$2"; shift ;;
    --write-env-file) [ $# -ge 2 ] || error "--write-env-file needs a value"; WRITE_ENV_FILE="$2"; shift ;;
    --print-home) PRINT_HOME=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --rebuild) REBUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) error "unknown option: $1" ;;
  esac
  shift
done

[ -n "$SPEC" ] || { usage >&2; error "--spark is required"; }
mkdir -p "$CACHE_DIR"
CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"

# Mirrors, newest-first: dlcdn only carries current releases, archive has all.
MIRRORS=("https://dlcdn.apache.org/spark" "https://archive.apache.org/dist/spark")
CHECKSUM_BASES=("https://downloads.apache.org/spark" "https://archive.apache.org/dist/spark")

# "4.3" -> newest "4.3.z" listed on any mirror; "4.3.0" stays as is.
resolve_release() {
  local spec="$1"
  if [[ "$spec" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "$spec"
    return
  fi
  local m listing best=""
  for m in "${MIRRORS[@]}"; do
    listing="$(curl -fsSL --max-time 30 "$m/" 2>/dev/null || true)"
    best="$(printf '%s\n' "$listing" \
      | grep -oE "spark-${spec//./\\.}\.[0-9]+/" \
      | sed -E 's#spark-(.*)/#\1#' | sort -uV | tail -n1)"
    [ -n "$best" ] && break
  done
  [ -n "$best" ] || error "no Spark $spec.x release found on ${MIRRORS[*]}"
  echo "$best"
}

fetch_release() {
  local version="$1"
  local name="spark-${version}-bin-hadoop3"
  local home="$CACHE_DIR/$name"
  if [ -f "$home/RELEASE" ] && [ -d "$home/jars" ]; then
    log "release $version already unpacked at $home"
    echo "$home"
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run: would download $name.tgz into $CACHE_DIR"
    echo "$home"
    return
  fi
  require_cmd curl
  require_cmd tar
  require_cmd sha512sum
  local tgz="$CACHE_DIR/$name.tgz" got=0 m
  for m in "${MIRRORS[@]}"; do
    log "downloading $m/spark-$version/$name.tgz"
    if curl -fL --progress-bar --retry 3 -o "$tgz.part" "$m/spark-$version/$name.tgz"; then
      mv "$tgz.part" "$tgz"
      got=1
      break
    fi
  done
  [ "$got" -eq 1 ] || { rm -f "$tgz.part"; error "could not download $name.tgz"; }

  # Apache's .sha512 files come in two layouts ("<hash>  <file>" and the
  # gpg --print-md block); squash whitespace and take the hex digest either way.
  local expected="" b
  for b in "${CHECKSUM_BASES[@]}"; do
    expected="$(curl -fsSL --max-time 30 "$b/spark-$version/$name.tgz.sha512" 2>/dev/null \
      | tr -d ' \n\r\t' | sed -E 's/^.*://; s/[^0-9a-fA-F].*$//' | tr 'A-F' 'a-f' || true)"
    [ "${#expected}" -eq 128 ] && break
    expected=""
  done
  [ -n "$expected" ] || { rm -f "$tgz"; error "no sha512 published for $name.tgz"; }
  local actual
  actual="$(sha512sum "$tgz" | awk '{print $1}')"
  [ "$actual" = "$expected" ] || { rm -f "$tgz"; error "sha512 mismatch for $name.tgz"; }

  tar -xzf "$tgz" -C "$CACHE_DIR"
  rm -f "$tgz"
  echo "$home"
}

fetch_ref() {
  local ref="$1"
  local home="$CACHE_DIR/src-${ref//\//_}"
  local stamp="$home/.ai_udf_transpile_built_sha"
  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run: would fetch $REPO_URL $ref into $home and build with -DskipTests"
    echo "$home"
    return
  fi
  require_cmd git
  if [ "$SKIP_BUILD" -eq 1 ]; then
    [ -d "$home/.git" ] || error "--skip-build but no checkout at $home"
    log "skip-build: using $home as is ($(git -C "$home" rev-parse --short HEAD))"
    echo "$home"
    return
  fi
  if [ ! -d "$home/.git" ]; then
    git init -q "$home"
    git -C "$home" remote add origin "$REPO_URL"
  fi
  log "fetching $ref from $REPO_URL"
  git -C "$home" fetch -q --depth=1 origin "$ref" || error "could not fetch $ref from $REPO_URL"
  local sha
  sha="$(git -C "$home" rev-parse FETCH_HEAD)"
  if [ "$REBUILD" -eq 0 ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sha" ]; then
    log "$ref already built at ${sha:0:12}"
    echo "$home"
    return
  fi
  git -C "$home" checkout -q --force --detach "$sha"
  require_cmd java
  log "building Spark $ref @ ${sha:0:12} (package, -DskipTests; first build is slow)"
  rm -f "$stamp"
  # Same module set as CI: enough for a classic pyspark session with the
  # transpiler hook, without building every connector.
  # -DskipTests still compiles test sources (the modules depend on each
  # other's test jars), so a branch whose tests do not compile cannot be
  # packaged; pin an earlier SHA with --spark <sha> until upstream fixes it.
  local build_log="$home/.ai_udf_transpile_build.log"
  if ! (cd "$home" && MAVEN_OPTS="${MAVEN_OPTS:--Xmx4g -XX:ReservedCodeCacheSize=1g}" \
    ./build/mvn -T1C -DskipTests -Phadoop-3 package -pl sql/core,assembly -am) >"$build_log" 2>&1; then
    grep -E '\[ERROR\]' "$build_log" | head -n 8 >&2 || true
    error "Spark build of $ref @ ${sha:0:12} failed (full log: $build_log)"
  fi
  echo "$sha" >"$stamp"
  echo "$home"
}

if [[ "$SPEC" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  KIND="release"
  VERSION="$(resolve_release "$SPEC")"
  HOME_DIR="$(fetch_release "$VERSION")"
  LABEL="spark-$VERSION"
  REV="$VERSION"
else
  KIND="ref"
  HOME_DIR="$(fetch_ref "$SPEC")"
  LABEL="$SPEC"
  REV="$( [ -d "$HOME_DIR/.git" ] && git -C "$HOME_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
fi

[ "$DRY_RUN" -eq 1 ] && { log "dry-run: SPARK_HOME would be $HOME_DIR"; exit 0; }

[ -f "$HOME_DIR/python/pyspark/sql/transpile.py" ] \
  || error "$LABEL ($HOME_DIR) has no pyspark/sql/transpile.py; the transpiler hook needs 4.3+"
PY4J_ZIP="$(ls "$HOME_DIR"/python/lib/py4j-*-src.zip 2>/dev/null | head -n1)"
[ -n "$PY4J_ZIP" ] || error "no py4j-*-src.zip under $HOME_DIR/python/lib"

if [ -n "$WRITE_ENV_FILE" ]; then
  cat >"$WRITE_ENV_FILE" <<EOF
export SPARK_HOME="$HOME_DIR"
export PYTHONPATH="$HOME_DIR/python:$PY4J_ZIP\${PYTHONPATH:+:\$PYTHONPATH}"
export AI_UDF_TRANSPILE_SPARK_LABEL="$LABEL"
export AI_UDF_TRANSPILE_SPARK_REV="$REV"
EOF
  log "wrote env file: $WRITE_ENV_FILE"
fi

if [ "$PRINT_HOME" -eq 1 ]; then
  echo "$HOME_DIR"
  exit 0
fi

cat <<EOF

Spark ($KIND) ready: $LABEL @ $REV
  export SPARK_HOME="$HOME_DIR"
  export PYTHONPATH="$HOME_DIR/python:$PY4J_ZIP\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
