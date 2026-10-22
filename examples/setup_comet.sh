#!/usr/bin/env bash
# Build Apache DataFusion Comet from source for the local Spark checkout and
# print the env exports needed by bench_micro's --rust-accel comet preset.
#
# Typical use:
#   examples/setup_comet.sh --write-env-file .comet-env
#   source .comet-env
#   python examples/bench_micro.py --rust-accel

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: examples/setup_comet.sh [options]

Options:
  --spark-home PATH        Spark checkout/build root (default: $SPARK_HOME or ~/spark)
  --comet-home PATH        Comet clone/build root (default: $COMET_HOME or ~/datafusion-comet)
  --ref REF                Comet git ref to checkout (default: main)
  --profile PROFILE        Override Comet Spark profile (spark-3.4|spark-3.5|spark-4.0|spark-4.1|spark-4.2)
  --scala-binary VERSION   Override scala binary version (2.12 or 2.13)
  --write-env-file PATH    Write COMET_JAR exports to PATH for `source PATH`
  --skip-build             Skip `make release`; only resolve jar path from existing build output
  --dry-run                Skip build and stop after showing detected settings
  -h, --help               Show this help

Examples:
  examples/setup_comet.sh
  examples/setup_comet.sh --ref main --write-env-file .comet-env
  examples/setup_comet.sh --profile spark-4.2 --scala-binary 2.13
EOF
}

error() {
  echo "error: $*" >&2
  exit 2
}

warn() {
  echo "warning: $*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || error "missing required command: $1"
}

pick_profile() {
  case "$1" in
    3.4|3.5|4.0|4.1|4.2)
      echo "spark-$1"
      ;;
    *)
      local major="${1%%.*}"
      if [[ "$major" =~ ^[0-9]+$ ]] && [ "$major" -ge 4 ]; then
        warn "Spark $1 detected; Comet has profiles up to 4.2, using spark-4.2"
        echo "spark-4.2"
      else
        error "unsupported Spark version '$1' for Comet profile auto-pick"
      fi
      ;;
  esac
}

resolve_latest_jar() {
  local profile_minor="$1"
  local scala_bin="$2"
  local comet_home="$3"
  local prefix="comet-spark-spark${profile_minor}_${scala_bin}-"
  local -a matches runtime_jars
  shopt -s nullglob
  matches=("$comet_home"/spark/target/"${prefix}"*.jar)
  shopt -u nullglob
  if [ "${#matches[@]}" -eq 0 ]; then
    return 1
  fi
  local candidate
  for candidate in "${matches[@]}"; do
    case "$candidate" in
      *-sources.jar|*-javadoc.jar|*-tests.jar|*-test.jar)
        continue
        ;;
      *)
        runtime_jars+=("$candidate")
        ;;
    esac
  done
  if [ "${#runtime_jars[@]}" -eq 0 ]; then
    return 1
  fi
  local newest="${runtime_jars[0]}"
  local candidate
  for candidate in "${runtime_jars[@]}"; do
    if [ "$candidate" -nt "$newest" ]; then
      newest="$candidate"
    fi
  done
  printf "%s\n" "$newest"
}

SPARK_HOME_ARG="${SPARK_HOME:-$HOME/spark}"
COMET_HOME_ARG="${COMET_HOME:-$HOME/datafusion-comet}"
COMET_REF="main"
PROFILE_OVERRIDE=""
SCALA_OVERRIDE=""
WRITE_ENV_FILE=""
SKIP_BUILD=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --spark-home)
      [ $# -ge 2 ] || error "--spark-home needs a value"
      SPARK_HOME_ARG="$2"
      shift
      ;;
    --comet-home)
      [ $# -ge 2 ] || error "--comet-home needs a value"
      COMET_HOME_ARG="$2"
      shift
      ;;
    --ref)
      [ $# -ge 2 ] || error "--ref needs a value"
      COMET_REF="$2"
      shift
      ;;
    --profile)
      [ $# -ge 2 ] || error "--profile needs a value"
      PROFILE_OVERRIDE="$2"
      shift
      ;;
    --scala-binary)
      [ $# -ge 2 ] || error "--scala-binary needs a value"
      SCALA_OVERRIDE="$2"
      shift
      ;;
    --write-env-file)
      [ $# -ge 2 ] || error "--write-env-file needs a value"
      WRITE_ENV_FILE="$2"
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error "unknown option: $1"
      ;;
  esac
  shift
done

[ -d "$SPARK_HOME_ARG" ] || error "spark home does not exist: $SPARK_HOME_ARG"
[ -f "$SPARK_HOME_ARG/pom.xml" ] || error "spark pom missing: $SPARK_HOME_ARG/pom.xml"

require_cmd git
require_cmd make
require_cmd java
if [ "$DRY_RUN" -eq 0 ] && [ "$SKIP_BUILD" -eq 0 ]; then
  require_cmd cargo
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    error "missing required command: python3 or python"
  fi
fi

read -r SPARK_MINOR SPARK_SCALA <<EOF
$("$PYTHON_BIN" - "$SPARK_HOME_ARG/pom.xml" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()

def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(node: ET.Element, name: str) -> str:
    for child in node:
        if local(child.tag) == name:
            return (child.text or "").strip()
    return ""


version = child_text(root, "version")
props = None
for child in root:
    if local(child.tag) == "properties":
        props = child
        break
scala = ""
if props is not None:
    scala = child_text(props, "scala.binary.version")
m = re.match(r"(\d+)\.(\d+)", version)
minor = f"{m.group(1)}.{m.group(2)}" if m else ""
print(minor, scala)
PY
)
EOF

[ -n "$SPARK_MINOR" ] || error "could not detect Spark version from $SPARK_HOME_ARG/pom.xml"
if [ -z "$SPARK_SCALA" ]; then
  warn "could not detect scala.binary.version from Spark pom; defaulting to 2.13"
  SPARK_SCALA="2.13"
fi

if [ -n "$PROFILE_OVERRIDE" ]; then
  PROFILE="$PROFILE_OVERRIDE"
else
  PROFILE="$(pick_profile "$SPARK_MINOR")"
fi

case "$PROFILE" in
  spark-3.4|spark-3.5|spark-4.0|spark-4.1|spark-4.2)
    ;;
  *)
    error "unsupported --profile '$PROFILE'"
    ;;
esac

SCALA_BIN="${SCALA_OVERRIDE:-$SPARK_SCALA}"
case "$SCALA_BIN" in
  2.12|2.13)
    ;;
  *)
    error "unsupported scala binary version '$SCALA_BIN' (expected 2.12 or 2.13)"
    ;;
esac

PROFILES="-P${PROFILE}"
if [ "$SCALA_BIN" = "2.12" ]; then
  case "$PROFILE" in
    spark-3.4|spark-3.5)
      PROFILES="${PROFILES} -Pscala-2.12"
      ;;
    *)
      error "scala-2.12 is not supported with $PROFILE"
      ;;
  esac
fi

if [ ! -d "$COMET_HOME_ARG/.git" ]; then
  echo "cloning Apache DataFusion Comet into $COMET_HOME_ARG"
  if command -v gh >/dev/null 2>&1; then
    gh repo clone apache/datafusion-comet "$COMET_HOME_ARG"
  else
    git clone https://github.com/apache/datafusion-comet.git "$COMET_HOME_ARG"
  fi
fi
[ -d "$COMET_HOME_ARG/.git" ] || error "comet git checkout missing at $COMET_HOME_ARG"

if ! git -C "$COMET_HOME_ARG" diff --quiet || ! git -C "$COMET_HOME_ARG" diff --cached --quiet; then
  error "comet checkout has local changes: $COMET_HOME_ARG (clean it first)"
fi

echo "updating Comet checkout at $COMET_HOME_ARG"
git -C "$COMET_HOME_ARG" fetch --tags origin
git -C "$COMET_HOME_ARG" checkout "$COMET_REF"
if git -C "$COMET_HOME_ARG" show-ref --verify --quiet "refs/remotes/origin/$COMET_REF"; then
  git -C "$COMET_HOME_ARG" pull --ff-only origin "$COMET_REF"
fi

echo "Spark detected: $SPARK_MINOR (scala $SPARK_SCALA)"
echo "Comet profile: $PROFILE (build profiles: $PROFILES)"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry-run: skipping build"
  echo "would run: (cd $COMET_HOME_ARG && make release PROFILES=\"$PROFILES\")"
  exit 0
elif [ "$SKIP_BUILD" -eq 1 ]; then
  echo "skip-build: not rebuilding Comet"
else
  (cd "$COMET_HOME_ARG" && make release PROFILES="$PROFILES")
fi

PROFILE_MINOR="${PROFILE#spark-}"
if ! COMET_JAR_PATH="$(resolve_latest_jar "$PROFILE_MINOR" "$SCALA_BIN" "$COMET_HOME_ARG")"; then
  error "no built jar found for spark-$PROFILE_MINOR scala-$SCALA_BIN in $COMET_HOME_ARG/spark/target"
fi

if [ -n "$WRITE_ENV_FILE" ]; then
  cat >"$WRITE_ENV_FILE" <<EOF
export COMET_JAR="$COMET_JAR_PATH"
export AI_UDF_TRANSPILE_COMET_JAR="$COMET_JAR_PATH"
EOF
  echo "wrote env file: $WRITE_ENV_FILE"
fi

cat <<EOF

Comet jar resolved:
  $COMET_JAR_PATH

Export in your shell:
  export COMET_JAR="$COMET_JAR_PATH"
  export AI_UDF_TRANSPILE_COMET_JAR="$COMET_JAR_PATH"

Then run:
  python examples/bench_micro.py --rust-accel --rust-accel-preset comet
EOF
