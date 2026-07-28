#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
ARTIFACT_ROOT=$(realpath "${2:-${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}}")
PYTHON_BIN=${CATENA_PYTHON:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}

[[ -d "$REPO_ROOT/src/catena" ]] || {
  echo "[ERROR] Not a CATENA repository: $REPO_ROOT" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] Python is unavailable: $PYTHON_BIN" >&2
  exit 1
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" tools/post_e21_status.py --artifact-root "$ARTIFACT_ROOT"
