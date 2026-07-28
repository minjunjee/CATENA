#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=${1:-/home/minjun_dev/CATENA}
ARTIFACT_ROOT=${2:-${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python tools/postcore_status.py --artifact-root "$ARTIFACT_ROOT"
