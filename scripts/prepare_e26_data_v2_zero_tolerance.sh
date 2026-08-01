#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26_STAGE3}"
CONFIG_PATH="${2:-$REPO_ROOT/configs/e26_data_lock_v2_zero_tolerance.yaml}"
TOOL_PYTHON="${3:-/data/minjun_dev/CATENA/e26_data_v1/.venv/bin/python}"

if [[ "${CATENA_E26_DATA_REPAIR_ACK:-}" != "E26_ZERO_TOLERANCE_REPAIR_AUTHORIZED" ]]; then
  echo "Set CATENA_E26_DATA_REPAIR_ACK=E26_ZERO_TOLERANCE_REPAIR_AUTHORIZED" >&2
  exit 2
fi
if [[ ! -x "$TOOL_PYTHON" ]]; then
  echo "Pinned E26 data-tool Python is missing: $TOOL_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

"$TOOL_PYTHON" "$REPO_ROOT/tools/repair_e26_zero_tolerance_data.py" \
  --config "$CONFIG_PATH" \
  --repo-root "$REPO_ROOT"

echo "E26 zero-tolerance data repair completed without GPU preflight or E26a."
