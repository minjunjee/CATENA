#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26_STAGE3}"
DATA_ROOT="${2:?pass the exact repaired-data root as argument 2}"
TOOL_PYTHON="${3:-/data/minjun_dev/CATENA/e26_data_v1/.venv/bin/python}"

if [[ ! -x "$TOOL_PYTHON" ]]; then
  echo "Pinned E26 data-tool Python is missing: $TOOL_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

"$TOOL_PYTHON" "$REPO_ROOT/tools/validate_e26_data_v2.py" \
  --data-lock "$REPO_ROOT/configs/e26_data_lock_v2_zero_tolerance.yaml" \
  --repair-receipt "$DATA_ROOT/zero_tolerance_repair_receipt.json" \
  --source-receipt "$DATA_ROOT/repair_source_receipt.json" \
  --expected-readiness "$DATA_ROOT/scientific_data_readiness_v3.json" \
  --check-only

echo "E26 zero-tolerance data readiness-v3 validation: PASS"
