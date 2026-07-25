#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh

# Launch four independent experiment lanes. Each command receives exactly one
# visible GPU and runs in the already-selected Conda environment.
CMD0="${CMD0:-python -m catena.cli smoke}"
CMD1="${CMD1:-python -m catena.cli smoke}"
CMD2="${CMD2:-python -m catena.cli smoke}"
CMD3="${CMD3:-python -m catena.cli smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: RUN_ID must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi

LOG_DIR="$ROOT/artifacts/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

declare -a PIDS

launch() {
  local gpu="$1"
  local cmd="$2"
  local log="$LOG_DIR/gpu${gpu}.log"

  echo "[GPU ${gpu}] ${cmd}"
  CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >"$log" 2>&1 &
  PIDS["$gpu"]=$!
  printf '%s\n' "${PIDS[$gpu]}" >"$LOG_DIR/gpu${gpu}.pid"
}

launch 0 "$CMD0"
launch 1 "$CMD1"
launch 2 "$CMD2"
launch 3 "$CMD3"

failed=0
for gpu in 0 1 2 3; do
  if wait "${PIDS[$gpu]}"; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s\n' "$status" >"$LOG_DIR/gpu${gpu}.status"
  printf '[GPU %s] exit status %s\n' "$gpu" "$status"
done

if (( failed != 0 )); then
  echo "ERROR: one or more GPU lanes failed. Logs: $LOG_DIR/" >&2
  exit 1
fi

echo "All lanes finished successfully. Logs: $LOG_DIR/"
