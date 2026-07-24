#!/usr/bin/env bash
set -euo pipefail

# Launch four independent experiment lanes. Each command is written to a log and
# receives exactly one visible GPU. Replace the defaults with any shell command.
CMD0="${CMD0:-python -m catena.cli smoke}"
CMD1="${CMD1:-python -m catena.cli smoke}"
CMD2="${CMD2:-python -m catena.cli smoke}"
CMD3="${CMD3:-python -m catena.cli smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "artifacts/logs/${RUN_ID}"

launch() {
  local gpu="$1"; shift
  local cmd="$*"
  echo "[GPU ${gpu}] ${cmd}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "${cmd}" > "artifacts/logs/${RUN_ID}/gpu${gpu}.log" 2>&1 &
  echo $! > "artifacts/logs/${RUN_ID}/gpu${gpu}.pid"
}

launch 0 "${CMD0}"
launch 1 "${CMD1}"
launch 2 "${CMD2}"
launch 3 "${CMD3}"
wait

echo "All lanes finished. Logs: artifacts/logs/${RUN_ID}/"
