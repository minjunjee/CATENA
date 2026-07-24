#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh
RUN_ID="toolcalls_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-toolcalls --config configs/experiments/e11_naturalized.yaml --run-index 0" \
CMD1="python -m catena.cli eval-toolcalls --config configs/experiments/e11_naturalized.yaml --run-index 1" \
CMD2="python -m catena.cli eval-toolcalls --config configs/experiments/e11_naturalized.yaml --run-index 2" \
CMD3="python -m catena.cli eval-toolcalls --config configs/experiments/e11_naturalized.yaml --run-index 3" \
bash scripts/launch_4gpu.sh
