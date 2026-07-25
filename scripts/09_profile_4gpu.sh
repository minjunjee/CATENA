#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh
RUN_ID="system_profile_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli profile-system --config configs/experiments/e10_profile.yaml --model-index 0" \
CMD1="python -m catena.cli profile-system --config configs/experiments/e10_profile.yaml --model-index 1" \
CMD2="python -m catena.cli eval-inference --config configs/experiments/e08_transformer.yaml --max-episodes 100" \
CMD3="python -m catena.cli eval-inference --config configs/experiments/e04_h2.yaml --max-episodes 100" \
bash scripts/launch_4gpu.sh
