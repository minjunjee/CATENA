#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

# Stage B: main three-seed confirmation at the validation-selected K plus
# one structural ablation. The remaining no-closure/untyped/generic ablations
# are launched in a second wave only after the main run is healthy.
RUN_ID="h3_main_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h3 --config configs/experiments/e07_h3_main.yaml --seed 11" \
CMD1="python -m catena.cli train-h3 --config configs/experiments/e07_h3_main.yaml --seed 22" \
CMD2="python -m catena.cli train-h3 --config configs/experiments/e07_h3_main.yaml --seed 33" \
CMD3="python -m catena.cli train-h3 --config configs/experiments/e07_h3_no_closure.yaml --seed 11" \
bash scripts/launch_4gpu.sh

RUN_ID="h3_ablation_wave2_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h3 --config configs/experiments/e07_h3_untyped.yaml --seed 11" \
CMD1="python -m catena.cli train-h3 --config configs/experiments/e06_h3_generic_slots8.yaml --seed 11" \
CMD2="python -m catena.cli train-h3 --config configs/experiments/e06_h3_generic_slots8.yaml --seed 22" \
CMD3="python -m catena.cli train-h3 --config configs/experiments/e06_h3_generic_slots8.yaml --seed 33" \
bash scripts/launch_4gpu.sh
