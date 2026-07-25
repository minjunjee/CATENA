#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

# Stage A: one-seed slot sweep and a parameter-matched generic control.
# Run 300 steps first. After checking finite gradients and validation improvement,
# rerun with SWEEP_STEPS=2000-4000. Test data is not opened here.
if [[ "${PILOT_ONLY:-0}" == "1" ]]; then
  STEPS="${SWEEP_STEPS:-300}"
else
  STEPS="${SWEEP_STEPS:-2500}"
fi

RUN_ID="h3_slot_sweep_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h3 --config configs/experiments/e06_h3_slots4.yaml --seed 11 --max-steps $STEPS" \
CMD1="python -m catena.cli train-h3 --config configs/experiments/e06_h3_slots8.yaml --seed 11 --max-steps $STEPS" \
CMD2="python -m catena.cli train-h3 --config configs/experiments/e06_h3_slots16.yaml --seed 11 --max-steps $STEPS" \
CMD3="python -m catena.cli train-h3 --config configs/experiments/e06_h3_generic_slots8.yaml --seed 11 --max-steps $STEPS" \
bash scripts/launch_4gpu.sh

cat <<MSG
H3 slot sweep finished.
Select K on validation C_joint first, with KL/update latency as tie breakers.
The default main config remains K=8; copy the validation-selected slot count into configs/experiments/e07_h3_main.yaml before E07.
MSG
