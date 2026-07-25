#!/usr/bin/env bash
# Evaluation schema: configs/experiments/e09_h4_eval.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

MODEL="configs/models/rwkv_fla_2.9b.yaml"
DATA="data/processed/chains_main/chains/test.jsonl"

# Wave 1
RUN_ID="h4_eval_wave1_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_composition/seed_11/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/composition_seed_11" \
CMD1="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_composition/seed_22/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/composition_seed_22" \
CMD2="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_composition/seed_33/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/composition_seed_33" \
CMD3="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_distill_only/seed_11/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/control_seed_11" \
bash scripts/launch_4gpu.sh

# Wave 2
RUN_ID="h4_eval_wave2_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_distill_only/seed_22/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/control_seed_22" \
CMD1="python -m catena.cli eval-h4 --model $MODEL --checkpoint artifacts/checkpoints/h4_distill_only/seed_33/encoder_final.pt --data $DATA --output artifacts/metrics/e09_h4/control_seed_33" \
CMD2="python -m catena.cli smoke" \
CMD3="python -m catena.cli smoke" \
bash scripts/launch_4gpu.sh

echo "H4 evaluation complete: artifacts/metrics/e09_h4/"
