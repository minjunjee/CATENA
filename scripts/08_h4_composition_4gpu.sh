#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

H3_INIT_CHECKPOINT="${H3_INIT_CHECKPOINT:-artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt}"
if [[ ! -f "$H3_INIT_CHECKPOINT" ]]; then
  echo "Missing validated H3 checkpoint: $H3_INIT_CHECKPOINT" >&2
  echo "Set H3_INIT_CHECKPOINT=/path/to/encoder_final.pt" >&2
  exit 2
fi

# Teacher distributions for chain train/validation sets.
for split in train val; do
  RUN_ID="h4_teacher_${split}_$(date +%Y%m%d_%H%M%S)" \
  CMD0="python -m catena.cli teacher-cache --config configs/experiments/e09_teacher_cache.yaml --split $split --shard-index 0 --num-shards 4" \
  CMD1="python -m catena.cli teacher-cache --config configs/experiments/e09_teacher_cache.yaml --split $split --shard-index 1 --num-shards 4" \
  CMD2="python -m catena.cli teacher-cache --config configs/experiments/e09_teacher_cache.yaml --split $split --shard-index 2 --num-shards 4" \
  CMD3="python -m catena.cli teacher-cache --config configs/experiments/e09_teacher_cache.yaml --split $split --shard-index 3 --num-shards 4" \
  bash scripts/launch_4gpu.sh
  python -m catena.cli teacher-merge --output-dir artifacts/teacher_cache/rwkv_chains --split "$split" --num-shards 4
 done

# Wave 1: three composition seeds and the first matched distillation-only seed.
RUN_ID="h4_train_wave1_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h4 --config configs/experiments/e09_h4_train.yaml --seed 11 --init-checkpoint $H3_INIT_CHECKPOINT" \
CMD1="python -m catena.cli train-h4 --config configs/experiments/e09_h4_train.yaml --seed 22 --init-checkpoint $H3_INIT_CHECKPOINT" \
CMD2="python -m catena.cli train-h4 --config configs/experiments/e09_h4_train.yaml --seed 33 --init-checkpoint $H3_INIT_CHECKPOINT" \
CMD3="python -m catena.cli train-h4 --config configs/experiments/e09_h4_distill_only.yaml --seed 11 --init-checkpoint $H3_INIT_CHECKPOINT" \
bash scripts/launch_4gpu.sh

# Wave 2: finish the parameter/data/step-matched control seeds.  Two GPUs are left
# intentionally idle so a failed composition run can be restarted without delaying
# the controls; override CMD2/CMD3 manually if all wave-1 jobs were healthy.
RUN_ID="h4_train_controls_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h4 --config configs/experiments/e09_h4_distill_only.yaml --seed 22 --init-checkpoint $H3_INIT_CHECKPOINT" \
CMD1="python -m catena.cli train-h4 --config configs/experiments/e09_h4_distill_only.yaml --seed 33 --init-checkpoint $H3_INIT_CHECKPOINT" \
CMD2="python -m catena.cli smoke" \
CMD3="python -m catena.cli smoke" \
bash scripts/launch_4gpu.sh

echo "H4 training complete. Run scripts/08_eval_h4_4gpu.sh next."
