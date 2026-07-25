#!/usr/bin/env bash
# Evaluation schema: configs/experiments/e07_h3_eval.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

MODEL="configs/models/rwkv_fla_2.9b.yaml"
DATA_MAIN="data/processed/main/test.jsonl"
DATA_STRESS="data/processed/stress/test.jsonl"

for split in main stress; do
  if [[ "$split" == "main" ]]; then DATA="$DATA_MAIN"; else DATA="$DATA_STRESS"; fi
  RUN_ID="h3_eval_${split}_$(date +%Y%m%d_%H%M%S)" \
  CMD0="python -m catena.cli eval-h3 --model $MODEL --checkpoint artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt --data $DATA --output artifacts/metrics/e07_h3/seed_11/$split --shard-index 0 --num-shards 1" \
  CMD1="python -m catena.cli eval-h3 --model $MODEL --checkpoint artifacts/checkpoints/e07_h3_main/seed_22/encoder_final.pt --data $DATA --output artifacts/metrics/e07_h3/seed_22/$split --shard-index 0 --num-shards 1" \
  CMD2="python -m catena.cli eval-h3 --model $MODEL --checkpoint artifacts/checkpoints/e07_h3_main/seed_33/encoder_final.pt --data $DATA --output artifacts/metrics/e07_h3/seed_33/$split --shard-index 0 --num-shards 1" \
  CMD3="python -m catena.cli eval-h3 --model $MODEL --checkpoint artifacts/checkpoints/e06_slot_sweep/generic_k8/seed_11/encoder_final.pt --data $DATA --output artifacts/metrics/e07_h3/generic_seed_11/$split --shard-index 0 --num-shards 1" \
  bash scripts/launch_4gpu.sh
done
