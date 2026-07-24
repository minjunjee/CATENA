#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

for split in train val; do
  RUN_ID="rwkv_teacher_${split}_$(date +%Y%m%d_%H%M%S)" \
  CMD0="python -m catena.cli teacher-cache --config configs/experiments/e05_rwkv_teacher.yaml --split $split --shard-index 0 --num-shards 4" \
  CMD1="python -m catena.cli teacher-cache --config configs/experiments/e05_rwkv_teacher.yaml --split $split --shard-index 1 --num-shards 4" \
  CMD2="python -m catena.cli teacher-cache --config configs/experiments/e05_rwkv_teacher.yaml --split $split --shard-index 2 --num-shards 4" \
  CMD3="python -m catena.cli teacher-cache --config configs/experiments/e05_rwkv_teacher.yaml --split $split --shard-index 3 --num-shards 4" \
  bash scripts/launch_4gpu.sh
  python -m catena.cli teacher-merge --output-dir artifacts/teacher_cache/main --split "$split" --num-shards 4
done
