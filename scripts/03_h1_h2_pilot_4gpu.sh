#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

RUN_ID="h1_h2_pilot_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-inference --config configs/experiments/e03_h1.yaml --shard-index 0 --num-shards 4" \
CMD1="python -m catena.cli eval-inference --config configs/experiments/e03_h1.yaml --shard-index 1 --num-shards 4" \
CMD2="python -m catena.cli eval-inference --config configs/experiments/e03_h1.yaml --shard-index 2 --num-shards 4" \
CMD3="python -m catena.cli eval-inference --config configs/experiments/e03_h1.yaml --shard-index 3 --num-shards 4" \
bash scripts/launch_4gpu.sh
python -m catena.cli predictions-merge --input-root artifacts/metrics/e03_h1

RUN_ID="h2_repr_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-inference --config configs/experiments/e04_h2.yaml --shard-index 0 --num-shards 4" \
CMD1="python -m catena.cli eval-inference --config configs/experiments/e04_h2.yaml --shard-index 1 --num-shards 4" \
CMD2="python -m catena.cli eval-inference --config configs/experiments/e04_h2.yaml --shard-index 2 --num-shards 4" \
CMD3="python -m catena.cli eval-inference --config configs/experiments/e04_h2.yaml --shard-index 3 --num-shards 4" \
bash scripts/launch_4gpu.sh
python -m catena.cli predictions-merge --input-root artifacts/metrics/e04_h2
