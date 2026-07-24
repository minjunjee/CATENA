#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

# 1) Inference-only repair policies. The config joins main and stress test sets.
RUN_ID="qwen_boundary_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-inference --config configs/experiments/e08_transformer.yaml --shard-index 0 --num-shards 4" \
CMD1="python -m catena.cli eval-inference --config configs/experiments/e08_transformer.yaml --shard-index 1 --num-shards 4" \
CMD2="python -m catena.cli eval-inference --config configs/experiments/e08_transformer.yaml --shard-index 2 --num-shards 4" \
CMD3="python -m catena.cli eval-inference --config configs/experiments/e08_transformer.yaml --shard-index 3 --num-shards 4" \
bash scripts/launch_4gpu.sh
python -m catena.cli predictions-merge --input-root artifacts/metrics/e08_transformer

# 2) Exact Qwen teacher distributions for a parameter-matched learned soft patch.
for split in train val; do
  RUN_ID="qwen_teacher_${split}_$(date +%Y%m%d_%H%M%S)" \
  CMD0="python -m catena.cli teacher-cache --config configs/experiments/e08_transformer_teacher.yaml --split $split --shard-index 0 --num-shards 4" \
  CMD1="python -m catena.cli teacher-cache --config configs/experiments/e08_transformer_teacher.yaml --split $split --shard-index 1 --num-shards 4" \
  CMD2="python -m catena.cli teacher-cache --config configs/experiments/e08_transformer_teacher.yaml --split $split --shard-index 2 --num-shards 4" \
  CMD3="python -m catena.cli teacher-cache --config configs/experiments/e08_transformer_teacher.yaml --split $split --shard-index 3 --num-shards 4" \
  bash scripts/launch_4gpu.sh
  python -m catena.cli teacher-merge --output-dir artifacts/teacher_cache/qwen_main --split "$split" --num-shards 4
done

# 3) Three learned-patch seeds; GPU3 profiles the Transformer while they train.
RUN_ID="qwen_soft_patch_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli train-h3 --config configs/experiments/e08_transformer_train.yaml --seed 11" \
CMD1="python -m catena.cli train-h3 --config configs/experiments/e08_transformer_train.yaml --seed 22" \
CMD2="python -m catena.cli train-h3 --config configs/experiments/e08_transformer_train.yaml --seed 33" \
CMD3="python -m catena.cli profile-system --config configs/experiments/e10_profile.yaml --model-index 1" \
bash scripts/launch_4gpu.sh

# 4) Evaluate learned patches on main test. Stress can be added after this gate.
RUN_ID="qwen_soft_patch_eval_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli eval-h3 --model configs/models/qwen2.5_3b.yaml --checkpoint artifacts/checkpoints/qwen_soft_patch/seed_11/encoder_final.pt --data data/processed/main/test.jsonl --output artifacts/metrics/e08_transformer/soft_patch_seed_11" \
CMD1="python -m catena.cli eval-h3 --model configs/models/qwen2.5_3b.yaml --checkpoint artifacts/checkpoints/qwen_soft_patch/seed_22/encoder_final.pt --data data/processed/main/test.jsonl --output artifacts/metrics/e08_transformer/soft_patch_seed_22" \
CMD2="python -m catena.cli eval-h3 --model configs/models/qwen2.5_3b.yaml --checkpoint artifacts/checkpoints/qwen_soft_patch/seed_33/encoder_final.pt --data data/processed/main/test.jsonl --output artifacts/metrics/e08_transformer/soft_patch_seed_33" \
CMD3="python -m catena.cli eval-h3 --model configs/models/qwen2.5_3b.yaml --checkpoint artifacts/checkpoints/qwen_soft_patch/seed_11/encoder_final.pt --data data/processed/stress/test.jsonl --output artifacts/metrics/e08_transformer/soft_patch_seed_11_stress --max-episodes 300" \
bash scripts/launch_4gpu.sh
