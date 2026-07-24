#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

python -m catena.cli data-generate --config configs/data/smoke.yaml
python -m catena.cli data-generate --config configs/data/pilot.yaml
python -m catena.cli data-generate --config configs/data/main.yaml
python -m catena.cli data-generate --config configs/data/stress.yaml
python -m catena.cli data-generate-chains --config configs/data/chains.yaml

for path in \
  data/processed/smoke/test.jsonl \
  data/processed/pilot/test.jsonl \
  data/processed/main/train.jsonl \
  data/processed/main/test.jsonl \
  data/processed/stress/test.jsonl; do
  python -m catena.cli data-validate --path "$path"
done
