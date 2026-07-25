#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh

export PYTHONPATH="$ROOT/src"

exec python -m catena.experiments.e00_audit \
  --root "$ROOT" \
  --config configs/experiments/e00_audit.yaml
