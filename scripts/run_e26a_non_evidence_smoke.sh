#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26}"
ARTIFACT_ROOT="${2:-/tmp/catena_e26_dry_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

[[ -x "$PYTHON_BIN" ]] || {
  echo "Configured CATENA Python is not executable: $PYTHON_BIN" >&2
  exit 2
}
[[ "$ARTIFACT_ROOT" == /tmp/catena_e26_dry_* ]] || {
  echo "Smoke artifacts must use a fresh /tmp/catena_e26_dry_* root" >&2
  exit 2
}
[[ ! -e "$ARTIFACT_ROOT" ]] || {
  echo "Smoke artifact root already exists: $ARTIFACT_ROOT" >&2
  exit 2
}

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" \
  experiments/e26a_operator_data_gate.py \
  --config configs/e26a_operator_data_gate.yaml \
  --artifact-root "$ARTIFACT_ROOT" \
  --device cuda:0 \
  --dry-run \
  --non-evidence-smoke \
  --candidate-id d512_ctx4096 \
  --batch-size 1 \
  --sequence-length 256 \
  --warmup-steps 20 \
  --measured-steps 100
