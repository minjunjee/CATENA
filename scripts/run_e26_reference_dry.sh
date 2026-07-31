#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${2:-/tmp/catena_e26_dry_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
[[ -x "$PYTHON_BIN" ]] || {
  echo "Configured CATENA Python is not executable: $PYTHON_BIN" >&2
  exit 2
}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$PYTHON_BIN" experiments/e26a_operator_data_gate.py \
  --config configs/e26a_operator_data_gate.yaml \
  --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run
"$PYTHON_BIN" experiments/e26b_lm_calibration.py \
  --config configs/e26b_calibration_lock.yaml \
  --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run --steps 1
"$PYTHON_BIN" experiments/e26c_matched_lm_train.py \
  --config configs/e26c_main_train.yaml \
  --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run --steps 1
"$PYTHON_BIN" experiments/e26d_transaction_eval.py \
  --config configs/e26d_frozen_eval.yaml \
  --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run
"$PYTHON_BIN" experiments/e26e_gate_interventions.py \
  --config configs/e26e_mechanism.yaml \
  --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run
printf 'E26 reference dry-run complete: %s\n' "$ARTIFACT_ROOT"
