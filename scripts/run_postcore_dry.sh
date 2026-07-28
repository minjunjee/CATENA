#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
ARTIFACT_ROOT_INPUT=${2:-/tmp/catena_postcore_dry_artifacts}

[[ -d "$REPO_ROOT/src/catena" ]] || {
  echo "[ERROR] Not a CATENA repository: $REPO_ROOT" >&2
  exit 1
}
if [[ -L "$ARTIFACT_ROOT_INPUT" ]]; then
  echo "[ERROR] Refusing a symlink dry-run target: $ARTIFACT_ROOT_INPUT" >&2
  exit 1
fi

ARTIFACT_ROOT=$(realpath -m -- "$ARTIFACT_ROOT_INPUT")
ARTIFACT_PARENT=$(dirname -- "$ARTIFACT_ROOT")
ARTIFACT_NAME=$(basename -- "$ARTIFACT_ROOT")
echo "[SAFETY] Canonical post-core dry target: $ARTIFACT_ROOT"
if [[ "$ARTIFACT_PARENT" != "/tmp" || "$ARTIFACT_NAME" != catena_postcore_dry* ]]; then
  echo "[ERROR] Refusing destructive dry cleanup outside a direct" >&2
  echo "        /tmp/catena_postcore_dry* child: $ARTIFACT_ROOT" >&2
  exit 1
fi
if [[ -e "$ARTIFACT_ROOT" ]] && mountpoint -q -- "$ARTIFACT_ROOT"; then
  echo "[ERROR] Refusing to remove a mounted dry-run target: $ARTIFACT_ROOT" >&2
  exit 1
fi
if [[ ${CATENA_DRY_VALIDATE_ONLY:-0} == 1 ]]; then
  echo "[PASS] Dry target validation only; no files were changed."
  exit 0
fi

rm -rf -- "$ARTIFACT_ROOT"
mkdir -p -- "$ARTIFACT_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Small matrix CPU dry-runs can become slower on hosts with very high default
# thread counts.  Bound BLAS/OpenMP threads for deterministic smoke latency.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

run() {
  local script=$1
  local config=$2
  shift 2
  echo "[DRY] $script"
  python "$script" --config "$config" --device cpu --artifact-root "$ARTIFACT_ROOT" --dry-run "$@"
}

run experiments/e10_learned_rank_scaling.py configs/e10_learned_rank_scaling.yaml
run experiments/e11_representation_control_coadaptation.py configs/e11_representation_control_coadaptation.yaml
run experiments/e12_control_algebra_lattice.py configs/e12_control_algebra_lattice.yaml
run experiments/e13a_r1_sequence_floor_throughput.py configs/e13a_r1_sequence_floor_throughput.yaml
run experiments/e13b_transactional_sequence_memory.py configs/e13b_transactional_sequence_memory.yaml --variant tied --seed 101 --ignore-calibration
run experiments/e13b_transactional_sequence_memory.py configs/e13b_transactional_sequence_memory.yaml --variant dual --seed 101 --ignore-calibration
python experiments/e13c_transactional_sequence_aggregate.py \
  --config configs/e13c_transactional_sequence_aggregate.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT" --dry-run
run experiments/e13a_r2_sequence_floor_throughput.py configs/e13a_r2_sequence_floor_throughput.yaml
run experiments/e13b_r1_transactional_sequence_memory.py configs/e13b_r1_transactional_sequence_memory.yaml --variant tied --seed 101 --ignore-calibration
run experiments/e13b_r1_transactional_sequence_memory.py configs/e13b_r1_transactional_sequence_memory.yaml --variant dual --seed 101 --ignore-calibration
python experiments/e13c_r1_transactional_sequence_aggregate.py \
  --config configs/e13c_r1_transactional_sequence_aggregate.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT" --dry-run
run experiments/e14_plan_continuation.py configs/e14_plan_continuation.yaml
run experiments/e15_official_backend_gate.py configs/e15_official_backend_gate.yaml
python experiments/e16_core_evidence_freeze.py \
  --config configs/e16_core_evidence_freeze.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT"

echo "[PASS] Post-core CPU dry run completed: $ARTIFACT_ROOT"
