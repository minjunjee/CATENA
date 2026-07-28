#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
ARTIFACT_ROOT=${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}
CATENA_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
PYTHON_BIN=$(realpath "${CATENA_PYTHON:-$CATENA_V6_PREFIX/bin/python}")

[[ -d "$REPO_ROOT/src/catena" ]] || {
  echo "[ERROR] Not a CATENA repository: $REPO_ROOT" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] catena-v6 Python is not executable: $PYTHON_BIN" >&2
  exit 1
}
PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
PYTHON_PREFIX=$(realpath "$PYTHON_PREFIX")
if [[ "$PYTHON_PREFIX" != "$CATENA_V6_PREFIX" ]]; then
  echo "[ERROR] Refusing a non-catena-v6 interpreter: $PYTHON_BIN" >&2
  echo "        expected prefix: $CATENA_V6_PREFIX" >&2
  echo "        actual prefix:   $PYTHON_PREFIX" >&2
  exit 1
fi

# Serialize the short preflight/spawn section so two launchers cannot both pass
# the process check before either has created its child processes.
exec 9>"/tmp/catena_postcore_wave1.launch.lock"
if ! flock -n 9; then
  echo "[ERROR] Another post-core wave-1 launcher is in preflight." >&2
  exit 1
fi

TARGET_EXPERIMENTS=(
  e10_learned_rank_scaling
  e11_representation_control_coadaptation
  e12_control_algebra_lattice
  e13a_r1_sequence_floor_throughput
)

assert_targets_idle() {
  local found=0
  local experiment_id proc pid cmdline
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    pid=${proc##*/}
    [[ "$pid" != "$$" ]] || continue
    cmdline=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    [[ "$cmdline" == *python* ]] || continue
    for experiment_id in "${TARGET_EXPERIMENTS[@]}"; do
      if [[ "$cmdline" == *"$experiment_id"* ]]; then
        echo "[ERROR] Target experiment is already alive:" >&2
        echo "        pid=$pid experiment=$experiment_id command=$cmdline" >&2
        found=1
      fi
    done
  done
  [[ "$found" -eq 0 ]]
}

assert_targets_idle || exit 1
echo "[SAFETY] catena-v6 interpreter: $PYTHON_BIN"
if [[ ${CATENA_LAUNCH_CHECK_ONLY:-0} == 1 ]]; then
  echo "[PASS] Wave-1 launch preflight only; no jobs were started."
  exit 0
fi

LOG_ROOT="$ARTIFACT_ROOT/_launcher_logs/postcore_wave1_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

launch() {
  local gpu=$1
  local name=$2
  local module=$3
  shift 3
  local log="$LOG_ROOT/${name}.log"
  echo "[LAUNCH] GPU ${gpu}: ${name} -> ${log}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    "$PYTHON_BIN" -m "$module" "$@" \
    --device cuda:0 --artifact-root "$ARTIFACT_ROOT" \
    >"$log" 2>&1 &
  echo $! > "$LOG_ROOT/${name}.pid"
}

launch 0 e10 experiments.e10_learned_rank_scaling \
  --config configs/e10_learned_rank_scaling.yaml
launch 1 e11 experiments.e11_representation_control_coadaptation \
  --config configs/e11_representation_control_coadaptation.yaml
launch 2 e12 experiments.e12_control_algebra_lattice \
  --config configs/e12_control_algebra_lattice.yaml
launch 3 e13a_r1 experiments.e13a_r1_sequence_floor_throughput \
  --config configs/e13a_r1_sequence_floor_throughput.yaml

echo "[DONE] Four independent jobs launched."
echo "[INFO] Logs: $LOG_ROOT"
echo "[INFO] Monitor: watch -n 2 nvidia-smi"
