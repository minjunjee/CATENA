#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
WAVE=${2:-1}
ARTIFACT_ROOT=${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}
CATENA_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
PYTHON_BIN=$(realpath "${CATENA_PYTHON:-$CATENA_V6_PREFIX/bin/python}")

case "$WAVE" in
  1)
    JOBS=("0 tied 101" "1 dual 101" "2 tied 211" "3 dual 211")
    ;;
  2)
    JOBS=("0 tied 307" "1 dual 307" "2 tied 401" "3 dual 401")
    ;;
  3)
    JOBS=("0 tied 503" "1 dual 503")
    ;;
  *)
    echo "Usage: $0 [repo_root] [1|2|3]" >&2
    exit 2
    ;;
esac

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

exec 9>"/tmp/catena_sequence_wave.launch.lock"
if ! flock -n 9; then
  echo "[ERROR] Another sequence-wave launcher is in preflight." >&2
  exit 1
fi

assert_sequence_job_idle() {
  local variant=$1
  local seed=$2
  local proc pid cmdline variant_match seed_match
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    pid=${proc##*/}
    [[ "$pid" != "$$" ]] || continue
    cmdline=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    [[ "$cmdline" == *python* ]] || continue
    [[ "$cmdline" == *e13b_transactional_sequence_memory* ]] || continue
    variant_match=0
    seed_match=0
    if [[ "$cmdline" == *"--variant $variant"* ]] || [[ "$cmdline" == *"--variant=$variant"* ]]; then
      variant_match=1
    fi
    if [[ "$cmdline" == *"--seed $seed"* ]] || [[ "$cmdline" == *"--seed=$seed"* ]]; then
      seed_match=1
    fi
    if [[ "$variant_match" -eq 1 && "$seed_match" -eq 1 ]]; then
      echo "[ERROR] Sequence target is already alive:" >&2
      echo "        pid=$pid variant=$variant seed=$seed command=$cmdline" >&2
      return 1
    fi
  done
}

# Check the complete selected wave before creating its log directory or
# starting any process. Different waves retain their registered seed schedule.
for job in "${JOBS[@]}"; do
  read -r _ variant seed <<<"$job"
  assert_sequence_job_idle "$variant" "$seed" || exit 1
done
echo "[SAFETY] catena-v6 interpreter: $PYTHON_BIN"
if [[ ${CATENA_LAUNCH_CHECK_ONLY:-0} == 1 ]]; then
  echo "[PASS] Sequence wave ${WAVE} preflight only; no jobs were started."
  exit 0
fi

LOG_ROOT="$ARTIFACT_ROOT/_launcher_logs/sequence_wave${WAVE}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for job in "${JOBS[@]}"; do
  read -r gpu variant seed <<<"$job"
  name="e13b_${variant}_seed${seed}"
  log="$LOG_ROOT/${name}.log"
  echo "[LAUNCH] GPU ${gpu}: ${name}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    "$PYTHON_BIN" -m experiments.e13b_transactional_sequence_memory \
      --config configs/e13b_transactional_sequence_memory.yaml \
      --variant "$variant" --seed "$seed" \
      --device cuda:0 --artifact-root "$ARTIFACT_ROOT" \
      >"$log" 2>&1 &
  echo $! > "$LOG_ROOT/${name}.pid"
done

echo "[DONE] Sequence wave ${WAVE} launched. Logs: $LOG_ROOT"
