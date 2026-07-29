#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
ARTIFACT_ROOT=$(realpath "${2:-/data/minjun_dev/CATENA/artifacts}")
E18_FREEZE=$(realpath "${3:?usage: launch_e23b_sharded.sh REPO ARTIFACT E18_FREEZE E23A_RUN E22B_RUN SOURCE_LOCK_TAG EQUIVALENCE_REPORT}")
E23A_RUN=$(realpath "${4:?explicit E23a completed run/report required}")
E22B_RUN=$(realpath "${5:?explicit E22b completed run/report required}")
SOURCE_LOCK_TAG=${6:?annotated E23b source-lock tag required}
EQUIVALENCE_REPORT=$(realpath "${7:?dependency-bound CPU equivalence report required}")
CATENA_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
PYTHON_BIN=$(realpath "${CATENA_PYTHON:-$CATENA_V6_PREFIX/bin/python}")
GPU_LIST=${CATENA_E23B_GPUS:-0,1,2,3}

[[ -d "$REPO_ROOT/src/catena" ]] || {
  echo "[ERROR] Not a CATENA repository: $REPO_ROOT" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] catena-v6 Python is unavailable: $PYTHON_BIN" >&2
  exit 1
}
[[ "$(realpath "$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')")" == "$CATENA_V6_PREFIX" ]] || {
  echo "[ERROR] E23b sharded MAIN must use catena-v6: $PYTHON_BIN" >&2
  exit 1
}
[[ ${CATENA_POST_E21_MAIN_ACK:-} == POST_E21_MAIN_AUTHORIZED ]] || {
  echo "[ERROR] Missing CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED" >&2
  exit 1
}

cd "$REPO_ROOT"
[[ -z "$(git status --porcelain=v1)" ]] || {
  echo "[ERROR] Commit and source-lock the sharding amendment before MAIN." >&2
  git status --short >&2
  exit 1
}
SOURCE_COMMIT=$(git rev-parse HEAD)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
[[ ${#GPUS[@]} -eq 4 ]] || {
  echo "[ERROR] E23b MAIN requires exactly four registered GPU shards." >&2
  exit 1
}
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Invalid GPU index: $gpu" >&2
    exit 1
  }
  [[ -z ${SEEN_GPUS[$gpu]:-} ]] || {
    echo "[ERROR] Duplicate GPU index: $gpu" >&2
    exit 1
  }
  SEEN_GPUS[$gpu]=1
  active_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z "$active_pids" ]] || {
    echo "[ERROR] GPU $gpu already has compute processes: $active_pids" >&2
    exit 1
  }
done

EXPECTED_GPU_COUNT=${#GPUS[@]} CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" - <<'PY'
import os

import torch

count = torch.cuda.device_count()
expected = int(os.environ["EXPECTED_GPU_COUNT"])
if count != expected:
    raise SystemExit(f"Expected {expected} selected CUDA devices, observed {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
capabilities = [torch.cuda.get_device_capability(index) for index in range(count)]
if len(set(names)) != 1 or len(set(capabilities)) != 1:
    raise SystemExit(
        f"E23b conservative sharding requires homogeneous GPUs: {names}, {capabilities}"
    )
print(f"[PASS] Homogeneous CUDA workers: {count} x {names[0]} capability={capabilities[0]}")
PY

for process in /proc/[0-9]*; do
  [[ -r "$process/cmdline" ]] || continue
  command_line=$(tr '\0' ' ' <"$process/cmdline" 2>/dev/null || true)
  if [[ "$command_line" == *python* && "$command_line" == *e23b_product_poset_confirmatory* ]]; then
    echo "[ERROR] E23b process already running: ${process##*/} $command_line" >&2
    exit 1
  fi
done

exec 9>"/tmp/catena_e23b_sharded.launch.lock"
flock -n 9 || {
  echo "[ERROR] Another E23b sharded launcher is active." >&2
  exit 1
}

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORKSPACE=$(
  "$PYTHON_BIN" -m experiments.e23b_product_poset_confirmatory_sharded prepare \
    --config configs/e23b_product_poset_confirmatory.yaml \
    --artifact-root "$ARTIFACT_ROOT" \
    --e18-freeze "$E18_FREEZE" \
    --e23a-screen "$E23A_RUN" \
    --e22b-run "$E22B_RUN" \
    --source-lock-tag "$SOURCE_LOCK_TAG" \
    --equivalence-report "$EQUIVALENCE_REPORT" \
    --gpu-indices "$GPU_LIST" \
    --shard-count "${#GPUS[@]}"
)
WORKSPACE=$(realpath "$WORKSPACE")
echo "$SOURCE_COMMIT" >"$WORKSPACE/source_commit.txt"
echo "[PLAN] E23b seed-sharded workspace: $WORKSPACE"
echo "[PLAN] Frozen source commit: $SOURCE_COMMIT"

PIDS=()
for index in "${!GPUS[@]}"; do
  shard_id=$(printf 'shard_%02d' "$index")
  gpu=${GPUS[$index]}
  log="$WORKSPACE/logs/$shard_id.log"
  echo "[LAUNCH] $shard_id on physical GPU $gpu -> $log"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -m experiments.e23b_product_poset_confirmatory_sharded worker \
      --workspace "$WORKSPACE" \
      --shard-id "$shard_id" \
      --device cuda:0 \
      >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "$pid" >"$WORKSPACE/logs/$shard_id.pid"
done

worker_failure=0
for index in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$index]}"; then
    shard_id=$(printf 'shard_%02d' "$index")
    echo "[ERROR] $shard_id failed; partial artifacts are preserved." >&2
    worker_failure=1
  fi
done
if [[ "$worker_failure" != 0 ]]; then
  echo "[BLOCKED] No aggregate or latest pointer was created: $WORKSPACE" >&2
  exit 1
fi

test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT" || {
  echo "[ERROR] Source commit changed while E23b shards were running." >&2
  exit 1
}
[[ -z "$(git status --porcelain=v1)" ]] || {
  echo "[ERROR] Source became dirty while E23b shards were running." >&2
  exit 1
}

RUN_DIR=$(
  "$PYTHON_BIN" -m experiments.e23b_product_poset_confirmatory_sharded aggregate \
    --workspace "$WORKSPACE" \
    --artifact-root "$ARTIFACT_ROOT" \
    --device cpu
)
echo "[DONE] Canonical E23b artifact: $RUN_DIR"
echo "[INFO] Shard workspace retained: $WORKSPACE"
