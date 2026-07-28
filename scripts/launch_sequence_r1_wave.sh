#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
WAVE=${2:-1}
ARTIFACT_ROOT=$(realpath -m "${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}")
CATENA_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
PYTHON_BIN=$(realpath "${CATENA_PYTHON:-$CATENA_V6_PREFIX/bin/python}")
R2_EXPERIMENT=e13a_r2_sequence_floor_throughput
B_R1_EXPERIMENT=e13b_r1_transactional_sequence_memory
B_R1_CONFIG="$REPO_ROOT/configs/e13b_r1_transactional_sequence_memory.yaml"

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
[[ -f "$B_R1_CONFIG" ]] || {
  echo "[ERROR] Missing E13b-R1 config: $B_R1_CONFIG" >&2
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

exec 9>"/tmp/catena_sequence_r1_wave.launch.lock"
if ! flock -n 9; then
  echo "[ERROR] Another E13b-R1 wave launcher is in preflight." >&2
  exit 1
fi

"$PYTHON_BIN" - "$ARTIFACT_ROOT" "$B_R1_CONFIG" "$R2_EXPERIMENT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


artifact_root = Path(sys.argv[1]).resolve()
source_config = Path(sys.argv[2]).resolve()
experiment_id = sys.argv[3]
experiment_root = (artifact_root / experiment_id).resolve()
pointer = experiment_root / "latest.json"
if not pointer.is_file():
    raise SystemExit("[BLOCKED] E13a-R2 has not produced a latest pointer.")
run_dir = Path(str(read_json(pointer)["run_dir"])).resolve()
if run_dir.parent != experiment_root:
    raise SystemExit("[BLOCKED] E13a-R2 latest pointer escapes its namespace.")
report_path = run_dir / "report.json"
manifest_path = run_dir / "run_manifest.json"
report = read_json(report_path)
manifest = read_json(manifest_path)
report_hash = sha256(report_path)
if (
    report.get("status") != "PASS"
    or not report.get("claim_gate", {}).get("go_for_e13b_r1", False)
    or not report.get("distractor_path_contract", {}).get("passed", False)
):
    raise SystemExit(f"[BLOCKED] E13a-R2 did not open E13b-R1: {run_dir}")
if (
    manifest.get("schema_version") != 2
    or manifest.get("experiment_id") != experiment_id
    or manifest.get("run_mode") != "MAIN"
    or manifest.get("run_id") != run_dir.name
    or manifest.get("report_sha256") != report_hash
):
    raise SystemExit("[BLOCKED] E13a-R2 MAIN manifest/report chain is invalid.")
expected_config_hash = (
    report.get("e13b_scale_feasibility", {})
    .get("source_config_file_sha256")
)
if expected_config_hash != sha256(source_config):
    raise SystemExit("[BLOCKED] E13b-R1 config changed after R2 calibration.")
print(f"[GO] Pinned E13a-R2 dependency: {run_dir}")
PY

assert_sequence_job_idle() {
  local variant=$1
  local seed=$2
  local proc pid cmdline
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    pid=${proc##*/}
    [[ "$pid" != "$$" ]] || continue
    cmdline=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    [[ "$cmdline" == *python* ]] || continue
    [[ "$cmdline" == *"$B_R1_EXPERIMENT"* ]] || continue
    if {
      [[ "$cmdline" == *"--variant $variant"* ]] \
        || [[ "$cmdline" == *"--variant=$variant"* ]];
    } && {
      [[ "$cmdline" == *"--seed $seed"* ]] \
        || [[ "$cmdline" == *"--seed=$seed"* ]];
    }; then
      echo "[ERROR] E13b-R1 target is already alive:" >&2
      echo "        pid=$pid variant=$variant seed=$seed command=$cmdline" >&2
      return 1
    fi
  done
}

assert_no_completed_target() {
  local variant=$1
  local seed=$2
  "$PYTHON_BIN" - "$ARTIFACT_ROOT" "$B_R1_EXPERIMENT" "$variant" "$seed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / sys.argv[2]
variant = sys.argv[3]
seed = int(sys.argv[4])
if not root.is_dir():
    raise SystemExit(0)
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "sequence_main_metrics.jsonl"
    if not (report_path.is_file() and manifest_path.is_file() and metrics_path.is_file()):
        continue
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if report.get("status") != "PASS" or manifest.get("run_mode") != "MAIN":
        continue
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("variant") == variant and int(row.get("seed", -1)) == seed:
            raise SystemExit(
                "[BLOCKED] Completed E13b-R1 target already exists: "
                f"{run_dir} variant={variant} seed={seed}"
            )
PY
}

for job in "${JOBS[@]}"; do
  read -r _ variant seed <<<"$job"
  assert_sequence_job_idle "$variant" "$seed" || exit 1
  assert_no_completed_target "$variant" "$seed" || exit 1
done

echo "[SAFETY] catena-v6 interpreter: $PYTHON_BIN"
echo "[SAFETY] artifact root: $ARTIFACT_ROOT"
if [[ ${CATENA_LAUNCH_CHECK_ONLY:-0} == 1 ]]; then
  echo "[PASS] E13b-R1 wave ${WAVE} preflight only; no jobs were started."
  exit 0
fi

LOG_ROOT="$ARTIFACT_ROOT/_launcher_logs/sequence_r1_wave${WAVE}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for job in "${JOBS[@]}"; do
  read -r gpu variant seed <<<"$job"
  name="e13b_r1_${variant}_seed${seed}"
  log="$LOG_ROOT/${name}.log"
  echo "[LAUNCH] GPU ${gpu}: ${name}"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    "$PYTHON_BIN" -m experiments.e13b_r1_transactional_sequence_memory \
      --config configs/e13b_r1_transactional_sequence_memory.yaml \
      --variant "$variant" --seed "$seed" \
      --device cuda:0 --artifact-root "$ARTIFACT_ROOT" \
      >"$log" 2>&1 &
  echo $! > "$LOG_ROOT/${name}.pid"
done

echo "[DONE] E13b-R1 wave ${WAVE} launched. Logs: $LOG_ROOT"
