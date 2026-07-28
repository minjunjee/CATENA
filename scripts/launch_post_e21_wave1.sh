#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "${1:-/home/minjun_dev/CATENA}")
ARTIFACT_ROOT=$(realpath "${2:-${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}}")
CATENA_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
PYTHON_BIN=$(realpath "${CATENA_PYTHON:-$CATENA_V6_PREFIX/bin/python}")

[[ -d "$REPO_ROOT/src/catena" ]] || {
  echo "[ERROR] Not a CATENA repository: $REPO_ROOT" >&2
  exit 1
}
[[ -x "$PYTHON_BIN" ]] || {
  echo "[ERROR] catena-v6 Python is unavailable: $PYTHON_BIN" >&2
  exit 1
}
[[ "$(realpath "$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')")" == "$CATENA_V6_PREFIX" ]] || {
  echo "[ERROR] E22-E24 launcher must use catena-v6: $PYTHON_BIN" >&2
  exit 1
}
[[ -f "$ARTIFACT_ROOT/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json" ]] || {
  echo "[ERROR] Missing immutable E18 dependency." >&2
  exit 1
}
[[ -f "$ARTIFACT_ROOT/E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json" ]] || {
  echo "[ERROR] Missing immutable E21 dependency." >&2
  exit 1
}

TARGETS=(
  e22a_locality_method_selection
  e23a_product_poset_screen
  e24a_approximate_rank_stress
  e24b_behavioral_attainability_stress
  e25a_official_gdn2_gate
  e25b_text_transaction_anchor
)
for process in /proc/[0-9]*; do
  [[ -r "$process/cmdline" ]] || continue
  command_line=$(tr '\0' ' ' < "$process/cmdline" 2>/dev/null || true)
  for target in "${TARGETS[@]}"; do
    if [[ "$command_line" == *python* && "$command_line" == *"$target"* ]]; then
      echo "[ERROR] Target already running: ${process##*/} $target" >&2
      exit 1
    fi
  done
done

echo "[SAFETY] Repository: $REPO_ROOT"
echo "[SAFETY] Canonical artifact root: $ARTIFACT_ROOT"
echo "[SAFETY] catena-v6 interpreter: $PYTHON_BIN"
echo "[PLAN] GPU0 E22a; GPU1 E23a; CPU E24a/E24b + E25b audit prep; GPU3 E25a parity"

if [[ ${CATENA_POST_E21_MAIN_ACK:-} != POST_E21_MAIN_AUTHORIZED ]]; then
  echo "[PASS] Preflight only. No process was started."
  echo "[INFO] Set CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED only after review."
  exit 0
fi

E25A_GATE_TERMINAL=0
if "$PYTHON_BIN" - "$ARTIFACT_ROOT" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pointer = root / "e25a_official_gdn2_gate" / "latest.json"
if not pointer.is_file():
    raise SystemExit(1)
run = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"]).resolve()
run.relative_to(root)
report = json.loads((run / "report.json").read_text(encoding="utf-8"))
terminal = (
    report.get("stage") == "GATE"
    and report.get("execution_status")
    in {"PASS", "FAIL", "NOT_CONFIGURED", "BLOCKED_DEPENDENCY"}
)
raise SystemExit(0 if terminal else 1)
PY
then
  E25A_GATE_TERMINAL=1
  echo "[SKIP] E25a already has a terminal parity-gate report; no automatic rerun."
fi

E25A_PREFIX=${CATENA_E25A_ENV_PREFIX:-}
if [[ "$E25A_GATE_TERMINAL" == 0 && ( -z "$E25A_PREFIX" || ! -x "$E25A_PREFIX/bin/python" ) ]]; then
  echo "[ERROR] E25a separate environment is not configured." >&2
  exit 1
fi
if [[ "$E25A_GATE_TERMINAL" == 0 ]]; then
  for variable in CATENA_GDN2_REPO CATENA_FLA_REPO CATENA_E25A_PLUGIN_SOURCE; do
    [[ -n ${!variable:-} ]] || {
      echo "[ERROR] Required E25a variable is unset: $variable" >&2
      exit 1
    }
  done
fi

exec 9>"/tmp/catena_post_e21_wave1.launch.lock"
flock -n 9 || {
  echo "[ERROR] Another Post-E21 launcher is in preflight." >&2
  exit 1
}

LOG_ROOT="$ARTIFACT_ROOT/_launcher_logs/post_e21_wave1_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

launch_catena() {
  local gpu=$1
  local name=$2
  local module=$3
  local config=$4
  shift 4
  local log="$LOG_ROOT/$name.log"
  echo "[LAUNCH] GPU $gpu: $name -> $log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    "$PYTHON_BIN" -m "$module" \
    --config "$config" --device cuda:0 --artifact-root "$ARTIFACT_ROOT" \
    "$@" \
    >"$log" 2>&1 &
  echo $! > "$LOG_ROOT/$name.pid"
}

launch_catena 0 e22a experiments.e22a_locality_method_selection \
  configs/e22a_locality_method_selection.yaml \
  --parent-e21-freeze "$ARTIFACT_ROOT/E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json"
launch_catena 1 e23a experiments.e23a_product_poset_screen \
  configs/e23a_product_poset_screen.yaml \
  --e18-freeze "$ARTIFACT_ROOT/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json"

E24A_LOG="$LOG_ROOT/e24a.log"
echo "[LAUNCH] CPU: e24a deterministic theory stress -> $E24A_LOG"
nohup env CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  "$PYTHON_BIN" -m experiments.e24a_approximate_rank_stress \
  --config configs/e24a_approximate_rank_stress.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT" \
  --allow-main --dependency-root "$ARTIFACT_ROOT" \
  >"$E24A_LOG" 2>&1 &
echo $! > "$LOG_ROOT/e24a.pid"

E24B_LOG="$LOG_ROOT/e24b.log"
echo "[LAUNCH] CPU: e24b deterministic theory stress -> $E24B_LOG"
nohup env CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  "$PYTHON_BIN" -m experiments.e24b_behavioral_attainability_stress \
  --config configs/e24b_behavioral_attainability_stress.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT" \
  --allow-main --dependency-root "$ARTIFACT_ROOT" \
  >"$E24B_LOG" 2>&1 &
echo $! > "$LOG_ROOT/e24b.pid"

E25B_LOG="$LOG_ROOT/e25b_audit_preparation.log"
echo "[LAUNCH] CPU: e25b locked 300-item audit preparation -> $E25B_LOG"
nohup env CATENA_ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  "$PYTHON_BIN" -m experiments.e25b_text_transaction_anchor \
  --config configs/e25b_text_transaction_anchor.yaml \
  --device cpu --artifact-root "$ARTIFACT_ROOT" \
  --prepare-audit \
  >"$E25B_LOG" 2>&1 &
echo $! > "$LOG_ROOT/e25b_audit_preparation.pid"

if [[ "$E25A_GATE_TERMINAL" == 0 ]]; then
  E25A_LOG="$LOG_ROOT/e25a.log"
  echo "[LAUNCH] GPU 3: e25a official parity -> $E25A_LOG"
  nohup env CUDA_VISIBLE_DEVICES=3 CATENA_OFFICIAL_DEVICE=cuda:0 \
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:/home/minjun_dev/CATENA_official/plugins${PYTHONPATH:+:$PYTHONPATH}" \
    "$E25A_PREFIX/bin/python" -m experiments.e25a_official_gdn2_gate \
    --config configs/e25a_official_gdn2_gate.yaml \
    --device cuda:0 --artifact-root "$ARTIFACT_ROOT" --stage gate \
    >"$E25A_LOG" 2>&1 &
  echo $! > "$LOG_ROOT/e25a.pid"
fi

echo "[DONE] Post-E21 wave-1 jobs started; terminal E25a gates are never rerun."
echo "[INFO] Logs: $LOG_ROOT"
