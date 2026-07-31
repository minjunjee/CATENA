#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/minjun_dev/CATENA_E26}"
ARTIFACT_ROOT="${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/e26a_operator_data_gate.yaml}"

if [[ "${CATENA_EXECUTE_E26A_GATE:-NO}" != "YES_I_HAVE_EXPLICIT_USER_APPROVAL" ]]; then
  cat <<'EOF'
DRY PRINT ONLY. This script never launches E26b or E26c.
After explicit user approval, set:
  CATENA_EXECUTE_E26A_GATE=YES_I_HAVE_EXPLICIT_USER_APPROVAL
and provide every hash-locked input variable printed below.
EOF
  cat <<EOF
$PYTHON_BIN $REPO_ROOT/experiments/e26a_operator_data_gate.py \\
  --config $CONFIG \\
  --artifact-root $ARTIFACT_ROOT \\
  --device <E26A_DEVICE_APPROVED_BY_RESOURCE_RECEIPT> \\
  --allow-main \\
  --execution-ack E26A_SCIENTIFIC_GATE_AUTHORIZED \\
  --protocol-lock <E26A_PROTOCOL_LOCK> \\
  --backend-candidate-lock <E26A_BACKEND_CANDIDATE_LOCK> \\
  --backend-manifest <E26A_BACKEND_MANIFEST> \\
  --tokenizer-manifest <E26A_TOKENIZER_MANIFEST> \\
  --corpus-manifest <E26A_CORPUS_MANIFEST> \\
  --data-lock <E26A_DATA_LOCK> \\
  --calibration-config $REPO_ROOT/configs/e26b_calibration_lock.yaml \\
  --data-readiness <E26A_DATA_READINESS> \\
  --transaction-manifest <E26A_TRANSACTION_MANIFEST> \\
  --validation-population-lock <E26A_VALIDATION_POPULATION_LOCK> \\
  --schedule-manifest <E26A_SCHEDULE_MANIFEST> \\
  --numerical-audit <E26A_NUMERICAL_AUDIT> \\
  --restart-audit <E26A_RESTART_AUDIT> \\
  --frozen-tree-receipt <E26A_FROZEN_TREE_RECEIPT> \\
  --resource-preflight <E26A_RESOURCE_PREFLIGHT> \\
  --expected-resource-preflight-sha256 <E26A_EXPECTED_RESOURCE_PREFLIGHT_SHA256>
EOF
  exit 0
fi

: "${E26A_PROTOCOL_LOCK:?set E26A_PROTOCOL_LOCK}"
: "${E26A_BACKEND_CANDIDATE_LOCK:?set E26A_BACKEND_CANDIDATE_LOCK}"
: "${E26A_BACKEND_MANIFEST:?set E26A_BACKEND_MANIFEST}"
: "${E26A_TOKENIZER_MANIFEST:?set E26A_TOKENIZER_MANIFEST}"
: "${E26A_CORPUS_MANIFEST:?set E26A_CORPUS_MANIFEST}"
: "${E26A_DATA_LOCK:?set E26A_DATA_LOCK}"
E26A_CALIBRATION_CONFIG="${E26A_CALIBRATION_CONFIG:-$REPO_ROOT/configs/e26b_calibration_lock.yaml}"
: "${E26A_DATA_READINESS:?set E26A_DATA_READINESS}"
: "${E26A_TRANSACTION_MANIFEST:?set E26A_TRANSACTION_MANIFEST}"
: "${E26A_VALIDATION_POPULATION_LOCK:?set E26A_VALIDATION_POPULATION_LOCK}"
: "${E26A_SCHEDULE_MANIFEST:?set E26A_SCHEDULE_MANIFEST}"
: "${E26A_NUMERICAL_AUDIT:?set E26A_NUMERICAL_AUDIT}"
: "${E26A_RESTART_AUDIT:?set E26A_RESTART_AUDIT}"
: "${E26A_FROZEN_TREE_RECEIPT:?set E26A_FROZEN_TREE_RECEIPT}"
: "${E26A_RESOURCE_PREFLIGHT:?set E26A_RESOURCE_PREFLIGHT}"
: "${E26A_EXPECTED_RESOURCE_PREFLIGHT_SHA256:?set E26A_EXPECTED_RESOURCE_PREFLIGHT_SHA256}"
: "${E26A_DEVICE:?set E26A_DEVICE to the cuda:N authorized by the resource receipt}"

[[ -x "$PYTHON_BIN" ]] || {
  echo "Configured Python is not executable: $PYTHON_BIN" >&2
  exit 2
}
[[ "$(realpath "$REPO_ROOT")" == "/home/minjun_dev/CATENA_E26" ]] || {
  echo "E26a must run from the isolated E26 worktree" >&2
  exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "E26a requires a clean committed worktree" >&2
  exit 2
}

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$PYTHON_BIN" experiments/e26a_operator_data_gate.py \
  --config "$CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --device "$E26A_DEVICE" \
  --allow-main \
  --execution-ack E26A_SCIENTIFIC_GATE_AUTHORIZED \
  --protocol-lock "$E26A_PROTOCOL_LOCK" \
  --backend-candidate-lock "$E26A_BACKEND_CANDIDATE_LOCK" \
  --backend-manifest "$E26A_BACKEND_MANIFEST" \
  --tokenizer-manifest "$E26A_TOKENIZER_MANIFEST" \
  --corpus-manifest "$E26A_CORPUS_MANIFEST" \
  --data-lock "$E26A_DATA_LOCK" \
  --calibration-config "$E26A_CALIBRATION_CONFIG" \
  --data-readiness "$E26A_DATA_READINESS" \
  --transaction-manifest "$E26A_TRANSACTION_MANIFEST" \
  --validation-population-lock "$E26A_VALIDATION_POPULATION_LOCK" \
  --schedule-manifest "$E26A_SCHEDULE_MANIFEST" \
  --numerical-audit "$E26A_NUMERICAL_AUDIT" \
  --restart-audit "$E26A_RESTART_AUDIT" \
  --frozen-tree-receipt "$E26A_FROZEN_TREE_RECEIPT" \
  --resource-preflight "$E26A_RESOURCE_PREFLIGHT" \
  --expected-resource-preflight-sha256 "$E26A_EXPECTED_RESOURCE_PREFLIGHT_SHA256"

# Intentional terminal boundary: this script has no E26b/E26c invocation.
