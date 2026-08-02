#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/minjun_dev/CATENA}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
ARTIFACT_ROOT="${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/e26_stage3d_fixed_layout_bf16_admissibility.yaml}"
STAGE3C_RESULT="${STAGE3C_RESULT:-$REPO_ROOT/docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_RESULT_KO.md}"
STAGE3C_PROTOCOL="${STAGE3C_PROTOCOL:-/data/minjun_dev/CATENA/e26_stage3c_preflight_fd22ea544139/stage3c_lock_bundle/e26a_protocol_lock.json}"
STAGE3C_ARTIFACT_ROOT="${STAGE3C_ARTIFACT_ROOT:-$ARTIFACT_ROOT/e26_stage3c_numerical_preflight/20260802T060323Z}"
FROZEN_RECEIPT="${FROZEN_RECEIPT:-/data/minjun_dev/CATENA/e26_stage3c_preflight_fd22ea544139/frozen_invariance_receipt.json}"
DEVICES="${DEVICES:-0,1,2}"

if [[ "${CATENA_E26_STAGE3D_ACK:-NO}" != "FIXED_LAYOUT_BF16_PREFLIGHT_AUTHORIZED" ]]; then
  echo "Stage-3D is authorization-gated. Set:" >&2
  echo "  CATENA_E26_STAGE3D_ACK=FIXED_LAYOUT_BF16_PREFLIGHT_AUTHORIZED" >&2
  echo "This launcher never starts Scientific E26a." >&2
  exit 2
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ "$(realpath "$REPO_ROOT")" == "/home/minjun_dev/CATENA" ]] || {
  echo "Unexpected repository root: $REPO_ROOT" >&2
  exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "Stage-3D requires a clean committed source tree." >&2
  exit 2
}

RUN_ID="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
INPUT_LOCK_DIR="/data/minjun_dev/CATENA/e26_stage3d_input_locks/$RUN_ID"
RUN_DIR="$ARTIFACT_ROOT/e26_stage3d_fixed_layout_bf16_admissibility/$RUN_ID"
mkdir -m 0700 -p "$INPUT_LOCK_DIR"
mkdir -p "$(dirname "$RUN_DIR")"
STAGE3C_ARTIFACT_MANIFEST="$INPUT_LOCK_DIR/stage3c_artifact_hash_manifest.json"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$REPO_ROOT/tools/prepare_e26_stage3d_artifact_manifest.py" \
  --repo-root "$REPO_ROOT" \
  --stage3c-artifact-root "$STAGE3C_ARTIFACT_ROOT" \
  --output "$STAGE3C_ARTIFACT_MANIFEST"

set +e
"$PYTHON_BIN" "$REPO_ROOT/tools/run_e26_stage3d_preflight.py" \
  --repo-root "$REPO_ROOT" \
  --output-root "$RUN_DIR" \
  --config "$CONFIG" \
  --stage3c-result "$STAGE3C_RESULT" \
  --stage3c-protocol-lock "$STAGE3C_PROTOCOL" \
  --stage3c-artifact-manifest "$STAGE3C_ARTIFACT_MANIFEST" \
  --stage3c-artifact-root "$STAGE3C_ARTIFACT_ROOT" \
  --e00-e25-manifest "$FROZEN_RECEIPT" \
  --devices "$DEVICES"
RUNNER_EXIT=$?
set -e

echo "Stage-3D input lock: $INPUT_LOCK_DIR"
echo "Stage-3D run:        $RUN_DIR"
echo "Stage-3D exit:       $RUNNER_EXIT"

if [[ "$RUNNER_EXIT" -ne 0 ]]; then
  echo "Resource preflight was not started because Stage-3D did not GO."
  echo "Scientific E26a was not started."
  exit "$RUNNER_EXIT"
fi

# Exit code alone is insufficient authorization.  Bind the post-GO resource
# run to the canonical Stage-3D receipt and to the exact E26a inputs recorded
# by the immutable Stage-3C protocol.
if ! "$PYTHON_BIN" - "$RUN_DIR/report.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("disposition") != "STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE"
    or payload.get("passed") is not True
    or payload.get("scientific_e26a_started") is not False
):
    raise SystemExit("Stage-3D report is not an admissible GO receipt")
PY
then
  echo "Stage-3D GO receipt validation failed; resource preflight was not started." >&2
  echo "Scientific E26a was not started." >&2
  exit 2
fi

mapfile -t RESOURCE_INPUTS < <(
  "$PYTHON_BIN" - "$STAGE3C_PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = payload.get("execution_input_paths", {})
for name in ("config", "tokenizer_manifest", "corpus_manifest"):
    value = paths.get(name)
    if not isinstance(value, str) or not value.startswith("/"):
        raise SystemExit(f"Stage-3C protocol lacks an absolute {name} path")
    print(value)
PY
)
[[ "${#RESOURCE_INPUTS[@]}" -eq 3 ]] || {
  echo "Could not resolve the three exact Stage-3C resource inputs." >&2
  exit 2
}

RESOURCE_RUN_DIR="$ARTIFACT_ROOT/e26_stage3d_resource_preflight/$RUN_ID"
set +e
"$PYTHON_BIN" "$REPO_ROOT/tools/run_e26_stage3d_resource_preflight.py" \
  --repo-root "$REPO_ROOT" \
  --output-root "$RESOURCE_RUN_DIR" \
  --config "${RESOURCE_INPUTS[0]}" \
  --stage3d-receipt "$RUN_DIR/report.json" \
  --tokenizer-manifest "${RESOURCE_INPUTS[1]}" \
  --corpus-manifest "${RESOURCE_INPUTS[2]}" \
  --devices "$DEVICES"
RESOURCE_EXIT=$?
set -e

echo "Resource run:        $RESOURCE_RUN_DIR"
echo "Resource exit:       $RESOURCE_EXIT"
echo "Scientific E26a was not started."
exit "$RESOURCE_EXIT"
