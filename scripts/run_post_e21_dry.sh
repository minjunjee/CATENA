#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 /tmp/catena_post_e21_dry_<fresh-name>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

POST_E21_REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
POST_E21_DRY_INPUT=$1
POST_E21_DRY_ROOT=$(realpath -m -- "$POST_E21_DRY_INPUT")
POST_E21_DRY_PARENT=$(dirname -- "$POST_E21_DRY_ROOT")
POST_E21_DRY_NAME=$(basename -- "$POST_E21_DRY_ROOT")
POST_E21_V6_PREFIX=$(realpath "${CATENA_V6_PREFIX:-/home/minjun_dev/miniconda3/envs/catena-v6}")
POST_E21_PYTHON=$(realpath "${CATENA_PYTHON:-$POST_E21_V6_PREFIX/bin/python}")

if [[ "$POST_E21_DRY_INPUT" != "$POST_E21_DRY_ROOT" ]]; then
  echo "[ERROR] Dry root must be an already-normalized absolute path." >&2
  exit 2
fi
if [[ "$POST_E21_DRY_PARENT" != "/tmp" || "$POST_E21_DRY_NAME" != catena_post_e21_dry_* ]]; then
  echo "[ERROR] Dry root must be a direct fresh /tmp/catena_post_e21_dry_* child." >&2
  exit 2
fi
if [[ -e "$POST_E21_DRY_ROOT" || -L "$POST_E21_DRY_ROOT" ]]; then
  echo "[ERROR] Dry root must not already exist: $POST_E21_DRY_ROOT" >&2
  exit 2
fi
if [[ ! -x "$POST_E21_PYTHON" ]]; then
  echo "[ERROR] catena-v6 Python is not executable: $POST_E21_PYTHON" >&2
  exit 2
fi
POST_E21_PYTHON_PREFIX=$("$POST_E21_PYTHON" -c 'import sys; print(sys.prefix)')
POST_E21_PYTHON_PREFIX=$(realpath "$POST_E21_PYTHON_PREFIX")
if [[ "$POST_E21_PYTHON_PREFIX" != "$POST_E21_V6_PREFIX" ]]; then
  echo "[ERROR] Refusing non-catena-v6 Python: $POST_E21_PYTHON" >&2
  exit 2
fi
if [[ ! -d "$POST_E21_REPO_ROOT/src/catena/post_e21" ]]; then
  echo "[ERROR] Not a Post-E21 CATENA repository: $POST_E21_REPO_ROOT" >&2
  exit 2
fi

echo "[SAFETY] Post-E21 dry root: $POST_E21_DRY_ROOT"
echo "[SAFETY] Python: $POST_E21_PYTHON"
if [[ ${CATENA_POST_E21_DRY_VALIDATE_ONLY:-0} == 1 ]]; then
  echo "[PASS] Validation only; no directory or experiment was created."
  exit 0
fi

umask 077
mkdir -- "$POST_E21_DRY_ROOT"
mkdir -- "$POST_E21_DRY_ROOT/_logs"
cd "$POST_E21_REPO_ROOT"
export PYTHONPATH="$POST_E21_REPO_ROOT/src:$POST_E21_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

declare -a POST_E21_RUN_RECORDS=()

latest_run() {
  local experiment_id=$1
  "$POST_E21_PYTHON" - "$POST_E21_DRY_ROOT" "$experiment_id" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
experiment_id = sys.argv[2]
pointer = root / experiment_id / "latest.json"
payload = json.loads(pointer.read_text(encoding="utf-8"))
run = Path(payload["run_dir"]).resolve()
run.relative_to(root)
if not (run / "report.json").is_file():
    raise SystemExit(f"missing completed report: {run}")
print(run)
PY
}

validate_dry_run() {
  local experiment_id=$1
  local run_dir=$2
  "$POST_E21_PYTHON" - "$POST_E21_DRY_ROOT" "$experiment_id" "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
experiment_id = sys.argv[2]
run = Path(sys.argv[3]).resolve()
run.relative_to(root)
report_path = run / "report.json"
manifest_path = run / "run_manifest.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("experiment_id") != experiment_id:
    raise SystemExit(f"experiment mismatch: {experiment_id}")
if manifest.get("run_mode") != "DRY_RUN" or report.get("run_mode") != "DRY_RUN":
    raise SystemExit(f"non-dry run detected: {experiment_id}")
if report.get("scientific_evidence") is True or report.get("claim_eligible") is True:
    raise SystemExit(f"claim-bearing dry run detected: {experiment_id}")
if not (run / "protocol_lock.json").is_file():
    raise SystemExit(f"protocol snapshot missing: {experiment_id}")
if not (run / "RESULTS_SUMMARY_KO.md").is_file():
    raise SystemExit(f"results summary missing: {experiment_id}")
PY
  POST_E21_RUN_RECORDS+=("$experiment_id=$run_dir")
}

validate_audit_preparation() {
  local experiment_id=$1
  local run_dir=$2
  "$POST_E21_PYTHON" - "$POST_E21_DRY_ROOT" "$experiment_id" "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
experiment_id = sys.argv[2]
run = Path(sys.argv[3]).resolve()
run.relative_to(root)
report = json.loads((run / "report.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
if manifest.get("experiment_id") != experiment_id:
    raise SystemExit(f"audit-preparation experiment mismatch: {experiment_id}")
if (
    manifest.get("run_mode") != "AUDIT_PREPARATION"
    or report.get("run_mode") != "AUDIT_PREPARATION"
    or report.get("stage") != "AUDIT_PREPARATION"
):
    raise SystemExit(f"invalid audit-preparation mode: {experiment_id}")
if report.get("scientific_evidence") is True or report.get("claim_eligible") is True:
    raise SystemExit(f"claim-bearing audit preparation detected: {experiment_id}")
required = (
    "protocol_lock.json",
    "RESULTS_SUMMARY_KO.md",
    "text_transaction_metrics.jsonl",
    "text_transaction_seed_metrics.jsonl",
)
missing = [name for name in required if not (run / name).is_file()]
if missing:
    raise SystemExit(f"audit-preparation artifacts missing: {missing}")
audit_artifacts = report.get("claim_gate", {}).get("audit_artifacts", {})
if set(audit_artifacts) != {"items", "population_lock", "review_template"}:
    raise SystemExit("audit-preparation report has an invalid artifact set")
for artifact_id, metadata in audit_artifacts.items():
    artifact_path = Path(metadata.get("path", "")).resolve()
    artifact_path.relative_to(run)
    if artifact_path.parent != run or not artifact_path.is_file():
        raise SystemExit(f"invalid audit-preparation artifact: {artifact_id}")
PY
  POST_E21_RUN_RECORDS+=("$experiment_id=$run_dir")
}

run_dry() {
  local experiment_id=$1
  local entrypoint=$2
  local config=$3
  shift 3
  echo "[DRY] $experiment_id"
  "$POST_E21_PYTHON" "$entrypoint" \
    --config "$config" \
    --device cpu \
    --artifact-root "$POST_E21_DRY_ROOT" \
    --dry-run \
    "$@" 2>&1 | tee "$POST_E21_DRY_ROOT/_logs/${experiment_id}.log"
  local completed
  completed=$(latest_run "$experiment_id")
  validate_dry_run "$experiment_id" "$completed"
  printf '%s\n' "$completed"
}

run_dry \
  e22a_locality_method_selection \
  experiments/e22a_locality_method_selection.py \
  configs/e22a_locality_method_selection.yaml
POST_E21_E22A_RUN=$(latest_run e22a_locality_method_selection)

run_dry \
  e22b_active_path_locality \
  experiments/e22b_active_path_locality.py \
  configs/e22b_active_path_locality.yaml \
  --selection-run "$POST_E21_E22A_RUN"
POST_E21_E22B_RUN=$(latest_run e22b_active_path_locality)

run_dry \
  e23a_product_poset_screen \
  experiments/e23a_product_poset_screen.py \
  configs/e23a_product_poset_screen.yaml

# E23b's locked dry path deliberately uses its internal non-evidence E22
# fixture.  A real E22b report is accepted only by E23b MAIN, which this
# orchestrator must never invoke.
run_dry \
  e23b_product_poset_confirmatory \
  experiments/e23b_product_poset_confirmatory.py \
  configs/e23b_product_poset_confirmatory.yaml

run_dry \
  e24a_approximate_rank_stress \
  experiments/e24a_approximate_rank_stress.py \
  configs/e24a_approximate_rank_stress.yaml

run_dry \
  e24b_behavioral_attainability_stress \
  experiments/e24b_behavioral_attainability_stress.py \
  configs/e24b_behavioral_attainability_stress.yaml

run_dry \
  e25a_official_gdn2_gate \
  experiments/e25a_official_gdn2_gate.py \
  configs/e25a_official_gdn2_gate.yaml \
  --stage gate

run_dry \
  e25b_text_transaction_anchor \
  experiments/e25b_text_transaction_anchor.py \
  configs/e25b_text_transaction_anchor.yaml
POST_E21_E25B_DRY_RUN=$(latest_run e25b_text_transaction_anchor)

echo "[AUDIT-PREP] e25b_text_transaction_anchor"
"$POST_E21_PYTHON" experiments/e25b_text_transaction_anchor.py \
  --config configs/e25b_text_transaction_anchor.yaml \
  --device cpu \
  --artifact-root "$POST_E21_DRY_ROOT" \
  --prepare-audit 2>&1 | tee \
  "$POST_E21_DRY_ROOT/_logs/e25b_text_transaction_anchor_audit_preparation.log"
POST_E21_E25B_AUDIT_RUN=$(latest_run e25b_text_transaction_anchor)
validate_audit_preparation \
  e25b_text_transaction_anchor \
  "$POST_E21_E25B_AUDIT_RUN"

"$POST_E21_PYTHON" - "$POST_E21_DRY_ROOT" "${POST_E21_RUN_RECORDS[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

root = Path(sys.argv[1]).resolve()
runs = []
for record in sys.argv[2:]:
    experiment_id, raw_path = record.split("=", 1)
    run = Path(raw_path).resolve()
    run.relative_to(root)
    report = run / "report.json"
    runs.append(
        {
            "experiment_id": experiment_id,
            "run_dir": str(run),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": 1,
    "run_mode": "DRY_RUN",
    "claim_eligible": False,
    "scientific_main_executed": False,
    "official_replication_executed": False,
    "audit_preparation_executed": True,
    "created_at_utc": datetime.now(UTC).isoformat(),
    "artifact_root": str(root),
    "runs": runs,
}
(root / "POST_E21_DRY_RUN_MANIFEST.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[PASS] Post-E21 dry pipeline completed: $POST_E21_DRY_ROOT"
echo "[INFO] E22a dependency: $POST_E21_E22A_RUN"
echo "[INFO] E22b dependency-validated dry run: $POST_E21_E22B_RUN"
echo "[INFO] E25b train/eval dry run: $POST_E21_E25B_DRY_RUN"
echo "[INFO] E25b audit preparation: $POST_E21_E25B_AUDIT_RUN"
