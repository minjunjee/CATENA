#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26}"
ARTIFACT_ROOT="${2:-/data/minjun_dev/CATENA/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
[[ -x "$PYTHON_BIN" ]] || {
  echo "Configured CATENA Python is not executable: $PYTHON_BIN" >&2
  exit 2
}
cd "$REPO_ROOT"
for experiment in \
  e26a_operator_data_gate e26b_lm_calibration e26c_matched_lm_train \
  e26d_transaction_eval e26e_gate_interventions e27_oracle_decomposition \
  e28a_locality_oracle_pareto e28b_locality_learned_main e28c_locality_transfer \
  e29a_policy_correctness e29b_quality_cost_regime e30a_scale_anchor \
  e30b_domain_transfer e30c_final_replication; do
  latest="$ARTIFACT_ROOT/$experiment/latest.json"
  if [[ -f "$latest" ]]; then
    printf '%-36s ' "$experiment"
    "$PYTHON_BIN" - "$latest" <<'PY'
import json, pathlib, sys
latest=json.loads(pathlib.Path(sys.argv[1]).read_text())
report=json.loads(pathlib.Path(latest['run_dir'],'report.json').read_text())
print(report.get('status'), report.get('disposition'), latest['run_id'])
PY
  else
    printf '%-36s NOT_RUN\n' "$experiment"
  fi
done
