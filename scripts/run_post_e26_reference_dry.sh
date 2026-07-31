#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${2:-/tmp/catena_e26_dry_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
for spec in \
  'e27_oracle_decomposition.py e27_oracle_decomposition.yaml' \
  'e28a_locality_oracle_pareto.py e28a_locality_oracle_pareto.yaml' \
  'e28b_locality_learned_main.py e28b_locality_learned_main.yaml' \
  'e28c_locality_transfer.py e28c_locality_transfer.yaml' \
  'e29a_policy_correctness.py e29a_policy_correctness.yaml' \
  'e29b_quality_cost_regime.py e29b_quality_cost_regime.yaml' \
  'e30a_scale_anchor.py e30a_scale_anchor.yaml' \
  'e30b_domain_transfer.py e30b_domain_transfer.yaml' \
  'e30c_final_replication.py e30c_replication.yaml'; do
  read -r entry config <<<"$spec"
  "$PYTHON_BIN" "experiments/$entry" --config "configs/$config" \
    --artifact-root "$ARTIFACT_ROOT" --device cpu --dry-run
done
printf 'Post-E26 reference dry-run complete: %s\n' "$ARTIFACT_ROOT"
