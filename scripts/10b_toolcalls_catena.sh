#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

CHECKPOINT="${CATENA_H3_CHECKPOINT:-artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing CATENA checkpoint: $CHECKPOINT" >&2
  exit 2
fi
TMP="artifacts/logs/e11_catena_$(date +%Y%m%d_%H%M%S).yaml"
mkdir -p "$(dirname "$TMP")"
python - "$CHECKPOINT" "$TMP" <<'PY'
import sys, yaml
checkpoint, out = sys.argv[1:]
config = {
  'experiment': 'e11_naturalized_toolcall_catena',
  'output_dir': 'artifacts/metrics/e11_naturalized',
  'data_dir': 'data/processed/stress',
  'max_episodes': 300,
  'max_new_tokens': 96,
  'runs': [{'model':'configs/models/rwkv_fla_2.9b.yaml','policy':'catena','checkpoint':checkpoint}],
  'metrics': ['tool_name_exact','argument_exact_match','schema_validity','simulator_success','stale_field_rate'],
}
with open(out,'w',encoding='utf-8') as f: yaml.safe_dump(config,f,sort_keys=False)
PY
python -m catena.cli eval-toolcalls --config "$TMP" --run-index 0
