#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26}"
DATA_ROOT="${2:-/data/minjun_dev/CATENA/e26_data_v1}"
TOOL_PYTHON="$DATA_ROOT/.venv/bin/python"

if [[ ! -x "$TOOL_PYTHON" ]]; then
  echo "Pinned E26 data-tool environment is missing: $TOOL_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

"$TOOL_PYTHON" - <<'PY'
import importlib.metadata as metadata
expected = {
    "huggingface_hub": "1.26.0",
    "numpy": "2.4.4",
    "pyarrow": "25.0.0",
    "tokenizers": "0.23.1",
}
observed = {name: metadata.version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"pinned data-tool environment mismatch: {observed}")
PY

"$TOOL_PYTHON" "$REPO_ROOT/tools/validate_e26_data_v1.py" \
  --data-lock "$REPO_ROOT/configs/e26_data_lock_v1.yaml" \
  --construction-receipt \
    "$DATA_ROOT/validation/construction_source_receipt.json" \
  --source-inventory "$DATA_ROOT/source_manifest/fineweb_inventory.json" \
  --source-metadata "$DATA_ROOT/source_manifest/metadata/metadata_receipt.json" \
  --download-receipt "$DATA_ROOT/source_manifest/download_receipt.json" \
  --tokenizer-manifest "$DATA_ROOT/tokenizer/build/canonical/tokenizer_manifest.json" \
  --tokenizer-replay "$DATA_ROOT/tokenizer/build/tokenizer_replay_receipt.json" \
  --dedup-receipt "$DATA_ROOT/content_lock/dedup_receipt.json" \
  --near-duplicate-audit "$DATA_ROOT/validation/near_duplicate_audit.json" \
  --memmap-receipt "$DATA_ROOT/general/general_memmaps_receipt.json" \
  --transaction-manifest "$DATA_ROOT/transaction/transaction_replay_manifest.json" \
  --schedule-manifest "$DATA_ROOT/schedule/paired_schedule_manifest.json" \
  --expected-readiness "$DATA_ROOT/validation/scientific_data_readiness_v2.json" \
  --check-only

echo "E26 Stage-2 scientific data validation: PASS"
