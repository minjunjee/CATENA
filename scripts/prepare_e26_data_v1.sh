#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/minjun_dev/CATENA_E26}"
DATA_ROOT="${2:-/data/minjun_dev/CATENA/e26_data_v1}"
BASE_PYTHON="${3:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
TOOL_ENV="$DATA_ROOT/.venv"

if [[ ! -d "$REPO_ROOT/.git" && ! -f "$REPO_ROOT/.git" ]]; then
  echo "E26 worktree not found: $REPO_ROOT" >&2
  exit 2
fi
if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base Python not executable: $BASE_PYTHON" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"
if [[ ! -x "$TOOL_ENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$TOOL_ENV"
  "$TOOL_ENV/bin/python" -m pip install \
    --disable-pip-version-check \
    -r "$REPO_ROOT/configs/e26_data_tooling_requirements.txt"
fi

"$TOOL_ENV/bin/python" - <<'PY'
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
print(f"E26 data tooling: {observed}")
PY

export PYTHONPATH="$REPO_ROOT/src"
export HF_HOME="$DATA_ROOT/.hf_home"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

mkdir -p \
  "$DATA_ROOT/source_manifest" \
  "$DATA_ROOT/source_parquet" \
  "$DATA_ROOT/content_lock" \
  "$DATA_ROOT/tokenizer" \
  "$DATA_ROOT/transaction" \
  "$DATA_ROOT/schedule" \
  "$DATA_ROOT/validation"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/resolve_e26_fineweb.py" \
  --inventory-output "$DATA_ROOT/source_manifest/fineweb_inventory.json" \
  --metadata-root "$DATA_ROOT/source_manifest/metadata" \
  --download-root "$DATA_ROOT/source_parquet" \
  --download-receipt "$DATA_ROOT/source_manifest/download_receipt.json" \
  --expansion-additions 1 \
  --capacity-prior-validation-tokens 4971104 \
  --capacity-required-validation-tokens 5000000 \
  --capacity-prior-build "$DATA_ROOT/general/build_v1"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/prepare_e26_document_index.py" \
  --download-receipt "$DATA_ROOT/source_manifest/download_receipt.json" \
  --sqlite-output "$DATA_ROOT/content_lock/documents.sqlite3" \
  --document-manifest-output "$DATA_ROOT/content_lock/documents.jsonl" \
  --dedup-receipt-output "$DATA_ROOT/content_lock/dedup_receipt.json"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/prepare_e26_tokenizer.py" \
  --download-receipt "$DATA_ROOT/source_manifest/download_receipt.json" \
  --document-index "$DATA_ROOT/content_lock/documents.sqlite3" \
  --output-root "$DATA_ROOT/tokenizer/build"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/audit_e26_near_duplicates.py" \
  --document-index "$DATA_ROOT/content_lock/documents.sqlite3" \
  --output "$DATA_ROOT/validation/near_duplicate_audit.json" \
  --invalid-attempt-command \
    "tools/audit_e26_near_duplicates.py --document-index $DATA_ROOT/content_lock/documents.sqlite3 --output $DATA_ROOT/validation/near_duplicate_audit_invalid_v1.json" \
  --invalid-attempt-output "NONE_NO_ARTIFACT"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/prepare_e26_memmaps.py" \
  --download-receipt "$DATA_ROOT/source_manifest/download_receipt.json" \
  --document-index "$DATA_ROOT/content_lock/documents.sqlite3" \
  --tokenizer-manifest "$DATA_ROOT/tokenizer/build/canonical/tokenizer_manifest.json" \
  --output-root "$DATA_ROOT/general"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/prepare_e26_transactions.py" \
  --output "$DATA_ROOT/transaction/transaction_replay_manifest.json"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/prepare_e26_schedule.py" \
  --train-corpus-manifest \
    "$DATA_ROOT/general/general_train/general_train.corpus_manifest.json" \
  --tokenizer-manifest "$DATA_ROOT/tokenizer/build/canonical/tokenizer_manifest.json" \
  --output "$DATA_ROOT/schedule/paired_schedule_manifest.json"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/lock_e26_construction_source.py" \
  --repo-root "$REPO_ROOT" \
  --output "$DATA_ROOT/validation/construction_source_receipt.json" \
  --artifact "data_lock=$REPO_ROOT/configs/e26_data_lock_v1.yaml" \
  --artifact "source_inventory=$DATA_ROOT/source_manifest/fineweb_inventory.json" \
  --artifact \
    "source_metadata=$DATA_ROOT/source_manifest/metadata/metadata_receipt.json" \
  --artifact "download_receipt=$DATA_ROOT/source_manifest/download_receipt.json" \
  --artifact "dedup_receipt=$DATA_ROOT/content_lock/dedup_receipt.json" \
  --artifact \
    "tokenizer_manifest=$DATA_ROOT/tokenizer/build/canonical/tokenizer_manifest.json" \
  --artifact \
    "tokenizer_replay=$DATA_ROOT/tokenizer/build/tokenizer_replay_receipt.json" \
  --artifact \
    "near_duplicate_audit=$DATA_ROOT/validation/near_duplicate_audit.json" \
  --artifact "memmap_receipt=$DATA_ROOT/general/general_memmaps_receipt.json" \
  --artifact \
    "transaction_manifest=$DATA_ROOT/transaction/transaction_replay_manifest.json" \
  --artifact \
    "schedule_manifest=$DATA_ROOT/schedule/paired_schedule_manifest.json"

"$TOOL_ENV/bin/python" "$REPO_ROOT/tools/validate_e26_data_v1.py" \
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
  --output "$DATA_ROOT/validation/scientific_data_readiness_v2.json"

echo "E26 Stage-2 data preparation completed without starting E26a."
