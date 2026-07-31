#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/minjun_dev/CATENA_E26}"
ARTIFACT_ROOT="${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
CONFIG="${CONFIG:-configs/e26a_operator_data_gate.yaml}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${CATENA_EXECUTE_MAIN:-NO}" != "YES_I_HAVE_APPROVED" ]]; then
  cat <<EOF2
DRY PRINT ONLY. Set CATENA_EXECUTE_MAIN=YES_I_HAVE_APPROVED after explicit user approval.
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN experiments/e26a_operator_data_gate.py \\
  --config $CONFIG --artifact-root $ARTIFACT_ROOT --device cuda:0 --allow-main \\
  --protocol-lock <e26a_protocol_lock.json> \\
  --backend-manifest <optimized_backend_manifest.json> \\
  --tokenizer-manifest <tokenizer_manifest.json> \\
  --corpus-manifest <token_memmap_manifest.json>
EOF2
  exit 0
fi
: "${BACKEND_MANIFEST:?set BACKEND_MANIFEST}"
: "${PROTOCOL_LOCK:?set PROTOCOL_LOCK}"
: "${TOKENIZER_MANIFEST:?set TOKENIZER_MANIFEST}"
: "${CORPUS_MANIFEST:?set CORPUS_MANIFEST}"
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" experiments/e26a_operator_data_gate.py \
  --config "$CONFIG" --artifact-root "$ARTIFACT_ROOT" --device cuda:0 --allow-main \
  --protocol-lock "$PROTOCOL_LOCK" \
  --backend-manifest "$BACKEND_MANIFEST" \
  --tokenizer-manifest "$TOKENIZER_MANIFEST" \
  --corpus-manifest "$CORPUS_MANIFEST"
