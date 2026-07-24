#!/usr/bin/env bash
# Source this file from the repository root before downloading models or compiling kernels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CATENA_ROOT="${CATENA_ROOT:-$ROOT}"
export CATENA_SCRATCH="${CATENA_SCRATCH:-$CATENA_ROOT/.scratch}"
export HF_HOME="${HF_HOME:-$CATENA_SCRATCH/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CATENA_SCRATCH/torch_extensions}"
export TMPDIR="${TMPDIR:-$CATENA_SCRATCH/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" artifacts/logs
