#!/usr/bin/env bash
# Source this file before running CATENA. Every writable path is repository-local.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

repo_local_path() {
  local label="$1"
  local value="$2"
  local resolved
  resolved="$(realpath -m -- "$value")"
  case "$resolved" in
    "$ROOT"|"$ROOT"/*)
      printf '%s\n' "$resolved"
      ;;
    *)
      echo "ERROR: $label must stay inside $ROOT (got $resolved)." >&2
      return 1
      ;;
  esac
}

export CATENA_ROOT="$ROOT"
export CATENA_SCRATCH
CATENA_SCRATCH="$(repo_local_path CATENA_SCRATCH "${CATENA_SCRATCH:-$ROOT/.scratch}")"
export CATENA_DATA_ROOT
CATENA_DATA_ROOT="$(repo_local_path CATENA_DATA_ROOT "${CATENA_DATA_ROOT:-$ROOT/artifacts}")"
export HF_HOME
HF_HOME="$(repo_local_path HF_HOME "${HF_HOME:-$CATENA_SCRATCH/huggingface}")"
export TRANSFORMERS_CACHE
TRANSFORMERS_CACHE="$(
  repo_local_path TRANSFORMERS_CACHE "${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
)"
export HF_DATASETS_CACHE
HF_DATASETS_CACHE="$(
  repo_local_path HF_DATASETS_CACHE "${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
)"
export TORCH_EXTENSIONS_DIR
TORCH_EXTENSIONS_DIR="$(
  repo_local_path TORCH_EXTENSIONS_DIR "${TORCH_EXTENSIONS_DIR:-$CATENA_SCRATCH/torch_extensions}"
)"
export TMPDIR
TMPDIR="$(repo_local_path TMPDIR "${TMPDIR:-$CATENA_SCRATCH/tmp}")"
export XDG_CACHE_HOME
XDG_CACHE_HOME="$(
  repo_local_path XDG_CACHE_HOME "${XDG_CACHE_HOME:-$CATENA_SCRATCH/xdg_cache}"
)"
export XDG_CONFIG_HOME
XDG_CONFIG_HOME="$(
  repo_local_path XDG_CONFIG_HOME "${XDG_CONFIG_HOME:-$CATENA_SCRATCH/xdg_config}"
)"
export XDG_DATA_HOME
XDG_DATA_HOME="$(
  repo_local_path XDG_DATA_HOME "${XDG_DATA_HOME:-$CATENA_SCRATCH/xdg_data}"
)"
export CUDA_CACHE_PATH
CUDA_CACHE_PATH="$(
  repo_local_path CUDA_CACHE_PATH "${CUDA_CACHE_PATH:-$CATENA_SCRATCH/cuda_cache}"
)"
export MPLCONFIGDIR
MPLCONFIGDIR="$(
  repo_local_path MPLCONFIGDIR "${MPLCONFIGDIR:-$CATENA_SCRATCH/matplotlib}"
)"
export NUMBA_CACHE_DIR
NUMBA_CACHE_DIR="$(
  repo_local_path NUMBA_CACHE_DIR "${NUMBA_CACHE_DIR:-$CATENA_SCRATCH/numba}"
)"
export TRITON_CACHE_DIR
TRITON_CACHE_DIR="$(
  repo_local_path TRITON_CACHE_DIR "${TRITON_CACHE_DIR:-$CATENA_SCRATCH/triton}"
)"
export TORCHINDUCTOR_CACHE_DIR
TORCHINDUCTOR_CACHE_DIR="$(
  repo_local_path TORCHINDUCTOR_CACHE_DIR \
    "${TORCHINDUCTOR_CACHE_DIR:-$CATENA_SCRATCH/torchinductor}"
)"
export WANDB_DIR
WANDB_DIR="$(repo_local_path WANDB_DIR "${WANDB_DIR:-$CATENA_SCRATCH/wandb}")"
export WANDB_CACHE_DIR
WANDB_CACHE_DIR="$(
  repo_local_path WANDB_CACHE_DIR "${WANDB_CACHE_DIR:-$CATENA_SCRATCH/wandb_cache}"
)"
export WANDB_CONFIG_DIR
WANDB_CONFIG_DIR="$(
  repo_local_path WANDB_CONFIG_DIR "${WANDB_CONFIG_DIR:-$CATENA_SCRATCH/wandb_config}"
)"
export WANDB_DATA_DIR
WANDB_DATA_DIR="$(
  repo_local_path WANDB_DATA_DIR "${WANDB_DATA_DIR:-$CATENA_SCRATCH/wandb_data}"
)"
export PIP_CACHE_DIR
PIP_CACHE_DIR="$(
  repo_local_path PIP_CACHE_DIR "${PIP_CACHE_DIR:-$CATENA_SCRATCH/pip_cache}"
)"
export PYTHONPYCACHEPREFIX
PYTHONPYCACHEPREFIX="$(
  repo_local_path PYTHONPYCACHEPREFIX \
    "${PYTHONPYCACHEPREFIX:-$CATENA_SCRATCH/pycache}"
)"
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$CATENA_DATA_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$CUDA_CACHE_PATH" \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" \
  "$ROOT/artifacts/logs"
