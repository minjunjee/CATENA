#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [REPOSITORY_LOCAL_OUTPUT_DIR]" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$(repo_local_path FREEZE_OUTPUT "${1:-artifacts/logs/environment_freeze_${STAMP}}")"
mkdir -p "$OUT"

python -m pip freeze | LC_ALL=C sort >"$OUT/pip-freeze.txt"
conda list --explicit >"$OUT/conda-explicit.txt"
python - <<'PY' >"$OUT/torch-environment.txt"
import torch
print(torch.__config__.show())
PY
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
nvcc --version >"$OUT/nvcc-version.txt" 2>&1 || true
git status --short >"$OUT/git-status.txt" 2>&1 || true
git rev-parse HEAD >"$OUT/git-commit.txt" 2>&1 || true

E00_ROOT="$ROOT/artifacts/profiles/e00_audit"
if [[ -f "$E00_ROOT/latest.json" ]]; then
  mkdir -p "$OUT/e00"
  cp "$E00_ROOT/latest.json" "$OUT/e00/latest.json"
  E00_RUN_REL="$(
    python - "$E00_ROOT/latest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["artifact_dir"])
PY
  )"
  E00_RUN="$(repo_local_path E00_LATEST_RUN "$E00_RUN_REL")"
  case "$E00_RUN" in
    "$E00_ROOT/runs/"*)
      [[ -d "$E00_RUN" ]] || {
        echo "ERROR: E00 latest run directory is missing: $E00_RUN" >&2
        exit 2
      }
      cp -a "$E00_RUN" "$OUT/e00/latest_run"
      ;;
    *)
      echo "ERROR: E00 latest run is outside the expected profile directory." >&2
      exit 2
      ;;
  esac
fi

shopt -s nullglob
E00_DOCS=("$ROOT"/docs/E00_*.md)
shopt -u nullglob
if (( ${#E00_DOCS[@]} > 0 )); then
  mkdir -p "$OUT/e00/docs"
  cp "${E00_DOCS[@]}" "$OUT/e00/docs/"
fi

(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)

echo "Environment freeze written to $OUT"
