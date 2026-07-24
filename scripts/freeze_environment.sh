#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts/logs
python -m pip freeze | sort > artifacts/logs/pip-freeze.txt
python - <<'PY' > artifacts/logs/torch-environment.txt
import torch
print(torch.__config__.show())
PY
nvidia-smi -q > artifacts/logs/nvidia-smi-q.txt
nvcc --version > artifacts/logs/nvcc-version.txt 2>&1 || true
git status --short > artifacts/logs/git-status.txt 2>&1 || true
git rev-parse HEAD > artifacts/logs/git-commit.txt 2>&1 || true
echo "Environment frozen under artifacts/logs/."
