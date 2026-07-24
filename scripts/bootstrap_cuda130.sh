#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} was not found. Install Python 3.11 or set PYTHON_BIN." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging ninja

# The PyTorch CUDA wheel bundles its own CUDA runtime. The system CUDA 13.0 toolkit
# is used only when a package builds custom CUDA extensions.
python -m pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[models,train,dev]'

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

echo
cat <<'MSG'
Base environment installed.

Next steps:
  source .venv/bin/activate
  bash scripts/audit_system.sh
  python -m pytest
  python -m catena.cli smoke

Install the FLA RWKV backend only after the base smoke tests pass:
  bash scripts/install_rwkv_fla.sh
MSG
