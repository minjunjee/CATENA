#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/setup_paths.sh

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the project virtual environment first." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-16}"
FLA_REF="${FLA_REF:-main}"

nvcc --version
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
print("torch", torch.__version__, "runtime", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

mkdir -p vendor artifacts/logs
if [[ ! -d vendor/flash-linear-attention/.git ]]; then
  git clone https://github.com/fla-org/flash-linear-attention.git vendor/flash-linear-attention
fi
git -C vendor/flash-linear-attention fetch --all --tags
git -C vendor/flash-linear-attention checkout "$FLA_REF"
FLA_COMMIT="$(git -C vendor/flash-linear-attention rev-parse HEAD)"
echo "$FLA_COMMIT" > artifacts/logs/fla-commit.txt

python -m pip install --no-build-isolation -e vendor/flash-linear-attention
python -m pip freeze | sort > artifacts/logs/pip-freeze-after-fla.txt

cat <<MSG
FLA installation completed at commit $FLA_COMMIT.
Run the hard gate next:
  python -m catena.cli model-smoke --model configs/models/rwkv_fla_0.4b_debug.yaml
  python -m catena.cli model-smoke --model configs/models/rwkv_fla_2.9b.yaml
MSG
