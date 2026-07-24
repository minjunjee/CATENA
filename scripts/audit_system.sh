#!/usr/bin/env bash
set -euo pipefail

ROOT="${CATENA_DATA_ROOT:-$(pwd)/artifacts}"
OUT="${1:-artifacts/logs/system_audit_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "# CATENA system audit"
  date -Is
  echo
  echo "## OS"
  uname -a
  [[ -f /etc/os-release ]] && cat /etc/os-release
  echo
  echo "## CPU and memory"
  lscpu || true
  free -h || true
  echo
  echo "## Storage"
  df -hT || true
  echo
  echo "## NVIDIA"
  nvidia-smi || true
  echo
  nvidia-smi -L || true
  echo
  nvidia-smi topo -m || true
  echo
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,compute_cap --format=csv || true
  echo
  echo "## CUDA toolkit"
  command -v nvcc || true
  nvcc --version || true
  echo
  echo "## Compiler"
  gcc --version | head -1 || true
  g++ --version | head -1 || true
  echo
  echo "## Python / PyTorch"
  python -V || true
  python - <<'PY' || true
import json
try:
    import torch
    payload = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            payload["devices"].append({
                "index": i,
                "name": p.name,
                "total_memory": p.total_memory,
                "capability": torch.cuda.get_device_capability(i),
            })
    print(json.dumps(payload, indent=2))
except Exception as e:
    print("PyTorch audit failed:", repr(e))
PY
  echo
  echo "## Limits"
  ulimit -a || true
  echo
  echo "## Environment paths"
  echo "CATENA_DATA_ROOT=$ROOT"
  echo "HF_HOME=${HF_HOME:-}"
  echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-}"
} | tee "$OUT"

echo "Audit written to $OUT"
