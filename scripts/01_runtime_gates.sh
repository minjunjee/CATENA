#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

# Four independent correctness lanes.  Scientific runs do not start until the
# differentiable main RWKV lane (GPU1) and the Qwen cache lane (GPU3) pass.
RUN_ID="runtime_gates_$(date +%Y%m%d_%H%M%S)" \
CMD0="python -m catena.cli runtime-gate --config configs/experiments/e01_runtime.yaml --model-index 0" \
CMD1="python -m catena.cli runtime-gate --config configs/experiments/e01_runtime.yaml --model-index 1" \
CMD2="python -m catena.cli runtime-gate --config configs/experiments/e01_runtime.yaml --model-index 2" \
CMD3="python -m catena.cli runtime-gate --config configs/experiments/e01_runtime.yaml --model-index 3" \
bash scripts/launch_4gpu.sh

echo "Runtime gates passed. Reports: artifacts/metrics/e01_runtime/model_*/runtime_gates.json"
