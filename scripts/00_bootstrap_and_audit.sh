#!/usr/bin/env bash
# Audit schema: configs/experiments/e00_audit.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/bootstrap_cuda130.sh
source .venv/bin/activate
source scripts/setup_paths.sh
bash scripts/audit_system.sh
python -m pytest
python -m catena.cli config-audit
python -m catena.cli smoke
bash scripts/freeze_environment.sh
