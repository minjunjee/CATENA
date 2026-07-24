#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="artifacts/logs/final_freeze_${STAMP}"
mkdir -p "$OUT"

bash scripts/audit_system.sh "$OUT/system_audit.txt"
python -m pip freeze | sort > "$OUT/pip_freeze.txt"
git rev-parse HEAD > "$OUT/git_sha.txt" 2>/dev/null || echo "not-a-git-checkout" > "$OUT/git_sha.txt"
git status --porcelain > "$OUT/git_status.txt" 2>/dev/null || true
python -m compileall -q src
PYTHONPATH=src pytest -q | tee "$OUT/tests.txt"
for f in scripts/*.sh; do bash -n "$f"; done
sha256sum configs/data/*.yaml configs/models/*.yaml configs/experiments/*.yaml > "$OUT/config_sha256.txt"

cat <<MSG
Environment and config freeze written to $OUT.
Run only validation-selected checkpoints on test/stress data after this point.
Do not change generator, metrics, or thresholds after opening final test results.
MSG
