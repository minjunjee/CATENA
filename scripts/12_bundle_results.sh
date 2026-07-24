#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
source scripts/setup_paths.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="artifacts/submission_bundle_${STAMP}"
mkdir -p "$OUT"

python -m compileall -q src
pytest -q
bash scripts/freeze_environment.sh

cp -a configs "$OUT/"
cp -a docs "$OUT/"
cp -a artifacts/metrics "$OUT/" 2>/dev/null || true
cp -a artifacts/profiles "$OUT/" 2>/dev/null || true
cp -a artifacts/figures "$OUT/" 2>/dev/null || true
cp -a artifacts/logs/host-audit* "$OUT/" 2>/dev/null || true
cp -a artifacts/logs/environment* "$OUT/" 2>/dev/null || true
cp BASELINE_STATUS.md README.md pyproject.toml "$OUT/"

git rev-parse HEAD > "$OUT/git-sha.txt" 2>/dev/null || echo "not-a-git-checkout" > "$OUT/git-sha.txt"
find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
tar -czf "${OUT}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "Bundle: ${OUT}.tar.gz"
