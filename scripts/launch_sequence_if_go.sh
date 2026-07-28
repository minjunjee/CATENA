#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=${1:-/home/minjun_dev/CATENA}
ARTIFACT_ROOT=${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - "$ARTIFACT_ROOT" <<'__CHECK__'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
pointer = root / "e13a_r1_sequence_floor_throughput" / "latest.json"
if not pointer.exists():
    raise SystemExit("[BLOCKED] Prospective E13a-R1 has not run.")
run_dir = Path(json.loads(pointer.read_text())["run_dir"])
report = json.loads((run_dir / "report.json").read_text())
if not report.get("claim_gate", {}).get("go_for_e13b", False):
    raise SystemExit(f"[BLOCKED] Prospective E13a-R1 did not open E13b: {run_dir}")
print(f"[GO] Prospective E13a-R1 opened E13b: {run_dir}")
__CHECK__
bash scripts/launch_sequence_wave.sh "$REPO_ROOT" 1
