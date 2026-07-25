#!/usr/bin/env bash
# Source after require_catena_conda.sh and setup_paths.sh in E01+ stage scripts.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source scripts/require_e00_pass.sh from a CATENA stage script." >&2
  exit 2
fi

python - "$ROOT" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from catena.experiments.e00_audit import (
    _parse_nvidia_inventory,
    _source_tree_hash,
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: E00 prerequisite failed: {message}")


root = Path(sys.argv[1]).resolve()
profile_root = (root / "artifacts" / "profiles" / "e00_audit").resolve()
latest_path = profile_root / "latest.json"
if not latest_path.is_file():
    fail("latest.json is missing; run scripts/00_bootstrap_and_audit.sh")

latest = json.loads(latest_path.read_text(encoding="utf-8"))
if latest.get("passed") is not True:
    fail("the latest E00 run is not PASS")

run_dir = (root / str(latest.get("artifact_dir", ""))).resolve()
try:
    run_dir.relative_to(profile_root / "runs")
except ValueError:
    fail("latest artifact_dir escapes the E00 runs directory")
if not run_dir.is_dir():
    fail("latest E00 run directory is missing")

report_path = run_dir / "report.json"
manifest_path = run_dir / "manifest.json"
checksums_path = run_dir / "SHA256SUMS"
for required in (report_path, manifest_path, checksums_path):
    if not required.is_file():
        fail(f"required artifact is missing: {required.name}")

report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("passed") is not True or report.get("run_id") != latest.get("run_id"):
    fail("latest pointer and report disagree")
if hashlib.sha256(report_path.read_bytes()).hexdigest() != latest.get(
    "report_sha256"
):
    fail("report hash does not match latest.json")
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != latest.get(
    "manifest_sha256"
):
    fail("manifest hash does not match latest.json")

for line in checksums_path.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
        fail("SHA256SUMS contains an invalid row")
    target = (run_dir / relative).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError:
        fail("SHA256SUMS contains a path escape")
    if not target.is_file():
        fail(f"checksummed artifact is missing: {relative}")
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        fail(f"artifact checksum mismatch: {relative}")

config_path = (root / str(report["config_path"])).resolve()
try:
    config_path.relative_to(root)
except ValueError:
    fail("report config path escapes the repository")
if hashlib.sha256(config_path.read_bytes()).hexdigest() != report.get(
    "config_sha256"
):
    fail("E00 config changed after the passing audit; rerun E00")

git_files = subprocess.run(
    [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=60,
    check=False,
)
if git_files.returncode != 0:
    fail("could not enumerate the current source tree")
relative_files = [item for item in git_files.stdout.split("\0") if item]
if _source_tree_hash(root, relative_files) != report.get("git", {}).get(
    "source_tree_sha256"
):
    fail("repository source changed after the passing audit; rerun E00")

def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


current_package_map = {
    canonical_name(name): distribution.version
    for distribution in importlib.metadata.distributions()
    if (name := distribution.metadata.get("Name"))
}
current_packages = sorted(current_package_map.items())
current_package_text = "".join(
    f"{name}=={version}\n" for name, version in current_packages
)
if current_package_text != (run_dir / "package_snapshot.txt").read_text(
    encoding="utf-8"
):
    fail("Python package set changed after the passing audit; rerun E00")

conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    fail("CONDA_PREFIX is unavailable")
conda_snapshot = subprocess.run(
    ["conda", "list", "--explicit", "--prefix", conda_prefix],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=120,
    check=False,
)
if conda_snapshot.returncode != 0:
    fail("current Conda explicit package list could not be captured")
if conda_snapshot.stdout != (run_dir / "conda-explicit.txt").read_text(
    encoding="utf-8"
):
    fail("Conda package set changed after the passing audit; rerun E00")

inventory_command = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,"
        "compute_cap,mig.mode.current",
        "--format=csv,noheader,nounits",
    ],
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=60,
    check=False,
)
if inventory_command.returncode != 0:
    fail("current NVIDIA inventory could not be read")
current_inventory = _parse_nvidia_inventory(inventory_command.stdout)
inventory_check = next(
    (
        check
        for check in report.get("checks", [])
        if check.get("check_id") == "host_gpu_inventory"
    ),
    None,
)
if inventory_check is None:
    fail("passing report lacks the host GPU inventory gate")
if current_inventory != inventory_check.get("observed", {}).get("host_gpus"):
    fail("GPU inventory changed after the passing audit; rerun E00")

print(f"E00 prerequisite PASS: {latest['run_id']}")
PY
