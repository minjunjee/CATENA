#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/minjun_dev/CATENA_E26}"
ARTIFACT_ROOT="${CATENA_ARTIFACT_ROOT:-/data/minjun_dev/CATENA/artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/home/minjun_dev/miniconda3/envs/catena-v6/bin/python}"
CONFIG="${CONFIG:-configs/e26c_main_train.yaml}"
BACKEND_MANIFEST="${BACKEND_MANIFEST:-}"
SEEDS=("${@:-26011 26022}")
if [[ "${CATENA_EXECUTE_MAIN:-NO}" != "YES_I_HAVE_APPROVED" ]]; then
  echo "DRY PRINT ONLY. This reference packet does not define live seed CLI wiring."
  echo "Codex must integrate --seed/--variant and E26b protocol-lock dependency before execution."
  exit 0
fi
[[ -n "$BACKEND_MANIFEST" ]] || { echo "BACKEND_MANIFEST is required" >&2; exit 2; }
cd "$REPO_ROOT"
# The live integration must replace this fail-closed stub with its registered
# 4-lane wave launcher after E26b GO. It is intentionally not auto-executable.
echo "Refusing to launch E26c from the packet stub; use the repository-integrated launcher." >&2
exit 3
