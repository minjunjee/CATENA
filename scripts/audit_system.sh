#!/usr/bin/env bash
# Deprecated compatibility entry point for the repository-local E00 audit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [REPOSITORY_LOCAL_POINTER_FILE]" >&2
  exit 2
fi

LEGACY_OUT=""
if [[ $# -eq 1 ]]; then
  LEGACY_OUT="$(repo_local_path AUDIT_POINTER "$1")"
fi

cat >&2 <<'MSG'
NOTICE: scripts/audit_system.sh is a compatibility shim.
The authoritative hardware/runtime audit is E00; no legacy host files are read.
MSG

if bash scripts/00_bootstrap_and_audit.sh; then
  status=0
else
  status=$?
fi

if [[ -n "$LEGACY_OUT" ]]; then
  mkdir -p "$(dirname "$LEGACY_OUT")"
  {
    echo "CATENA E00 compatibility pointer"
    echo "exit_status=$status"
    echo "latest=$ROOT/artifacts/profiles/e00_audit/latest.json"
    if [[ -f "$ROOT/artifacts/profiles/e00_audit/latest.json" ]]; then
      cat "$ROOT/artifacts/profiles/e00_audit/latest.json"
    fi
  } >"$LEGACY_OUT"
  echo "E00 pointer written to $LEGACY_OUT"
fi

exit "$status"
