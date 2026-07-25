#!/usr/bin/env bash
# Deprecated compatibility entry point.  It intentionally installs nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh

cat >&2 <<'MSG'
NOTICE: scripts/bootstrap_cuda130.sh is deprecated.
No environment or package changes will be made.
Running the E00 audit in the existing Conda environment 'catena' instead.
MSG

exec bash scripts/00_bootstrap_and_audit.sh "$@"
