#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SESSION_NAME COMMAND [ARG ...]" >&2
  exit 2
fi
SESSION="$1"; shift
COMMAND="$*"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/artifacts/logs/$SESSION"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && source scripts/setup_paths.sh && $COMMAND 2>&1 | tee '$LOG_DIR/session.log'"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_DIR/session.log"
