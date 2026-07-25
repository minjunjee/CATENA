#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh

validate_session() {
  local session="$1"
  if [[ ! "$session" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: session name must contain only letters, digits, dot, underscore, or hyphen." >&2
    return 2
  fi
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 2 ]]; then
    echo "ERROR: invalid internal tmux worker invocation." >&2
    exit 2
  fi
  SESSION="$2"
  validate_session "$SESSION"
  LOG_DIR="$ROOT/artifacts/logs/$SESSION"
  COMMAND_FILE="$LOG_DIR/command.sh"
  LOG_FILE="$LOG_DIR/session.log"
  STATUS_FILE="$LOG_DIR/exit.status"

  if [[ ! -f "$COMMAND_FILE" ]]; then
    echo "ERROR: missing tmux command file: $COMMAND_FILE" >&2
    exit 2
  fi

  COMMAND="$(<"$COMMAND_FILE")"
  set +e
  set -o pipefail
  bash -c "$COMMAND" 2>&1 | tee "$LOG_FILE"
  status=${PIPESTATUS[0]}
  printf '%s\n' "$status" >"$STATUS_FILE"
  exit "$status"
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SESSION_NAME COMMAND [ARG ...]" >&2
  exit 2
fi

SESSION="$1"
shift
validate_session "$SESSION"
COMMAND="$*"
LOG_DIR="$ROOT/artifacts/logs/$SESSION"
COMMAND_FILE="$LOG_DIR/command.sh"
LOG_FILE="$LOG_DIR/session.log"
STATUS_FILE="$LOG_DIR/exit.status"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  exit 1
fi

CONDA_EXE="$(command -v conda)"
printf '%s\n' "$COMMAND" >"$COMMAND_FILE"
rm -f "$STATUS_FILE"

printf -v tmux_command \
  'cd %q && exec %q run --no-capture-output -n catena bash scripts/run_in_tmux.sh --worker %q' \
  "$ROOT" "$CONDA_EXE" "$SESSION"
tmux new-session -d -s "$SESSION" "$tmux_command"

echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Log: $LOG_FILE"
echo "Exit status (written on completion): $STATUS_FILE"
