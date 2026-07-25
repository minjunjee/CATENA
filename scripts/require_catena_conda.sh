#!/usr/bin/env bash
# Source from a repository stage script after changing to the repository root.
# If necessary, the calling script is re-executed once in the existing Conda
# environment.  This helper never creates or modifies an environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source scripts/require_catena_conda.sh from a CATENA stage script." >&2
  exit 2
fi

catena_active=0
if [[ "${CONDA_DEFAULT_ENV:-}" == "catena" && -n "${CONDA_PREFIX:-}" ]]; then
  resolved_prefix="$(realpath -m -- "$CONDA_PREFIX")"
  python_command="$(command -v python 2>/dev/null || true)"
  if [[ -n "$python_command" ]]; then
    resolved_python="$(realpath -m -- "$python_command")"
    case "$resolved_python" in
      "$resolved_prefix"/bin/python|"$resolved_prefix"/bin/python[0-9]*)
        if [[ "${resolved_prefix##*/}" == "catena" ]]; then
          catena_active=1
        fi
        ;;
    esac
  fi
fi

if (( catena_active == 1 )); then
  unset CATENA_CONDA_REEXEC
  return 0
fi

if [[ "${CATENA_CONDA_REEXEC:-0}" == "1" ]]; then
  echo "ERROR: conda run did not enter the required environment 'catena'." >&2
  return 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found; the existing environment 'catena' is required." >&2
  return 2
fi

caller="$0"
if [[ "$caller" != /* ]]; then
  if [[ -f "$caller" ]]; then
    caller="$PWD/$caller"
  elif [[ -n "${ROOT:-}" && -f "$ROOT/scripts/${caller##*/}" ]]; then
    caller="$ROOT/scripts/${caller##*/}"
  else
    echo "ERROR: cannot resolve the calling stage script: $0" >&2
    return 2
  fi
fi

export CATENA_CONDA_REEXEC=1
exec conda run --no-capture-output -n catena bash "$caller" "$@"
