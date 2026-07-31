#!/usr/bin/env bash
set -euo pipefail
if [[ "${CATENA_EXECUTE_MAIN:-NO}" != "YES_I_HAVE_APPROVED" ]]; then
  echo "DRY PRINT ONLY: E26d requires frozen E26b benchmark and E26c checkpoint manifests."
  exit 0
fi
echo "Refusing to evaluate from packet stub; use repository-integrated dependency resolver." >&2
exit 3
