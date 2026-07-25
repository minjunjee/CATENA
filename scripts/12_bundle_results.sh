#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/require_catena_conda.sh
source scripts/setup_paths.sh
source scripts/require_e00_pass.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/artifacts/submission_bundle_${STAMP}"
mkdir -p "$OUT"

python -m compileall -q src
python -m pytest -q

cp -a configs "$OUT/"
cp -a docs "$OUT/"
cp -a artifacts/metrics "$OUT/" 2>/dev/null || true
cp -a artifacts/figures "$OUT/" 2>/dev/null || true
cp BASELINE_STATUS.md README.md pyproject.toml "$OUT/"

E00_ROOT="$ROOT/artifacts/profiles/e00_audit"
if [[ ! -f "$E00_ROOT/latest.json" ]]; then
  echo "ERROR: E00 latest.json is missing; run scripts/00_bootstrap_and_audit.sh first." >&2
  exit 2
fi

E00_RUN_REL="$(
  python - "$E00_ROOT/latest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if not payload.get("passed", False):
    raise SystemExit("latest E00 audit did not pass")
print(payload["artifact_dir"])
PY
)"
E00_RUN="$(repo_local_path E00_LATEST_RUN "$E00_RUN_REL")"
case "$E00_RUN" in
  "$E00_ROOT/runs/"*)
    [[ -d "$E00_RUN" ]] || {
      echo "ERROR: E00 latest run directory is missing: $E00_RUN" >&2
      exit 2
    }
    mkdir -p "$OUT/e00"
    cp "$E00_RUN/report.md" "$OUT/e00/report.md"
    python - "$E00_RUN/report.json" "$OUT/e00/report.sanitized.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

sanitized = {
    "schema_version": report["schema_version"],
    "experiment": report["experiment"],
    "run_id": report["run_id"],
    "created_at": report["created_at"],
    "passed": report["passed"],
    "status": report["status"],
    "config_path": report["config_path"],
    "config_sha256": report["config_sha256"],
    "resolved_config_sha256": report["resolved_config_sha256"],
    "git": {
        "commit": report.get("git", {}).get("commit"),
        "dirty": report.get("git", {}).get("dirty"),
        "source_tree_sha256": report.get("git", {}).get("source_tree_sha256"),
    },
    "environment": report.get("environment"),
    "host_gpu_count": report.get("host_gpu_count"),
    "selected_physical_gpus": report.get("selected_physical_gpus"),
    "public_gpu_rows": report.get("public_gpu_rows"),
    "storage": {
        key: report.get("storage", {}).get(key)
        for key in (
            "free_gib_before",
            "recommended_free_gib",
            "probe_size_mib",
            "repeats",
            "median_write_mib_s",
            "median_read_mib_s",
        )
    },
    "checks": [
        {
            "check_id": check["check_id"],
            "category": check["category"],
            "status": check["status"],
            "required": check["required"],
            "summary": check["summary"],
        }
        for check in report["checks"]
    ],
    "interpretation": report["interpretation"],
    "scientific_plan_change": report.get("scientific_plan_change"),
    "plan_changes": report["plan_changes"],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(sanitized, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
    shopt -s nullglob
    E00_DOCS=("$ROOT"/docs/E00_*.md)
    shopt -u nullglob
    if (( ${#E00_DOCS[@]} > 0 )); then
      mkdir -p "$OUT/e00/docs"
      cp "${E00_DOCS[@]}" "$OUT/e00/docs/"
    fi
    ;;
  *)
    echo "ERROR: E00 latest run is outside the expected profile directory." >&2
    exit 2
    ;;
esac

git rev-parse HEAD > "$OUT/git-sha.txt" 2>/dev/null || echo "not-a-git-checkout" > "$OUT/git-sha.txt"
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)
tar -czf "${OUT}.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "Bundle: ${OUT}.tar.gz"
