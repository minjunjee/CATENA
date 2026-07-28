from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e15_official_backend_gate"
DEFAULT_CONFIG = "configs/e15_official_backend_gate.yaml"
_PLUGIN_RESERVED_KEYS = {
    "name",
    "repo_path",
    "status",
    "commit",
    "passed",
    "scientific_evidence",
}


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _authoritative_plugin_row(
    *,
    base: dict[str, Any],
    result: dict[str, Any],
    commit: str,
) -> tuple[dict[str, Any], bool]:
    passed = result.get("passed")
    scientific_evidence = result.get("scientific_evidence")
    if not isinstance(passed, bool):
        raise TypeError("backend gate plugin result.passed must be a boolean")
    if not isinstance(scientific_evidence, bool):
        raise TypeError(
            "backend gate plugin result.scientific_evidence must be a boolean"
        )
    if passed and not scientific_evidence:
        raise ValueError(
            "a passing official backend must explicitly set "
            "scientific_evidence=true"
        )
    plugin_payload = {
        key: value
        for key, value in result.items()
        if key not in _PLUGIN_RESERVED_KEYS
    }
    row = {
        **base,
        **plugin_payload,
        "status": "PASS" if passed else "FAIL",
        "commit": commit,
        "passed": passed,
        "scientific_evidence": bool(passed and scientific_evidence),
    }
    return row, bool(passed and scientific_evidence)


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    rows: list[dict[str, Any]] = []
    all_passed = True
    for backend in config["backends"]:
        name = str(backend["name"])
        repo_path = Path(str(backend["repo_path"]))
        expected_commit = str(backend["expected_commit"])
        plugin_module = str(backend["plugin_module"])
        row: dict[str, Any] = {"name": name, "repo_path": str(repo_path)}
        if args.dry_run:
            row.update({"status": "DRY_RUN", "scientific_evidence": False})
            rows.append(row)
            all_passed = False
            continue
        try:
            if not repo_path.exists():
                raise FileNotFoundError(f"backend repo not found: {repo_path}")
            head = _git_head(repo_path)
            if not expected_commit or head != expected_commit:
                raise RuntimeError(
                    f"commit mismatch for {name}: expected={expected_commit!r}, actual={head!r}"
                )
            module = importlib.import_module(plugin_module)
            if not hasattr(module, "run_backend_gate"):
                raise AttributeError(
                    f"{plugin_module} must expose run_backend_gate(config: dict)"
                )
            result = module.run_backend_gate(dict(backend))
            if not isinstance(result, dict):
                raise TypeError("backend gate plugin must return a dictionary")
            row, passed = _authoritative_plugin_row(
                base=row,
                result=result,
                commit=head,
            )
            all_passed = all_passed and passed
        except Exception as error:  # strict boundary report; no reference fallback
            row.update({
                "status": "NOT_CONFIGURED",
                "error": f"{type(error).__name__}: {error}",
                "scientific_evidence": False,
            })
            all_passed = False
        rows.append(row)

    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "backends": rows,
        "claim_gate": {
            "official_backend_ready": bool(not args.dry_run and all_passed),
            "allowed_claim": (
                "Official-backend claims only for rows with status=PASS and "
                "a pinned full commit SHA."
            ),
            "forbidden_claim": (
                "Reference or dry-run backends as official "
                "GDN2/KDA/KVEraser evidence."
            ),
        },
    }
    with (run_dir / "backend_gate_rows.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
