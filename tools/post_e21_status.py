#!/usr/bin/env python3
"""Read-only status view for the additive E22--E25 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENTS = (
    "e22a_locality_method_selection",
    "e22b_active_path_locality",
    "e23a_product_poset_screen",
    "e23b_product_poset_confirmatory",
    "e24a_approximate_rank_stress",
    "e24b_behavioral_attainability_stress",
    "e25a_official_gdn2_gate",
    "e25b_text_transaction_anchor",
)

DEPENDENCY_ANCHORS = (
    "E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json",
    "E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json",
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def experiment_status(root: Path, experiment_id: str) -> dict[str, Any]:
    pointer_path = root / experiment_id / "latest.json"
    if not pointer_path.is_file():
        return {
            "experiment_id": experiment_id,
            "state": "NOT_RUN",
            "latest": None,
        }
    pointer = _read_object(pointer_path)
    raw_run_dir = pointer.get("run_dir")
    if not isinstance(raw_run_dir, str):
        return {
            "experiment_id": experiment_id,
            "state": "INVALID_LATEST",
            "latest": str(pointer_path),
        }
    run_dir = Path(raw_run_dir).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError:
        return {
            "experiment_id": experiment_id,
            "state": "UNSAFE_LATEST",
            "latest": str(run_dir),
        }
    report_path = run_dir / "report.json"
    report = _read_object(report_path)
    if not report:
        return {
            "experiment_id": experiment_id,
            "state": "INCOMPLETE",
            "latest": str(run_dir),
        }
    claim = report.get("claim_gate")
    claim_status = claim.get("status") if isinstance(claim, dict) else None
    return {
        "experiment_id": experiment_id,
        "state": str(report.get("execution_status", report.get("status", "UNKNOWN"))),
        "run_mode": report.get("run_mode"),
        "claim_status": claim_status,
        "claim_eligible": bool(report.get("claim_eligible", False)),
        "evidence_tier": report.get("evidence_tier"),
        "latest": str(run_dir),
    }


def build_status(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    dependencies = {
        name: {
            "path": str((resolved / name).resolve()),
            "exists": (resolved / name).is_file(),
        }
        for name in DEPENDENCY_ANCHORS
    }
    return {
        "artifact_root": str(resolved),
        "dependencies": dependencies,
        "experiments": [
            experiment_status(resolved, experiment_id) for experiment_id in EXPERIMENTS
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/artifacts"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_status(args.artifact_root)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"Artifact root: {payload['artifact_root']}")
    for name, descriptor in payload["dependencies"].items():
        state = "PASS" if descriptor["exists"] else "MISSING"
        print(f"{name:<52} {state}")
    print()
    print(f"{'experiment':<43} {'execution':<18} {'claim':<48} {'eligible':<8}")
    for row in payload["experiments"]:
        print(
            f"{row['experiment_id']:<43} {row['state']:<18} "
            f"{str(row.get('claim_status') or '-'):<48} "
            f"{str(row.get('claim_eligible', False)):<8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
