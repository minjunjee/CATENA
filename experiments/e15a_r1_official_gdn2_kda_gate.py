from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.io import write_json
from catena.eval.official_operator_gate import run_official_operator_gate
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e15a_r1_official_gdn2_kda_gate"
DEFAULT_CONFIG = "configs/e15a_r1_official_gdn2_kda_gate.yaml"


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
    row = run_official_operator_gate(
        backend=dict(config["backend"]),
        dry_run=bool(args.dry_run),
    )
    write_json(run_dir / "backend_gate_row.json", row)
    if args.dry_run:
        status = "DRY_RUN"
    elif row["status"] == "PASS":
        status = "PASS"
    elif row["status"] == "NOT_CONFIGURED":
        status = "NOT_CONFIGURED"
    else:
        status = "FAIL"
    report = {
        "status": status,
        "repair": dict(config["repair"]),
        "backend": row,
        "claim_gate": {
            "official_operator_claim_eligible": row["status"] == "PASS",
            "scientific_evidence": bool(row["scientific_evidence"]),
            "required_checks": list(dict(config["backend"]["checks"])),
            "allowed_claim": (
                "Official GDN2/KDA operator evidence only after the pinned "
                "GDN2 and FLA revisions and every registered parity, state, "
                "gradient, and intervention check pass."
            ),
            "forbidden_claim": (
                "Dry-run, reference, mock, unpinned, partially passing, or "
                "controlled-reference results as official evidence."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {status}: {run_dir}")


if __name__ == "__main__":
    main()
