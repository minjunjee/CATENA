from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.io import file_sha256
from catena.post_e21.contracts import (
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.product_poset_eval import (
    mean_locality_method_payload,
    resolve_e18b_freeze,
    summarize_seed_predictions,
    validate_theory_prediction_lock,
)
from catena.post_e21.product_poset_runner import (
    data_manifest_payload,
    generate_product_poset_rows,
    product_poset_runtime,
    results_summary_ko,
    validate_e23_config,
    write_cell_rows,
    write_theory_predictions,
    write_training_rows,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e23a_product_poset_screen"
DEFAULT_CONFIG = "configs/e23a_product_poset_screen.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E23A_PRODUCT_POSET_SCREEN_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--e18-freeze")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _summary_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.device != "cpu":
        raise ValueError("E23a dry-run must use --device cpu")
    snapshot = validate_protocol_lock(
        lock_path=LOCK_PATH,
        config_path=args.config,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    config, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    validate_e23_config(
        config,
        experiment_id=EXPERIMENT_ID,
        expected_seed_count=3,
    )
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    theory = validate_theory_prediction_lock(
        snapshot=snapshot,
        config=config,
    )
    dependency = resolve_e18b_freeze(
        freeze_path=args.e18_freeze,
        dry_run=args.dry_run,
    )
    runtime = product_poset_runtime(config, dry_run=args.dry_run)
    locality_method = mean_locality_method_payload()
    locality_risk_scale = float(config["adequacy"]["locality_margin"])
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload=data_manifest_payload(
            config,
            phase="SCREEN",
            boundary_mode="capacity_only",
            dependency={"e18": dependency.as_dict()},
            locality_method=locality_method,
            locality_risk_scale=locality_risk_scale,
            runtime=runtime,
        ),
    )
    theory_artifact = write_theory_predictions(
        run_dir=run_dir,
        config=config,
        locked_sha256=str(theory["sha256"]),
    )

    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] | None = None
    checkpoint_hashes: dict[str, str] = {}
    training_rows: list[dict[str, Any]] = []
    if dependency.execution_status == "PASS":
        learned = generate_product_poset_rows(
            config,
            boundary_mode="capacity_only",
            locality_method_payload=locality_method,
            locality_risk_scale=locality_risk_scale,
            device=device,
            run_dir=run_dir,
            dry_run=args.dry_run,
        )
        rows = learned.rows
        training_rows = learned.training_rows
        checkpoint_hashes = learned.checkpoint_hashes
        for row in rows:
            row["e18_freeze_sha256"] = dependency.freeze_sha256
        for row in training_rows:
            row["e18_freeze_sha256"] = dependency.freeze_sha256
        seed_rows, detail = summarize_seed_predictions(
            rows,
            seeds=[int(value) for value in runtime["seeds"]],
            intensities=[float(value) for value in runtime["intensities"]],
            updates=[int(value) for value in runtime["updates"]],
            gap_events=[int(value) for value in runtime["gap_events"]],
            affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
            target_margin=float(config["adequacy"]["target_margin"]),
            retention_margin=float(config["adequacy"]["retention_margin"]),
            locality_margin=float(config["adequacy"]["locality_margin"]),
            minimum_single_axis_exact_matches=int(
                config["adequacy"]["minimum_single_axis_exact_matches"]
            ),
            minimum_pairwise_exact_matches=int(
                config["adequacy"]["minimum_pairwise_exact_matches"]
            ),
            incomparable_direction_margin=float(
                config["adequacy"]["incomparable_direction_margin"]
            ),
            maximal_simpler_degradation_margin=float(
                config["adequacy"]["maximal_simpler_degradation_margin"]
            ),
            boundary_mode="capacity_only",
        )
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=rows,
        seed_rows=seed_rows,
        raw_filename="product_poset_raw_metrics.jsonl",
        seed_filename="product_poset_seed_metrics.jsonl",
    )
    cell_artifact = write_cell_rows(
        run_dir=run_dir,
        cell_rows=[] if detail is None else detail["cells"],
    )
    training_artifact = write_training_rows(
        run_dir=run_dir,
        rows=training_rows,
    )
    run_mode = "DRY_RUN" if args.dry_run else "MAIN"
    claim_status = (
        "BLOCKED_DEPENDENCY"
        if dependency.execution_status != "PASS"
        else ("DRY_RUN_ONLY" if args.dry_run else "SCREEN_ONLY_NO_CONFIRMATORY_CLAIM")
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        results_summary_ko(
            phase="E23a Screen",
            run_mode=run_mode,
            status=claim_status,
            boundary_mode="capacity_only",
            assessment=None if detail is None else detail["assessment"],
            dependency_reason=dependency.reason,
        ),
        encoding="utf-8",
    )
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=False,
    )
    report = {
        "status": ("PASS" if dependency.execution_status == "PASS" else "BLOCKED_DEPENDENCY"),
        "execution_status": dependency.execution_status,
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "phase": "SCREEN",
        **metadata,
        "e18_dependency": dependency.as_dict(),
        "boundary_mode": "capacity_only",
        "theory_prediction": {
            "locked_before_outcomes": True,
            "sha256": theory["sha256"],
            "poset_minimal_sets": theory["poset_minimal_sets"],
            "result_independent_boundary_sets": theory["confirmatory_boundary_sets"],
        },
        "summary": None if detail is None else detail["assessment"],
        "artifacts": {
            "data_manifest": {
                "path": str(data_manifest_path.resolve()),
                "sha256": file_sha256(data_manifest_path),
            },
            "theory_predictions": theory_artifact,
            "rows": row_artifacts,
            "training_runs": training_artifact,
            "poset_minimal_demands": cell_artifact,
            "results_summary_ko": _summary_descriptor(summary_path),
        },
        "claim_gate": {
            "status": claim_status,
            "supported": False,
            "screen_claim_eligible": False,
            "e23b_boundary_selected_from_screen_outcomes": False,
            "allowed_claim": ("Pipeline diagnostic for a controlled 4-bit controller poset."),
            "forbidden_claim": (
                "Confirmatory minimality, safe locality, semantic, language-model, "
                "agent, official-backend, or runtime transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {dependency.execution_status}/{claim_status}: {run_dir}")


if __name__ == "__main__":
    main()
