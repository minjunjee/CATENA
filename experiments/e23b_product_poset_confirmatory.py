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
    resolve_e18b_freeze,
    resolve_e22b_dependency,
    resolve_e23a_screen_dependency,
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

EXPERIMENT_ID = "e23b_product_poset_confirmatory"
DEFAULT_CONFIG = "configs/e23b_product_poset_confirmatory.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E23B_PRODUCT_POSET_CONFIRMATORY_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--e18-freeze")
    parser.add_argument("--e23a-screen")
    parser.add_argument("--e22b-run")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _summary_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def _claim_status(
    *,
    dry_run: bool,
    execution_status: str,
    boundary_mode: str | None,
    assessment: dict[str, Any] | None,
) -> tuple[str, bool]:
    if execution_status != "PASS":
        return "BLOCKED_DEPENDENCY", False
    if dry_run:
        return "DRY_RUN_ONLY", False
    if assessment is None:
        raise AssertionError("PASS E23b requires an assessment")
    if boundary_mode == "safe_minimality":
        supported = bool(assessment["safe_minimality_supported"])
        return (
            "SUPPORTED_SAFE_EPSILON_MINIMALITY_CONTROLLED" if supported else "NOT_SUPPORTED",
            supported,
        )
    if boundary_mode == "capacity_only":
        supported = bool(assessment["capacity_supported"])
        return (
            "SUPPORTED_CAPACITY_EPSILON_MINIMALITY_CONTROLLED" if supported else "NOT_SUPPORTED",
            supported,
        )
    raise AssertionError("Unknown E23b boundary mode")


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run and args.device != "cpu":
        raise ValueError("E23b dry-run must use --device cpu")
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
        expected_seed_count=8,
    )
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    theory = validate_theory_prediction_lock(
        snapshot=snapshot,
        config=config,
    )

    e18_dependency = resolve_e18b_freeze(
        freeze_path=args.e18_freeze,
        dry_run=args.dry_run,
    )
    screen_dependency = resolve_e23a_screen_dependency(
        screen_run=args.e23a_screen,
        dry_run=args.dry_run,
        expected_e18_freeze_sha256=e18_dependency.freeze_sha256,
    )
    e22_dependency = resolve_e22b_dependency(
        e22b_run=args.e22b_run,
        dry_run=args.dry_run,
    )
    dependency_statuses = (
        e18_dependency.execution_status,
        screen_dependency.execution_status,
        e22_dependency.execution_status,
    )
    overall_execution = (
        "PASS" if all(status == "PASS" for status in dependency_statuses) else "BLOCKED_DEPENDENCY"
    )
    dependency_payload = {
        "e18": e18_dependency.as_dict(),
        "e23a_screen": screen_dependency.as_dict(),
        "e22": e22_dependency.as_dict(),
        "overall_execution_status": overall_execution,
    }
    runtime = product_poset_runtime(config, dry_run=args.dry_run)
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload=data_manifest_payload(
            config,
            phase="CONFIRMATORY",
            boundary_mode=e22_dependency.boundary_mode,
            dependency=dependency_payload,
            locality_method=e22_dependency.locality_method,
            locality_risk_scale=e22_dependency.locality_risk_scale,
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
    if overall_execution == "PASS":
        if e22_dependency.boundary_mode is None:
            raise AssertionError("PASS dependency requires a boundary mode")
        learned = generate_product_poset_rows(
            config,
            boundary_mode=e22_dependency.boundary_mode,
            locality_method_payload=e22_dependency.locality_method,
            locality_risk_scale=e22_dependency.locality_risk_scale,
            device=device,
            run_dir=run_dir,
            dry_run=args.dry_run,
        )
        rows = learned.rows
        training_rows = learned.training_rows
        checkpoint_hashes = learned.checkpoint_hashes
        for row in rows:
            row["e18_freeze_sha256"] = e18_dependency.freeze_sha256
            row["e23a_screen_report_sha256"] = screen_dependency.report_sha256
            row["e22_report_sha256"] = e22_dependency.report_sha256
            row["e22_protocol_lock_sha256"] = e22_dependency.protocol_lock_sha256
            row["safe_objective_implemented"] = e22_dependency.safe_objective_implemented
        for row in training_rows:
            row["e18_freeze_sha256"] = e18_dependency.freeze_sha256
            row["e23a_screen_report_sha256"] = screen_dependency.report_sha256
            row["e22_report_sha256"] = e22_dependency.report_sha256
            row["e22_protocol_lock_sha256"] = e22_dependency.protocol_lock_sha256
            row["safe_objective_implemented"] = e22_dependency.safe_objective_implemented
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
            boundary_mode=e22_dependency.boundary_mode,
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
    assessment = None if detail is None else detail["assessment"]
    claim_status, supported = _claim_status(
        dry_run=args.dry_run,
        execution_status=overall_execution,
        boundary_mode=e22_dependency.boundary_mode,
        assessment=assessment,
    )
    run_mode = "DRY_RUN" if args.dry_run else "MAIN"
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        results_summary_ko(
            phase="E23b Confirmatory",
            run_mode=run_mode,
            status=claim_status,
            boundary_mode=e22_dependency.boundary_mode,
            assessment=assessment,
            dependency_reason=";".join(
                (
                    e18_dependency.reason,
                    screen_dependency.reason,
                    e22_dependency.reason,
                )
            ),
        ),
        encoding="utf-8",
    )
    claim_eligible = bool(not args.dry_run and overall_execution == "PASS" and supported)
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=claim_eligible,
    )
    report = {
        "status": ("PASS" if overall_execution == "PASS" else "BLOCKED_DEPENDENCY"),
        "execution_status": overall_execution,
        "experiment_id": EXPERIMENT_ID,
        "run_mode": run_mode,
        "phase": "CONFIRMATORY",
        **metadata,
        "dependency": dependency_payload,
        "e18_dependency": e18_dependency.as_dict(),
        "e23a_screen_dependency": screen_dependency.as_dict(),
        "e22_dependency": e22_dependency.as_dict(),
        "boundary_mode": e22_dependency.boundary_mode,
        "boundary_selection": {
            "rule": "theory_boundary_only_v1",
            "result_independent": True,
            "selected_before_e23_outcomes": True,
            "e23a_outcomes_used": False,
            "e23a_screen_recorded_for_pipeline_provenance_only": True,
            "sets": theory["confirmatory_boundary_sets"],
        },
        "theory_prediction": {
            "locked_before_outcomes": True,
            "sha256": theory["sha256"],
            "poset_minimal_sets": theory["poset_minimal_sets"],
        },
        "summary": assessment,
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
            "supported": supported,
            "safe_locality_supported": bool(
                assessment is not None and assessment["safe_minimality_supported"]
            ),
            "capacity_supported": bool(assessment is not None and assessment["capacity_supported"]),
            "allowed_claim": (
                "Safe absolute-adequacy minimal controller recovery in a controlled "
                "four-axis sequence poset."
                if e22_dependency.boundary_mode == "safe_minimality"
                else "Capacity-only absolute-adequacy minimal controller recovery in a "
                "controlled four-axis sequence poset."
            ),
            "forbidden_claim": (
                "Locality when boundary_mode=capacity_only; semantic, natural-"
                "language, language-model, agent, official-backend, or runtime "
                "transfer in every mode."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {overall_execution}/{claim_status}: {run_dir}")


if __name__ == "__main__":
    main()
