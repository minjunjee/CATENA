"""Prospectively locked E24b behavioral-attainability theory stress."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json, write_jsonl
from catena.post_e21.contracts import (
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.e24_protocol import (
    E24_EVIDENCE_TIER,
    protocol_lock_path,
    select_e24_run_mode,
    validate_e24_main_dependencies,
    validate_e24_snapshot,
    validate_e24b_config,
)
from catena.post_e21.e24b_behavioral_attainability import (
    BehavioralCell,
    build_holdout_plan,
    factor_sensitivity_rows,
    family_level_scatter_rows,
    holdout_plan_payload,
    precompute_controller_bounds,
    precompute_holdout_predictions,
    registered_behavioral_cells,
    score_precomputed_predictions,
    simulate_behavioral_rows,
)
from catena.systems.device import resolve_device
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e24b_behavioral_attainability_stress"
DEFAULT_CONFIG = "configs/e24b_behavioral_attainability_stress.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--dependency-root")
    return parser


def _artifact_descriptor(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def _write_results_summary(
    *,
    run_dir: Path,
    config: dict[str, Any],
    run_mode: str,
    cell_count: int,
    prediction_count: int,
    fold_count: int,
    claim_assessment: dict[str, Any],
    maximum_class_specific_excess: float,
) -> Path:
    reporting = config["reporting"]
    dry_run = run_mode == "DRY_RUN"
    status_label = reporting["dry_run_status_label"] if dry_run else reporting["main_status_label"]
    lines = [
        "# E24b behavioral-attainability 결과 요약",
        "",
        f"- 상태: **{status_label}**",
        f"- 실행 모드: `{run_mode}` / CPU",
        "- 증거 등급: `CONTROLLED_THEORY_STRESS`",
        f"- explicit `--allow-main`: `{'true' if not dry_run else 'false'}`",
        "",
        "## 파이프라인 구조 확인",
        "",
        f"- cell 수: `{cell_count}`",
        f"- outcome join 전 기록한 test prediction 수: `{prediction_count}`",
        f"- outcome-independent holdout fold 수: `{fold_count}`",
        "- leave-one-demand/controller/geometry split을 모두 생성함",
        "- 각 predictor에는 해당 fold의 train outcome만 전달함",
        "- clean application target과 noisy teacher를 분리함",
        "- affected/retained structural row block을 별도로 평가함",
        "- lambda/noise/horizon sensitivity를 seed 단위로 별도 기록함",
        "- controller-class별 clean-target lower bound와 excess를 기록함",
        f"- maximum class-specific excess: `{maximum_class_specific_excess:.6g}`",
        "- prospective gate의 upper unit은 seed cluster임",
        (
            "- H1/E10b/E11b dependency: `PASS`"
            if not dry_run
            else "- H1/E10b/E11b anchor만 기록; canonical artifact는 읽지 않음"
        ),
        "",
        "## 증거 및 주장 경계",
        "",
        (
            "- 이 dry-run 파일은 artifact/leakage 계약 확인용이며 과학 결과가 아니다."
            if dry_run
            else f"- claim disposition: `{claim_assessment['claim_disposition']}`"
        ),
        (
            "- dry-run OOS metric이나 gate 상태를 평가 또는 주장에 사용하지 않는다."
            if dry_run
            else f"- allowed claim: {claim_assessment['allowed_claim']}"
        ),
        "- causal, universal reachability, NL/LM, agent, 공식 backend 주장은 닫혀 있다.",
    ]
    maximum_lines = int(reporting["maximum_lines"])
    if len(lines) > maximum_lines:
        raise RuntimeError(f"E24b Korean summary exceeds {maximum_lines} lines")
    path = run_dir / str(reporting["results_summary_filename"])
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite E24b summary: {path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary_descriptor(path: Path) -> dict[str, Any]:
    return {
        **_artifact_descriptor(path),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def _manifest_payload(
    config: dict[str, Any],
    *,
    cells: Sequence[BehavioralCell],
    plan_sha256: str,
    dry_run: bool,
    dependency_validation: dict[str, Any],
) -> dict[str, Any]:
    design = config["design"]
    override = config["dry_run_overrides"]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outcome_independent": True,
        "run_mode": "DRY_RUN" if dry_run else "MAIN",
        "seeds": (
            [int(override["seed"])] if dry_run else [int(value) for value in config["seeds"]]
        ),
        "dimension": (int(override["dimension"]) if dry_run else int(design["dimension"])),
        "batch_size": (int(override["batch_size"]) if dry_run else int(design["batch_size"])),
        "cell_count": len(cells),
        "cells": [cell.as_dict() for cell in cells],
        "holdout_axes": [str(value) for value in design["holdouts"]],
        "noise_conditions": list(design["noise_conditions"]),
        "geometry_profiles": list(design["geometry_profiles"]),
        "simulation_contract": dict(config["simulation"]),
        "holdout_plan_sha256": plan_sha256,
        "fold_membership_precomputed_without_outcomes": True,
        "prediction_feature_order": list(config["predictor"]["feature_order"]),
        "predictions_written_before_test_outcome_join": True,
        "optimization_gap": dict(config["optimization_gap"]),
        "inference": dict(config["inference"]),
        "sensitivity_factors": list(config["reporting"]["sensitivity_factors"]),
        "claim_assessment": dict(config["claim_assessment"]),
        "dependencies": dependency_validation,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    """Run E24b dry, or main only with explicit authorization and dependencies."""

    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    preflight_config = load_config(config_path)
    validate_e24b_config(preflight_config)
    device = resolve_device(args.device)
    lock_path = protocol_lock_path(preflight_config, repo_root=REPO_ROOT)
    snapshot = validate_protocol_lock(
        lock_path=lock_path,
        config_path=config_path,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    validate_e24_snapshot(snapshot)
    run_mode, dependency_validation = select_e24_run_mode(
        config=preflight_config,
        dry_run=bool(args.dry_run),
        allow_main=bool(args.allow_main),
        dependency_root=args.dependency_root,
        artifact_root=args.artifact_root,
        device=device,
    )
    dry_run = run_mode == "DRY_RUN"
    cells = registered_behavioral_cells(preflight_config, dry_run=dry_run)
    holdout_axes = [str(value) for value in preflight_config["design"]["holdouts"]]
    folds = build_holdout_plan(cells, holdout_axes=holdout_axes)
    plan_payload = holdout_plan_payload(folds)

    config, run_dir, initialized_device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=str(config_path),
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode=run_mode,
    )
    validate_e24b_config(config)
    if initialized_device != device:
        raise RuntimeError("E24b device changed after preflight")
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    plan_path = run_dir / "holdout_plan.json"
    write_json(plan_path, plan_payload)
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload=_manifest_payload(
            config,
            cells=cells,
            plan_sha256=str(plan_payload["plan_sha256"]),
            dry_run=dry_run,
            dependency_validation=dependency_validation,
        ),
    )

    bound_rows = precompute_controller_bounds(
        config,
        cells=cells,
        dry_run=dry_run,
        device=initialized_device,
    )
    bound_path = run_dir / "precomputed_controller_bounds.jsonl"
    write_jsonl(bound_path, bound_rows)
    simulation_result = simulate_behavioral_rows(
        config,
        cells=cells,
        dry_run=dry_run,
        device=initialized_device,
        precomputed_bound_rows=bound_rows,
    )
    feature_path = run_dir / "teacher_side_features.jsonl"
    write_jsonl(feature_path, simulation_result.feature_rows)
    predictor = precompute_holdout_predictions(
        feature_rows=simulation_result.feature_rows,
        outcome_rows=simulation_result.outcome_rows,
        folds=folds,
        ridge=float(config["predictor"]["ridge"]),
    )
    prediction_path = run_dir / "precomputed_predictions.jsonl"
    write_jsonl(prediction_path, predictor.predictions)
    predictor_path = run_dir / "predictor_checkpoints.json"
    write_json(
        predictor_path,
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_mode": run_mode,
            "test_outcome_used": False,
            "models": predictor.checkpoints,
        },
    )

    metric_rows, seed_rows, evaluation = score_precomputed_predictions(
        predictions=predictor.predictions,
        outcome_rows=simulation_result.outcome_rows,
        config=config,
        dry_run=dry_run,
    )
    metric_path = run_dir / "oos_metrics.jsonl"
    write_jsonl(metric_path, metric_rows)
    scatter_rows = family_level_scatter_rows(
        predictions=predictor.predictions,
        outcome_rows=simulation_result.outcome_rows,
    )
    scatter_path = run_dir / "family_level_scatter.jsonl"
    write_jsonl(scatter_path, scatter_rows)
    sensitivity_rows = factor_sensitivity_rows(
        outcome_rows=simulation_result.outcome_rows,
        factors=tuple(str(value) for value in config["reporting"]["sensitivity_factors"]),
    )
    sensitivity_path = run_dir / "factor_sensitivity.jsonl"
    write_jsonl(sensitivity_path, sensitivity_rows)
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=simulation_result.outcome_rows,
        seed_rows=seed_rows,
    )
    checkpoint_hashes = {
        predictor_path.name: file_sha256(predictor_path),
    }
    maximum_class_specific_excess = max(
        float(row["excess_over_controller_specific_lower_bound"])
        for row in simulation_result.outcome_rows
    )
    summary_path = _write_results_summary(
        run_dir=run_dir,
        config=config,
        run_mode=run_mode,
        cell_count=len(cells),
        prediction_count=len(predictor.predictions),
        fold_count=len(folds),
        claim_assessment=dict(evaluation["claim_assessment"]),
        maximum_class_specific_excess=maximum_class_specific_excess,
    )

    closing_snapshot = validate_protocol_lock(
        lock_path=lock_path,
        config_path=config_path,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    validate_e24_snapshot(closing_snapshot)
    if closing_snapshot != snapshot:
        raise RuntimeError("E24b protocol snapshot changed during execution")
    if not dry_run:
        closing_dependencies = validate_e24_main_dependencies(
            config,
            artifact_root=args.dependency_root,
        )
        if closing_dependencies != dependency_validation:
            raise RuntimeError("E24b dependency evidence changed during execution")
    computed_disposition = str(evaluation["claim_assessment"]["computed_disposition"])
    claim_eligible = (
        not dry_run and computed_disposition != "CONSTRUCTION_ROBUST_PREDICTION_FAILURE"
    )
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier=E24_EVIDENCE_TIER,
        claim_eligible=claim_eligible,
    )
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "DRY_RUN_COMPLETE" if dry_run else "MAIN_COMPLETE",
        "execution_status": "PASS",
        "run_mode": run_mode,
        **metadata,
        "dependencies": dependency_validation,
        "evaluation": {
            **evaluation,
            "scientific_status": (
                "NOT_EVALUATED_DRY_RUN" if dry_run else "PROSPECTIVE_GATE_EVALUATED"
            ),
            "prospective_gate_reported_as_claim": not dry_run,
            "test_outcome_join_after_prediction_artifact": True,
            "sensitivity": {
                "factors": list(config["reporting"]["sensitivity_factors"]),
                "aggregation": "registered_factor_level_by_seed",
                "upper_unit": "seed",
                "descriptive_only": True,
            },
            "optimization_gap": {
                "lower_bound_name": (
                    "controller_specific_clean_target_analytic_behavioral_lower_bound"
                ),
                "lower_bound_scope": "same_controller_class_clean_target_projection",
                "observed_application_outcome_used_in_lower_bound": False,
                "bound_frozen_before_observed_application_outcome": True,
                "mean_observed_application_error": (
                    sum(
                        float(row["observed_application_error"])
                        for row in simulation_result.outcome_rows
                    )
                    / len(simulation_result.outcome_rows)
                ),
                "mean_controller_specific_behavioral_lower_bound": (
                    sum(
                        float(row["controller_specific_behavioral_lower_bound"])
                        for row in simulation_result.outcome_rows
                    )
                    / len(simulation_result.outcome_rows)
                ),
                "mean_controller_specific_analytic_lower_bound": (
                    sum(
                        float(
                            row["controller_specific_clean_target_analytic_behavioral_lower_bound"]
                        )
                        for row in simulation_result.outcome_rows
                    )
                    / len(simulation_result.outcome_rows)
                ),
                "mean_excess_over_controller_specific_lower_bound": (
                    sum(
                        float(row["excess_over_controller_specific_lower_bound"])
                        for row in simulation_result.outcome_rows
                    )
                    / len(simulation_result.outcome_rows)
                ),
                "maximum_excess_over_controller_specific_lower_bound": (
                    maximum_class_specific_excess
                ),
                "predictor_feature_excluded": True,
                "clean_target_full_controller_maximum_error": max(
                    float(row["clean_oracle_attainable_behavioral_error"])
                    for row in simulation_result.outcome_rows
                    if row["controller_class"] == "full"
                ),
            },
        },
        "artifacts": {
            "data_manifest": _artifact_descriptor(data_manifest_path),
            "holdout_plan": _artifact_descriptor(plan_path),
            "teacher_side_features": _artifact_descriptor(
                feature_path,
                rows=len(simulation_result.feature_rows),
            ),
            "precomputed_controller_bounds": _artifact_descriptor(
                bound_path,
                rows=len(simulation_result.bound_rows),
            ),
            "precomputed_predictions": _artifact_descriptor(
                prediction_path,
                rows=len(predictor.predictions),
            ),
            "predictor_checkpoints": _artifact_descriptor(predictor_path),
            "oos_metrics": _artifact_descriptor(
                metric_path,
                rows=len(metric_rows),
            ),
            "family_level_scatter": _artifact_descriptor(
                scatter_path,
                rows=len(scatter_rows),
            ),
            "factor_sensitivity": _artifact_descriptor(
                sensitivity_path,
                rows=len(sensitivity_rows),
            ),
            "rows": row_artifacts,
            "results_summary_ko": _summary_descriptor(summary_path),
        },
        "claim_boundary": {
            "claim_eligible": claim_eligible,
            "explicit_allow_main_received": not dry_run,
            "claim_disposition": (evaluation["claim_assessment"]["claim_disposition"]),
            "allowed_claim": evaluation["claim_assessment"]["allowed_claim"],
            "forbidden": list(config["claim_ceiling"]["forbidden"]),
            "interpretation": (
                "Pipeline and leakage-control integrity only; no E24b scientific conclusion."
                if dry_run
                else "Claim is limited by the predeclared subset disposition."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
