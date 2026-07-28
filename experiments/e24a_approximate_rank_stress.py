"""Prospectively locked E24a approximate-spectrum theory stress."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

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
    validate_e24a_config,
)
from catena.post_e21.e24a_approximate_rank import (
    SpectrumFamilyFold,
    SpectrumInstance,
    build_spectrum_family_folds,
    build_spectrum_instances,
    run_approximate_rank_stress,
    score_ood_spectrum_predictions,
    train_ood_spectrum_predictors,
)
from catena.systems.device import resolve_device
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e24a_approximate_rank_stress"
DEFAULT_CONFIG = "configs/e24a_approximate_rank_stress.yaml"
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
    seed_count: int,
    raw_row_count: int,
    ood_match_fraction: float,
    mean_ood_excess: float,
    claim_disposition: str,
) -> Path:
    reporting = config["reporting"]
    dry_run = run_mode == "DRY_RUN"
    status_label = reporting["dry_run_status_label"] if dry_run else reporting["main_status_label"]
    lines = [
        "# E24a approximate-rank 결과 요약",
        "",
        f"- 상태: **{status_label}**",
        f"- 실행 모드: `{run_mode}` / CPU",
        "- 증거 등급: `CONTROLLED_THEORY_STRESS`",
        f"- explicit `--allow-main`: `{'true' if not dry_run else 'false'}`",
        "",
        "## 파이프라인 구조 확인",
        "",
        f"- seed 수: `{seed_count}`",
        f"- raw row 수: `{raw_row_count}`",
        "- oracle floor와 learned excess를 분리 기록함",
        "- effective/stable rank와 epsilon-minimal 상태를 기록함",
        "- 세 spectrum family를 leave-one-family-out으로 각각 평가함",
        (
            f"- OOD epsilon-minimal rank match fraction: `{ood_match_fraction:.6g}`"
            if not dry_run
            else "- dry-run metric은 과학 평가에 사용하지 않음"
        ),
        (
            f"- mean OOD excess over oracle: `{mean_ood_excess:.6g}`"
            if not dry_run
            else "- held-out family는 해당 learner training에서 완전히 제외됨"
        ),
        (
            "- H1/E10b/E11b dependency: `PASS`"
            if not dry_run
            else "- H1/E10b/E11b anchor만 기록; canonical artifact는 읽지 않음"
        ),
        "",
        "## 증거 및 주장 경계",
        "",
        (
            "- 이 dry-run 파일은 계약 무결성 확인용이며 과학 결과가 아니다."
            if dry_run
            else f"- claim disposition: `{claim_disposition}`"
        ),
        "- transfer 범위는 등록된 세 synthetic family와 seed별 shared basis에 한정됨",
        "- universal rank, parameter efficiency, 공식 backend, LM 주장은 닫혀 있다.",
    ]
    maximum_lines = int(reporting["maximum_lines"])
    if len(lines) > maximum_lines:
        raise RuntimeError(f"E24a Korean summary exceeds {maximum_lines} lines")
    path = run_dir / str(reporting["results_summary_filename"])
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite E24a summary: {path}")
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
    dry_run: bool,
    dependency_validation: dict[str, Any],
    instances: Sequence[SpectrumInstance],
    folds: Sequence[SpectrumFamilyFold],
) -> dict[str, Any]:
    design = config["design"]
    learning = config["learning"]
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "outcome_independent": True,
        "run_mode": "DRY_RUN" if dry_run else "MAIN",
        "dimension": int(design["dimension"]),
        "controller_ranks": [int(value) for value in design["controller_ranks"]],
        "primary_estimator": str(learning["primary_estimator"]),
        "instances": [instance.manifest_dict() for instance in instances],
        "leave_one_family_out_folds": [fold.as_dict() for fold in folds],
        "test_outcomes_used_for_training": False,
        "predictions_written_before_test_outcome_join": True,
        "epsilon_minimal": dict(config["epsilon_minimal"]),
        "spectrum_family_transfer": dict(design["spectrum_family_transfer"]),
        "dependencies": dependency_validation,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    """Run E24a dry, or main only with explicit authorization and dependencies."""

    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    preflight_config = load_config(config_path)
    validate_e24a_config(preflight_config)
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
    instances = build_spectrum_instances(
        preflight_config,
        dry_run=dry_run,
        device=device,
    )
    folds = build_spectrum_family_folds(instances)

    config, run_dir, initialized_device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=str(config_path),
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode=run_mode,
    )
    validate_e24a_config(config)
    if initialized_device != device:
        raise RuntimeError("E24a device changed after preflight")
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload=_manifest_payload(
            config,
            dry_run=dry_run,
            dependency_validation=dependency_validation,
            instances=instances,
            folds=folds,
        ),
    )
    fold_plan_path = run_dir / "spectrum_family_fold_plan.json"
    write_json(
        fold_plan_path,
        {
            "schema_version": 1,
            "outcome_independent": True,
            "fold_rule": "leave_one_spectrum_family_out",
            "folds": [fold.as_dict() for fold in folds],
        },
    )

    prediction_bundle = train_ood_spectrum_predictors(
        config,
        instances=instances,
        folds=folds,
        dry_run=dry_run,
        device=initialized_device,
    )
    prediction_rows_path = run_dir / "precomputed_ood_predictions.jsonl"
    write_jsonl(prediction_rows_path, prediction_bundle.prediction_rows)
    prediction_tensors_path = run_dir / "precomputed_ood_prediction_tensors.pt"
    torch.save(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_mode": run_mode,
            "test_outcomes_used": False,
            "predictions": prediction_bundle.predictions,
        },
        prediction_tensors_path,
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_index: list[dict[str, Any]] = []
    for name, payload in sorted(prediction_bundle.checkpoint_payloads.items()):
        path = checkpoint_dir / name
        torch.save(payload, path)
        relative_name = f"checkpoints/{name}"
        checkpoint_hash = file_sha256(path)
        checkpoint_hashes[relative_name] = checkpoint_hash
        checkpoint_index.append(
            {
                "path": str(path.resolve()),
                "relative_path": relative_name,
                "sha256": checkpoint_hash,
                "held_out_family": payload["held_out_family"],
                "controller_rank": payload["controller_rank"],
                "test_outcomes_used_for_training": False,
            }
        )
    checkpoint_index_path = run_dir / "checkpoint_index.json"
    write_json(
        checkpoint_index_path,
        {
            "schema_version": 1,
            "checkpoint_count": len(checkpoint_index),
            "checkpoints": checkpoint_index,
        },
    )

    score = score_ood_spectrum_predictions(
        config,
        instances=instances,
        folds=folds,
        bundle=prediction_bundle,
        dry_run=dry_run,
    )
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=score.raw_rows,
        seed_rows=score.seed_rows,
    )
    fold_metrics_path = run_dir / "ood_spectrum_family_metrics.jsonl"
    write_jsonl(fold_metrics_path, score.fold_rows)

    diagnostic = run_approximate_rank_stress(
        config,
        dry_run=dry_run,
        device=initialized_device,
    )
    diagnostic_rows_path = run_dir / "direct_empirical_svd_diagnostic.jsonl"
    write_jsonl(diagnostic_rows_path, diagnostic.raw_rows)
    diagnostic_factors_path = run_dir / "direct_empirical_svd_factors.pt"
    torch.save(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "run_mode": run_mode,
            "primary_estimand": False,
            "diagnostic_only": True,
            "factors": diagnostic.learned_factors,
        },
        diagnostic_factors_path,
    )
    ood_match_fraction = sum(
        float(row["ood_epsilon_minimal_rank_match_fraction"]) for row in score.seed_rows
    ) / len(score.seed_rows)
    mean_ood_excess = sum(
        float(row["mean_ood_learned_excess_over_oracle"]) for row in score.seed_rows
    ) / len(score.seed_rows)
    summary_path = _write_results_summary(
        run_dir=run_dir,
        config=config,
        run_mode=run_mode,
        seed_count=len(score.seed_rows),
        raw_row_count=len(score.raw_rows),
        ood_match_fraction=ood_match_fraction,
        mean_ood_excess=mean_ood_excess,
        claim_disposition=str(score.assessment["claim_disposition"]),
    )

    closing_snapshot = validate_protocol_lock(
        lock_path=lock_path,
        config_path=config_path,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    validate_e24_snapshot(closing_snapshot)
    if closing_snapshot != snapshot:
        raise RuntimeError("E24a protocol snapshot changed during execution")
    if not dry_run:
        closing_dependencies = validate_e24_main_dependencies(
            config,
            artifact_root=args.dependency_root,
        )
        if closing_dependencies != dependency_validation:
            raise RuntimeError("E24a dependency evidence changed during execution")
    claim_eligible = not dry_run and bool(score.assessment["computed_supported"])
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
            "scientific_status": (
                "NOT_EVALUATED_DRY_RUN" if dry_run else "CONTROLLED_OOD_SPECTRUM_TRANSFER_EVALUATED"
            ),
            "seed_count": len(score.seed_rows),
            "raw_row_count": len(score.raw_rows),
            "epsilon_minimal_unresolved_is_match": False,
            "ood_epsilon_minimal_rank_match_fraction": ood_match_fraction,
            "mean_ood_learned_excess_over_oracle": mean_ood_excess,
            **score.assessment,
            "direct_empirical_svd_role": "diagnostic_only_non_primary",
        },
        "artifacts": {
            "data_manifest": _artifact_descriptor(data_manifest_path),
            "rows": row_artifacts,
            "spectrum_family_fold_plan": _artifact_descriptor(fold_plan_path),
            "precomputed_ood_predictions": _artifact_descriptor(
                prediction_rows_path,
                rows=len(prediction_bundle.prediction_rows),
            ),
            "precomputed_ood_prediction_tensors": _artifact_descriptor(prediction_tensors_path),
            "checkpoint_index": _artifact_descriptor(
                checkpoint_index_path,
                rows=len(checkpoint_index),
            ),
            "ood_spectrum_family_metrics": _artifact_descriptor(
                fold_metrics_path,
                rows=len(score.fold_rows),
            ),
            "direct_empirical_svd_diagnostic": _artifact_descriptor(
                diagnostic_rows_path,
                rows=len(diagnostic.raw_rows),
            ),
            "direct_empirical_svd_factors": _artifact_descriptor(diagnostic_factors_path),
            "results_summary_ko": _summary_descriptor(summary_path),
        },
        "claim_boundary": {
            "claim_eligible": claim_eligible,
            "explicit_allow_main_received": not dry_run,
            "allowed_claim": (
                "None; dry-run is non-evidence."
                if dry_run or not claim_eligible
                else config["claim_ceiling"]["allowed_after_authorized_main"]
            ),
            "forbidden": list(config["claim_ceiling"]["forbidden"]),
            "interpretation": (
                "Pipeline integrity only; no E24a scientific conclusion."
                if dry_run
                else (
                    "Leave-one-spectrum-family-out learned-controller "
                    "transfer within shared registered seed bases only."
                )
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
