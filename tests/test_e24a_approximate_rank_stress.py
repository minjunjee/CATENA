from __future__ import annotations

import json
import runpy
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from catena.core.config import load_config
from catena.core.io import file_sha256
from catena.post_e21.contracts import PostE21ContractError
from catena.post_e21.e24_protocol import (
    E24DependencyError,
    dependency_expectation_payload,
    validate_e24_main_dependencies,
    validate_e24a_config,
)
from catena.post_e21.e24a_approximate_rank import (
    OodPredictionBundle,
    OodScoreResult,
    SpectrumFamilyFold,
    SpectrumInstance,
    SpectrumSpec,
    build_spectrum_family_folds,
    build_spectrum_instances,
    epsilon_minimal_rank,
    learned_truncated_approximations,
    normalized_error,
    normalized_oracle_floor,
    population_operator,
    registered_spectrum_specs,
    run_approximate_rank_stress,
    score_ood_spectrum_predictions,
    singular_values,
    spectrum_statistics,
    train_ood_spectrum_predictors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e24a_approximate_rank_stress.yaml"
LOCK_PATH = REPO_ROOT / "docs/E24A_APPROXIMATE_RANK_STRESS_LOCK.json"
main = cast(
    Callable[[Sequence[str] | None], Path],
    runpy.run_path(str(REPO_ROOT / "experiments/e24a_approximate_rank_stress.py"))["main"],
)


@pytest.fixture(scope="module")
def learned_e24a() -> tuple[
    dict[str, Any],
    tuple[SpectrumInstance, ...],
    tuple[SpectrumFamilyFold, ...],
    OodPredictionBundle,
    OodScoreResult,
]:
    config = load_config(CONFIG_PATH)
    instances = build_spectrum_instances(
        config,
        dry_run=True,
        device=torch.device("cpu"),
    )
    folds = build_spectrum_family_folds(instances)
    bundle = train_ood_spectrum_predictors(
        config,
        instances=instances,
        folds=folds,
        dry_run=True,
        device=torch.device("cpu"),
    )
    score = score_ood_spectrum_predictions(
        config,
        instances=instances,
        folds=folds,
        bundle=bundle,
        dry_run=True,
    )
    return config, instances, folds, bundle, score


def test_e24a_registered_grid_and_spectrum_diagnostics() -> None:
    config = load_config(CONFIG_PATH)
    validate_e24a_config(config)
    assert len(registered_spectrum_specs(config, dry_run=False)) == 20
    assert len(registered_spectrum_specs(config, dry_run=True)) == 4

    spec = SpectrumSpec(
        spectrum_id="test_exact_rank_8",
        family="low_rank_plus_noise",
        parameter=0.0,
        base_rank=8,
        split="exact_rank_reference",
    )
    values = singular_values(dimension=64, spec=spec)
    floors = [normalized_oracle_floor(values, rank) for rank in (1, 2, 4, 8, 16)]
    assert floors == sorted(floors, reverse=True)
    assert floors[3] == pytest.approx(0.0, abs=1e-14)
    statistics = spectrum_statistics(values)
    assert 1.0 <= statistics["effective_rank"] <= 8.0
    assert 1.0 <= statistics["stable_rank"] <= 8.0


def test_e24a_learned_error_respects_oracle_floor() -> None:
    spec = SpectrumSpec(
        spectrum_id="test_power",
        family="power_law",
        parameter=1.5,
        base_rank=None,
        split="construction_spectrum_stress",
    )
    values = singular_values(dimension=32, spec=spec)
    target = population_operator(values, seed=17)
    ranks = (1, 2, 4, 8, 16)
    learned = learned_truncated_approximations(
        target,
        ranks=ranks,
        observation_count=8,
        relative_observation_noise=0.05,
        seed=19,
    )
    for rank in ranks:
        assert normalized_error(learned[rank], target) + 1e-10 >= (
            normalized_oracle_floor(values, rank)
        )


def test_e24a_epsilon_minimal_unresolved_is_not_a_match() -> None:
    assert epsilon_minimal_rank({1: 0.25, 2: 0.01}, epsilon=0.05) is None
    assert epsilon_minimal_rank({1: 0.25, 2: 0.0024}, epsilon=0.05) == 2


def test_e24a_dry_core_has_oracle_learned_and_construction_rows() -> None:
    config = load_config(CONFIG_PATH)
    result = run_approximate_rank_stress(
        config,
        dry_run=True,
        device=torch.device("cpu"),
    )
    assert len(result.raw_rows) == 24
    assert len(result.seed_rows) == 1
    assert result.seed_rows[0]["spectrum_count"] == 4
    assert result.seed_rows[0]["construction_spectrum_count"] == 3
    assert result.seed_rows[0]["learned_spectrum_family_transfer_evaluated"] is False
    assert all(
        float(row["normalized_learned_error"]) + 1e-10 >= float(row["normalized_oracle_floor"])
        for row in result.raw_rows
    )
    assert {str(row["split"]) for row in result.raw_rows} == {
        "exact_rank_reference",
        "construction_spectrum_stress",
    }
    assert all(
        row["empirical_estimator_scope"] == "direct_per_target_factorization_no_family_transfer"
        for row in result.raw_rows
    )


def test_e24a_primary_learner_holds_out_entire_spectrum_families(
    learned_e24a: tuple[
        dict[str, Any],
        tuple[SpectrumInstance, ...],
        tuple[SpectrumFamilyFold, ...],
        OodPredictionBundle,
        OodScoreResult,
    ],
) -> None:
    _config, instances, folds, bundle, score = learned_e24a
    assert len(instances) == 4
    assert {instance.descriptor.numel() for instance in instances} == {16}
    assert len(folds) == 3
    for fold in folds:
        assert fold.held_out_family not in fold.training_families
        assert set(fold.train_instance_ids).isdisjoint(fold.test_instance_ids)
    assert len(bundle.checkpoint_payloads) == 9
    assert len(bundle.prediction_rows) == 12
    assert all(
        row["test_family_seen_during_training"] is False and row["test_outcome_used"] is False
        for row in bundle.prediction_rows
    )
    assert all(
        checkpoint["optimizer_trace"]["steps"] == 8
        and checkpoint["test_outcomes_used_for_training"] is False
        and checkpoint["model_state_dict"]
        for checkpoint in bundle.checkpoint_payloads.values()
    )
    assert len(score.raw_rows) == 12
    assert len(score.fold_rows) == 3
    assert all(
        row["primary_estimand"] is True
        and row["fold_rule"] == "leave_one_spectrum_family_out"
        and float(row["normalized_ood_learned_error"]) + 1e-8
        >= float(row["normalized_oracle_floor"])
        for row in score.raw_rows
    )
    assert score.assessment["claim_disposition"] == "DRY_RUN_NON_EVIDENCE"


def test_e24a_heldout_target_mutation_cannot_change_fold_predictions(
    learned_e24a: tuple[
        dict[str, Any],
        tuple[SpectrumInstance, ...],
        tuple[SpectrumFamilyFold, ...],
        OodPredictionBundle,
        OodScoreResult,
    ],
) -> None:
    config, instances, folds, _bundle, _score = learned_e24a
    fold = next(item for item in folds if item.held_out_family == "exponential")
    baseline = train_ood_spectrum_predictors(
        config,
        instances=instances,
        folds=(fold,),
        dry_run=True,
        device=torch.device("cpu"),
    )
    held_out_ids = set(fold.test_instance_ids)
    mutated = tuple(
        replace(instance, target=instance.target + 1000.0)
        if instance.instance_id in held_out_ids
        else instance
        for instance in instances
    )
    repeated = train_ood_spectrum_predictors(
        config,
        instances=mutated,
        folds=(fold,),
        dry_run=True,
        device=torch.device("cpu"),
    )
    assert baseline.prediction_rows == repeated.prediction_rows
    assert all(
        torch.equal(baseline.predictions[row_id], repeated.predictions[row_id])
        for row_id in baseline.predictions
    )


def test_e24_dependency_expectations_are_dry_only_and_main_blocking(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    declaration = dependency_expectation_payload(config)
    assert declaration["canonical_artifacts_read"] is False
    assert declaration["validation_status"] == "NOT_READ_DRY_RUN"
    assert len(declaration["required"]) == 3
    with pytest.raises(E24DependencyError, match="BLOCKED_DEPENDENCY") as error:
        validate_e24_main_dependencies(config, artifact_root=tmp_path)
    assert error.value.status == "BLOCKED_DEPENDENCY"
    first_anchor = config["dependencies"]["required"][0]
    fake_report = tmp_path / str(first_anchor["relative_report_path"])
    fake_report.parent.mkdir(parents=True)
    fake_report.write_text(
        json.dumps({"status": "PASS", "claim_gate": {"supported": True}}),
        encoding="utf-8",
    )
    with pytest.raises(E24DependencyError, match="SHA-256 mismatch") as mismatch:
        validate_e24_main_dependencies(config, artifact_root=tmp_path)
    assert mismatch.value.status == "BLOCKED_DEPENDENCY"


def test_e24a_main_is_blocked_before_artifact_creation(tmp_path: Path) -> None:
    artifact_root = tmp_path / "main_must_not_exist"
    with pytest.raises(PostE21ContractError, match="explicit --allow-main"):
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--device",
                "cpu",
                "--artifact-root",
                str(artifact_root),
            ]
        )
    assert not artifact_root.exists()


def test_e24a_allow_main_still_requires_dependency_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "main_missing_dependency_root"
    with pytest.raises(
        PostE21ContractError,
        match="requires an explicit --dependency-root",
    ):
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--device",
                "cpu",
                "--artifact-root",
                str(artifact_root),
                "--allow-main",
            ]
        )
    assert not artifact_root.exists()


def test_e24a_dry_run_contract_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "e24a_dry"
    run_dir = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--device",
            "cpu",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ]
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "DRY_RUN_COMPLETE"
    assert report["claim_eligible"] is False
    assert report["scientific_evidence"] is False
    assert report["evaluation"]["scientific_status"] == "NOT_EVALUATED_DRY_RUN"
    assert report["evaluation"]["claim_disposition"] == "DRY_RUN_NON_EVIDENCE"
    assert report["evaluation"]["test_outcomes_used_for_training"] is False
    assert report["evaluation"]["predictions_written_before_test_outcome_join"] is True
    assert report["dependencies"]["canonical_artifacts_read"] is False
    assert (run_dir / "protocol_lock.json").read_bytes() == LOCK_PATH.read_bytes()
    assert (run_dir / "raw_metrics.jsonl").is_file()
    assert (run_dir / "seed_metrics.jsonl").is_file()
    assert (run_dir / "precomputed_ood_predictions.jsonl").is_file()
    assert (run_dir / "precomputed_ood_prediction_tensors.pt").is_file()
    assert (run_dir / "checkpoint_index.json").is_file()
    assert (run_dir / "ood_spectrum_family_metrics.jsonl").is_file()
    assert (run_dir / "direct_empirical_svd_diagnostic.jsonl").is_file()
    assert len(report["checkpoint_hashes"]) == 9
    assert (run_dir / "precomputed_ood_predictions.jsonl").stat().st_mtime_ns <= (
        run_dir / "raw_metrics.jsonl"
    ).stat().st_mtime_ns
    raw_rows = [
        json.loads(line)
        for line in (run_dir / "raw_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows
    assert all(
        row["primary_estimand"] is True and row["test_family_seen_during_training"] is False
        for row in raw_rows
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    assert "DRY_RUN_NON_EVIDENCE" in "\n".join(summary_lines)
    assert len(summary_lines) <= 45
    summary_descriptor = report["artifacts"]["results_summary_ko"]
    assert summary_descriptor["sha256"] == file_sha256(summary_path)
    assert summary_descriptor["line_count"] == len(summary_lines)
