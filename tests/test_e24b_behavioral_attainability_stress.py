from __future__ import annotations

import json
import math
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
from catena.post_e21.e24_protocol import validate_e24b_config
from catena.post_e21.e24b_behavioral_attainability import (
    BehavioralCell,
    HoldoutFold,
    build_holdout_plan,
    build_target_teacher_sequence,
    factor_sensitivity_rows,
    family_level_scatter_rows,
    fit_fold_predictor,
    precompute_controller_bounds,
    precompute_holdout_predictions,
    registered_behavioral_cells,
    score_precomputed_predictions,
    simulate_behavioral_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e24b_behavioral_attainability_stress.yaml"
LOCK_PATH = REPO_ROOT / "docs/E24B_BEHAVIORAL_ATTAINABILITY_STRESS_LOCK.json"
main = cast(
    Callable[[Sequence[str] | None], Path],
    runpy.run_path(str(REPO_ROOT / "experiments/e24b_behavioral_attainability_stress.py"))["main"],
)

DryE24bFixture = tuple[
    dict[str, Any],
    tuple[BehavioralCell, ...],
    tuple[HoldoutFold, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]


@pytest.fixture(scope="module")
def dry_e24b() -> DryE24bFixture:
    config = load_config(CONFIG_PATH)
    validate_e24b_config(config)
    cells = registered_behavioral_cells(config, dry_run=True)
    folds = build_holdout_plan(
        cells,
        holdout_axes=tuple(str(value) for value in config["design"]["holdouts"]),
    )
    bound_rows = precompute_controller_bounds(
        config,
        cells=cells,
        dry_run=True,
        device=torch.device("cpu"),
    )
    result = simulate_behavioral_rows(
        config,
        cells=cells,
        dry_run=True,
        device=torch.device("cpu"),
        precomputed_bound_rows=bound_rows,
    )
    return (
        config,
        cells,
        folds,
        result.feature_rows,
        bound_rows,
        result.outcome_rows,
    )


def test_e24b_registered_grid_and_outcome_independent_folds(
    dry_e24b: DryE24bFixture,
) -> None:
    _config, cells, folds, feature_rows, bound_rows, rows = dry_e24b
    assert len(cells) == 1_280
    main_cells = registered_behavioral_cells(
        load_config(CONFIG_PATH),
        dry_run=False,
    )
    assert len(main_cells) == 69_120
    assert {float(cell.target_noise_factor) for cell in main_cells} == {0.0, 0.01, 0.05, 0.10}
    assert {float(cell.teacher_noise_factor) for cell in main_cells} == {0.0, 0.01, 0.05, 0.10}
    assert {
        (
            float(cell.target_noise_factor),
            float(cell.teacher_noise_factor),
        )
        for cell in main_cells
    } == {
        (0.0, 0.0),
        (0.01, 0.0),
        (0.05, 0.0),
        (0.10, 0.0),
        (0.0, 0.01),
        (0.0, 0.05),
        (0.0, 0.10),
        (0.01, 0.01),
        (0.05, 0.05),
        (0.10, 0.10),
    }
    assert len(feature_rows) == len(cells)
    assert len(bound_rows) == len(cells)
    assert len(rows) == len(cells)
    assert len(folds) == 10
    row_ids = {str(row["row_id"]) for row in rows}
    test_counts = {row_id: 0 for row_id in row_ids}
    for fold in folds:
        assert set(fold.train_row_ids).isdisjoint(fold.test_row_ids)
        assert set(fold.train_row_ids) | set(fold.test_row_ids) == row_ids
        for row_id in fold.test_row_ids:
            test_counts[row_id] += 1
    assert set(test_counts.values()) == {3}
    assert {fold.held_out_value for fold in folds if fold.axis == "geometry_block"} == {
        "baseline_geometry",
        "unseen_key_correlation",
        "unseen_operator_norm",
        "unseen_key_load",
    }
    geometry_values = {
        cell.geometry_block: (
            cell.key_correlation,
            cell.target_operator_norm,
            cell.key_load_fraction,
        )
        for cell in cells
    }
    assert geometry_values == {
        "baseline_geometry": (0.0, 1.0, 0.5),
        "unseen_key_correlation": (0.75, 1.0, 0.5),
        "unseen_operator_norm": (0.0, 1.5, 0.5),
        "unseen_key_load": (0.0, 1.0, 1.0),
    }
    assert {float(cell.target_noise_factor) for cell in cells} == {0.0, 0.05}
    assert {float(cell.teacher_noise_factor) for cell in cells} == {0.0, 0.01, 0.05, 0.10}
    assert all(
        row["clean_target_features_included"] is False
        and "log_behavioral_mse" not in row
        and "clean_target_sequence_sha256" not in row
        for row in feature_rows
    )
    assert {str(row["row_id"]) for row in bound_rows} == row_ids
    assert all(
        row["observed_application_outcome_used_in_lower_bound"] is False
        and row["bound_frozen_before_observed_application_outcome"] is True
        and row["predictor_feature_used"] is False
        and "observed_application_error" not in row
        and "behavioral_mse" not in row
        and "log_behavioral_mse" not in row
        for row in bound_rows
    )
    realized_geometry = {
        str(row["geometry_block"]): (
            float(row["realized_mean_key_correlation"]),
            float(row["realized_high_load_fraction"]),
        )
        for row in feature_rows
        if row["noise_condition"] == "clean_teacher"
    }
    assert realized_geometry["baseline_geometry"] == pytest.approx(
        (0.0, 0.5),
        abs=1e-12,
    )
    assert realized_geometry["unseen_key_correlation"] == pytest.approx(
        (0.75, 0.5),
        abs=1e-12,
    )
    assert realized_geometry["unseen_key_load"] == pytest.approx(
        (0.0, 1.0),
        abs=1e-12,
    )
    assert all(math.isfinite(float(row["behavioral_mse"])) for row in rows)
    for row in rows:
        readout_lambda = float(row["readout_lambda"])
        expected = readout_lambda * float(row["affected_behavioral_mse"]) + (
            1.0 - readout_lambda
        ) * float(row["unaffected_behavioral_mse"])
        assert float(row["behavioral_mse"]) == pytest.approx(
            expected,
            abs=1e-15,
        )
        expected_linearized = readout_lambda * float(
            row["affected_linearized_behavioral_regret"]
        ) + (1.0 - readout_lambda) * float(row["unaffected_linearized_behavioral_regret"])
        expected_bound = readout_lambda * float(row["affected_lipschitz_upper_bound"]) + (
            1.0 - readout_lambda
        ) * float(row["unaffected_lipschitz_upper_bound"])
        assert float(row["linearized_behavioral_regret"]) == pytest.approx(
            expected_linearized,
            abs=1e-15,
        )
        assert float(row["lipschitz_upper_bound"]) == pytest.approx(
            expected_bound,
            abs=1e-15,
        )
        assert float(row["affected_behavioral_mse"]) >= 0.0
        assert float(row["unaffected_behavioral_mse"]) >= 0.0
        assert row["readout_weighting_formula"] == (
            "lambda * affected_mse + (1 - lambda) * unaffected_mse"
        )
        assert row["readout_row_block_method"] == (
            "structural_affected_and_retained_state_rows_with_within_block_qr_whitening"
        )
    assert all(
        float(row["behavioral_mse"]) <= float(row["lipschitz_upper_bound"]) + 1e-12 for row in rows
    )
    component_rows: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["seed"],
            row["demand_family"],
            row["controller_class"],
            row["geometry_block"],
            row["noise_condition"],
            row["horizon"],
            row["readout"],
        )
        component_rows.setdefault(key, []).append(row)
    assert all(len(group) == 2 for group in component_rows.values())
    for group in component_rows.values():
        assert {float(row["affected_behavioral_mse"]) for row in group} == {
            float(group[0]["affected_behavioral_mse"])
        }
        assert {float(row["unaffected_behavioral_mse"]) for row in group} == {
            float(group[0]["unaffected_behavioral_mse"])
        }
    for row in rows:
        lower_bound = float(row["controller_specific_behavioral_lower_bound"])
        registered_lower_bound = float(
            row["controller_specific_clean_target_analytic_behavioral_lower_bound"]
        )
        observed = float(row["observed_application_error"])
        assert lower_bound == pytest.approx(registered_lower_bound, abs=1e-15)
        assert registered_lower_bound <= observed + 1e-10
        assert float(row["excess_over_controller_specific_lower_bound"]) == pytest.approx(
            observed - registered_lower_bound,
            abs=1e-15,
        )
        assert row["lower_bound_controller_class"] == row["controller_class"]
        assert row["lower_bound_independent_of_predictor"] is True
        assert row["observed_application_outcome_used_in_lower_bound"] is False
        assert row["bound_frozen_before_observed_application_outcome"] is True
        assert float(row["clean_target_retained_row_max_abs"]) == 0.0
    assert any(
        float(row["controller_specific_clean_target_analytic_behavioral_lower_bound"]) > 1e-8
        for row in rows
        if row["controller_class"] != "full"
    )
    assert (
        max(
            float(row["clean_oracle_attainable_behavioral_error"])
            for row in rows
            if row["controller_class"] == "full"
        )
        <= 1e-20
    )


def test_e24b_common_family_is_shared_basis_diagonalizable(
    dry_e24b: DryE24bFixture,
) -> None:
    config, cells, _folds, _feature_rows, _bound_rows, _rows = dry_e24b
    base = next(
        cell
        for cell in cells
        if cell.noise_condition == "clean_teacher"
        and cell.geometry_block == "baseline_geometry"
        and cell.horizon == 4
        and cell.readout_lambda == 0.25
        and cell.readout == "linear"
    )
    common_cell = replace(
        base,
        row_id="common_shared_projection_check",
        demand_family="common_rotated_commuting",
        controller_class="shared_basis_diagonal",
    )
    noncommuting_cell = replace(
        common_cell,
        row_id="noncommuting_shared_projection_check",
        demand_family="noncommuting",
    )
    sequence = build_target_teacher_sequence(
        common_cell,
        dimension=16,
        affected_row_fraction=0.75,
    )
    assert (
        max(float((target - target.mT).abs().max()) for target in sequence.nominal_targets) <= 1e-12
    )
    assert (
        max(
            float((left @ right - right @ left).abs().max())
            for left in sequence.nominal_targets
            for right in sequence.nominal_targets
        )
        <= 1e-12
    )
    first_diagonal = torch.diagonal(
        sequence.shared_basis.mT @ sequence.nominal_targets[0] @ sequence.shared_basis
    )
    second_diagonal = torch.diagonal(
        sequence.shared_basis.mT @ sequence.nominal_targets[1] @ sequence.shared_basis
    )
    ratios = second_diagonal[:12] / first_diagonal[:12]
    assert float(ratios.std()) > 1e-3
    noncommuting_sequence = build_target_teacher_sequence(
        noncommuting_cell,
        dimension=16,
        affected_row_fraction=0.75,
    )
    assert (
        float(
            (noncommuting_sequence.nominal_targets[0] - noncommuting_sequence.nominal_targets[1])
            .abs()
            .max()
        )
        > 1e-3
    )
    assert (
        float(
            (
                noncommuting_sequence.nominal_targets[0] @ noncommuting_sequence.nominal_targets[1]
                - noncommuting_sequence.nominal_targets[1]
                @ noncommuting_sequence.nominal_targets[0]
            )
            .abs()
            .max()
        )
        > 1e-6
    )
    result = simulate_behavioral_rows(
        config,
        cells=(common_cell, noncommuting_cell),
        dry_run=True,
        device=torch.device("cpu"),
    )
    by_id = {str(row["row_id"]): row for row in result.outcome_rows}
    assert float(by_id[common_cell.row_id]["observed_application_error"]) <= 1e-20
    assert float(by_id[noncommuting_cell.row_id]["observed_application_error"]) > 1e-8


def test_e24b_geometry_blocks_change_only_the_registered_factor(
    dry_e24b: DryE24bFixture,
) -> None:
    _config, cells, _folds, _features, _bounds, _outcomes = dry_e24b
    matched_cells = {
        cell.geometry_block: cell
        for cell in cells
        if cell.seed == 993119
        and cell.demand_family == "axis_commuting"
        and cell.controller_class == "full"
        and cell.noise_condition == "clean_teacher"
        and cell.readout_lambda == 0.25
        and cell.horizon == 4
        and cell.readout == "linear"
    }
    sequences = {
        name: build_target_teacher_sequence(
            cell,
            dimension=16,
            affected_row_fraction=0.75,
        )
        for name, cell in matched_cells.items()
    }
    baseline = sequences["baseline_geometry"]
    correlated = sequences["unseen_key_correlation"]
    rescaled = sequences["unseen_operator_norm"]
    high_load = sequences["unseen_key_load"]
    for base_target, correlated_target, rescaled_target, load_target in zip(
        baseline.nominal_targets,
        correlated.nominal_targets,
        rescaled.nominal_targets,
        high_load.nominal_targets,
        strict=True,
    ):
        assert torch.equal(base_target, correlated_target)
        assert torch.equal(base_target, load_target)
        assert torch.allclose(
            1.5 * base_target,
            rescaled_target,
            atol=1e-15,
            rtol=1e-15,
        )
        assert float(torch.linalg.matrix_norm(base_target)) == pytest.approx(
            1.0,
            abs=1e-14,
        )
        assert float(torch.linalg.matrix_norm(rescaled_target)) == pytest.approx(
            1.5,
            abs=1e-14,
        )
    assert torch.equal(baseline.key_transform, rescaled.key_transform)
    assert not torch.equal(baseline.key_transform, correlated.key_transform)
    assert not torch.equal(baseline.key_transform, high_load.key_transform)


def test_e24b_clean_target_is_invariant_to_teacher_corruption(
    dry_e24b: DryE24bFixture,
) -> None:
    _config, _cells, _folds, _feature_rows, bound_rows, rows = dry_e24b
    candidates = [
        row
        for row in rows
        if row["demand_family"] == "noncommuting"
        and row["controller_class"] == "shared_basis_diagonal"
        and row["geometry_block"] == "baseline_geometry"
        and float(row["readout_lambda"]) == 0.25
        and int(row["horizon"]) == 1
        and row["readout"] == "linear"
        and float(row["target_noise_factor"]) == 0.0
    ]
    clean = next(row for row in candidates if float(row["teacher_noise_factor"]) == 0.0)
    corrupted = next(row for row in candidates if float(row["teacher_noise_factor"]) == 0.10)
    assert clean["clean_target_sequence_sha256"] == corrupted["clean_target_sequence_sha256"]
    assert clean["teacher_sequence_sha256"] != corrupted["teacher_sequence_sha256"]
    assert float(clean["teacher_retained_row_frobenius_norm"]) == 0.0
    assert float(corrupted["teacher_retained_row_frobenius_norm"]) > 0.0
    lower_bound_name = "controller_specific_clean_target_analytic_behavioral_lower_bound"
    assert float(clean[lower_bound_name]) > 0.0
    assert float(clean[lower_bound_name]) == pytest.approx(
        float(corrupted[lower_bound_name]),
        abs=1e-15,
    )
    bounds_by_id = {str(row["row_id"]): row for row in bound_rows}
    clean_bound = bounds_by_id[str(clean["row_id"])]
    corrupted_bound = bounds_by_id[str(corrupted["row_id"])]
    assert float(clean_bound[lower_bound_name]) == pytest.approx(
        float(corrupted_bound[lower_bound_name]),
        abs=1e-15,
    )
    assert "observed_application_error" not in clean_bound
    assert "observed_application_error" not in corrupted_bound
    frozen_bound_payload = json.dumps(
        bound_rows,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert float(clean["observed_application_error"]) != pytest.approx(
        float(corrupted["observed_application_error"]),
        abs=1e-12,
    )
    mutated = dict(corrupted)
    mutated["observed_application_error"] = 1e9
    mutated["behavioral_mse"] = 1e9
    assert mutated[lower_bound_name] == corrupted[lower_bound_name]
    assert (
        json.dumps(
            bound_rows,
            sort_keys=True,
            separators=(",", ":"),
        )
        == frozen_bound_payload
    )


def test_e24b_fold_predictions_do_not_consume_test_outcomes(
    dry_e24b: DryE24bFixture,
) -> None:
    config, _cells, folds, teacher_features, _bound_rows, outcome_rows = dry_e24b
    fold = folds[0]
    feature_rows = {str(row["row_id"]): dict(row) for row in teacher_features}
    outcomes = {str(row["row_id"]): row for row in outcome_rows}
    training = {
        row_id: float(outcomes[row_id]["log_behavioral_mse"]) for row_id in fold.train_row_ids
    }
    first, _checkpoint = fit_fold_predictor(
        feature_rows=feature_rows,
        fold=fold,
        training_outcomes=training,
        ridge=float(config["predictor"]["ridge"]),
    )
    test_id = fold.test_row_ids[0]
    feature_rows[test_id]["log_behavioral_mse"] = 1e9
    second, _checkpoint = fit_fold_predictor(
        feature_rows=feature_rows,
        fold=fold,
        training_outcomes=training,
        ridge=float(config["predictor"]["ridge"]),
    )
    assert first == second
    assert all(row["test_outcome_used"] is False for row in first)
    with pytest.raises(ValueError, match="exactly the registered training"):
        fit_fold_predictor(
            feature_rows=feature_rows,
            fold=fold,
            training_outcomes={**training, test_id: 0.0},
            ridge=float(config["predictor"]["ridge"]),
        )


def test_e24b_oos_metrics_cover_all_holdout_axes(
    dry_e24b: DryE24bFixture,
) -> None:
    config, _cells, folds, feature_rows, _bound_rows, rows = dry_e24b
    predictor = precompute_holdout_predictions(
        feature_rows=feature_rows,
        outcome_rows=rows,
        folds=folds,
        ridge=float(config["predictor"]["ridge"]),
    )
    assert len(predictor.predictions) == len(rows) * 3
    metrics, seed_rows, summary = score_precomputed_predictions(
        predictions=predictor.predictions,
        outcome_rows=rows,
        config=config,
        dry_run=True,
    )
    axis_rows = [row for row in metrics if row["aggregation"] == "holdout_axis"]
    assert {row["holdout_axis"] for row in axis_rows} == {
        "demand_family",
        "controller_class",
        "geometry_block",
    }
    assert all(
        row["seed_cluster_count"] == 1
        and row["cluster_bootstrap_replicates"] == 32
        and row["cluster_bootstrap_resample_unit"] == "seed"
        and row["episode_row_resampling_used"] is False
        and "cluster_bootstrap_ci" in row
        for row in axis_rows
    )
    assert all(
        math.isfinite(float(row[name]))
        for row in metrics
        for name in (
            "r2",
            "rmse",
            "mae",
            "pearson_r",
            "spearman_r",
            "calibration_slope",
            "calibration_intercept",
        )
    )
    assert len(seed_rows) == 1
    scatter_rows = family_level_scatter_rows(
        predictions=predictor.predictions,
        outcome_rows=rows,
    )
    assert scatter_rows
    assert all(
        row["upper_unit"] == "seed" and row["outcome_join_after_prediction_artifact"] is True
        for row in scatter_rows
    )
    sensitivity_rows = factor_sensitivity_rows(
        outcome_rows=rows,
        factors=tuple(str(value) for value in config["reporting"]["sensitivity_factors"]),
    )
    assert len(sensitivity_rows) == 9
    assert {row["factor"] for row in sensitivity_rows} == {
        "readout_lambda",
        "noise_condition",
        "horizon",
    }
    assert all(
        row["upper_unit"] == "seed"
        and row["descriptive_sensitivity_only"] is True
        and math.isfinite(float(row["mean_observed_application_error"]))
        and math.isfinite(float(row["mean_excess_over_controller_specific_lower_bound"]))
        for row in sensitivity_rows
    )
    assert summary["scientific_status"] == "NOT_EVALUATED_DRY_RUN"
    subset_rows = [row for row in metrics if row["aggregation"] == "claim_subset_holdout_axis"]
    assert len(subset_rows) == 6
    assert summary["claim_assessment"]["claim_disposition"] == "DRY_RUN_NON_EVIDENCE"
    assert summary["claim_assessment"]["computed_disposition"] in {
        "BROAD_NOISY_NONLINEAR_MULTISTEP_PASS",
        "ONLY_LINEAR_H1_PASS",
        "CONSTRUCTION_ROBUST_PREDICTION_FAILURE",
    }


def test_e24b_main_claim_disposition_decision_order(
    dry_e24b: DryE24bFixture,
) -> None:
    config, _cells, folds, feature_rows, _bound_rows, rows = dry_e24b
    predictor = precompute_holdout_predictions(
        feature_rows=feature_rows,
        outcome_rows=rows,
        folds=folds,
        ridge=float(config["predictor"]["ridge"]),
    )
    outcome_by_id = {str(row["row_id"]): float(row["log_behavioral_mse"]) for row in rows}
    metadata = {str(row["row_id"]): row for row in rows}

    perfect = [
        {
            **prediction,
            "predicted_log_behavioral_mse": outcome_by_id[str(prediction["row_id"])],
        }
        for prediction in predictor.predictions
    ]
    _metrics, _seeds, broad = score_precomputed_predictions(
        predictions=perfect,
        outcome_rows=rows,
        config=config,
        dry_run=False,
    )
    assert broad["claim_assessment"]["claim_disposition"] == "BROAD_NOISY_NONLINEAR_MULTISTEP_PASS"

    linear_only = []
    for prediction in predictor.predictions:
        row_id = str(prediction["row_id"])
        row = metadata[row_id]
        is_linear_h1 = row["readout"] == "linear" and int(row["horizon"]) == 1
        linear_only.append(
            {
                **prediction,
                "predicted_log_behavioral_mse": (outcome_by_id[row_id] if is_linear_h1 else 0.0),
            }
        )
    _metrics, _seeds, linear = score_precomputed_predictions(
        predictions=linear_only,
        outcome_rows=rows,
        config=config,
        dry_run=False,
    )
    assert linear["claim_assessment"]["claim_disposition"] == "ONLY_LINEAR_H1_PASS"

    failed = [
        {**prediction, "predicted_log_behavioral_mse": 0.0} for prediction in predictor.predictions
    ]
    _metrics, _seeds, failure = score_precomputed_predictions(
        predictions=failed,
        outcome_rows=rows,
        config=config,
        dry_run=False,
    )
    assert (
        failure["claim_assessment"]["claim_disposition"] == "CONSTRUCTION_ROBUST_PREDICTION_FAILURE"
    )


def test_e24b_main_is_blocked_before_artifact_creation(tmp_path: Path) -> None:
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


def test_e24b_dry_run_writes_predictions_before_outcome_join(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "e24b_dry"
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
    assert report["evaluation"]["test_outcome_join_after_prediction_artifact"] is True
    assert report["evaluation"]["claim_assessment"]["claim_disposition"] == "DRY_RUN_NON_EVIDENCE"
    assert (
        report["evaluation"]["optimization_gap"]["clean_target_full_controller_maximum_error"]
        <= 1e-20
    )
    assert report["dependencies"]["canonical_artifacts_read"] is False
    assert (run_dir / "protocol_lock.json").read_bytes() == LOCK_PATH.read_bytes()
    feature_path = run_dir / "teacher_side_features.jsonl"
    bound_path = run_dir / "precomputed_controller_bounds.jsonl"
    prediction_path = run_dir / "precomputed_predictions.jsonl"
    raw_path = run_dir / "raw_metrics.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    assert predictions
    assert all(row["test_outcome_used"] is False for row in predictions)
    teacher_features = [
        json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines()
    ]
    assert teacher_features
    assert all(
        row["clean_target_features_included"] is False and "log_behavioral_mse" not in row
        for row in teacher_features
    )
    precomputed_bounds = [
        json.loads(line) for line in bound_path.read_text(encoding="utf-8").splitlines()
    ]
    assert precomputed_bounds
    assert all(
        row["observed_application_outcome_used_in_lower_bound"] is False
        and row["bound_frozen_before_observed_application_outcome"] is True
        and row["predictor_feature_used"] is False
        and "observed_application_error" not in row
        and "behavioral_mse" not in row
        and "log_behavioral_mse" not in row
        for row in precomputed_bounds
    )
    assert (
        bound_path.stat().st_mtime_ns
        <= feature_path.stat().st_mtime_ns
        <= prediction_path.stat().st_mtime_ns
        <= raw_path.stat().st_mtime_ns
    )
    oos_rows = [
        json.loads(line)
        for line in (run_dir / "oos_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        row["cluster_bootstrap_resample_unit"] == "seed"
        and row["episode_row_resampling_used"] is False
        for row in oos_rows
        if row["aggregation"]
        in {
            "holdout_axis",
            "claim_subset_holdout_axis",
        }
    )
    assert (run_dir / "family_level_scatter.jsonl").is_file()
    sensitivity_path = run_dir / "factor_sensitivity.jsonl"
    assert sensitivity_path.is_file()
    sensitivity_rows = [
        json.loads(line) for line in sensitivity_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sensitivity_rows) == 9
    assert {row["factor"] for row in sensitivity_rows} == {
        "readout_lambda",
        "noise_condition",
        "horizon",
    }
    assert report["evaluation"]["sensitivity"] == {
        "factors": ["readout_lambda", "noise_condition", "horizon"],
        "aggregation": "registered_factor_level_by_seed",
        "upper_unit": "seed",
        "descriptive_only": True,
    }
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    assert "DRY_RUN_NON_EVIDENCE" in "\n".join(summary_lines)
    assert len(summary_lines) <= 45
    summary_descriptor = report["artifacts"]["results_summary_ko"]
    assert summary_descriptor["sha256"] == file_sha256(summary_path)
    assert summary_descriptor["line_count"] == len(summary_lines)
