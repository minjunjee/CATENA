from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch

from catena.core.provenance_v61 import sha256_file, write_jsonl_strict
from catena.core.randomness import seed_everything
from catena.core.schema import CandidateMode, ControllerKind, MemoryEpisode, Operation
from catena.data.geometry_sweep import generate_geometry_grid
from catena.eval.metrics import evaluate_episode
from catena.eval.seed_inference import (
    calibration_slope,
    exact_sign_flip_test,
    r2_score,
)
from catena.eval.statistics_v61 import (
    conditional_oos_r2,
    fit_full_fixed_effect,
    fit_operation_only_fixed_effect,
    fixed_seed_operation_stratified_bootstrap,
    predict_operation_fixed_effect,
)
from catena.models.matched_controllers import MatchedScalarController, ScalarConstraint
from catena.theory.reachability import (
    ReadoutMode,
    behavioral_mse,
    constrained_reachability,
)
from catena.training.matched_probe import (
    MatchedTrainConfig,
    apply_matched_controller,
    train_matched_controller,
)
from catena.training.schedules_v61 import (
    balanced_geometry_schedule,
    schedule_cell_counts,
    schedule_sha256,
)
from experiments.common import build_parser
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00_e01,
)

EXPERIMENT_ID = "e01b_constrained_behavioral_reachability"
DEFAULT_CONFIG = "configs/e01b_constrained_behavioral_reachability.yaml"
PRIMARY_CONDITION = "oracle_candidate/tied"
OPERATION_ORDER = tuple(operation.value for operation in Operation)
PREDICTORS = (
    "behavior_feasible_mse",
    "state_feasible_mse",
    "state_span_mse",
)


def _kind(constraint: ScalarConstraint) -> ControllerKind:
    return (
        ControllerKind.TIED_SCALAR
        if constraint is ScalarConstraint.TIED
        else ControllerKind.DUAL_SCALAR
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _numeric_field(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}.")
    return float(value)


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    local = tensor.detach().cpu().contiguous()
    return (
        str(local.dtype).encode("utf-8")
        + b"\0"
        + repr(tuple(local.shape)).encode("utf-8")
        + b"\0"
        + local.numpy().tobytes()
    )


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_digest(state[name]))
        digest.update(b"\0")
    return digest.hexdigest()


def _paired_episode_digest(
    oracle: Sequence[MemoryEpisode],
    recurrent: Sequence[MemoryEpisode],
) -> dict[str, object]:
    if len(oracle) != len(recurrent):
        raise AssertionError("Oracle/Recurrent episode collections must be equally sized.")
    digest = hashlib.sha256()
    erase_differences: list[float] = []
    tensor_names = (
        "keys",
        "values",
        "state",
        "target_state",
        "old_value",
        "new_value",
        "write_candidate",
        "unaffected_indices",
    )
    for oracle_episode, recurrent_episode in zip(oracle, recurrent, strict=True):
        if oracle_episode.operation is not recurrent_episode.operation:
            raise AssertionError("Paired candidate modes changed the operation.")
        if oracle_episode.affected_index != recurrent_episode.affected_index:
            raise AssertionError("Paired candidate modes changed the affected address.")
        for name in tensor_names:
            left = getattr(oracle_episode, name)
            right = getattr(recurrent_episode, name)
            if not torch.equal(left, right):
                raise AssertionError(f"Paired candidate modes changed base tensor: {name}")
            digest.update(_tensor_digest(left))
        erase_differences.append(
            float(
                torch.mean(
                    (oracle_episode.erase_candidate - recurrent_episode.erase_candidate)
                    ** 2
                ).item()
            )
        )
    return {
        "episodes": len(oracle),
        "base_tensor_sha256": digest.hexdigest(),
        "mean_erase_candidate_mse": float(np.mean(erase_differences)),
        "only_erase_candidate_varies": True,
    }


def _evaluate(
    model: MatchedScalarController,
    episodes: Sequence[MemoryEpisode],
    *,
    seed: int,
    mode: CandidateMode,
    constraint: ScalarConstraint,
    split: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    kind = _kind(constraint)
    for episode in episodes:
        output = apply_matched_controller(model, episode).cpu()
        metrics = evaluate_episode(output, episode)
        state = constrained_reachability(episode, kind, mode=ReadoutMode.STATE)
        behavior = constrained_reachability(
            episode, kind, mode=ReadoutMode.BEHAVIORAL
        )
        affected = constrained_reachability(
            episode, kind, mode=ReadoutMode.AFFECTED
        )
        retention = constrained_reachability(
            episode, kind, mode=ReadoutMode.RETENTION
        )
        learned_behavior = float(behavioral_mse(output, episode).item())
        if not math.isfinite(learned_behavior):
            raise FloatingPointError(
                f"Non-finite behavioral error for {episode.episode_id}."
            )
        source_seed = int(episode.metadata["seed"])
        rows.append(
            {
                "seed": seed,
                "split": split,
                "candidate_mode": mode.value,
                "constraint": constraint.value,
                "episode_id": episode.episode_id,
                "base_episode_id": f"{source_seed}:{episode.operation.value}",
                "episode_seed": source_seed,
                "operation": episode.operation.value,
                "state_span_mse": state.span_mse,
                "state_feasible_mse": state.feasible_mse,
                "behavior_span_mse": behavior.span_mse,
                "behavior_feasible_mse": behavior.feasible_mse,
                "affected_feasible_mse": affected.feasible_mse,
                "retention_feasible_mse": retention.feasible_mse,
                "behavior_rank": behavior.rank,
                "behavior_condition_number": _finite_or_none(
                    behavior.condition_number
                ),
                "behavior_condition_number_finite": math.isfinite(
                    behavior.condition_number
                ),
                "behavior_principal_angle_deg": behavior.principal_angle_deg,
                "learned_behavior_mse": learned_behavior,
                "learned_excess_over_behavioral_bound": (
                    learned_behavior - behavior.feasible_mse
                ),
                "old_scale": float(episode.metadata["old_scale"]),
                "new_scale": float(episode.metadata["new_scale"]),
                "old_new_cosine": float(episode.metadata["old_new_cosine"]),
                "key_correlation": float(episode.metadata["key_correlation"]),
                "state_load": float(episode.metadata["state_load"]),
                "candidate_contamination": float(
                    episode.metadata["candidate_contamination"]
                ),
                **metrics.to_dict(),
            }
        )
    return rows


def _row_arrays(
    rows: Sequence[dict[str, object]], predictor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [_numeric_field(row, predictor) for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [_numeric_field(row, "learned_behavior_mse") for row in rows],
        dtype=np.float64,
    )
    operations = np.asarray([str(row["operation"]) for row in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError(f"Non-finite analysis input for {predictor}.")
    return x, y, operations


def _calibrate(
    train_rows: Sequence[dict[str, object]],
    test_rows: Sequence[dict[str, object]],
    predictor: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    x_train, y_train, operation_train = _row_arrays(train_rows, predictor)
    x_test, y_test, operation_test = _row_arrays(test_rows, predictor)
    operation_only = fit_operation_only_fixed_effect(
        y_train,
        operation_train,
        operation_order=OPERATION_ORDER,
    )
    operation_prediction = predict_operation_fixed_effect(
        operation_only, operation_test
    )
    train_residual_x = np.concatenate(
        [
            x_train[operation_train == operation]
            - np.mean(x_train[operation_train == operation])
            for operation in OPERATION_ORDER
        ]
    )
    test_residual_x = np.concatenate(
        [
            x_test[operation_test == operation]
            - np.mean(x_test[operation_test == operation])
            for operation in OPERATION_ORDER
        ]
    )
    train_slope_estimable = bool(
        float(np.dot(train_residual_x, train_residual_x)) > 1e-18
    )
    test_slope_estimable = bool(
        float(np.dot(test_residual_x, test_residual_x)) > 1e-18
    )
    if train_slope_estimable:
        full = fit_full_fixed_effect(
            x_train,
            y_train,
            operation_train,
            operation_order=OPERATION_ORDER,
        )
        full_prediction = predict_operation_fixed_effect(
            full, operation_test, x=x_test
        )
        if full.slope is None:
            raise AssertionError("Full fixed-effect model must expose a slope.")
        train_slope = float(full.slope)
    else:
        full_prediction = operation_prediction.copy()
        train_slope = 0.0
    if test_slope_estimable:
        unseen_fit = fit_full_fixed_effect(
            x_test,
            y_test,
            operation_test,
            operation_order=OPERATION_ORDER,
        )
        if unseen_fit.slope is None:
            raise AssertionError("Full unseen fixed-effect model must expose a slope.")
        test_slope = float(unseen_fit.slope)
    else:
        test_slope = 0.0
    report: dict[str, object] = {
        "train_operation_adjusted_slope": train_slope,
        "train_slope_estimable": train_slope_estimable,
        "unseen_operation_adjusted_slope": test_slope,
        "unseen_slope_estimable": test_slope_estimable,
        "unseen_operation_only_r2": r2_score(y_test, operation_prediction),
        "unseen_full_raw_r2": r2_score(y_test, full_prediction),
        "unseen_conditional_r2": conditional_oos_r2(
            y_test, full_prediction, operation_prediction
        ),
        "unseen_calibration_slope": calibration_slope(
            y_test, full_prediction
        ),
        "test_mean_learned_excess_over_bound": float(
            np.mean(y_test - x_test)
        ),
        "test_mean_error_to_bound_ratio": float(
            np.mean(y_test) / max(float(np.mean(x_test)), 1e-12)
        ),
        "test_mean_predictor": float(np.mean(x_test)),
        "test_mean_learned_behavior_mse": float(np.mean(y_test)),
    }
    if not all(
        math.isfinite(_numeric_field(report, key))
        for key, value in report.items()
        if not isinstance(value, bool)
    ):
        raise FloatingPointError(f"Non-finite calibration output for {predictor}.")
    payload = {
        "x": x_test,
        "y": y_test,
        "operations": operation_test,
        "operation_prediction": operation_prediction,
        "full_prediction": full_prediction,
    }
    return report, payload


def _interval_dict(interval: Any) -> dict[str, object]:
    return {
        "estimate": float(interval.estimate),
        "ci95": [float(interval.low), float(interval.high)],
        "resampling_unit": "episode_within_fixed_seed_and_operation",
    }


def _bootstrap_primary(
    payloads: Mapping[int, dict[str, np.ndarray]],
    *,
    samples: int,
) -> dict[str, object]:
    operations = {
        seed: payload["operations"] for seed, payload in payloads.items()
    }

    def conditional_statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        values = []
        for seed, indices in indices_by_seed.items():
            payload = payloads[seed]
            values.append(
                conditional_oos_r2(
                    payload["y"][indices],
                    payload["full_prediction"][indices],
                    payload["operation_prediction"][indices],
                )
            )
        return float(np.mean(values))

    def slope_statistic(indices_by_seed: Mapping[int, np.ndarray]) -> float:
        values = []
        for seed, indices in indices_by_seed.items():
            payload = payloads[seed]
            fit = fit_full_fixed_effect(
                payload["x"][indices],
                payload["y"][indices],
                payload["operations"][indices],
                operation_order=OPERATION_ORDER,
            )
            if fit.slope is None:
                raise AssertionError("Bootstrap full model must expose a slope.")
            values.append(float(fit.slope))
        return float(np.mean(values))

    def calibration_statistic(
        indices_by_seed: Mapping[int, np.ndarray],
    ) -> float:
        values = []
        for seed, indices in indices_by_seed.items():
            payload = payloads[seed]
            values.append(
                calibration_slope(
                    payload["y"][indices],
                    payload["full_prediction"][indices],
                )
            )
        return float(np.mean(values))

    return {
        "conditional_oos_r2": _interval_dict(
            fixed_seed_operation_stratified_bootstrap(
                operations,
                conditional_statistic,
                samples=samples,
                seed=6101,
            )
        ),
        "unseen_operation_adjusted_slope": _interval_dict(
            fixed_seed_operation_stratified_bootstrap(
                operations,
                slope_statistic,
                samples=samples,
                seed=6102,
            )
        ),
        "calibration_slope_descriptive": _interval_dict(
            fixed_seed_operation_stratified_bootstrap(
                operations,
                calibration_statistic,
                samples=samples,
                seed=6103,
            )
        ),
    }


def _claim_eligibility(
    *,
    dry_run: bool,
    seeds: Sequence[int],
    configured_seeds: Sequence[int],
    condition_count: int,
    row_count: int,
    expected_row_count: int,
    checkpoint_count: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if dry_run:
        reasons.append("dry_run")
    if list(seeds) != list(configured_seeds):
        reasons.append("seed_set_does_not_match_preregistered_order")
    if len(set(seeds)) != 8:
        reasons.append("requires_exactly_8_unique_training_seeds")
    if condition_count != 4:
        reasons.append("requires_all_4_candidate_constraint_conditions")
    if row_count != expected_row_count:
        reasons.append("episode_row_count_mismatch")
    if checkpoint_count != len(seeds) * 4:
        reasons.append("checkpoint_count_mismatch")
    return not reasons, reasons


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    dependencies = validate_legacy_e00_e01(
        args.artifact_root,
        require_full=not args.dry_run,
    )
    config, run_dir, device, run_context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    configured_seeds = [int(value) for value in config["seeds"]]
    seeds = list(configured_seeds)
    steps = int(config["training"]["steps"])
    train_count = int(config["data"]["train_count_per_cell"])
    test_count = int(config["data"]["test_count_per_cell"])
    train_grid = config["data"]["train_grid"]
    test_grid = config["data"]["test_grid"]
    bootstrap_samples = int(config["statistics"]["bootstrap_samples"])
    if args.dry_run:
        seeds = seeds[:1]
        steps = min(steps, 20)
        train_count = test_count = 2
        bootstrap_samples = min(bootstrap_samples, 100)
        train_grid = {
            "key_dim": [12],
            "value_dim": [12],
            "num_associations": [6],
            "key_correlations": [0.2],
            "old_scales": [0.8, 1.2],
            "new_scales": [1.0],
            "old_new_cosines": [0.0],
        }
        test_grid = {
            "key_dim": [12],
            "value_dim": [12],
            "num_associations": [8],
            "key_correlations": [0.35],
            "old_scales": [0.7, 1.3],
            "new_scales": [1.2],
            "old_new_cosines": [0.25],
        }

    rows: list[dict[str, object]] = []
    condition_reports: dict[str, list[dict[str, object]]] = defaultdict(list)
    primary_bootstrap_payloads: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_records: list[dict[str, object]] = []
    pairing_records: list[dict[str, object]] = []
    schedule_records: list[dict[str, object]] = []
    expected_rows = 0

    for seed in seeds:
        seed_everything(seed)
        generated: dict[
            CandidateMode, tuple[list[MemoryEpisode], list[MemoryEpisode]]
        ] = {}
        for mode in (CandidateMode.ORACLE, CandidateMode.RECURRENT_READ):
            generated[mode] = (
                generate_geometry_grid(
                    seed=seed * 100000,
                    candidate_mode=mode,
                    grid=train_grid,
                    count_per_cell=train_count,
                ),
                generate_geometry_grid(
                    seed=seed * 100000 + 50000,
                    candidate_mode=mode,
                    grid=test_grid,
                    count_per_cell=test_count,
                ),
            )

        pairing_records.append(
            {
                "seed": seed,
                "train": _paired_episode_digest(
                    generated[CandidateMode.ORACLE][0],
                    generated[CandidateMode.RECURRENT_READ][0],
                ),
                "test": _paired_episode_digest(
                    generated[CandidateMode.ORACLE][1],
                    generated[CandidateMode.RECURRENT_READ][1],
                ),
            }
        )

        for mode in (CandidateMode.ORACLE, CandidateMode.RECURRENT_READ):
            train_episodes, test_episodes = generated[mode]
            schedule = balanced_geometry_schedule(
                train_episodes,
                steps=steps,
                seed=seed * 101
                + int(mode is CandidateMode.RECURRENT_READ),
            )
            counts = schedule_cell_counts(schedule)
            if steps >= len(counts) and min(counts.values()) < 1:
                raise AssertionError("Balanced schedule omitted a geometry/operation cell.")
            order_hash = schedule_sha256(schedule)
            schedule_records.append(
                {
                    "seed": seed,
                    "candidate_mode": mode.value,
                    "steps": steps,
                    "geometry_operation_cells": len(counts),
                    "minimum_cell_exposure": min(counts.values()),
                    "maximum_cell_exposure": max(counts.values()),
                    "episode_order_sha256": order_hash,
                    "shared_by_tied_and_dual": True,
                }
            )
            expected_rows += 2 * (len(train_episodes) + len(test_episodes))

            seed_everything(seed * 1000 + int(mode is CandidateMode.RECURRENT_READ))
            template = MatchedScalarController(
                10, int(config["model"]["hidden_dim"]), ScalarConstraint.DUAL
            )
            initial_state = {
                name: value.detach().clone()
                for name, value in template.state_dict().items()
            }
            initial_sha = _state_dict_sha256(initial_state)

            for constraint in (ScalarConstraint.TIED, ScalarConstraint.DUAL):
                model = MatchedScalarController(
                    10, int(config["model"]["hidden_dim"]), constraint
                )
                model.load_state_dict(initial_state)
                if _state_dict_sha256(model.state_dict()) != initial_sha:
                    raise AssertionError("Matched condition failed initial-state parity.")
                losses = train_matched_controller(
                    model=model,
                    episodes=schedule,
                    config=MatchedTrainConfig(
                        steps=steps,
                        learning_rate=float(config["training"]["learning_rate"]),
                    ),
                    device=device,
                )
                if not np.isfinite(np.asarray(losses)).all():
                    raise FloatingPointError("Training trace contains a non-finite loss.")

                train_rows = _evaluate(
                    model,
                    train_episodes,
                    seed=seed,
                    mode=mode,
                    constraint=constraint,
                    split="train_geometry",
                )
                test_rows = _evaluate(
                    model,
                    test_episodes,
                    seed=seed,
                    mode=mode,
                    constraint=constraint,
                    split="unseen_geometry",
                )
                rows.extend(train_rows + test_rows)
                key = f"{mode.value}/{constraint.value}"
                predictor_reports: dict[str, dict[str, object]] = {}
                for predictor in PREDICTORS:
                    predictor_report, payload = _calibrate(
                        train_rows, test_rows, predictor
                    )
                    predictor_reports[predictor] = predictor_report
                    if (
                        key == PRIMARY_CONDITION
                        and predictor == "behavior_feasible_mse"
                    ):
                        primary_bootstrap_payloads[seed] = payload

                condition_reports[key].append(
                    {
                        "seed": seed,
                        "predictors": predictor_reports,
                        "training": {
                            "steps": steps,
                            "first_loss": losses[0],
                            "final_loss": losses[-1],
                            "mean_loss": float(np.mean(losses)),
                            "episode_order_sha256": order_hash,
                            "initial_state_sha256": initial_sha,
                            "final_state_sha256": _state_dict_sha256(
                                model.state_dict()
                            ),
                        },
                    }
                )

                checkpoint = (
                    run_dir / f"seed{seed}_{mode.value}_{constraint.value}.pt"
                )
                torch.save(model.state_dict(), checkpoint)
                loaded = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                checkpoint_records.append(
                    {
                        "seed": seed,
                        "candidate_mode": mode.value,
                        "constraint": constraint.value,
                        "path": checkpoint.name,
                        "file_sha256": sha256_file(checkpoint),
                        "state_dict_sha256": _state_dict_sha256(loaded),
                        "round_trip_match": (
                            _state_dict_sha256(loaded)
                            == _state_dict_sha256(model.state_dict())
                        ),
                    }
                )

    summary: dict[str, dict[str, Any]] = {}
    for key, reports in condition_reports.items():
        predictors: dict[str, object] = {}
        for predictor in PREDICTORS:
            seed_values = [
                cast(dict[str, Any], report["predictors"])[predictor]
                for report in reports
            ]
            conditional = np.asarray(
                [value["unseen_conditional_r2"] for value in seed_values]
            )
            slopes = np.asarray(
                [value["unseen_operation_adjusted_slope"] for value in seed_values]
            )
            calibration = np.asarray(
                [value["unseen_calibration_slope"] for value in seed_values]
            )
            predictors[predictor] = {
                "seed_reports": seed_values,
                "mean_unseen_conditional_r2": float(conditional.mean()),
                "mean_unseen_operation_adjusted_slope": float(slopes.mean()),
                "unseen_slope_sign_flip_p": exact_sign_flip_test(
                    slopes, alternative="greater"
                ),
                "mean_unseen_calibration_slope": float(calibration.mean()),
            }
        summary[key] = {
            "seed_reports": reports,
            "predictors": predictors,
        }

    primary_predictor = cast(
        dict[str, Any],
        summary[PRIMARY_CONDITION]["predictors"],
    )["behavior_feasible_mse"]
    minimum_r2 = float(config["claim_gate"]["minimum_oos_r2"])
    alpha = float(config["claim_gate"]["alpha"])
    mean_r2 = float(primary_predictor["mean_unseen_conditional_r2"])
    slope_p = float(primary_predictor["unseen_slope_sign_flip_p"])
    eligible, eligibility_reasons = _claim_eligibility(
        dry_run=args.dry_run,
        seeds=seeds,
        configured_seeds=configured_seeds,
        condition_count=len(condition_reports),
        row_count=len(rows),
        expected_row_count=expected_rows,
        checkpoint_count=len(checkpoint_records),
    )
    supported = bool(
        eligible and mean_r2 >= minimum_r2 and slope_p <= alpha
    )

    oracle_tied = cast(
        dict[str, Any],
        summary["oracle_candidate/tied"]["predictors"],
    )
    state_span_r2 = float(
        oracle_tied["state_span_mse"]["mean_unseen_conditional_r2"]
    )
    state_feasible_r2 = float(
        oracle_tied["state_feasible_mse"]["mean_unseen_conditional_r2"]
    )
    reachability_comparison = {
        "r_span_conditional_oos_r2": state_span_r2,
        "r_feas_conditional_oos_r2": state_feasible_r2,
        "r_beh_conditional_oos_r2": mean_r2,
        "r_beh_minus_r_span": mean_r2 - state_span_r2,
        "r_beh_minus_r_feas": mean_r2 - state_feasible_r2,
        "behavioral_predictor_observed_higher_than_state_span": (
            mean_r2 > state_span_r2
        ),
        "confirmatory_gate": False,
    }

    candidate_gap: dict[str, list[dict[str, float]]] = defaultdict(list)
    for constraint in (ScalarConstraint.TIED, ScalarConstraint.DUAL):
        oracle_reports = condition_reports[
            f"{CandidateMode.ORACLE.value}/{constraint.value}"
        ]
        recurrent_reports = condition_reports[
            f"{CandidateMode.RECURRENT_READ.value}/{constraint.value}"
        ]
        for oracle_report, recurrent_report in zip(
            oracle_reports, recurrent_reports, strict=True
        ):
            oracle_metrics = cast(
                dict[str, Any],
                oracle_report["predictors"],
            )["behavior_feasible_mse"]
            recurrent_metrics = cast(
                dict[str, Any],
                recurrent_report["predictors"],
            )["behavior_feasible_mse"]
            candidate_gap[constraint.value].append(
                {
                    "seed": _numeric_field(oracle_report, "seed"),
                    "recurrent_minus_oracle_bound_mse": float(
                        recurrent_metrics["test_mean_predictor"]
                        - oracle_metrics["test_mean_predictor"]
                    ),
                    "recurrent_minus_oracle_learned_mse": float(
                        recurrent_metrics["test_mean_learned_behavior_mse"]
                        - oracle_metrics["test_mean_learned_behavior_mse"]
                    ),
                    "recurrent_minus_oracle_excess_over_bound": float(
                        recurrent_metrics[
                            "test_mean_learned_excess_over_bound"
                        ]
                        - oracle_metrics[
                            "test_mean_learned_excess_over_bound"
                        ]
                    ),
                }
            )

    episode_metrics_path = run_dir / "episode_geometry_metrics.jsonl"
    write_jsonl_strict(episode_metrics_path, rows)
    report = {
        "status": "PASS",
        "execution": {
            "dry_run": bool(args.dry_run),
            "row_count": len(rows),
            "expected_row_count": expected_rows,
            "unique_seeds": seeds,
            "condition_count": len(condition_reports),
            "checkpoint_count": len(checkpoint_records),
            "scientific_evidence": False,
        },
        "primary": {
            "condition": PRIMARY_CONDITION,
            "predictor": (
                "box_constrained_equal_weight_behavioral_mse_lower_bound"
            ),
            "oos_estimand": (
                "1-SSE(full_operation_plus_predictor)/SSE(operation_only)"
            ),
            "mean_unseen_geometry_conditional_r2": mean_r2,
            "minimum_oos_r2": minimum_r2,
            "unseen_positive_slope_sign_flip_p": slope_p,
            "alpha": alpha,
            "mean_test_calibration_slope": float(
                primary_predictor["mean_unseen_calibration_slope"]
            ),
            "calibration_confirmatory_gate": False,
            "eligible": eligible,
            "eligibility_failures": eligibility_reasons,
            "supported": supported,
        },
        "episode_uncertainty": _bootstrap_primary(
            primary_bootstrap_payloads,
            samples=bootstrap_samples,
        ),
        "training_seed_uncertainty": {
            "method": "exact_sign_flip",
            "unit": "8_independent_training_seeds",
            "p_value": slope_p,
        },
        "conditions": summary,
        "reachability_comparison": reachability_comparison,
        "candidate_mode_paired_gaps": {
            "meaning": (
                "candidate recovery/content interference diagnostic; not addressing"
            ),
            "base_episode_pairing": pairing_records,
            "seed_gaps": candidate_gap,
        },
        "training_schedules": schedule_records,
        "checkpoints": checkpoint_records,
        "artifacts": {
            "episode_geometry_metrics": {
                "path": episode_metrics_path.name,
                "rows": len(rows),
                "sha256": sha256_file(episode_metrics_path),
            }
        },
        "dependency_lineage": dependencies,
        "claim_gate": {
            "supported": supported,
            "eligible": eligible,
            "constrained_behavioral_reachability_predicts_error": supported,
            "state_predictor_comparison_is_diagnostic": True,
            "recurrent_read_is_candidate_recovery_content_interference": True,
        },
        "scientific_evidence": False,
    }
    finalize_v61_run(
        context=run_context,
        report=report,
        main_eligible=eligible,
        full_eligible=eligible,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
