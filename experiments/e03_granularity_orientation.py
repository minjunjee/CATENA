from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import sha256_file, write_jsonl_strict
from catena.data.operator_families import OperatorFamily, generate_operator_set
from catena.eval.seed_inference import exact_sign_flip_test
from catena.systems.device import resolve_device
from catena.theory.joint_diagonalization import (
    analyze_joint_diagonalization,
    joint_diagonalization_objective,
)
from experiments.common import build_parser
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e03_granularity_orientation"
DEFAULT_CONFIG = "configs/e03_granularity_orientation.yaml"


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite.")
    return result


def _validated_seeds(value: object, name: str, *, exact_count: int | None) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    seeds: list[int] = []
    for seed in value:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{name} must contain only integers.")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must contain unique values.")
    if exact_count is not None and len(seeds) != exact_count:
        raise ValueError(f"{name} must contain exactly {exact_count} values.")
    return seeds


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("The config experiment_id does not match E03.")
    main_seeds = _validated_seeds(config.get("seeds"), "seeds", exact_count=8)
    dry_seeds = _validated_seeds(
        config.get("dry_run_seeds"),
        "dry_run_seeds",
        exact_count=2,
    )
    if set(main_seeds) & set(dry_seeds):
        raise ValueError("Main and dry-run seeds must be disjoint.")

    data = config.get("data")
    model = config.get("model")
    optimization = config.get("optimization")
    statistics = config.get("statistics")
    runtime = config.get("runtime")
    if not all(
        isinstance(section, dict)
        for section in (data, model, optimization, statistics, runtime)
    ):
        raise ValueError(
            "data, model, optimization, statistics, and runtime must be mappings."
        )
    assert isinstance(data, dict)
    assert isinstance(model, dict)
    assert isinstance(optimization, dict)
    assert isinstance(statistics, dict)
    assert isinstance(runtime, dict)

    dimension = _positive_integer(data.get("dimension"), "data.dimension")
    projector_rank = _positive_integer(
        data.get("projector_rank"),
        "data.projector_rank",
    )
    if projector_rank >= dimension:
        raise ValueError("data.projector_rank must be smaller than data.dimension.")
    _positive_integer(
        data.get("train_operators_per_family"),
        "data.train_operators_per_family",
    )
    _positive_integer(
        data.get("test_operators_per_family"),
        "data.test_operators_per_family",
    )
    _positive_integer(data.get("probes_per_operator"), "data.probes_per_operator")
    low_rank = _positive_integer(model.get("low_rank"), "model.low_rank")
    if low_rank != projector_rank:
        raise ValueError(
            "model.low_rank must equal data.projector_rank for the registered "
            "transaction-conditioned oracle recovery check."
        )

    for key in ("steps", "restarts", "dry_run_steps", "dry_run_restarts"):
        _positive_integer(optimization.get(key), f"optimization.{key}")
    _finite_float(
        optimization.get("learning_rate"),
        "optimization.learning_rate",
        positive=True,
    )
    cpu_threads = _positive_integer(runtime.get("cpu_threads"), "runtime.cpu_threads")
    if cpu_threads > 16:
        raise ValueError("runtime.cpu_threads must not exceed 16 for this small-matrix run.")

    alpha = _finite_float(statistics.get("alpha"), "statistics.alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("statistics.alpha must lie strictly between zero and one.")
    for key in (
        "minimum_practical_regret_contrast",
        "maximum_commuting_commutator_norm",
        "maximum_sufficient_regret",
        "minimum_noncommuting_commutator_norm",
        "minimum_noncommuting_regret",
        "maximum_oracle_regret",
        "maximum_orthogonality_error",
        "minimum_empirical_prediction_r2",
        "minimum_empirical_calibration_slope",
        "maximum_empirical_calibration_slope",
        "maximum_empirical_calibration_intercept",
        "empirical_calibration_floor",
    ):
        _finite_float(statistics.get(key), f"statistics.{key}", positive=True)
    minimum_slope = float(statistics["minimum_empirical_calibration_slope"])
    maximum_slope = float(statistics["maximum_empirical_calibration_slope"])
    if minimum_slope >= maximum_slope:
        raise ValueError("Empirical calibration slope bounds are not ordered.")
    minimum_r2 = float(statistics["minimum_empirical_prediction_r2"])
    if not 0.0 < minimum_r2 <= 1.0:
        raise ValueError("minimum_empirical_prediction_r2 must lie in (0, 1].")


def _projector_diagnostics(projectors: list[torch.Tensor]) -> dict[str, Any]:
    symmetry_errors = [
        float(
            torch.linalg.matrix_norm(
                projector - projector.transpose(0, 1),
                ord="fro",
            ).item()
        )
        for projector in projectors
    ]
    idempotence_errors = [
        float(
            torch.linalg.matrix_norm(
                projector @ projector - projector,
                ord="fro",
            ).item()
        )
        for projector in projectors
    ]
    ranks = [int(torch.linalg.matrix_rank(projector).item()) for projector in projectors]
    return {
        "maximum_symmetry_error": max(symmetry_errors),
        "maximum_idempotence_error": max(idempotence_errors),
        "minimum_numerical_rank": min(ranks),
        "maximum_numerical_rank": max(ranks),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Correlation inputs must be aligned vectors.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation inputs must be finite.")
    if len(x) < 3:
        return {
            "evaluable": False,
            "n": int(len(x)),
            "pearson": None,
            "reason": "fewer_than_three_observations",
        }
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return {
            "evaluable": False,
            "n": int(len(x)),
            "pearson": None,
            "reason": "near_constant_input",
        }
    return {
        "evaluable": True,
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "reason": None,
    }


def _empirical_calibration(
    rows: list[dict[str, Any]],
    *,
    floor: float,
) -> dict[str, Any]:
    analytic: list[float] = []
    empirical: list[float] = []
    for row in rows:
        for method in ("fixed_diagonal", "learned_basis_diagonal"):
            predicted = float(row[f"{method}_regret"])
            observed = float(row[f"{method}_empirical_error"])
            if predicted > floor:
                analytic.append(predicted)
                empirical.append(observed)
    x = np.asarray(analytic, dtype=np.float64)
    y = np.asarray(empirical, dtype=np.float64)
    if len(x) < 3:
        raise ValueError("Too few above-floor values for empirical calibration.")
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.asarray([intercept, slope])
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 1e-20:
        raise ValueError("Empirical calibration target has no usable variance.")
    r2 = float(1.0 - np.sum((y - prediction) ** 2) / denominator)
    return {
        "n": int(len(x)),
        "analytic_metric": "heldout_mean_operator_entry_squared_error",
        "empirical_metric": (
            "heldout_isotropic_probe_application_mse_with_component_variance_1/d"
        ),
        "floor": floor,
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": r2,
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "maximum_absolute_prediction_error": float(np.max(np.abs(y - x))),
    }


def _paired_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["family"]))
        if key in indexed:
            raise ValueError(f"Duplicate E03 row key: {key}.")
        indexed[key] = row
    return indexed


def _contrast_report(
    values: np.ndarray,
    seeds: list[int],
    *,
    sesoi: float,
    alpha: float,
) -> dict[str, Any]:
    if values.shape != (len(seeds),) or not np.isfinite(values).all():
        raise ValueError("Contrast values must be one finite value per seed.")
    p_value = exact_sign_flip_test(values - sesoi, "greater")
    estimate = float(values.mean())
    return {
        "estimate": estimate,
        "minimum_seed_value": float(values.min()),
        "maximum_seed_value": float(values.max()),
        "minimum_practical_regret_contrast": sesoi,
        "seed_exact_sign_flip_p_greater_than_sesoi": p_value,
        "seed_values": {
            str(seed): float(value)
            for seed, value in zip(seeds, values, strict=True)
        },
        "passed": bool(estimate > sesoi and p_value <= alpha),
    }


def _build_frontier(
    rows: list[dict[str, Any]],
    *,
    dimension: int,
    low_rank: int,
) -> list[dict[str, Any]]:
    methods = {
        "fixed_diagonal": {
            "active_coefficients_per_transaction": dimension,
            "shared_stored_parameters": 0,
            "shared_intrinsic_degrees_of_freedom": 0,
            "matrix_vector_multiply_accumulates": dimension,
            "role": "oracle projection in the fixed coordinate basis",
        },
        "learned_basis_diagonal": {
            "active_coefficients_per_transaction": dimension,
            "shared_stored_parameters": dimension * dimension,
            "shared_intrinsic_degrees_of_freedom": dimension * (dimension - 1) // 2,
            "matrix_vector_multiply_accumulates": 2 * dimension * dimension + dimension,
            "role": "multi-restart learned shared basis with oracle diagonal coefficients",
        },
        "transaction_conditioned_low_rank": {
            "active_coefficients_per_transaction": 2 * dimension * low_rank,
            "shared_stored_parameters": 0,
            "shared_intrinsic_degrees_of_freedom": 0,
            "matrix_vector_multiply_accumulates": 2 * dimension * low_rank,
            "role": "richer-control oracle upper bound; not parameter matched",
        },
        "full_matrix": {
            "active_coefficients_per_transaction": dimension * dimension,
            "shared_stored_parameters": 0,
            "shared_intrinsic_degrees_of_freedom": 0,
            "matrix_vector_multiply_accumulates": dimension * dimension,
            "role": "full-matrix oracle",
        },
    }
    field_prefixes = {
        "fixed_diagonal": "fixed_diagonal",
        "learned_basis_diagonal": "learned_basis_diagonal",
        "transaction_conditioned_low_rank": "low_rank",
        "full_matrix": "full_matrix",
    }
    frontier: list[dict[str, Any]] = []
    for family in OperatorFamily:
        family_rows = [row for row in rows if row["family"] == family.value]
        if not family_rows:
            raise ValueError(f"Missing frontier rows for {family.value}.")
        for method, cost in methods.items():
            prefix = field_prefixes[method]
            frontier.append(
                {
                    "family": family.value,
                    "control_class": method,
                    "mean_analytic_regret": float(
                        np.mean(
                            [
                                float(row[f"{prefix}_regret"])
                                for row in family_rows
                            ]
                        )
                    ),
                    "mean_empirical_error": float(
                        np.mean(
                            [
                                float(row[f"{prefix}_empirical_error"])
                                for row in family_rows
                            ]
                        )
                    ),
                    **cost,
                }
            )
    return frontier


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    preview = load_config(Path(args.config).resolve(strict=True))
    _validate_config(preview)
    preflight_device = resolve_device(args.device)
    if preflight_device.type != "cpu":
        raise ValueError("E03 is a deterministic CPU theory experiment; use --device cpu.")

    runtime = preview["runtime"]
    assert isinstance(runtime, dict)
    torch.set_num_threads(int(runtime["cpu_threads"]))
    dependencies = [
        validate_legacy_e00(
            args.artifact_root,
            require_full=not args.dry_run,
        )
    ]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    if device.type != "cpu":
        raise AssertionError("E03 device changed after CPU preflight.")

    data = config["data"]
    model = config["model"]
    optimization = config["optimization"]
    statistics = config["statistics"]
    assert isinstance(data, dict)
    assert isinstance(model, dict)
    assert isinstance(optimization, dict)
    assert isinstance(statistics, dict)

    seeds = [
        int(value)
        for value in (
            config["dry_run_seeds"] if args.dry_run else config["seeds"]
        )
    ]
    steps = int(
        optimization["dry_run_steps"] if args.dry_run else optimization["steps"]
    )
    restarts = int(
        optimization["dry_run_restarts"]
        if args.dry_run
        else optimization["restarts"]
    )
    dimension = int(data["dimension"])
    projector_rank = int(data["projector_rank"])
    train_count = int(data["train_operators_per_family"])
    test_count = int(data["test_operators_per_family"])
    probe_count = int(data["probes_per_operator"])
    low_rank = int(model["low_rank"])

    rows: list[dict[str, Any]] = []
    experiment_started = time.perf_counter()
    for seed in seeds:
        for family_index, family in enumerate(OperatorFamily):
            family_started = time.perf_counter()
            operator_set = generate_operator_set(
                family=family,
                dim=dimension,
                rank=projector_rank,
                count=train_count + test_count,
                seed=seed,
                dtype=torch.float64,
            )
            training_projectors = operator_set.projectors[:train_count]
            evaluation_projectors = operator_set.projectors[train_count:]
            result = analyze_joint_diagonalization(
                training_projectors,
                evaluation_projectors=evaluation_projectors,
                steps=steps,
                learning_rate=float(optimization["learning_rate"]),
                low_rank=low_rank,
                restarts=restarts,
                seed=seed + 10_000 * family_index,
                probe_count=probe_count,
                probe_seed=seed + 1_000_000 * (family_index + 1),
            )
            certified_objective: float | None = None
            certified_regret: float | None = None
            if operator_set.certified_shared_basis is not None:
                certified_objective = joint_diagonalization_objective(
                    operator_set.certified_shared_basis,
                    evaluation_projectors,
                )
                certified_regret = certified_objective / (
                    test_count * dimension * dimension
                )
            diagnostics = _projector_diagnostics(operator_set.projectors)
            row = {
                "seed": seed,
                "family": family.value,
                "dtype": str(evaluation_projectors[0].dtype),
                "train_operator_count": train_count,
                "test_operator_count": test_count,
                "projector_rank": projector_rank,
                "certified_shared_basis_available": (
                    operator_set.certified_shared_basis is not None
                ),
                "certified_shared_basis_rjd_objective": certified_objective,
                "certified_shared_basis_regret": certified_regret,
                "family_wall_seconds": time.perf_counter() - family_started,
                **diagnostics,
                **asdict(result),
            }
            if not _all_finite(row):
                raise FloatingPointError(
                    f"E03 generated a non-finite row for seed={seed}, family={family.value}."
                )
            rows.append(row)

    indexed = _paired_rows(rows)

    def values_for(
        left_family: OperatorFamily,
        left_metric: str,
        right_family: OperatorFamily,
        right_metric: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                float(indexed[(seed, left_family.value)][left_metric])
                - float(indexed[(seed, right_family.value)][right_metric])
                for seed in seeds
            ],
            dtype=np.float64,
        )

    alpha = float(statistics["alpha"])
    contrast_sesoi = float(statistics["minimum_practical_regret_contrast"])
    contrasts = {
        "fixed_basis_rotation_penalty": _contrast_report(
            values_for(
                OperatorFamily.COMMON_ROTATED_COMMUTING,
                "fixed_diagonal_regret",
                OperatorFamily.AXIS_COMMUTING,
                "fixed_diagonal_regret",
            ),
            seeds,
            sesoi=contrast_sesoi,
            alpha=alpha,
        ),
        "shared_basis_recovery": _contrast_report(
            values_for(
                OperatorFamily.COMMON_ROTATED_COMMUTING,
                "fixed_diagonal_regret",
                OperatorFamily.COMMON_ROTATED_COMMUTING,
                "learned_basis_diagonal_regret",
            ),
            seeds,
            sesoi=contrast_sesoi,
            alpha=alpha,
        ),
        "noncommuting_joint_diagonalization_gap": _contrast_report(
            values_for(
                OperatorFamily.NONCOMMUTING,
                "learned_basis_diagonal_regret",
                OperatorFamily.COMMON_ROTATED_COMMUTING,
                "learned_basis_diagonal_regret",
            ),
            seeds,
            sesoi=contrast_sesoi,
            alpha=alpha,
        ),
        "transaction_conditioned_low_rank_recovery": {
            **_contrast_report(
                values_for(
                    OperatorFamily.NONCOMMUTING,
                    "learned_basis_diagonal_regret",
                    OperatorFamily.NONCOMMUTING,
                    "low_rank_regret",
                ),
                seeds,
                sesoi=contrast_sesoi,
                alpha=alpha,
            ),
            "interpretation": "oracle richer-control upper bound; not parameter matched",
        },
    }

    axis_rows = [
        indexed[(seed, OperatorFamily.AXIS_COMMUTING.value)] for seed in seeds
    ]
    common_rows = [
        indexed[(seed, OperatorFamily.COMMON_ROTATED_COMMUTING.value)]
        for seed in seeds
    ]
    noncommuting_rows = [
        indexed[(seed, OperatorFamily.NONCOMMUTING.value)] for seed in seeds
    ]
    commuting_rows = [*axis_rows, *common_rows]
    maximum_commuting_commutator = max(
        float(row["commutator_norm"]) for row in commuting_rows
    )
    maximum_axis_fixed_regret = max(
        float(row["fixed_diagonal_regret"]) for row in axis_rows
    )
    maximum_common_learned_regret = max(
        float(row["learned_basis_diagonal_regret"]) for row in common_rows
    )
    maximum_certified_shared_basis_regret = max(
        float(row["certified_shared_basis_regret"]) for row in commuting_rows
    )
    minimum_noncommuting_commutator = min(
        float(row["commutator_norm"]) for row in noncommuting_rows
    )
    minimum_noncommuting_regret = min(
        float(row["learned_basis_diagonal_regret"]) for row in noncommuting_rows
    )
    maximum_low_rank_regret = max(float(row["low_rank_regret"]) for row in rows)
    maximum_full_matrix_regret = max(
        float(row["full_matrix_regret"]) for row in rows
    )
    maximum_orthogonality_error = max(
        float(row["learned_basis_orthogonality_error"]) for row in rows
    )
    optimizer_nesting_holds = all(
        float(row["train_learned_basis_diagonal_regret"])
        <= float(row["optimizer_identity_candidate_regret"]) + 1e-15
        for row in rows
    )
    projector_invariants_hold = all(
        float(row["maximum_symmetry_error"])
        <= float(statistics["maximum_oracle_regret"])
        and float(row["maximum_idempotence_error"])
        <= float(statistics["maximum_oracle_regret"])
        and int(row["minimum_numerical_rank"]) == projector_rank
        and int(row["maximum_numerical_rank"]) == projector_rank
        for row in rows
    )

    absolute_gates: dict[str, dict[str, Any]] = {
        "commuting_family_commutator_zero": {
            "value": maximum_commuting_commutator,
            "maximum": float(statistics["maximum_commuting_commutator_norm"]),
            "passed": maximum_commuting_commutator
            <= float(statistics["maximum_commuting_commutator_norm"]),
        },
        "axis_fixed_diagonal_sufficient": {
            "value": maximum_axis_fixed_regret,
            "maximum": float(statistics["maximum_sufficient_regret"]),
            "passed": maximum_axis_fixed_regret
            <= float(statistics["maximum_sufficient_regret"]),
        },
        "common_rotation_learned_basis_sufficient": {
            "value": maximum_common_learned_regret,
            "maximum": float(statistics["maximum_sufficient_regret"]),
            "passed": maximum_common_learned_regret
            <= float(statistics["maximum_sufficient_regret"]),
        },
        "construction_shared_basis_certificate": {
            "value": maximum_certified_shared_basis_regret,
            "maximum": float(statistics["maximum_oracle_regret"]),
            "passed": maximum_certified_shared_basis_regret
            <= float(statistics["maximum_oracle_regret"]),
        },
        "noncommuting_family_certificate": {
            "value": minimum_noncommuting_commutator,
            "minimum": float(statistics["minimum_noncommuting_commutator_norm"]),
            "passed": minimum_noncommuting_commutator
            >= float(statistics["minimum_noncommuting_commutator_norm"]),
        },
        "noncommuting_shared_basis_residual": {
            "value": minimum_noncommuting_regret,
            "minimum": float(statistics["minimum_noncommuting_regret"]),
            "passed": minimum_noncommuting_regret
            >= float(statistics["minimum_noncommuting_regret"]),
        },
        "low_rank_oracle_recovery": {
            "value": maximum_low_rank_regret,
            "maximum": float(statistics["maximum_oracle_regret"]),
            "passed": maximum_low_rank_regret
            <= float(statistics["maximum_oracle_regret"]),
        },
        "full_matrix_oracle_recovery": {
            "value": maximum_full_matrix_regret,
            "maximum": float(statistics["maximum_oracle_regret"]),
            "passed": maximum_full_matrix_regret
            <= float(statistics["maximum_oracle_regret"]),
        },
        "optimizer_identity_nesting": {
            "passed": optimizer_nesting_holds,
        },
        "optimizer_orthogonality": {
            "value": maximum_orthogonality_error,
            "maximum": float(statistics["maximum_orthogonality_error"]),
            "passed": maximum_orthogonality_error
            <= float(statistics["maximum_orthogonality_error"]),
        },
        "projector_invariants": {
            "passed": projector_invariants_hold,
        },
    }

    empirical_calibration = _empirical_calibration(
        rows,
        floor=float(statistics["empirical_calibration_floor"]),
    )
    empirical_gate = {
        **empirical_calibration,
        "minimum_r2": float(statistics["minimum_empirical_prediction_r2"]),
        "slope_interval": [
            float(statistics["minimum_empirical_calibration_slope"]),
            float(statistics["maximum_empirical_calibration_slope"]),
        ],
        "maximum_absolute_intercept": float(
            statistics["maximum_empirical_calibration_intercept"]
        ),
    }
    empirical_gate["passed"] = bool(
        float(empirical_calibration["r2"])
        >= float(statistics["minimum_empirical_prediction_r2"])
        and float(statistics["minimum_empirical_calibration_slope"])
        <= float(empirical_calibration["slope"])
        <= float(statistics["maximum_empirical_calibration_slope"])
        and abs(float(empirical_calibration["intercept"]))
        <= float(statistics["maximum_empirical_calibration_intercept"])
    )

    commutators = np.asarray(
        [float(row["commutator_norm"]) for row in rows],
        dtype=np.float64,
    )
    jd_regrets = np.asarray(
        [float(row["learned_basis_diagonal_regret"]) for row in rows],
        dtype=np.float64,
    )
    noncommuting_commutators = np.asarray(
        [float(row["commutator_norm"]) for row in noncommuting_rows],
        dtype=np.float64,
    )
    noncommuting_jd_regrets = np.asarray(
        [
            float(row["learned_basis_diagonal_regret"])
            for row in noncommuting_rows
        ],
        dtype=np.float64,
    )
    geometry_prediction = {
        "pooled_family_clustered_descriptive": _correlation(
            commutators,
            jd_regrets,
        ),
        "noncommuting_seed_level_descriptive": _correlation(
            noncommuting_commutators,
            noncommuting_jd_regrets,
        ),
        "confirmatory": False,
        "reason": (
            "Correlation is descriptive because pooled values are family-clustered "
            "and the noncommuting subset has only one aggregate value per seed."
        ),
    }

    frontier = _build_frontier(
        rows,
        dimension=dimension,
        low_rank=low_rank,
    )
    metrics_path = run_dir / "operator_family_metrics.jsonl"
    frontier_path = run_dir / "control_frontier.jsonl"
    write_jsonl_strict(metrics_path, rows)
    write_jsonl_strict(frontier_path, frontier)

    expected_rows = len(seeds) * len(OperatorFamily)
    main_execution_complete = bool(
        not args.dry_run
        and len(seeds) == 8
        and len(rows) == 24
        and expected_rows == 24
        and _all_finite(rows)
    )
    contrast_gate_passed = all(
        bool(contrast["passed"]) for contrast in contrasts.values()
    )
    absolute_gate_passed = all(
        bool(gate["passed"]) for gate in absolute_gates.values()
    )
    supported = bool(
        main_execution_complete
        and contrast_gate_passed
        and absolute_gate_passed
        and empirical_gate["passed"]
    )
    report = {
        "status": "PASS",
        "execution": {
            "dry_run": bool(args.dry_run),
            "expected_row_count": expected_rows,
            "row_count": len(rows),
            "unique_seeds": seeds,
            "family_count": len(OperatorFamily),
            "training_operators_per_family": train_count,
            "heldout_operators_per_family": test_count,
            "probes_per_heldout_operator": probe_count,
            "optimization_steps": steps,
            "optimization_restarts": restarts,
            "cpu_threads": torch.get_num_threads(),
            "wall_seconds": time.perf_counter() - experiment_started,
            "main_execution_complete": main_execution_complete,
        },
        "dependency_lineage": dependencies,
        "metric_definitions": {
            "commutator_norm": (
                "mean over unordered operator pairs of the Frobenius norm "
                "||P_i P_j - P_j P_i||_F"
            ),
            "rjd_objective": (
                "sum over heldout operators of "
                "||offdiag(Q^T P_tau Q)||_F^2"
            ),
            "regret": (
                "rjd_objective divided by heldout_operator_count * dimension^2; "
                "equivalently mean squared operator-entry reconstruction error"
            ),
            "empirical_error": (
                "heldout isotropic probe application MSE; each input component "
                "has variance 1/d, so its expectation equals regret"
            ),
            "demand_descriptor": (
                "the target projector P_tau itself; every control class receives "
                "the same descriptor"
            ),
        },
        "contrasts": contrasts,
        "absolute_gates": absolute_gates,
        "empirical_regret_prediction": empirical_gate,
        "geometry_prediction": geometry_prediction,
        "control_cost_frontier": frontier,
        "artifacts": {
            "operator_family_metrics": {
                "path": metrics_path.name,
                "rows": len(rows),
                "sha256": sha256_file(metrics_path),
            },
            "control_frontier": {
                "path": frontier_path.name,
                "rows": len(frontier),
                "sha256": sha256_file(frontier_path),
            },
        },
        "claim_gate": {
            "evaluated": main_execution_complete,
            "supported": supported,
            "four_preregistered_contrasts_passed": contrast_gate_passed,
            "absolute_geometry_gates_passed": absolute_gate_passed,
            "empirical_regret_prediction_passed": bool(empirical_gate["passed"]),
            "allowed_claim": (
                "Within the registered synthetic projector families, fixed or "
                "shared-basis diagonal control was sufficient for jointly "
                "diagonalizable demands, whereas noncommuting demands retained a "
                "shared-diagonal residual recovered by a richer oracle."
                if supported
                else None
            ),
            "forbidden_claims": [
                "Channel-wise control is universally superior.",
                "The low-rank oracle is a parameter-matched learned controller.",
                "This theory experiment establishes a language-model or official-backend claim.",
            ],
        },
        "evidence_scope": {
            "evidence_tier": "CONTROLLED_REFERENCE",
            "controlled_geometry_claim_eligible": supported,
            "official_backend_claim_eligible": False,
            "language_model_claim_eligible": False,
            "architecture_transfer_claim_eligible": False,
        },
    }
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=main_execution_complete,
        full_eligible=main_execution_complete,
    )
    print(
        f"[{EXPERIMENT_ID}] PASS: {run_dir} "
        f"(H3={'SUPPORTED' if supported else 'NOT_EVALUATED_OR_NOT_SUPPORTED'})"
    )


if __name__ == "__main__":
    main()
