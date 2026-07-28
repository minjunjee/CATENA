from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from catena.eval.seed_inference import calibration_slope, r2_score


@dataclass(slots=True, frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(slots=True, frozen=True)
class OperationFixedEffectModel:
    """A small OLS model with operation intercepts and an optional predictor."""

    operation_order: tuple[str, ...]
    coefficients: np.ndarray
    includes_predictor: bool

    @property
    def slope(self) -> float | None:
        if not self.includes_predictor:
            return None
        return float(self.coefficients[1])


@dataclass(slots=True, frozen=True)
class FixedEffectOOSReport:
    """Out-of-sample diagnostics conditional on an operation-only baseline."""

    operation_only_r2: float
    full_r2: float
    conditional_r2: float
    train_slope: float
    test_slope: float
    calibration_slope: float
    operation_only_sse: float
    full_sse: float


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _operation_vector(operations: np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(operations)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(
            "operations must be a one-dimensional vector aligned with the observations"
        )
    return array.astype(str)


def _operation_order(
    operations: np.ndarray,
    operation_order: list[str] | tuple[str, ...],
    *,
    require_all: bool,
) -> tuple[str, ...]:
    order = tuple(str(value) for value in operation_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("operation_order must contain unique operation names")
    observed = set(operations.tolist())
    unknown = observed - set(order)
    if unknown:
        raise ValueError(
            f"operations contains values absent from operation_order: {sorted(unknown)}"
        )
    if require_all:
        missing = set(order) - observed
        if missing:
            raise ValueError(
                f"fit data is missing configured operations: {sorted(missing)}"
            )
    return order


def _operation_columns(
    operations: np.ndarray, operation_order: tuple[str, ...]
) -> list[np.ndarray]:
    columns = [np.ones(len(operations), dtype=np.float64)]
    columns.extend(
        (operations == operation).astype(np.float64)
        for operation in operation_order[1:]
    )
    return columns


def _fit_design(
    design: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
) -> np.ndarray:
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"{model_name} design is rank deficient")
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise FloatingPointError(f"{model_name} fit produced non-finite coefficients")
    return np.asarray(coefficients, dtype=np.float64)


def fit_operation_only_fixed_effect(
    y: np.ndarray,
    operations: np.ndarray,
    *,
    operation_order: list[str] | tuple[str, ...],
) -> OperationFixedEffectModel:
    """Fit ``y ~ operation`` with the first configured operation as baseline."""

    y = _finite_vector(y, "y")
    operations = _operation_vector(operations, len(y))
    order = _operation_order(operations, operation_order, require_all=True)
    design = np.column_stack(_operation_columns(operations, order))
    coefficients = _fit_design(design, y, model_name="operation-only")
    return OperationFixedEffectModel(order, coefficients, False)


def fit_full_fixed_effect(
    x: np.ndarray,
    y: np.ndarray,
    operations: np.ndarray,
    *,
    operation_order: list[str] | tuple[str, ...],
) -> OperationFixedEffectModel:
    """Fit ``y ~ x + operation`` with one common within-operation slope."""

    x = _finite_vector(x, "x")
    y = _finite_vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    operations = _operation_vector(operations, len(y))
    order = _operation_order(operations, operation_order, require_all=True)
    operation_columns = _operation_columns(operations, order)
    design = np.column_stack([operation_columns[0], x, *operation_columns[1:]])
    coefficients = _fit_design(design, y, model_name="full fixed-effect")
    return OperationFixedEffectModel(order, coefficients, True)


def predict_operation_fixed_effect(
    model: OperationFixedEffectModel,
    operations: np.ndarray,
    *,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Predict from a fitted operation-only or full fixed-effect model."""

    if x is None:
        if model.includes_predictor:
            raise ValueError("x is required for a full fixed-effect model")
        operations_array = np.asarray(operations)
        expected_length = len(operations_array)
        if expected_length == 0:
            raise ValueError("operations must not be empty")
    else:
        x = _finite_vector(x, "x")
        expected_length = len(x)
    operations_array = _operation_vector(operations, expected_length)
    _operation_order(operations_array, model.operation_order, require_all=False)
    operation_columns = _operation_columns(operations_array, model.operation_order)
    if model.includes_predictor:
        if x is None:
            raise AssertionError("Validated full model unexpectedly lacks x.")
        columns: list[np.ndarray] = [
            operation_columns[0],
            x,
            *operation_columns[1:],
        ]
    else:
        columns = operation_columns
    design = np.column_stack(columns)
    if design.shape[1] != len(model.coefficients):
        raise ValueError("model coefficient count does not match its design")
    prediction = design @ model.coefficients
    if not np.isfinite(prediction).all():
        raise FloatingPointError("fixed-effect prediction contains a non-finite value")
    return np.asarray(prediction, dtype=np.float64)


def conditional_oos_r2(
    observed: np.ndarray,
    full_prediction: np.ndarray,
    operation_only_prediction: np.ndarray,
    *,
    minimum_baseline_sse: float = 1e-12,
) -> float:
    """Return predictor-specific OOS R2 relative to an operation-only model.

    This is ``1 - SSE(full) / SSE(operation-only)``. It excludes predictive
    power due solely to operation identity.
    """

    observed = _finite_vector(observed, "observed")
    full_prediction = _finite_vector(full_prediction, "full_prediction")
    operation_only_prediction = _finite_vector(
        operation_only_prediction, "operation_only_prediction"
    )
    if not (
        observed.shape == full_prediction.shape == operation_only_prediction.shape
    ):
        raise ValueError("observed and both predictions must have identical shapes")
    if not np.isfinite(minimum_baseline_sse) or minimum_baseline_sse <= 0.0:
        raise ValueError("minimum_baseline_sse must be positive and finite")
    baseline_sse = float(np.sum((observed - operation_only_prediction) ** 2))
    if baseline_sse <= minimum_baseline_sse:
        raise ValueError(
            "operation-only baseline has no residual OOS error for conditional R2"
        )
    full_sse = float(np.sum((observed - full_prediction) ** 2))
    return float(1.0 - full_sse / baseline_sse)


def evaluate_fixed_effect_oos(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_operations: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_operations: np.ndarray,
    *,
    operation_order: list[str] | tuple[str, ...],
) -> FixedEffectOOSReport:
    """Fit on train geometry and evaluate predictor-specific test performance."""

    operation_only = fit_operation_only_fixed_effect(
        train_y, train_operations, operation_order=operation_order
    )
    full = fit_full_fixed_effect(
        train_x, train_y, train_operations, operation_order=operation_order
    )
    operation_prediction = predict_operation_fixed_effect(
        operation_only, test_operations
    )
    full_prediction = predict_operation_fixed_effect(
        full, test_operations, x=test_x
    )
    test_y = _finite_vector(test_y, "test_y")
    test_x = _finite_vector(test_x, "test_x")
    if test_x.shape != test_y.shape:
        raise ValueError("test_x and test_y must have identical shapes")
    test_fit = fit_full_fixed_effect(
        test_x, test_y, test_operations, operation_order=operation_order
    )
    if full.slope is None or test_fit.slope is None:
        raise AssertionError("Full fixed-effect models must expose a slope.")
    operation_sse = float(np.sum((test_y - operation_prediction) ** 2))
    full_sse = float(np.sum((test_y - full_prediction) ** 2))
    return FixedEffectOOSReport(
        operation_only_r2=r2_score(test_y, operation_prediction),
        full_r2=r2_score(test_y, full_prediction),
        conditional_r2=conditional_oos_r2(
            test_y, full_prediction, operation_prediction
        ),
        train_slope=float(full.slope),
        test_slope=float(test_fit.slope),
        calibration_slope=calibration_slope(test_y, full_prediction),
        operation_only_sse=operation_sse,
        full_sse=full_sse,
    )


def fixed_seed_operation_stratified_bootstrap(
    seed_operations: Mapping[int, np.ndarray],
    statistic: Callable[[Mapping[int, np.ndarray]], float],
    *,
    samples: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    """Bootstrap episodes within operation strata while keeping every seed.

    ``statistic`` receives a mapping from each original seed to episode indices.
    Seeds are never resampled or omitted. Each replicate samples, with
    replacement, the original number of episodes in every seed/operation
    stratum. This keeps fixed-checkpoint episode uncertainty separate from
    training-seed inference.
    """

    if not seed_operations:
        raise ValueError("seed_operations must not be empty")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    ordered_seeds = tuple(sorted(seed_operations))
    original_indices: dict[int, np.ndarray] = {}
    strata: dict[int, tuple[np.ndarray, ...]] = {}
    for seed_value in ordered_seeds:
        operations = np.asarray(seed_operations[seed_value])
        if operations.ndim != 1 or len(operations) == 0:
            raise ValueError(
                f"seed_operations[{seed_value}] must be a nonempty vector"
            )
        operations = operations.astype(str)
        labels = tuple(sorted(set(operations.tolist())))
        original_indices[seed_value] = np.arange(len(operations), dtype=np.int64)
        strata[seed_value] = tuple(
            np.flatnonzero(operations == label).astype(np.int64) for label in labels
        )

    point = float(statistic(original_indices))
    if not np.isfinite(point):
        raise ValueError("statistic returned a non-finite point estimate")

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_indices: dict[int, np.ndarray] = {}
        for seed_value in ordered_seeds:
            sampled_indices[seed_value] = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in strata[seed_value]
                ]
            )
        estimate = float(statistic(sampled_indices))
        if not np.isfinite(estimate):
            raise ValueError(
                f"statistic returned a non-finite bootstrap estimate at sample "
                f"{sample_index}"
            )
        estimates[sample_index] = estimate

    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=point,
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
    )
