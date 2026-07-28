from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegressionReport:
    slope: float
    r2: float


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _group_vector(groups: np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(groups)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError("groups must be a one-dimensional vector aligned with the values")
    return array.astype(str)


def _validated_group_order(groups: np.ndarray, group_order: list[str]) -> list[str]:
    if not group_order or len(set(group_order)) != len(group_order):
        raise ValueError("group_order must contain unique group names")
    observed = set(groups.tolist())
    unknown = observed - set(group_order)
    if unknown:
        raise ValueError(f"groups contains values absent from group_order: {sorted(unknown)}")
    used = [group for group in group_order if group in observed]
    if not used:
        raise ValueError("groups does not contain any configured group")
    return used


def exact_sign_flip_test(values: np.ndarray, alternative: str = "greater") -> float:
    values = _finite_vector(values, "values")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(
            "alternative must be one of 'greater', 'less', or 'two-sided'"
        )
    observed = float(values.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(float(np.mean(values * np.asarray(signs))))
    null_array = np.asarray(null)
    if alternative == "greater":
        return float(np.mean(null_array >= observed - 1e-15))
    if alternative == "less":
        return float(np.mean(null_array <= observed + 1e-15))
    return float(np.mean(np.abs(null_array) >= abs(observed) - 1e-15))


def fit_fixed_effect_regression(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    group_order: list[str],
) -> tuple[np.ndarray, RegressionReport, list[str]]:
    x = _finite_vector(x, "x")
    y = _finite_vector(y, "y")
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    groups = _group_vector(groups, len(x))
    used = _validated_group_order(groups, group_order)
    columns = [np.ones_like(x), x]
    for group in used[1:]:
        columns.append((groups == group).astype(np.float64))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("fixed-effect regression produced non-finite coefficients")
    prediction = design @ coefficients
    return (
        coefficients,
        RegressionReport(float(coefficients[1]), r2_score(y, prediction)),
        used,
    )


def predict_fixed_effect(
    coefficients: np.ndarray, x: np.ndarray, groups: np.ndarray, group_order: list[str]
) -> np.ndarray:
    x = _finite_vector(x, "x")
    groups = _group_vector(groups, len(x))
    _validated_group_order(groups, group_order)
    coefficients = _finite_vector(coefficients, "coefficients")
    expected_coefficients = 2 + max(len(group_order) - 1, 0)
    if len(coefficients) != expected_coefficients:
        raise ValueError(
            f"expected {expected_coefficients} coefficients, got {len(coefficients)}"
        )
    columns = [np.ones_like(x), x]
    for group in group_order[1:]:
        columns.append((groups == group).astype(np.float64))
    prediction = np.column_stack(columns) @ coefficients
    if not np.isfinite(prediction).all():
        raise FloatingPointError("fixed-effect prediction contains a non-finite value")
    return np.asarray(prediction, dtype=np.float64)


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Slope from observed ~ intercept + slope * predicted.

    A value near one indicates that an analytic lower-bound predictor has the
    right scale, whereas R2 alone only measures ranking/predictive fit.
    """

    observed = _finite_vector(observed, "observed")
    predicted = _finite_vector(predicted, "predicted")
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have identical shapes")
    centered = predicted - predicted.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(centered, observed - observed.mean()) / denominator)


def r2_score(y: np.ndarray, prediction: np.ndarray) -> float:
    y = _finite_vector(y, "y")
    prediction = _finite_vector(prediction, "prediction")
    if y.shape != prediction.shape:
        raise ValueError("y and prediction must have identical shapes")
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator)
