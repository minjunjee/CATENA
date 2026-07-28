from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RegretBin:
    label: str
    lower: float
    upper: float
    include_upper: bool = False

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Regret-bin label must not be empty.")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Regret-bin bounds must be finite.")
        if self.lower < 0.0 or self.upper <= self.lower:
            raise ValueError("Regret-bin bounds must satisfy 0 <= lower < upper.")

    def contains(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("Analytic regret must be finite.")
        if self.include_upper:
            return self.lower <= value <= self.upper
        return self.lower <= value < self.upper


@dataclass(frozen=True, slots=True)
class AnalyticCandidate:
    candidate_id: str
    construction_sha256: str
    analytic_regret: float
    alpha: float
    generation_seed: int

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.construction_sha256:
            raise ValueError("Candidate identity and construction hash are required.")
        if not math.isfinite(self.analytic_regret) or self.analytic_regret < 0.0:
            raise ValueError("analytic_regret must be non-negative and finite.")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be non-negative and finite.")
        if isinstance(self.generation_seed, bool) or not isinstance(
            self.generation_seed,
            int,
        ):
            raise ValueError("generation_seed must be an integer.")


@dataclass(frozen=True, slots=True)
class CalibrationThresholds:
    minimum_r2: float
    minimum_slope: float
    maximum_slope: float
    maximum_absolute_intercept: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_r2,
            self.minimum_slope,
            self.maximum_slope,
            self.maximum_absolute_intercept,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Calibration thresholds must be finite.")
        if not 0.0 < self.minimum_r2 <= 1.0:
            raise ValueError("minimum_r2 must lie in (0, 1].")
        if self.minimum_slope >= self.maximum_slope:
            raise ValueError("Calibration slope bounds must be ordered.")
        if self.maximum_absolute_intercept <= 0.0:
            raise ValueError("maximum_absolute_intercept must be positive.")


def validate_regret_bins(bins: Sequence[RegretBin]) -> tuple[RegretBin, ...]:
    if not bins:
        raise ValueError("At least one regret bin is required.")
    ordered = tuple(bins)
    if len({item.label for item in ordered}) != len(ordered):
        raise ValueError("Regret-bin labels must be unique.")
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.lower < previous.upper:
            raise ValueError("Regret bins must not overlap.")
    if any(item.include_upper for item in ordered[:-1]):
        raise ValueError("Only the final regret bin may include its upper bound.")
    return ordered


def assign_regret_bin(value: float, bins: Sequence[RegretBin]) -> str | None:
    ordered = validate_regret_bins(bins)
    matches = [item.label for item in ordered if item.contains(value)]
    if len(matches) > 1:
        raise AssertionError("Validated non-overlapping bins produced multiple matches.")
    return matches[0] if matches else None


def select_first_valid_candidates(
    candidates: Iterable[AnalyticCandidate],
    bins: Sequence[RegretBin],
    *,
    families_per_bin: int,
) -> dict[str, list[AnalyticCandidate]]:
    """Select the first analytic-only candidates that fill every registered bin."""

    if (
        isinstance(families_per_bin, bool)
        or not isinstance(families_per_bin, int)
        or families_per_bin <= 0
    ):
        raise ValueError("families_per_bin must be a positive integer.")
    ordered = validate_regret_bins(bins)
    selected: dict[str, list[AnalyticCandidate]] = {
        item.label: [] for item in ordered
    }
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate.candidate_id}.")
        seen_ids.add(candidate.candidate_id)
        label = assign_regret_bin(candidate.analytic_regret, ordered)
        if label is None or len(selected[label]) >= families_per_bin:
            continue
        if candidate.generation_seed in seen_seeds:
            continue
        selected[label].append(candidate)
        seen_seeds.add(candidate.generation_seed)
        if all(len(values) == families_per_bin for values in selected.values()):
            break
    return selected


def validate_selected_design(
    selected: Mapping[str, Sequence[AnalyticCandidate]],
    bins: Sequence[RegretBin],
    *,
    families_per_bin: int,
    minimum_nonzero_range: float,
) -> dict[str, Any]:
    ordered = validate_regret_bins(bins)
    if not math.isfinite(minimum_nonzero_range) or minimum_nonzero_range <= 0.0:
        raise ValueError("minimum_nonzero_range must be positive and finite.")
    expected_labels = {item.label for item in ordered}
    if set(selected) != expected_labels:
        raise ValueError("Selected design labels do not match the registered bins.")
    flattened = [candidate for item in ordered for candidate in selected[item.label]]
    per_key_membership = {
        item.label: all(
            item.contains(candidate.analytic_regret)
            for candidate in selected[item.label]
        )
        for item in ordered
    }
    counts = Counter(
        assign_regret_bin(candidate.analytic_regret, ordered)
        for candidate in flattened
    )
    unique_candidate_ids = len({candidate.candidate_id for candidate in flattened})
    unique_seeds = len({candidate.generation_seed for candidate in flattened})
    regrets = np.asarray(
        [candidate.analytic_regret for candidate in flattened],
        dtype=np.float64,
    )
    nonzero = regrets[regrets > 0.0]
    observed_range = (
        float(nonzero.max() - nonzero.min()) if len(nonzero) >= 2 else 0.0
    )
    exact_counts = all(
        counts[item.label] == families_per_bin for item in ordered
    )
    expected_total = len(ordered) * families_per_bin
    passed = bool(
        len(flattened) == expected_total
        and all(per_key_membership.values())
        and exact_counts
        and unique_candidate_ids == expected_total
        and unique_seeds == expected_total
        and len(nonzero) == expected_total
        and observed_range >= minimum_nonzero_range
    )
    return {
        "passed": passed,
        "bin_counts": {
            item.label: int(counts[item.label]) for item in ordered
        },
        "per_key_bin_membership": per_key_membership,
        "all_candidates_in_mapping_key_bin": all(per_key_membership.values()),
        "expected_families_per_bin": families_per_bin,
        "expected_total_families": expected_total,
        "actual_total_families": len(flattened),
        "unique_candidate_ids": unique_candidate_ids,
        "unique_generation_seeds": unique_seeds,
        "exact_zero_values_excluded": bool(len(nonzero) == len(flattened)),
        "minimum_nonzero_regret": (
            float(nonzero.min()) if len(nonzero) else None
        ),
        "maximum_nonzero_regret": (
            float(nonzero.max()) if len(nonzero) else None
        ),
        "observed_nonzero_range": observed_range,
        "minimum_required_nonzero_range": minimum_nonzero_range,
    }


def fit_jd_application_calibration(
    analytic_regret: np.ndarray,
    empirical_error: np.ndarray,
    *,
    thresholds: CalibrationThresholds,
) -> dict[str, Any]:
    x = np.asarray(analytic_regret, dtype=np.float64)
    y = np.asarray(empirical_error, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 3:
        raise ValueError("Calibration inputs must be aligned vectors of length >= 3.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Calibration inputs must contain only finite values.")
    if np.any(x <= 0.0):
        raise ValueError("Exact-zero analytic values must be excluded from calibration.")
    if float(np.ptp(x)) <= 0.0:
        raise ValueError("Analytic predictor has no range.")

    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ np.asarray([intercept, slope])
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("Empirical application error has no variance.")
    r2 = float(1.0 - np.sum((y - prediction) ** 2) / denominator)
    pearson = float(np.corrcoef(x, y)[0, 1])
    r2_gate_passed = bool(r2 >= thresholds.minimum_r2)
    absolute_calibration_passed = bool(
        thresholds.minimum_slope <= slope <= thresholds.maximum_slope
        and abs(intercept) <= thresholds.maximum_absolute_intercept
    )
    return {
        "n": int(len(x)),
        "r2": r2,
        "pearson": pearson,
        "slope": float(slope),
        "intercept": float(intercept),
        "analytic_minimum": float(x.min()),
        "analytic_maximum": float(x.max()),
        "analytic_range": float(np.ptp(x)),
        "mean_absolute_error": float(np.mean(np.abs(y - x))),
        "maximum_absolute_error": float(np.max(np.abs(y - x))),
        "mean_relative_absolute_error": float(np.mean(np.abs(y - x) / x)),
        "maximum_relative_absolute_error": float(np.max(np.abs(y - x) / x)),
        "thresholds": {
            "minimum_r2": thresholds.minimum_r2,
            "minimum_slope": thresholds.minimum_slope,
            "maximum_slope": thresholds.maximum_slope,
            "maximum_absolute_intercept": thresholds.maximum_absolute_intercept,
        },
        "r2_gate_passed": r2_gate_passed,
        "absolute_calibration_passed": absolute_calibration_passed,
        "passed": bool(r2_gate_passed and absolute_calibration_passed),
    }
