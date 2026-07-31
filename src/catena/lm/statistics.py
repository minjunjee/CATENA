from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def exact_sign_flip_pvalue(
    effects: Sequence[float],
    *,
    null: float = 0.0,
    alternative: str = "greater",
) -> float:
    """Exact paired sign-flip randomization test on the mean.

    Suitable for the registered CATENA seed counts. Zero-centered values are
    retained and all 2^n sign assignments are enumerated for n <= 20.
    """

    values = np.asarray(effects, dtype=np.float64) - float(null)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("effects must be a non-empty one-dimensional sequence")
    if values.size > 20:
        raise ValueError("Exact enumeration is intentionally limited to n <= 20")
    observed = float(values.mean())
    distribution = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        distribution.append(float((values * np.asarray(signs)).mean()))
    tolerance = 1.0e-15
    if alternative == "greater":
        count = sum(value >= observed - tolerance for value in distribution)
    elif alternative == "less":
        count = sum(value <= observed + tolerance for value in distribution)
    elif alternative == "two-sided":
        count = sum(abs(value) >= abs(observed) - tolerance for value in distribution)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return count / len(distribution)


def bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 20260731,
) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be non-empty and one-dimensional")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (alpha, 1.0 - alpha))
    return Interval(estimate=float(array.mean()), lower=float(lower), upper=float(upper))


def did_by_seed(
    improvements: Mapping[int, Mapping[str, float]],
) -> dict[int, float]:
    required = {"PRESERVE", "ADD", "INVALIDATE", "SUPERSEDE"}
    output: dict[int, float] = {}
    for seed, row in improvements.items():
        if set(row) < required:
            missing = required - set(row)
            raise ValueError(f"Seed {seed} is missing operations: {sorted(missing)}")
        asymmetric = 0.5 * (float(row["ADD"]) + float(row["INVALIDATE"]))
        symmetric = 0.5 * (float(row["PRESERVE"]) + float(row["SUPERSEDE"]))
        output[int(seed)] = asymmetric - symmetric
    return output


def equivalence_pass(interval: Interval, margin: float) -> bool:
    if margin <= 0:
        raise ValueError("margin must be positive")
    return interval.lower >= -margin and interval.upper <= margin


def noninferiority_pass(interval: Interval, degradation_margin: float) -> bool:
    """Pass when higher-is-better effect is not worse than -margin."""

    if degradation_margin < 0:
        raise ValueError("degradation_margin must be non-negative")
    return interval.lower >= -degradation_margin


def relative_degradation(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("denominator must be finite and positive")
    return numerator / denominator - 1.0


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must lie in [0,1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
