from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def exact_sign_flip(values: Iterable[float], *, alternative: str = "greater") -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional collection")
    observed = float(array.mean())
    count = 0
    extreme = 0
    for mask in range(1 << array.size):
        signs = np.ones(array.size, dtype=np.float64)
        for index in range(array.size):
            if mask & (1 << index):
                signs[index] = -1.0
        statistic = float((array * signs).mean())
        count += 1
        if alternative == "greater" and statistic >= observed - 1e-15:
            extreme += 1
        elif alternative == "less" and statistic <= observed + 1e-15:
            extreme += 1
        elif alternative == "two-sided" and abs(statistic) >= abs(observed) - 1e-15:
            extreme += 1
    return extreme / count


def monotonic_fraction(values: list[float], *, decreasing: bool = True) -> float:
    if len(values) < 2:
        return 1.0
    passed = 0
    for left, right in zip(values[:-1], values[1:], strict=True):
        if (decreasing and right <= left + 1e-12) or (
            not decreasing and right >= left - 1e-12
        ):
            passed += 1
    return passed / (len(values) - 1)


def normalized_recovery(*, baseline: float, model: float, oracle: float) -> float | None:
    denominator = baseline - oracle
    if abs(denominator) <= 1e-12:
        return None
    return (baseline - model) / denominator
