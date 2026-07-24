from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 13,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Paired bootstrap inputs must have equal length")
    if not left:
        raise ValueError("Paired bootstrap inputs are empty")
    rng = random.Random(seed)
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    diffs = np.empty(samples, dtype=float)
    n = len(left_arr)
    for i in range(samples):
        indices = np.asarray([rng.randrange(n) for _ in range(n)])
        diffs[i] = statistic(left_arr[indices]) - statistic(right_arr[indices])
    return {
        "difference": float(statistic(left_arr) - statistic(right_arr)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_left_le_right": float(np.mean(diffs <= 0.0)),
    }
