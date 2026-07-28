from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Interval:
    estimate: float
    low: float
    high: float


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have identical shapes.")
    if a.ndim != 1:
        raise ValueError("Paired bootstrap expects one-dimensional arrays.")
    rng = np.random.default_rng(seed)
    differences = a - b
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample_idx = rng.integers(0, len(differences), size=len(differences))
        estimates[index] = differences[sample_idx].mean()
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(differences.mean()),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
    )


def hierarchical_seed_episode_bootstrap(
    seed_episode_effects: dict[int, np.ndarray],
    *,
    samples: int = 5000,
    seed: int = 0,
) -> Interval:
    if not seed_episode_effects:
        raise ValueError("No seed-level effects provided.")
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(seed_episode_effects))
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        per_seed: list[float] = []
        for sampled_seed in sampled_seeds:
            values = seed_episode_effects[int(sampled_seed)]
            sampled_values = rng.choice(values, size=len(values), replace=True)
            per_seed.append(float(sampled_values.mean()))
        estimates[index] = float(np.mean(per_seed))
    point = float(np.mean([values.mean() for values in seed_episode_effects.values()]))
    return Interval(
        estimate=point,
        low=float(np.quantile(estimates, 0.025)),
        high=float(np.quantile(estimates, 0.975)),
    )


def equivalence_within(interval: Interval, margin: float) -> bool:
    return interval.low >= -margin and interval.high <= margin



def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be paired one-dimensional arrays.")
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(centered_x, y - y.mean()) / denominator)


def hierarchical_seed_episode_slope_bootstrap(
    seed_episode_xy: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    samples: int = 5000,
    seed: int = 0,
) -> Interval:
    """Cluster bootstrap an OLS slope with training seed as the upper-level unit."""
    if not seed_episode_xy:
        raise ValueError("No seed-level pairs provided.")
    seeds = np.array(sorted(seed_episode_xy))
    for x, y in seed_episode_xy.values():
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("Each seed must contain paired one-dimensional arrays.")
    point_x = np.concatenate([seed_episode_xy[int(s)][0] for s in seeds])
    point_y = np.concatenate([seed_episode_xy[int(s)][1] for s in seeds])
    point = _ols_slope(point_x, point_y)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for sampled_seed in sampled_seeds:
            x, y = seed_episode_xy[int(sampled_seed)]
            sampled_indices = rng.integers(0, len(x), size=len(x))
            x_parts.append(x[sampled_indices])
            y_parts.append(y[sampled_indices])
        estimates[index] = _ols_slope(np.concatenate(x_parts), np.concatenate(y_parts))
    finite = estimates[np.isfinite(estimates)]
    if len(finite) == 0:
        return Interval(estimate=point, low=float("nan"), high=float("nan"))
    return Interval(
        estimate=point,
        low=float(np.quantile(finite, 0.025)),
        high=float(np.quantile(finite, 0.975)),
    )

def permutation_interaction_test(
    values: np.ndarray,
    controller: np.ndarray,
    orientation: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    if not (values.shape == controller.shape == orientation.shape):
        raise ValueError("All arrays must share the same shape.")
    rng = np.random.default_rng(seed)

    def interaction(v: np.ndarray, c: np.ndarray, o: np.ndarray) -> float:
        cells: dict[tuple[int, int], float] = {}
        for c_value in np.unique(c):
            for o_value in np.unique(o):
                mask = (c == c_value) & (o == o_value)
                cells[(int(c_value), int(o_value))] = float(v[mask].mean())
        if len(cells) != 4:
            raise ValueError("Interaction test currently expects a 2x2 design.")
        return (
            cells[(1, 1)]
            - cells[(1, 0)]
            - cells[(0, 1)]
            + cells[(0, 0)]
        )

    observed = interaction(values, controller, orientation)
    null = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        shuffled = rng.permutation(controller)
        null[index] = interaction(values, shuffled, orientation)
    p_value = float((np.abs(null) >= abs(observed)).mean())
    return {"interaction": float(observed), "p_value": p_value}
