from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product


def rank_cell_seed_provenance(seed: int, intrinsic_rank: int) -> dict[str, int]:
    """Deterministic E10 streams, paired across learned-rank variants."""
    seed = int(seed)
    intrinsic_rank = int(intrinsic_rank)
    return {
        "family_seed": 10_000 * seed + intrinsic_rank,
        "train_descriptor_seed": 20_000 * seed + intrinsic_rank,
        "test_descriptor_seed": 30_000 * seed + intrinsic_rank,
        "model_seed": 40_000 * seed + 100 * intrinsic_rank,
        "optimizer_sampling_seed": 50_000 * seed + intrinsic_rank,
    }


def aggregate_intrinsic_rank_effects_by_seed(
    cell_effects: Mapping[tuple[int, int], float],
    *,
    seeds: Sequence[int],
    intrinsic_ranks: Sequence[int],
) -> list[dict[str, int | float]]:
    """Equal-weight intrinsic-rank cell effects within each independent seed."""
    ordered_seeds = tuple(int(seed) for seed in seeds)
    ordered_ranks = tuple(int(rank) for rank in intrinsic_ranks)
    if not ordered_seeds or not ordered_ranks:
        raise ValueError("seeds and intrinsic_ranks must both be non-empty")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be unique")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError("intrinsic_ranks must be unique")

    expected = set(product(ordered_seeds, ordered_ranks))
    observed = set(cell_effects)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "cell_effects must contain exactly one value per seed × intrinsic-rank "
            f"cell; missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, int | float]] = []
    for seed in ordered_seeds:
        effects = [float(cell_effects[(seed, rank)]) for rank in ordered_ranks]
        if not all(math.isfinite(effect) for effect in effects):
            raise ValueError(f"cell_effects for seed={seed} contain a non-finite value")
        rows.append(
            {
                "seed": seed,
                "intrinsic_rank_cell_count": len(effects),
                "mean_low_vs_high_rank_gain": sum(effects) / len(effects),
                "minimum_cell_gain": min(effects),
                "maximum_cell_gain": max(effects),
            }
        )
    return rows


def evaluate_minimum_rank_tracking(
    minimum_qualifying_ranks: Mapping[tuple[int, int], int | None],
    *,
    seeds: Sequence[int],
    intrinsic_ranks: Sequence[int],
    max_rank_factor: float,
    max_available_rank: int,
) -> tuple[list[dict[str, int | bool | None]], list[dict[str, int | bool]]]:
    """Evaluate lower/upper rank tracking and within-seed monotonicity."""
    ordered_seeds = tuple(int(seed) for seed in seeds)
    ordered_ranks = tuple(sorted(int(rank) for rank in intrinsic_ranks))
    if not ordered_seeds or not ordered_ranks:
        raise ValueError("seeds and intrinsic_ranks must both be non-empty")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("seeds must be unique")
    if len(set(ordered_ranks)) != len(ordered_ranks):
        raise ValueError("intrinsic_ranks must be unique")
    expected = set(product(ordered_seeds, ordered_ranks))
    observed = set(minimum_qualifying_ranks)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "minimum_qualifying_ranks must contain every registered cell exactly once; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if max_rank_factor < 1.0:
        raise ValueError("max_rank_factor must be at least 1")
    if max_available_rank < max(ordered_ranks):
        raise ValueError("max_available_rank must cover the largest intrinsic rank")

    cell_rows: list[dict[str, int | bool | None]] = []
    seed_rows: list[dict[str, int | bool]] = []
    for seed in ordered_seeds:
        ordered_minima: list[int] = []
        complete = True
        for intrinsic_rank in ordered_ranks:
            minimum = minimum_qualifying_ranks[(seed, intrinsic_rank)]
            upper_bound = min(
                int(math.floor(max_rank_factor * intrinsic_rank)),
                int(max_available_rank),
            )
            matched = minimum is not None and intrinsic_rank <= int(minimum) <= upper_bound
            cell_rows.append(
                {
                    "seed": seed,
                    "intrinsic_rank": intrinsic_rank,
                    "minimum_qualifying_rank": minimum,
                    "registered_lower_bound": intrinsic_rank,
                    "registered_upper_bound": upper_bound,
                    "rank_tracking_matched": matched,
                }
            )
            if minimum is None:
                complete = False
            else:
                ordered_minima.append(int(minimum))

        nondecreasing = complete and all(
            right >= left
            for left, right in zip(ordered_minima[:-1], ordered_minima[1:], strict=True)
        )
        seed_rows.append(
            {
                "seed": seed,
                "intrinsic_rank_cell_count": len(ordered_ranks),
                "all_cells_have_qualifying_rank": complete,
                "minimum_qualifying_rank_nondecreasing": nondecreasing,
            }
        )
    return cell_rows, seed_rows


def minimum_sufficient_rank_from_exact_target_recovery(
    recoveries: Mapping[int, float],
    *,
    threshold: float,
) -> int | None:
    """Return the smallest learned rank meeting the exact-target recovery gate."""
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not recoveries:
        raise ValueError("recoveries must be non-empty")
    normalized = {int(rank): float(value) for rank, value in recoveries.items()}
    if any(rank <= 0 for rank in normalized):
        raise ValueError("learned ranks must be positive")
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError("exact-target recoveries must be finite")
    qualifying = sorted(rank for rank, recovery in normalized.items() if recovery >= threshold)
    return qualifying[0] if qualifying else None


def oracle_normalized_rank_recovery(
    *,
    baseline_error: float,
    model_error: float,
    oracle_error: float,
    tolerance: float = 1e-8,
) -> float:
    """Validate reachable-floor metrics and return oracle-normalized recovery."""
    values = (float(baseline_error), float(model_error), float(oracle_error))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rank errors must be finite")
    baseline, model, oracle = values
    if min(values) < -tolerance:
        raise ValueError("rank errors must be non-negative")
    if oracle > baseline + tolerance:
        raise ValueError("best-rank oracle error cannot exceed the zero-predictor baseline")
    if oracle > model + tolerance:
        raise ValueError("rank-constrained model error cannot beat its best-rank oracle")
    headroom = baseline - oracle
    if headroom <= tolerance:
        raise ValueError("oracle-normalized recovery requires positive oracle headroom")
    return (baseline - model) / headroom
