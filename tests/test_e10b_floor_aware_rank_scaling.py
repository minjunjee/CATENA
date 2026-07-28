from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from catena.eval.rank_saturation import (
    classify_pre_saturation_pairs,
    eligible_pre_saturation_monotonic_fraction,
)
from experiments.e10b_floor_aware_rank_scaling import (
    fresh_test_descriptor_seed,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(rank: int, error: float, recovery: float) -> dict[str, float | int]:
    return {
        "learned_rank": rank,
        "test_error": error,
        "exact_target_recovery": recovery,
    }


def test_only_pairs_with_unqualified_lower_rank_enter_monotonicity_gate() -> None:
    pairs = classify_pre_saturation_pairs(
        [
            _row(1, 0.10, 0.80),
            _row(2, 0.04, 0.96),
            _row(4, 0.041, 0.97),
            _row(8, 0.042, 0.99),
        ],
        recovery_threshold=0.95,
    )

    assert [item["pair_disposition"] for item in pairs] == [
        "ELIGIBLE_PRE_SATURATION",
        "SATURATED_EXCLUDED",
        "SATURATED_EXCLUDED",
    ]
    assert pairs[0]["non_increasing"] is True
    assert pairs[1]["non_increasing"] is False
    fraction, passed, eligible = eligible_pre_saturation_monotonic_fraction(
        pairs
    )
    assert (fraction, passed, eligible) == (1.0, 1, 1)


def test_pre_saturation_increase_fails_without_a_numerical_tolerance() -> None:
    pairs = classify_pre_saturation_pairs(
        [
            _row(1, 0.10, 0.40),
            _row(2, 0.11, 0.60),
            _row(4, 0.05, 0.96),
        ],
        recovery_threshold=0.95,
    )

    fraction, passed, eligible = eligible_pre_saturation_monotonic_fraction(
        pairs
    )
    assert fraction == pytest.approx(0.5)
    assert (passed, eligible) == (1, 2)
    assert pairs[0]["error_decrease"] == pytest.approx(-0.01)


def test_e10b_preserves_original_grid_thresholds_and_statistical_unit() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/e10b_floor_aware_rank_scaling.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["seeds"] == [101, 211, 307, 401, 503, 601, 701, 809]
    assert config["data"]["intrinsic_ranks"] == [1, 2, 4, 8, 16]
    assert config["model"]["learned_ranks"] == [1, 2, 4, 8, 16, 32]
    assert config["claim_gate"] == {
        "oracle_normalized_recovery": 0.95,
        "max_rank_factor": 2.0,
        "minimum_rank_match_fraction": 0.8,
        "eligible_pre_saturation_monotonic_fraction": 1.0,
    }
    assert config["statistics"] == {
        "alpha": 0.05,
        "statistical_unit": "frozen_training_seed",
    }
    assert config["protocol"]["original_test_rows_reused"] is False
    assert config["protocol"]["checkpoints_retrained"] is False
    assert config["source_e10"]["checkpoint_count"] == 240


def test_fresh_descriptor_namespace_cannot_reuse_original_e10_test_seed() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/e10b_floor_aware_rank_scaling.yaml").read_text(
            encoding="utf-8"
        )
    )
    namespace = config["data"]["fresh_test_namespace"]
    fresh_seeds: set[int] = set()
    original_seeds: set[int] = set()
    for seed in config["seeds"]:
        for intrinsic_rank in config["data"]["intrinsic_ranks"]:
            fresh = fresh_test_descriptor_seed(
                source_training_seed=seed,
                intrinsic_rank=intrinsic_rank,
                seed_offset=namespace["seed_offset"],
                seed_multiplier=namespace["seed_multiplier"],
            )
            original = 30_000 * seed + intrinsic_rank
            assert fresh != original
            fresh_seeds.add(fresh)
            original_seeds.add(original)

    assert len(fresh_seeds) == 40
    assert fresh_seeds.isdisjoint(original_seeds)
