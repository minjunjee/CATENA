from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catena.core.config import load_config
from catena.data.operator_families import OperatorFamily
from experiments.e03_granularity_orientation import (
    _build_frontier,
    _contrast_report,
    _correlation,
    _empirical_calibration,
    _validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registered_e03_config_has_disjoint_dry_and_main_seeds() -> None:
    config = load_config(REPO_ROOT / "configs/e03_granularity_orientation.yaml")
    _validate_config(config)
    assert len(config["seeds"]) == 8
    assert set(config["seeds"]).isdisjoint(config["dry_run_seeds"])


def test_contrast_gate_tests_effect_above_registered_sesoi() -> None:
    seeds = list(range(8))
    passing = _contrast_report(
        np.full(8, 0.0011),
        seeds,
        sesoi=0.001,
        alpha=0.05,
    )
    failing = _contrast_report(
        np.full(8, 0.0009),
        seeds,
        sesoi=0.001,
        alpha=0.05,
    )
    assert passing["passed"] is True
    assert passing["seed_exact_sign_flip_p_greater_than_sesoi"] == 1 / 256
    assert failing["passed"] is False


def test_dry_run_correlation_is_explicitly_unevaluable() -> None:
    result = _correlation(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    assert result == {
        "evaluable": False,
        "n": 2,
        "pearson": None,
        "reason": "fewer_than_three_observations",
    }


def test_empirical_calibration_and_frontier_contract() -> None:
    rows = []
    for family_index, family in enumerate(OperatorFamily, start=1):
        fixed = 0.001 * family_index
        learned = 0.0005 * family_index
        rows.append(
            {
                "family": family.value,
                "fixed_diagonal_regret": fixed,
                "fixed_diagonal_empirical_error": fixed,
                "learned_basis_diagonal_regret": learned,
                "learned_basis_diagonal_empirical_error": learned,
                "low_rank_regret": 0.0,
                "low_rank_empirical_error": 0.0,
                "full_matrix_regret": 0.0,
                "full_matrix_empirical_error": 0.0,
            }
        )

    calibration = _empirical_calibration(rows, floor=1e-10)
    assert calibration["slope"] == pytest.approx(1.0)
    assert calibration["r2"] == pytest.approx(1.0)

    frontier = _build_frontier(rows, dimension=32, low_rank=8)
    assert len(frontier) == 12
    low_rank = next(
        row
        for row in frontier
        if row["control_class"] == "transaction_conditioned_low_rank"
    )
    assert low_rank["active_coefficients_per_transaction"] == 512
    assert "not parameter matched" in low_rank["role"]
