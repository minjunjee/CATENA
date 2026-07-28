from __future__ import annotations

from experiments.e11b_scale_normalized_coadaptation import (
    normalized_seed_contrasts,
)


def test_normalized_contrasts_are_scale_invariant() -> None:
    errors = {
        ("axis_commuting", "fixed_diagonal"): 0.0100,
        ("axis_commuting", "learned_basis_diagonal"): 0.0101,
        ("common_rotated_commuting", "fixed_diagonal"): 0.40,
        ("common_rotated_commuting", "learned_basis_diagonal"): 0.02,
        ("noncommuting", "learned_basis_diagonal"): 0.20,
        ("noncommuting", "low_rank"): 0.01,
    }
    energy = {
        "axis_commuting": 1.0,
        "common_rotated_commuting": 1.0,
        "noncommuting": 1.0,
    }
    first = normalized_seed_contrasts(errors, energy)
    scaled = normalized_seed_contrasts(
        {key: value * 7.0 for key, value in errors.items()},
        {key: value * 7.0 for key, value in energy.items()},
    )

    normalized_keys = {
        key for key in first if not key.endswith("_raw_gain") and key != "noncommuting_raw_gap"
    }
    for key in normalized_keys:
        assert abs(first[key] - scaled[key]) < 1e-12


def test_normalized_contrasts_encode_registered_regimes() -> None:
    errors = {
        ("axis_commuting", "fixed_diagonal"): 0.00100,
        ("axis_commuting", "learned_basis_diagonal"): 0.00101,
        ("common_rotated_commuting", "fixed_diagonal"): 0.50,
        ("common_rotated_commuting", "learned_basis_diagonal"): 0.01,
        ("noncommuting", "learned_basis_diagonal"): 0.20,
        ("noncommuting", "low_rank"): 0.01,
    }
    energy = {
        "axis_commuting": 1.0,
        "common_rotated_commuting": 1.0,
        "noncommuting": 1.0,
    }

    contrasts = normalized_seed_contrasts(errors, energy)

    assert contrasts["axis_equivalence_fraction"] < 0.01
    assert contrasts["common_rotation_recovery_fraction"] > 0.95
    assert contrasts["common_shared_residual_fraction"] <= 0.01
    assert contrasts["noncommuting_gap_fraction"] >= 0.10
    assert contrasts["noncommuting_shared_residual_fraction"] >= 0.10
    assert contrasts["low_rank_recovery_fraction"] >= 0.90
    assert contrasts["low_rank_residual_fraction"] <= 0.01
