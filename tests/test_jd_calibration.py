from __future__ import annotations

import numpy as np
import pytest

from catena.eval.jd_calibration import (
    AnalyticCandidate,
    CalibrationThresholds,
    RegretBin,
    assign_regret_bin,
    fit_jd_application_calibration,
    select_first_valid_candidates,
    validate_selected_design,
)


def _bins() -> tuple[RegretBin, ...]:
    return (
        RegretBin("low", 1e-5, 1e-3),
        RegretBin("high", 1e-3, 6.5e-3, include_upper=True),
    )


def _candidate(index: int, regret: float) -> AnalyticCandidate:
    return AnalyticCandidate(
        candidate_id=f"candidate-{index}",
        construction_sha256=f"{index:064x}",
        analytic_regret=regret,
        alpha=0.1 * index,
        generation_seed=1000 + index,
    )


def test_regret_bin_assignment_has_registered_boundaries() -> None:
    bins = _bins()
    assert assign_regret_bin(1e-5, bins) == "low"
    assert assign_regret_bin(1e-3, bins) == "high"
    assert assign_regret_bin(6.5e-3, bins) == "high"
    assert assign_regret_bin(0.0, bins) is None


def test_first_valid_selection_and_design_gate() -> None:
    candidates = [
        _candidate(1, 0.0),
        _candidate(2, 2e-4),
        _candidate(3, 2e-3),
        _candidate(4, 3e-4),
        _candidate(5, 6e-3),
    ]
    selected = select_first_valid_candidates(
        candidates,
        _bins(),
        families_per_bin=2,
    )
    assert [item.candidate_id for item in selected["low"]] == [
        "candidate-2",
        "candidate-4",
    ]
    assert [item.candidate_id for item in selected["high"]] == [
        "candidate-3",
        "candidate-5",
    ]
    design = validate_selected_design(
        selected,
        _bins(),
        families_per_bin=2,
        minimum_nonzero_range=0.004,
    )
    assert design["passed"] is True
    assert design["observed_nonzero_range"] == pytest.approx(0.0058)


def test_design_gate_rejects_missing_bin_and_duplicate_seed() -> None:
    selected = {
        "low": [_candidate(1, 2e-4), _candidate(2, 3e-4)],
        "high": [_candidate(3, 2e-3)],
    }
    design = validate_selected_design(
        selected,
        _bins(),
        families_per_bin=2,
        minimum_nonzero_range=0.001,
    )
    assert design["passed"] is False


def test_design_gate_rejects_candidates_stored_under_the_wrong_bin_key() -> None:
    selected = {
        "low": [_candidate(1, 2e-3)],
        "high": [_candidate(2, 2e-4)],
    }
    design = validate_selected_design(
        selected,
        _bins(),
        families_per_bin=1,
        minimum_nonzero_range=0.001,
    )
    assert design["passed"] is False
    assert design["all_candidates_in_mapping_key_bin"] is False


def test_calibration_gate_distinguishes_absolute_and_rank_prediction() -> None:
    x = np.linspace(1e-4, 6e-3, 48)
    thresholds = CalibrationThresholds(0.99, 0.95, 1.05, 1e-4)
    passing = fit_jd_application_calibration(x, x, thresholds=thresholds)
    assert passing["passed"] is True
    assert passing["slope"] == pytest.approx(1.0)
    assert passing["intercept"] == pytest.approx(0.0, abs=1e-12)

    rank_only = fit_jd_application_calibration(
        x,
        1.1 * x,
        thresholds=thresholds,
    )
    assert rank_only["r2_gate_passed"] is True
    assert rank_only["absolute_calibration_passed"] is False
    assert rank_only["passed"] is False


def test_calibration_rejects_exact_zero_and_range_free_inputs() -> None:
    thresholds = CalibrationThresholds(0.99, 0.95, 1.05, 1e-4)
    with pytest.raises(ValueError, match="Exact-zero"):
        fit_jd_application_calibration(
            np.asarray([0.0, 0.1, 0.2]),
            np.asarray([0.0, 0.1, 0.2]),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="no range"):
        fit_jd_application_calibration(
            np.asarray([0.1, 0.1, 0.1]),
            np.asarray([0.1, 0.1, 0.1]),
            thresholds=thresholds,
        )
