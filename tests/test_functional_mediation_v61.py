import numpy as np
import pytest
import torch

from catena.core.provenance_v61 import dumps_json_strict
from catena.core.schema import Operation
from catena.eval.functional_mediation import (
    GateChannel,
    dose_gate,
    exact_feasible_l2_norm_match,
    monotonic_nonincreasing_fraction,
    recovery_fraction_from_means,
    relevant_channels,
    restore_relevant,
    scalarize_gate,
)
from catena.models.memory import GateOutput
from experiments.e04_functional_mediation import (
    BootstrapRatioHeadroomError,
    _bootstrap_ratio_interval,
    _donor_base_index,
    _equivalence_gate,
    _expected_rows,
    _noninferiority_gate,
    _positive_gate,
)


def _gates(erase: float, write: float) -> GateOutput:
    return GateOutput(torch.tensor(erase), torch.tensor(write))


def test_physical_channel_dose_and_relevance_mapping():
    gates = _gates(0.8, 0.6)
    erase = dose_gate(gates, GateChannel.ERASE, 0.25)
    write = dose_gate(gates, GateChannel.WRITE, 0.5)
    joint = dose_gate(gates, GateChannel.JOINT, 0.0)

    assert (erase.erase.item(), erase.write.item()) == pytest.approx((0.2, 0.6))
    assert (write.erase.item(), write.write.item()) == pytest.approx((0.8, 0.3))
    assert (joint.erase.item(), joint.write.item()) == pytest.approx((0.0, 0.0))
    assert relevant_channels(Operation.PRESERVE) == ()
    assert relevant_channels(Operation.ADD) == (GateChannel.WRITE,)
    assert relevant_channels(Operation.INVALIDATE) == (GateChannel.ERASE,)
    assert relevant_channels(Operation.SUPERSEDE) == (
        GateChannel.ERASE,
        GateChannel.WRITE,
    )


def test_exact_feasible_norm_match_has_no_clipping_or_residual():
    donor = _gates(0.8, 0.1)
    recipient = _gates(0.5, 0.5)
    result = exact_feasible_l2_norm_match(donor, recipient, tolerance=1e-6)

    assert result.matched_norm == pytest.approx(result.recipient_norm, abs=1e-6)
    assert result.absolute_mismatch <= 1e-6
    assert result.gates.erase.item() < 1.0
    assert result.gates.write.item() > 0.0


def test_exact_norm_match_rejects_infeasible_or_zero_donor():
    with pytest.raises(ValueError, match="near-zero donor"):
        exact_feasible_l2_norm_match(
            _gates(0.0, 0.0),
            _gates(1.0, 0.0),
            tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="feasible gate box"):
        exact_feasible_l2_norm_match(
            _gates(0.01, 0.99),
            _gates(1.0, 1.0),
            tolerance=1e-6,
        )


def test_restore_relevant_changes_only_demanded_components():
    damaged = _gates(0.2, 0.3)
    donor = _gates(0.8, 0.9)

    add = restore_relevant(damaged, donor, Operation.ADD)
    invalidate = restore_relevant(damaged, donor, Operation.INVALIDATE)
    supersede = restore_relevant(damaged, donor, Operation.SUPERSEDE)
    preserve = restore_relevant(damaged, donor, Operation.PRESERVE)

    assert (add.erase.item(), add.write.item()) == pytest.approx((0.2, 0.9))
    assert (invalidate.erase.item(), invalidate.write.item()) == pytest.approx((0.8, 0.3))
    assert (supersede.erase.item(), supersede.write.item()) == pytest.approx((0.8, 0.9))
    assert (preserve.erase.item(), preserve.write.item()) == pytest.approx((0.2, 0.3))


def test_scalarization_recovery_and_monotonic_helpers():
    scalarized = scalarize_gate(_gates(0.2, 0.8))
    assert (scalarized.erase.item(), scalarized.write.item()) == pytest.approx((0.5, 0.5))
    assert recovery_fraction_from_means(
        damaged_error=0.04,
        rescued_error=0.01,
        baseline_error=0.0,
        minimum_headroom=0.001,
    ) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="does not exceed"):
        recovery_fraction_from_means(
            damaged_error=0.0005,
            rescued_error=0.0,
            baseline_error=0.0,
            minimum_headroom=0.001,
        )
    assert monotonic_nonincreasing_fraction(
        np.asarray([0.04, 0.03, 0.03, 0.01, 0.0]),
        tolerance=1e-12,
    ) == 1.0
    assert monotonic_nonincreasing_fraction(
        np.asarray([0.04, 0.03, 0.031, 0.01, 0.0]),
        tolerance=1e-12,
    ) == pytest.approx(0.75)


def test_e04_row_contract_and_adjacent_two_cycle_pairing():
    assert _expected_rows(1, 8) == 480
    assert _expected_rows(8, 128) == 61_440
    assert [_donor_base_index(index, 4) for index in range(4)] == [1, 0, 3, 2]


def test_seed_level_positive_equivalence_and_noninferiority_gates():
    positive_values = {
        seed: np.full(4, 0.002, dtype=np.float64)
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
    }
    null_values = {
        seed: np.zeros(4, dtype=np.float64)
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
    }

    positive = _positive_gate(
        positive_values,
        threshold=0.001,
        alpha=0.05,
        bootstrap_samples=100,
        bootstrap_seed=1,
        confidence=0.95,
        inference_eligible=True,
    )
    equivalent = _equivalence_gate(
        null_values,
        margin=0.0005,
        alpha=0.05,
        bootstrap_samples=100,
        bootstrap_seed=2,
        confidence=0.95,
        inference_eligible=True,
    )
    noninferior = _noninferiority_gate(
        null_values,
        margin=0.0005,
        alpha=0.05,
        bootstrap_samples=100,
        bootstrap_seed=3,
        confidence=0.95,
        inference_eligible=True,
    )

    assert positive["supported"] is True
    assert equivalent["supported"] is True
    assert noninferior["supported"] is True
    serialized = dumps_json_strict(
        {
            "positive": positive,
            "equivalent": equivalent,
            "noninferior": noninferior,
        }
    )
    assert '"11"' in serialized


def test_ineligible_execution_cannot_open_statistical_gate():
    values = {
        seed: np.full(4, 0.002, dtype=np.float64)
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
    }
    gate = _positive_gate(
        values,
        threshold=0.001,
        alpha=0.05,
        bootstrap_samples=20,
        bootstrap_seed=4,
        confidence=0.95,
        inference_eligible=False,
    )

    assert gate["bootstrap_lower_above_threshold"] is True
    assert gate["supported"] is False


def test_ratio_bootstrap_reports_registered_headroom_failure():
    numerator = {
        seed: np.asarray([0.002, 0.002], dtype=np.float64)
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
    }
    denominator = {
        seed: np.asarray([0.0, 0.004], dtype=np.float64)
        for seed in (11, 22, 33, 44, 55, 66, 77, 88)
    }

    with pytest.raises(BootstrapRatioHeadroomError, match="headroom"):
        _bootstrap_ratio_interval(
            numerator,
            denominator,
            minimum_denominator=0.001,
            samples=500,
            seed=7,
            confidence=0.95,
        )
