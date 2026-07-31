import torch

from catena.lm.oracle_control import (
    OracleLevel,
    first_substantial_rescue,
    fit_bounded_erase_write,
    operation_gate_target,
)


def test_bounded_oracle_fit_recovers_known_gate() -> None:
    torch.manual_seed(1)
    state = torch.randn(3, 3)
    erase = torch.randn(3, 3)
    write = torch.randn(3, 3)
    target = state - 0.7 * erase + 0.2 * write
    fit = fit_bounded_erase_write(state, erase, write, target, grid_points=101)
    assert abs(fit.erase - 0.7) <= 0.011
    assert abs(fit.write - 0.2) <= 0.011


def test_oracle_level_classification() -> None:
    levels = [
        (OracleLevel.METADATA_GATE, 0.45),
        (OracleLevel.METADATA_GATE_ADDRESS, 0.70),
        (OracleLevel.METADATA_GATE_ADDRESS_CANDIDATE, 0.90),
    ]
    first = first_substantial_rescue(levels, learned=0.40, exact=0.95)
    assert first == OracleLevel.METADATA_GATE_ADDRESS
    assert operation_gate_target("ADD") == (0.0, 1.0)
