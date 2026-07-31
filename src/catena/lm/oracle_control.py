from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch.nn import functional as F


class OracleLevel(StrEnum):
    LEARNED = "learned"
    METADATA_GATE = "metadata_oracle_gate"
    METADATA_GATE_ADDRESS = "metadata_oracle_gate_address"
    METADATA_GATE_ADDRESS_CANDIDATE = "metadata_oracle_gate_address_candidate"
    BEHAVIORAL_UPPER_BOUND = "outcome_behavioral_upper_bound"
    EXACT_REFRESH = "exact_canonical_refresh"


@dataclass(frozen=True)
class OracleFit:
    erase: float
    write: float
    residual_mse: float
    bounded: bool


def operation_gate_target(operation: str) -> tuple[float, float]:
    mapping = {
        "PRESERVE": (0.0, 0.0),
        "ADD": (0.0, 1.0),
        "INVALIDATE": (1.0, 0.0),
        "SUPERSEDE": (1.0, 1.0),
        "ADD_EXCEPTION": (0.0, 1.0),
    }
    try:
        return mapping[operation]
    except KeyError as exc:
        raise ValueError(f"Unknown operation: {operation}") from exc


def fit_bounded_erase_write(
    state: torch.Tensor,
    erase_component: torch.Tensor,
    write_component: torch.Tensor,
    target_state: torch.Tensor,
    *,
    grid_points: int = 101,
) -> OracleFit:
    """Outcome-using bounded 2-D grid fit for diagnostic upper bounds.

    This function is deliberately simple and transparent. It must not be
    reported as a deployable controller because ``target_state`` can depend on
    the future behavioral outcome.
    """

    if not (state.shape == erase_component.shape == write_component.shape == target_state.shape):
        raise ValueError("All state tensors must have identical shapes")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    values = torch.linspace(0.0, 1.0, grid_points, device=state.device, dtype=state.dtype)
    best: tuple[float, float, float] | None = None
    with torch.no_grad():
        for erase in values:
            # Vectorize the write-axis while retaining a transparent bounded scan.
            candidates = (
                state.unsqueeze(0)
                - erase * erase_component.unsqueeze(0)
                + values.view(-1, *([1] * state.ndim)) * write_component.unsqueeze(0)
            )
            errors = (
                (candidates - target_state.unsqueeze(0))
                .float()
                .pow(2)
                .mean(dim=tuple(range(1, candidates.ndim)))
            )
            index = int(errors.argmin().item())
            record = (float(errors[index].item()), float(erase.item()), float(values[index].item()))
            if best is None or record[0] < best[0]:
                best = record
    assert best is not None
    return OracleFit(erase=best[1], write=best[2], residual_mse=best[0], bounded=True)


def metadata_address_from_span(keys: torch.Tensor, span: slice) -> torch.Tensor:
    """Average and normalize target materialization keys.

    ``keys`` has shape ``[tokens, heads, head_dim]`` or
    ``[batch, tokens, heads, head_dim]``.
    """

    if keys.ndim == 4:
        selected = keys[:, span]
        address = selected.mean(dim=1)
    elif keys.ndim == 3:
        selected = keys[span]
        address = selected.mean(dim=0)
    else:
        raise ValueError("keys must have three or four dimensions")
    return F.normalize(address, dim=-1)


def exact_candidate_from_span(values: torch.Tensor, span: slice) -> torch.Tensor:
    if values.ndim == 4:
        return values[:, span].mean(dim=1)
    if values.ndim == 3:
        return values[span].mean(dim=0)
    raise ValueError("values must have three or four dimensions")


def headroom_recovery(learned: float, intervention: float, exact: float) -> float | None:
    denominator = exact - learned
    if abs(denominator) < 1.0e-8:
        return None
    return (intervention - learned) / denominator


def first_substantial_rescue(
    levels: Iterable[tuple[OracleLevel, float]],
    *,
    learned: float,
    exact: float,
    absolute_gain_min: float = 0.02,
    headroom_fraction_min: float = 0.20,
) -> OracleLevel | None:
    for level, score in levels:
        if level in (OracleLevel.LEARNED, OracleLevel.EXACT_REFRESH):
            continue
        recovery = headroom_recovery(learned, score, exact)
        if (
            score - learned >= absolute_gain_min
            and recovery is not None
            and recovery >= headroom_fraction_min
        ):
            return level
    return None
