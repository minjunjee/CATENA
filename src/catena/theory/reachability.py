from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from catena.core.schema import ControllerKind, MemoryEpisode


class ReadoutMode(str, Enum):
    STATE = "state"
    BEHAVIORAL = "behavioral"
    AFFECTED = "affected"
    RETENTION = "retention"


@dataclass(slots=True)
class ReachabilityReport:
    """Dimension-normalized local reachability diagnostics.

    ``span_mse`` allows unconstrained coefficients and diagnoses whether the
    required direction lies in the linear span of the local control Jacobian.
    ``feasible_mse`` restricts post-sigmoid gates to [0, 1] and therefore
    diagnoses the actually reachable bounded set for the scalar probe.
    """

    span_mse: float
    feasible_mse: float
    rank: int
    condition_number: float
    principal_angle_deg: float
    optimal_gates: list[float]


def _columns(episode: MemoryEpisode, kind: ControllerKind) -> list[torch.Tensor]:
    erase = -episode.erase_candidate
    write = episode.write_candidate
    if kind is ControllerKind.TIED_SCALAR:
        return [erase + write]
    if kind is ControllerKind.DUAL_SCALAR:
        return [erase, write]
    raise ValueError(f"Only tied/dual scalar reachability is implemented, got {kind}")


def _weighted_readout(
    tensor: torch.Tensor,
    episode: MemoryEpisode,
    mode: ReadoutMode,
) -> torch.Tensor:
    """Return a vector whose squared L2 norm is the desired normalized error.

    State mode is normalized so ||r||^2 equals state MSE. Behavioral mode gives
    equal 1/2 weight to correction and unaffected retention, irrespective of
    the number of unaffected keys. This prevents the retention block from
    dominating merely because it contains more coordinates.
    """

    if mode is ReadoutMode.STATE:
        flat = tensor.reshape(-1)
        return flat / math.sqrt(max(flat.numel(), 1))

    affected = (
        episode.keys[episode.affected_index : episode.affected_index + 1] @ tensor
    ).reshape(-1)
    unaffected = (episode.keys[episode.unaffected_indices] @ tensor).reshape(-1)

    if mode is ReadoutMode.AFFECTED:
        return affected / math.sqrt(max(affected.numel(), 1))
    if mode is ReadoutMode.RETENTION:
        return unaffected / math.sqrt(max(unaffected.numel(), 1))

    affected = affected / math.sqrt(2.0 * max(affected.numel(), 1))
    unaffected = unaffected / math.sqrt(2.0 * max(unaffected.numel(), 1))
    return torch.cat([affected, unaffected])


def _bounded_lstsq(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Solve the one- or two-gate box-constrained least-squares problem exactly."""

    p = design.shape[1]
    if p == 1:
        denom = torch.dot(design[:, 0], design[:, 0]).clamp_min(1e-12)
        return torch.clamp(torch.dot(design[:, 0], target) / denom, 0.0, 1.0).reshape(1)
    if p != 2:
        raise ValueError("Exact bounded solver supports one or two gates.")

    candidates: list[torch.Tensor] = []
    unconstrained = torch.linalg.lstsq(design, target).solution
    if bool(torch.all((unconstrained >= 0.0) & (unconstrained <= 1.0))):
        candidates.append(unconstrained)

    # A convex quadratic over a rectangle attains its optimum either at the
    # unconstrained solution or on one of the four edges. Optimize each edge.
    for first in (0.0, 1.0):
        residual = target - first * design[:, 0]
        denom = torch.dot(design[:, 1], design[:, 1]).clamp_min(1e-12)
        second = torch.clamp(torch.dot(design[:, 1], residual) / denom, 0.0, 1.0)
        candidates.append(
            torch.stack(
                [
                    torch.as_tensor(first, dtype=design.dtype, device=design.device),
                    second,
                ]
            )
        )
    for second in (0.0, 1.0):
        residual = target - second * design[:, 1]
        denom = torch.dot(design[:, 0], design[:, 0]).clamp_min(1e-12)
        first = torch.clamp(torch.dot(design[:, 0], residual) / denom, 0.0, 1.0)
        candidates.append(
            torch.stack(
                [
                    first,
                    torch.as_tensor(second, dtype=design.dtype, device=design.device),
                ]
            )
        )

    errors = [torch.sum((design @ candidate - target) ** 2) for candidate in candidates]
    return candidates[int(torch.argmin(torch.stack(errors)).item())]


def constrained_reachability(
    episode: MemoryEpisode,
    kind: ControllerKind,
    *,
    mode: ReadoutMode,
) -> ReachabilityReport:
    columns = [_weighted_readout(column, episode, mode) for column in _columns(episode, kind)]
    design = torch.stack(columns, dim=1)
    target = _weighted_readout(episode.target_state - episode.state, episode, mode)

    singular_values = torch.linalg.svdvals(design)
    tolerance = max(design.shape) * torch.finfo(design.dtype).eps * singular_values.max().clamp_min(1e-12)
    rank = int((singular_values > tolerance).sum().item())
    if rank:
        nonzero = singular_values[:rank]
        condition = float((nonzero[0] / nonzero[-1]).item())
    else:
        condition = float("inf")

    unconstrained = torch.linalg.lstsq(design, target).solution
    span_prediction = design @ unconstrained
    span_mse = float(torch.sum((span_prediction - target) ** 2).item())

    feasible = _bounded_lstsq(design, target)
    feasible_mse = float(torch.sum((design @ feasible - target) ** 2).item())

    target_norm = target.norm()
    if float(target_norm.item()) <= 1e-12:
        angle = 0.0
    else:
        projection_norm = span_prediction.norm()
        if float(projection_norm.item()) <= 1e-12:
            angle = 90.0
        else:
            cosine = torch.clamp(
                torch.dot(span_prediction, target) / (projection_norm * target_norm),
                -1.0,
                1.0,
            )
            angle = float(torch.rad2deg(torch.acos(cosine)).item())

    return ReachabilityReport(
        span_mse=span_mse,
        feasible_mse=feasible_mse,
        rank=rank,
        condition_number=condition,
        principal_angle_deg=angle,
        optimal_gates=[float(value) for value in feasible],
    )


def behavioral_mse(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    """Equal-weight correction/retention behavioral MSE."""

    difference = output_state - episode.target_state
    vector = _weighted_readout(difference, episode, ReadoutMode.BEHAVIORAL)
    return torch.sum(vector**2)
