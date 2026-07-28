from __future__ import annotations

from dataclasses import dataclass

import torch

from catena.core.schema import ControllerKind, MemoryEpisode


@dataclass(slots=True)
class ControlGeometry:
    rank: int
    singular_values: torch.Tensor
    condition_number: float
    projection_regret: float
    principal_angle_deg: float


def _basis_matrices(episode: MemoryEpisode, kind: ControllerKind) -> list[torch.Tensor]:
    erase = -episode.erase_candidate
    write = episode.write_candidate
    if kind is ControllerKind.TIED_SCALAR:
        return [erase + write]
    if kind is ControllerKind.DUAL_SCALAR:
        return [erase, write]
    if kind is ControllerKind.VECTOR:
        key = episode.keys[episode.affected_index]
        old = episode.erase_candidate.transpose(0, 1) @ key
        new = episode.new_value
        basis: list[torch.Tensor] = []
        for index in range(old.numel()):
            coordinate = torch.zeros_like(old)
            coordinate[index] = old[index]
            basis.append(-torch.outer(key, coordinate))
        for index in range(new.numel()):
            coordinate = torch.zeros_like(new)
            coordinate[index] = new[index]
            basis.append(torch.outer(key, coordinate))
        return basis
    raise ValueError(f"Geometry basis not implemented for {kind}")


def local_control_geometry(
    episode: MemoryEpisode,
    kind: ControllerKind,
    tolerance: float = 1e-8,
) -> ControlGeometry:
    bases = _basis_matrices(episode, kind)
    jacobian = torch.stack([basis.reshape(-1) for basis in bases], dim=1)
    delta = (episode.target_state - episode.state).reshape(-1)
    u, singular_values, _ = torch.linalg.svd(jacobian, full_matrices=False)
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        projection = torch.zeros_like(delta)
        smallest = 0.0
        condition = float("inf")
        principal_angle = 90.0 if delta.norm() > 0 else 0.0
    else:
        basis = u[:, :rank]
        projection = basis @ (basis.transpose(0, 1) @ delta)
        nonzero = singular_values[:rank]
        smallest = float(nonzero[-1].item())
        condition = float((nonzero[0] / nonzero[-1]).item()) if smallest > 0 else float("inf")
        denom = delta.norm().clamp_min(1e-12)
        cosine = (projection.norm() / denom).clamp(0.0, 1.0)
        principal_angle = float(torch.rad2deg(torch.acos(cosine)).item())
    regret = float((delta - projection).norm().item())
    return ControlGeometry(
        rank=rank,
        singular_values=singular_values.detach().cpu(),
        condition_number=condition,
        projection_regret=regret,
        principal_angle_deg=principal_angle,
    )


def analytic_optimal_controls(
    episode: MemoryEpisode,
    kind: ControllerKind,
) -> tuple[torch.Tensor, float]:
    bases = _basis_matrices(episode, kind)
    design = torch.stack([basis.reshape(-1) for basis in bases], dim=1)
    target = (episode.target_state - episode.state).reshape(-1)
    solution = torch.linalg.lstsq(design, target).solution
    residual = target - design @ solution
    return solution, float(residual.pow(2).mean().item())
