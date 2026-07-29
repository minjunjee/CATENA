from __future__ import annotations

from dataclasses import dataclass

import torch

from catena.core.schema import DemandOrientation


@dataclass(slots=True)
class SubspaceDemand:
    orientation: DemandOrientation
    projector: torch.Tensor
    descriptor: torch.Tensor
    oracle_mask: torch.Tensor | None


def _orthogonal_matrix(dim: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    raw = torch.randn(dim, dim, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(raw)
    return q


def build_subspace_demand(
    *,
    dim: int,
    active_dim: int,
    orientation: DemandOrientation,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> SubspaceDemand:
    if not 0 < active_dim <= dim:
        raise ValueError("active_dim must be within [1, dim]")
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(dim, dtype=dtype)
    if orientation in {DemandOrientation.AXIS_CONTIGUOUS, DemandOrientation.ORACLE_MASK}:
        start = int(torch.randint(0, dim - active_dim + 1, (1,), generator=generator).item())
        mask[start : start + active_dim] = 1.0
        projector = torch.diag(mask)
        descriptor = torch.nn.functional.one_hot(
            torch.tensor(start), num_classes=dim
        ).to(dtype=dtype)
        oracle_mask = mask if orientation is DemandOrientation.ORACLE_MASK else None
    elif orientation is DemandOrientation.AXIS_SPARSE:
        indices = torch.randperm(dim, generator=generator)[:active_dim]
        mask[indices] = 1.0
        projector = torch.diag(mask)
        descriptor = torch.zeros(dim, dtype=dtype)
        descriptor[indices] = 1.0 / active_dim
        oracle_mask = None
    elif orientation is DemandOrientation.ROTATED:
        q = _orthogonal_matrix(dim, generator, dtype)
        basis = q[:, :active_dim]
        projector = basis @ basis.transpose(0, 1)
        descriptor = basis.mean(dim=1)
        oracle_mask = None
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return SubspaceDemand(
        orientation=orientation,
        projector=projector,
        descriptor=descriptor,
        oracle_mask=oracle_mask,
    )
