from __future__ import annotations

import torch
from torch import nn

from catena.models.operator_controllers import LowRankOperatorController


def _orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix)
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    return q * signs


class FixedBasisDiagonalController(nn.Module):
    def __init__(self, *, descriptor_dim: int, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.diagonal = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        values = self.diagonal(descriptor)
        return torch.diag_embed(values)


class LearnedBasisDiagonalController(nn.Module):
    def __init__(self, *, descriptor_dim: int, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.raw_basis = nn.Parameter(torch.eye(dimension) + 0.01 * torch.randn(dimension, dimension))
        self.diagonal = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )

    def basis(self) -> torch.Tensor:
        return _orthogonalize(self.raw_basis)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        basis = self.basis()
        values = self.diagonal(descriptor)
        diagonal = torch.diag_embed(values)
        return basis @ diagonal @ basis.transpose(0, 1)


class BlockDiagonalController(nn.Module):
    def __init__(
        self,
        *,
        descriptor_dim: int,
        dimension: int,
        hidden_dim: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if dimension % block_size != 0:
            raise ValueError("dimension must be divisible by block_size")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.block_count = dimension // block_size
        self.raw_basis = nn.Parameter(torch.eye(dimension) + 0.01 * torch.randn(dimension, dimension))
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.block_count * block_size * block_size),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        basis = _orthogonalize(self.raw_basis)
        raw = self.network(descriptor).view(
            -1, self.block_count, self.block_size, self.block_size
        )
        matrix = torch.zeros(
            descriptor.shape[0],
            self.dimension,
            self.dimension,
            dtype=descriptor.dtype,
            device=descriptor.device,
        )
        for index in range(self.block_count):
            start = index * self.block_size
            stop = start + self.block_size
            matrix[:, start:stop, start:stop] = raw[:, index]
        return basis @ matrix @ basis.transpose(0, 1)


class LowRankCoadaptationController(LowRankOperatorController):
    pass
