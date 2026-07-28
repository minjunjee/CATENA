from __future__ import annotations

import torch
from torch import nn


class LowRankOperatorController(nn.Module):
    def __init__(
        self,
        *,
        descriptor_dim: int,
        dimension: int,
        rank: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.backbone = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.left_head = nn.Linear(hidden_dim, dimension * rank)
        self.right_head = nn.Linear(hidden_dim, dimension * rank)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(descriptor)
        left = self.left_head(hidden).view(-1, self.dimension, self.rank)
        right = self.right_head(hidden).view(-1, self.dimension, self.rank)
        return torch.bmm(left, right.transpose(1, 2)) / max(self.rank, 1) ** 0.5


class FullMatrixOperatorController(nn.Module):
    def __init__(
        self,
        *,
        descriptor_dim: int,
        dimension: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension * dimension),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.network(descriptor).view(-1, self.dimension, self.dimension)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
