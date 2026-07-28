from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from catena.models.memory import GateOutput


class ScalarConstraint(str, Enum):
    TIED = "tied"
    DUAL = "dual"


class MatchedScalarController(nn.Module):
    """One parameterization, two control constraints.

    Both conditions own exactly the same two-output head.  The tied condition
    projects the two logits onto the diagonal subspace before the sigmoid.
    """

    def __init__(self, input_dim: int, hidden_dim: int, constraint: ScalarConstraint) -> None:
        super().__init__()
        self.constraint = constraint
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> GateOutput:
        logits = self.head(self.encoder(features))
        if self.constraint is ScalarConstraint.TIED:
            beta_logit = logits.mean(dim=-1)
            beta = torch.sigmoid(beta_logit)
            return GateOutput(erase=beta, write=beta)
        gates = torch.sigmoid(logits)
        return GateOutput(erase=gates[..., 0], write=gates[..., 1])
