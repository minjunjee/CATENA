from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from catena.models.memory import GateOutput


class SemanticConstraint(str, Enum):
    FACTORIZED = "factorized"
    SHARED = "shared"


class MatchedSemanticController(nn.Module):
    """Parameter-matched shared/factorized semantic controller.

    Both constraints own the same tensors. Factorized routing masks the first
    half of the hidden representation from the write output and the second
    half from the erase output; shared routing uses all features.
    """

    def __init__(self, input_dim: int, hidden_dim: int, constraint: SemanticConstraint) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        self.constraint = constraint
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> GateOutput:
        hidden = self.encoder(features)
        if self.constraint is SemanticConstraint.FACTORIZED:
            half = hidden.shape[-1] // 2
            erase_hidden = torch.cat([hidden[..., :half], torch.zeros_like(hidden[..., half:])], dim=-1)
            write_hidden = torch.cat([torch.zeros_like(hidden[..., :half]), hidden[..., half:]], dim=-1)
            erase_logit = torch.nn.functional.linear(erase_hidden, self.head.weight[0:1], self.head.bias[0:1]).squeeze(-1)
            write_logit = torch.nn.functional.linear(write_hidden, self.head.weight[1:2], self.head.bias[1:2]).squeeze(-1)
        else:
            logits = self.head(hidden)
            erase_logit, write_logit = logits[..., 0], logits[..., 1]
        return GateOutput(torch.sigmoid(erase_logit), torch.sigmoid(write_logit))
