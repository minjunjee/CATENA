from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from catena.core.schema import ControllerKind
from catena.models.memory import GateOutput


@dataclass(slots=True)
class ControllerSpec:
    kind: ControllerKind
    input_dim: int
    value_dim: int
    hidden_dim: int = 64
    low_rank: int = 4


class GateController(nn.Module):
    def __init__(self, spec: ControllerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = nn.Sequential(
            nn.Linear(spec.input_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
        )
        if spec.kind is ControllerKind.TIED_SCALAR:
            self.head = nn.Linear(spec.hidden_dim, 1)
        elif spec.kind is ControllerKind.DUAL_SCALAR:
            self.head = nn.Linear(spec.hidden_dim, 2)
        elif spec.kind is ControllerKind.VECTOR:
            self.head = nn.Linear(spec.hidden_dim, 2 * spec.value_dim)
        elif spec.kind is ControllerKind.LOW_RANK:
            self.head = nn.Linear(spec.hidden_dim, 2 * spec.value_dim * spec.low_rank)
        else:
            raise ValueError(f"Unsupported controller kind: {spec.kind}")

    def forward(self, features: torch.Tensor) -> GateOutput:
        hidden = self.encoder(features)
        raw = self.head(hidden)
        if self.spec.kind is ControllerKind.TIED_SCALAR:
            beta = torch.sigmoid(raw[..., 0])
            return GateOutput(erase=beta, write=beta)
        if self.spec.kind is ControllerKind.DUAL_SCALAR:
            gates = torch.sigmoid(raw)
            return GateOutput(erase=gates[..., 0], write=gates[..., 1])
        if self.spec.kind is ControllerKind.VECTOR:
            gates = torch.sigmoid(raw).view(*raw.shape[:-1], 2, self.spec.value_dim)
            return GateOutput(erase=gates[..., 0, :], write=gates[..., 1, :])
        if self.spec.kind is ControllerKind.LOW_RANK:
            matrices = raw.view(
                *raw.shape[:-1], 2, self.spec.value_dim, self.spec.low_rank
            )
            # Low-rank factors are returned in GateOutput for experiment-specific assembly.
            return GateOutput(erase=matrices[..., 0, :, :], write=matrices[..., 1, :, :])
        raise AssertionError("Unreachable controller kind")


class HashedTextEncoder(nn.Module):
    def __init__(self, vocab_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vocab_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        return self.projection(bag)


class FactorizedSemanticController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.erase_branch = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.write_branch = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> GateOutput:
        hidden = self.shared(features)
        return GateOutput(
            erase=torch.sigmoid(self.erase_branch(hidden).squeeze(-1)),
            write=torch.sigmoid(self.write_branch(hidden).squeeze(-1)),
        )


class SharedSemanticController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> GateOutput:
        raw = torch.sigmoid(self.network(features))
        return GateOutput(erase=raw[..., 0], write=raw[..., 1])
