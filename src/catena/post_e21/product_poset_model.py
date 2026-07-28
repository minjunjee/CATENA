"""Matched learned sequence controllers for the E23 product poset."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from catena.data.controller_poset import ControllerSpec, missing_required_axes
from catena.data.sequence_control_lattice import (
    SequenceControlLatticeInput,
    demand_feature_dim,
)


@dataclass(frozen=True, slots=True)
class ProductPosetProbeConfig:
    """Theory-floor coefficients; never used as an application outcome."""

    missing_axis_floor: float
    numerical_floor: float

    def __post_init__(self) -> None:
        if (
            self.missing_axis_floor < 0
            or self.numerical_floor < 0
            or not math.isfinite(self.missing_axis_floor)
            or not math.isfinite(self.numerical_floor)
        ):
            raise ValueError("theory coefficients must be finite and non-negative")


def theoretical_affected_error(
    *,
    controller: ControllerSpec,
    demand_family: str,
    intensity: float,
    updates: int,
    gap_events: int,
    config: ProductPosetProbeConfig,
) -> float:
    """Outcome-independent qualitative floor retained for theory diagnostics."""

    if not math.isfinite(intensity) or intensity <= 0:
        raise ValueError("intensity must be positive and finite")
    if updates <= 0 or gap_events < 0:
        raise ValueError("invalid sequence cell")
    missing = len(missing_required_axes(controller, demand_family))
    stress = 1.0 + 0.04 * (updates - 1) + 0.00002 * gap_events
    return config.numerical_floor + missing * config.missing_axis_floor * intensity**2 * stress


@dataclass(slots=True)
class ProductPosetSequenceOutput:
    state: torch.Tensor
    erase_gates: torch.Tensor
    write_gates: torch.Tensor


class MatchedProductPosetSequenceController(nn.Module):
    """The E18 maximal head projected by an arbitrary four-bit controller."""

    def __init__(
        self,
        *,
        controller: ControllerSpec,
        num_entities: int,
        value_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.value_dim = int(value_dim)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        encoder_input_dim = (
            2 * embedding_dim + self.value_dim + demand_feature_dim(self.value_dim) + self.value_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Identical maximal parameter surface for all 16 projections.
        self.maximal_head = nn.Linear(hidden_dim, 2 + 2 * self.value_dim)

    def forward(
        self,
        inputs: SequenceControlLatticeInput,
    ) -> ProductPosetSequenceOutput:
        magnitude, value, address, conditioning = (bool(bit) for bit in self.controller.bits)
        state = inputs.initial_state.clone()
        batch_size, sequence_length = inputs.erase_entity_ids.shape
        batch_index = torch.arange(batch_size, device=state.device)
        erase_rows: list[torch.Tensor] = []
        write_rows: list[torch.Tensor] = []
        for time_index in range(sequence_length):
            erase_address = inputs.erase_entity_ids[:, time_index]
            oracle_write_address = inputs.write_entity_ids[:, time_index]
            candidate = inputs.candidate_values[:, time_index]
            old_read = state[batch_index, erase_address]
            state_summary = old_read if conditioning else torch.zeros_like(old_read)
            features = torch.cat(
                (
                    self.entity_embedding(erase_address),
                    self.entity_embedding(oracle_write_address),
                    candidate,
                    inputs.demand_features[:, time_index],
                    state_summary,
                ),
                dim=-1,
            )
            hidden = self.encoder(features)
            raw = self.maximal_head(hidden)
            scalars = torch.sigmoid(raw[:, :2])
            diagonals = torch.sigmoid(raw[:, 2:]).view(
                batch_size,
                2,
                self.value_dim,
            )
            if value:
                if magnitude:
                    erase_gate = diagonals[:, 0]
                    write_gate = diagonals[:, 1]
                else:
                    beta = diagonals.mean(dim=1)
                    erase_gate = beta
                    write_gate = beta
            elif magnitude:
                erase_gate = scalars[:, 0:1].expand(-1, self.value_dim)
                write_gate = scalars[:, 1:2].expand(-1, self.value_dim)
            else:
                beta = scalars.mean(dim=-1, keepdim=True)
                erase_gate = beta.expand(-1, self.value_dim)
                write_gate = beta.expand(-1, self.value_dim)

            write_address = oracle_write_address if address else erase_address
            erased = old_read - erase_gate * old_read
            next_state = state.clone()
            next_state[batch_index, erase_address] = erased
            same = erase_address == write_address
            different = ~same
            if bool(different.any()):
                next_state[batch_index[different], write_address[different]] = (
                    state[batch_index[different], write_address[different]]
                    + write_gate[different] * candidate[different]
                )
            if bool(same.any()):
                next_state[batch_index[same], erase_address[same]] = (
                    erased[same] + write_gate[same] * candidate[same]
                )
            state = next_state
            erase_rows.append(erase_gate)
            write_rows.append(write_gate)
        return ProductPosetSequenceOutput(
            state=state,
            erase_gates=torch.stack(erase_rows, dim=1),
            write_gates=torch.stack(write_rows, dim=1),
        )


def product_poset_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
