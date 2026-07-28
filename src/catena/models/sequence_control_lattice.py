from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from catena.data.sequence_control_lattice import (
    SequenceControlLatticeInput,
    demand_feature_dim,
)


class SequenceControlFreedom(StrEnum):
    TIED_SCALAR = "tied_scalar"
    DUAL_SCALAR = "dual_scalar"
    DIAGONAL_VALUE = "diagonal_value"
    SEPARATE_ADDRESS = "separate_address"
    STATE_AWARE = "state_aware"


@dataclass(slots=True)
class SequenceControlLatticeOutput:
    state: torch.Tensor
    erase_gates: torch.Tensor
    write_gates: torch.Tensor


class MatchedSequenceControlLattice(nn.Module):
    """A shared maximal surface with nested forward projection constraints."""

    def __init__(
        self,
        *,
        freedom: SequenceControlFreedom,
        num_entities: int,
        value_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.freedom = freedom
        self.value_dim = int(value_dim)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        encoder_input_dim = (
            2 * embedding_dim
            + self.value_dim
            + demand_feature_dim(self.value_dim)
            + self.value_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Every freedom registers both scalar and value-diagonal outputs.
        self.maximal_head = nn.Linear(hidden_dim, 2 + 2 * self.value_dim)

    def forward(
        self,
        inputs: SequenceControlLatticeInput,
    ) -> SequenceControlLatticeOutput:
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
            state_summary = (
                old_read
                if self.freedom is SequenceControlFreedom.STATE_AWARE
                else torch.zeros_like(old_read)
            )
            features = torch.cat(
                [
                    self.entity_embedding(erase_address),
                    self.entity_embedding(oracle_write_address),
                    candidate,
                    inputs.demand_features[:, time_index],
                    state_summary,
                ],
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

            if self.freedom is SequenceControlFreedom.TIED_SCALAR:
                beta = scalars.mean(dim=-1, keepdim=True)
                erase_gate = beta.expand(-1, self.value_dim)
                write_gate = beta.expand(-1, self.value_dim)
            elif self.freedom is SequenceControlFreedom.DUAL_SCALAR:
                erase_gate = scalars[:, 0:1].expand(-1, self.value_dim)
                write_gate = scalars[:, 1:2].expand(-1, self.value_dim)
            else:
                erase_gate = diagonals[:, 0]
                write_gate = diagonals[:, 1]

            write_address = (
                oracle_write_address
                if self.freedom
                in {
                    SequenceControlFreedom.SEPARATE_ADDRESS,
                    SequenceControlFreedom.STATE_AWARE,
                }
                else erase_address
            )
            erased = old_read - erase_gate * old_read
            same_address = erase_address == write_address
            next_state = state.clone()
            next_state[batch_index, erase_address] = erased
            different = ~same_address
            if different.any():
                next_state[
                    batch_index[different],
                    write_address[different],
                ] = (
                    state[batch_index[different], write_address[different]]
                    + write_gate[different] * candidate[different]
                )
            if same_address.any():
                next_state[
                    batch_index[same_address],
                    erase_address[same_address],
                ] = (
                    erased[same_address]
                    + write_gate[same_address] * candidate[same_address]
                )
            state = next_state
            erase_rows.append(erase_gate)
            write_rows.append(write_gate)

        return SequenceControlLatticeOutput(
            state=state,
            erase_gates=torch.stack(erase_rows, dim=1),
            write_gates=torch.stack(write_rows, dim=1),
        )


def sequence_lattice_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
