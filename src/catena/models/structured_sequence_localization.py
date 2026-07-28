from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from catena.data.structured_sequence_localization import (
    StructuredSequenceTransferBatch,
    StructuredTransferCondition,
    concatenate_structured_event_features,
    structured_event_feature_dim,
)


class StructuredSequenceFreedom(StrEnum):
    BASE = "base"
    SEPARATE_ADDRESS = "separate_address"
    STATE_AWARE = "state_aware"
    FULL = "full"

    @property
    def has_separate_address(self) -> bool:
        return self in {self.SEPARATE_ADDRESS, self.FULL}

    @property
    def has_state_read(self) -> bool:
        return self in {self.STATE_AWARE, self.FULL}


@dataclass(slots=True)
class StructuredSequenceTransferOutput:
    state: torch.Tensor
    erase_address_weights: torch.Tensor
    write_address_weights: torch.Tensor
    erase_candidates: torch.Tensor
    activity_gates: torch.Tensor
    raw_address_logits: torch.Tensor
    raw_candidates: torch.Tensor
    raw_activity_logits: torch.Tensor


class MatchedStructuredSequenceController(nn.Module):
    """Common maximal event surface with paired address/state-read projections."""

    def __init__(
        self,
        *,
        freedom: StructuredSequenceFreedom,
        slots: int,
        identifier_dim: int,
        value_dim: int,
        hidden_dim: int,
        address_temperature: float,
    ) -> None:
        super().__init__()
        if address_temperature <= 0:
            raise ValueError("address_temperature must be positive")
        self.freedom = freedom
        self.slots = int(slots)
        self.value_dim = int(value_dim)
        self.address_temperature = float(address_temperature)
        feature_dim = structured_event_feature_dim(identifier_dim, value_dim)
        self.event_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.address_head = nn.Linear(hidden_dim, 2 * self.slots)
        self.candidate_head = nn.Linear(hidden_dim, self.value_dim)
        self.activity_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _visible_demand_gates(
        *,
        family_one_hot: torch.Tensor,
        operation_one_hot: torch.Tensor,
        channel_mask: torch.Tensor,
        erase_candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = family_one_hot[:, 0:1]
        granularity = family_one_hot[:, 1:2]
        address = family_one_hot[:, 2:3]
        state_conditioning = family_one_hot[:, 3:4]

        erase_scalar = operation_one_hot[:, 2:3] + operation_one_hot[:, 3:4]
        write_scalar = operation_one_hot[:, 1:2] + operation_one_hot[:, 3:4]
        state_erase = (erase_candidate[:, 0:1] > 0.0).to(
            erase_candidate.dtype
        )
        state_write = 1.0 - state_erase
        erase_gate = (
            magnitude * erase_scalar
            + granularity * channel_mask
            + address
            + state_conditioning * state_erase
        )
        write_gate = (
            magnitude * write_scalar
            + granularity * channel_mask
            + address
            + state_conditioning * state_write
        )
        return erase_gate, write_gate

    def forward(
        self,
        batch: StructuredSequenceTransferBatch,
        condition: StructuredTransferCondition,
    ) -> StructuredSequenceTransferOutput:
        event_features = concatenate_structured_event_features(batch.inputs)
        state = batch.inputs.initial_state.clone()
        batch_size, sequence_length = batch.erase_addresses.shape
        erase_weights_rows: list[torch.Tensor] = []
        write_weights_rows: list[torch.Tensor] = []
        candidate_rows: list[torch.Tensor] = []
        activity_rows: list[torch.Tensor] = []
        raw_address_rows: list[torch.Tensor] = []
        raw_candidate_rows: list[torch.Tensor] = []
        raw_activity_rows: list[torch.Tensor] = []

        for time_index in range(sequence_length):
            hidden = self.event_encoder(event_features[:, time_index])
            raw_address = self.address_head(hidden).view(
                batch_size,
                2,
                self.slots,
            )
            raw_candidate = self.candidate_head(hidden)
            raw_activity = self.activity_head(hidden).squeeze(-1)

            if self.freedom.has_separate_address:
                erase_logits = raw_address[:, 0]
                write_logits = raw_address[:, 1]
            else:
                shared_logits = raw_address.mean(dim=1)
                erase_logits = shared_logits
                write_logits = shared_logits

            if condition.uses_oracle_address:
                erase_weights = nn.functional.one_hot(
                    batch.erase_addresses[:, time_index],
                    num_classes=self.slots,
                ).to(state.dtype)
                write_weights = nn.functional.one_hot(
                    batch.write_addresses[:, time_index],
                    num_classes=self.slots,
                ).to(state.dtype)
            else:
                erase_weights = torch.softmax(
                    erase_logits / self.address_temperature,
                    dim=-1,
                )
                write_weights = torch.softmax(
                    write_logits / self.address_temperature,
                    dim=-1,
                )

            state_read = torch.einsum("bs,bsv->bv", erase_weights, state)
            if condition.uses_oracle_candidate:
                erase_candidate = batch.old_candidates[:, time_index]
            elif self.freedom.has_state_read:
                erase_candidate = state_read
            else:
                erase_candidate = raw_candidate

            erase_gate, write_gate = self._visible_demand_gates(
                family_one_hot=batch.inputs.family_one_hot[:, time_index],
                operation_one_hot=batch.inputs.operation_one_hot[:, time_index],
                channel_mask=batch.inputs.channel_masks[:, time_index],
                erase_candidate=erase_candidate,
            )
            activity = torch.sigmoid(raw_activity)
            erase_delta = (
                activity[:, None] * erase_gate * erase_candidate
            )
            write_delta = (
                activity[:, None]
                * write_gate
                * batch.inputs.new_candidates[:, time_index]
            )
            state = (
                state
                - erase_weights[:, :, None] * erase_delta[:, None, :]
                + write_weights[:, :, None] * write_delta[:, None, :]
            )

            erase_weights_rows.append(erase_weights)
            write_weights_rows.append(write_weights)
            candidate_rows.append(erase_candidate)
            activity_rows.append(activity)
            raw_address_rows.append(raw_address)
            raw_candidate_rows.append(raw_candidate)
            raw_activity_rows.append(raw_activity)

        return StructuredSequenceTransferOutput(
            state=state,
            erase_address_weights=torch.stack(erase_weights_rows, dim=1),
            write_address_weights=torch.stack(write_weights_rows, dim=1),
            erase_candidates=torch.stack(candidate_rows, dim=1),
            activity_gates=torch.stack(activity_rows, dim=1),
            raw_address_logits=torch.stack(raw_address_rows, dim=1),
            raw_candidates=torch.stack(raw_candidate_rows, dim=1),
            raw_activity_logits=torch.stack(raw_activity_rows, dim=1),
        )


def structured_sequence_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
