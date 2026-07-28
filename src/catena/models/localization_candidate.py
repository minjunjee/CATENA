from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from catena.data.localization_candidate import (
    LocalizationCandidateBatch,
    LocalizationCandidateCondition,
)


class LocalizationCandidateFreedom(StrEnum):
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
class LocalizationCandidateOutput:
    state: torch.Tensor
    erase_address_weights: torch.Tensor
    write_address_weights: torch.Tensor
    erase_candidate: torch.Tensor
    raw_address_logits: torch.Tensor
    raw_candidate: torch.Tensor


class MatchedLocalizationCandidateController(nn.Module):
    """A common maximal surface projected into four control freedoms."""

    def __init__(
        self,
        *,
        freedom: LocalizationCandidateFreedom,
        descriptor_dim: int,
        slots: int,
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
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.address_head = nn.Linear(hidden_dim, 2 * slots)
        self.candidate_head = nn.Linear(hidden_dim, value_dim)

    def forward(
        self,
        batch: LocalizationCandidateBatch,
        condition: LocalizationCandidateCondition,
    ) -> LocalizationCandidateOutput:
        batch_size = batch.state.shape[0]
        hidden = self.encoder(batch.descriptor)
        raw_address_logits = self.address_head(hidden).view(
            batch_size,
            2,
            self.slots,
        )
        raw_candidate = self.candidate_head(hidden)

        if self.freedom.has_separate_address:
            erase_logits = raw_address_logits[:, 0]
            write_logits = raw_address_logits[:, 1]
        else:
            shared_logits = raw_address_logits.mean(dim=1)
            erase_logits = shared_logits
            write_logits = shared_logits

        if condition.uses_oracle_address:
            erase_weights = nn.functional.one_hot(
                batch.erase_address,
                num_classes=self.slots,
            ).to(batch.state.dtype)
            write_weights = nn.functional.one_hot(
                batch.write_address,
                num_classes=self.slots,
            ).to(batch.state.dtype)
        else:
            erase_weights = torch.softmax(
                erase_logits / self.address_temperature,
                dim=-1,
            )
            write_weights = torch.softmax(
                write_logits / self.address_temperature,
                dim=-1,
            )

        if condition.uses_oracle_candidate:
            erase_candidate = batch.old_candidate
        elif self.freedom.has_state_read:
            erase_candidate = torch.einsum(
                "bs,bsv->bv",
                erase_weights,
                batch.state,
            )
        else:
            erase_candidate = raw_candidate

        state = (
            batch.state
            - erase_weights[:, :, None] * erase_candidate[:, None, :]
            + write_weights[:, :, None] * batch.new_candidate[:, None, :]
        )
        return LocalizationCandidateOutput(
            state=state,
            erase_address_weights=erase_weights,
            write_address_weights=write_weights,
            erase_candidate=erase_candidate,
            raw_address_logits=raw_address_logits,
            raw_candidate=raw_candidate,
        )


def localization_candidate_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
