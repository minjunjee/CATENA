from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from catena.data.control_lattice import ControlLatticeBatch


class ControlFreedom(StrEnum):
    TIED = "tied_scalar"
    DUAL = "dual_scalar"
    DIAGONAL = "diagonal_value"
    SEPARATE_ADDRESS = "separate_address"
    STATE_AWARE = "state_aware"


@dataclass(slots=True)
class LatticeOutput:
    state: torch.Tensor
    erase_gate: torch.Tensor
    write_gate: torch.Tensor


class MatchedControlLatticeController(nn.Module):
    """One maximal head with projection constraints defining each control class.

    All variants register the same tensors.  The `freedom` flag only changes
    how the maximal output is projected into a reachable update operator.
    """

    def __init__(
        self,
        *,
        freedom: ControlFreedom,
        descriptor_dim: int,
        value_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.freedom = freedom
        self.value_dim = int(value_dim)
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim + value_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 2 + 2 * value_dim)

    def forward(self, batch: ControlLatticeBatch) -> LatticeOutput:
        batch_size = batch.state.shape[0]
        batch_index = torch.arange(batch_size, device=batch.state.device)
        old_read = batch.state[batch_index, batch.erase_address]
        state_summary = (
            old_read
            if self.freedom is ControlFreedom.STATE_AWARE
            else torch.zeros_like(old_read)
        )
        hidden = self.encoder(torch.cat([batch.descriptor, state_summary], dim=-1))
        raw = self.head(hidden)
        scalars = torch.sigmoid(raw[:, :2])
        vectors = torch.sigmoid(raw[:, 2:]).view(batch_size, 2, self.value_dim)

        if self.freedom is ControlFreedom.TIED:
            beta = scalars.mean(dim=-1, keepdim=True)
            erase_gate = beta
            write_gate = beta
        elif self.freedom is ControlFreedom.DUAL:
            erase_gate = scalars[:, 0:1]
            write_gate = scalars[:, 1:2]
        else:
            erase_gate = vectors[:, 0]
            write_gate = vectors[:, 1]

        erase_address = batch.erase_address
        write_address = (
            batch.write_address
            if self.freedom in {ControlFreedom.SEPARATE_ADDRESS, ControlFreedom.STATE_AWARE}
            else batch.erase_address
        )
        state = batch.state.clone()
        if erase_gate.ndim == 2 and erase_gate.shape[-1] == 1:
            erase_term = erase_gate * batch.old_value
            write_term = write_gate * batch.new_value
        else:
            erase_term = erase_gate * batch.old_value
            write_term = write_gate * batch.new_value
        state[batch_index, erase_address] = state[batch_index, erase_address] - erase_term
        state[batch_index, write_address] = state[batch_index, write_address] + write_term
        return LatticeOutput(state=state, erase_gate=erase_gate, write_gate=write_gate)


def controller_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
