from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

GRANULARITY_WIDTH_DESCRIPTOR_INDEX = 5
GRANULARITY_START_DESCRIPTOR_INDEX = 11


class DemandAxis(StrEnum):
    MAGNITUDE = "magnitude_factorization"
    GRANULARITY = "value_granularity"
    ADDRESS = "address_decoupling"
    STATE_CONDITIONED = "state_conditioning"


@dataclass(slots=True)
class ControlLatticeBatch:
    state: torch.Tensor
    descriptor: torch.Tensor
    old_value: torch.Tensor
    new_value: torch.Tensor
    erase_address: torch.Tensor
    write_address: torch.Tensor
    erase_mask: torch.Tensor
    write_mask: torch.Tensor
    target: torch.Tensor
    family_id: torch.Tensor


def _one_hot(indices: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, num_classes=size).to(torch.float32)


def generate_control_lattice_batch(
    *,
    family: DemandAxis,
    batch_size: int,
    slots: int,
    value_dim: int,
    generator: torch.Generator,
    device: torch.device,
) -> ControlLatticeBatch:
    state = torch.randn(batch_size, slots, value_dim, generator=generator) * 0.15
    erase_address = torch.randint(slots, (batch_size,), generator=generator)
    write_address = erase_address.clone()
    old_value = state[torch.arange(batch_size), erase_address].clone()
    new_value = torch.randn(batch_size, value_dim, generator=generator)
    new_value = new_value / new_value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    erase_mask = torch.ones(batch_size, value_dim)
    write_mask = torch.ones(batch_size, value_dim)

    # Descriptor is shared across controller classes.  It describes the update
    # demand but not the target state itself.
    descriptor = torch.zeros(batch_size, 12)
    descriptor[:, 0] = 1.0

    if family is DemandAxis.MAGNITUDE:
        operation = torch.randint(4, (batch_size,), generator=generator)
        # PRESERVE, ADD, INVALIDATE, SUPERSEDE corners.
        erase = torch.tensor([0.0, 0.0, 1.0, 1.0])[operation]
        write = torch.tensor([0.0, 1.0, 0.0, 1.0])[operation]
        descriptor[:, 1:5] = _one_hot(operation, 4)
    elif family is DemandAxis.GRANULARITY:
        erase = torch.ones(batch_size)
        write = torch.ones(batch_size)
        mask_width = torch.randint(1, max(2, value_dim // 2), (batch_size,), generator=generator)
        mask_start = torch.empty(batch_size, dtype=torch.long)
        erase_mask.zero_()
        write_mask.zero_()
        for index, width in enumerate(mask_width.tolist()):
            start = int(torch.randint(value_dim - width + 1, (1,), generator=generator))
            mask_start[index] = start
            erase_mask[index, start : start + width] = 1.0
            write_mask[index, start : start + width] = 1.0
        descriptor[:, GRANULARITY_WIDTH_DESCRIPTOR_INDEX] = mask_width.to(torch.float32) / value_dim
        descriptor[:, 6] = 1.0
        descriptor[:, GRANULARITY_START_DESCRIPTOR_INDEX] = mask_start.to(torch.float32) / max(
            value_dim - 1, 1
        )
    elif family is DemandAxis.ADDRESS:
        erase = torch.ones(batch_size)
        write = torch.ones(batch_size)
        offset = torch.randint(1, slots, (batch_size,), generator=generator)
        write_address = (erase_address + offset) % slots
        descriptor[:, 7] = 1.0
        descriptor[:, 8] = offset.to(torch.float32) / slots
    elif family is DemandAxis.STATE_CONDITIONED:
        # The same external descriptor requires different updates depending on
        # a latent bit stored in the affected value.  Only state-aware control
        # receives that old-value read.
        marker = (old_value[:, 0] > 0).to(torch.float32)
        erase = marker
        write = 1.0 - marker
        descriptor[:, 9] = 1.0
        descriptor[:, 10] = 0.5  # deliberately ambiguous without state read
    else:
        raise AssertionError(f"Unhandled family: {family}")

    target = state.clone()
    batch_index = torch.arange(batch_size)
    target[batch_index, erase_address] = (
        target[batch_index, erase_address] - erase[:, None] * erase_mask * old_value
    )
    target[batch_index, write_address] = (
        target[batch_index, write_address] + write[:, None] * write_mask * new_value
    )
    return ControlLatticeBatch(
        state=state.to(device),
        descriptor=descriptor.to(device),
        old_value=old_value.to(device),
        new_value=new_value.to(device),
        erase_address=erase_address.to(device),
        write_address=write_address.to(device),
        erase_mask=erase_mask.to(device),
        write_mask=write_mask.to(device),
        target=target.to(device),
        family_id=torch.full((batch_size,), list(DemandAxis).index(family), device=device),
    )
