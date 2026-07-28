"""Learned repeated-sequence data for the E23 four-axis product poset.

The tensors intentionally reuse the E18 ``SequenceControlLatticeInput`` and
``SequenceControlLatticeBatch`` contracts.  E23 extends the E18 four
single-axis families to their six pairwise combinations plus PRESERVE while
keeping oracle addresses/candidates and the model-visible verified-event bit.
"""

from __future__ import annotations

import hashlib
import math
from typing import cast

import torch

from catena.data.controller_poset import (
    CONTROLLER_AXES,
    DEMAND_FAMILIES,
    demand_required_axes,
)
from catena.data.sequence_control_lattice import (
    SequenceControlLatticeBatch,
    SequenceControlLatticeInput,
    demand_feature_dim,
)


def _stream_seed(seed: int, stream: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    digest = hashlib.sha256(f"{seed}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, stream: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_stream_seed(seed, stream))


def _normalized(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.randn(*shape, generator=generator)
    return cast(
        torch.Tensor,
        values / values.norm(dim=-1, keepdim=True).clamp_min(1.0e-8),
    )


def _verified_positions(*, updates: int, gap_events: int) -> torch.Tensor:
    if updates == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.arange(1, updates, dtype=torch.long) + gap_events,
        )
    )


def _contiguous_masks(
    *,
    batch_size: int,
    updates: int,
    value_dim: int,
    seed: int,
) -> torch.Tensor:
    widths = torch.randint(
        1,
        value_dim,
        (batch_size, updates),
        generator=_generator(seed, "value-mask-width"),
    )
    uniform = torch.rand(
        batch_size,
        updates,
        generator=_generator(seed, "value-mask-start"),
    )
    starts = (uniform * (value_dim - widths + 1)).to(torch.long)
    coordinates = torch.arange(value_dim).view(1, 1, value_dim)
    return (
        (coordinates >= starts.unsqueeze(-1)) & (coordinates < (starts + widths).unsqueeze(-1))
    ).to(torch.float32)


def _validate(
    *,
    demand_family: str,
    intensity: float,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
) -> None:
    if demand_family not in DEMAND_FAMILIES:
        raise ValueError(f"unknown E23 demand family: {demand_family}")
    if not math.isfinite(intensity) or not 0.0 < intensity <= 1.0:
        raise ValueError("intensity must lie in (0, 1]")
    if batch_size <= 0 or num_entities < 2 or value_dim < 2:
        raise ValueError("invalid E23 batch or state dimensions")
    if updates <= 0 or gap_events < 0:
        raise ValueError("invalid E23 sequence cell")


def generate_product_poset_sequence_batch(
    *,
    demand_family: str,
    intensity: float,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    seed: int,
    device: torch.device,
) -> SequenceControlLatticeBatch:
    """Generate one deterministic E18-compatible learned application batch."""

    _validate(
        demand_family=demand_family,
        intensity=intensity,
        batch_size=batch_size,
        num_entities=num_entities,
        value_dim=value_dim,
        updates=updates,
        gap_events=gap_events,
    )
    required = set(demand_required_axes(demand_family))
    initial = (
        torch.randn(
            batch_size,
            num_entities,
            value_dim,
            generator=_generator(seed, "base-initial-state"),
        )
        * 0.15
    )
    erase_base = torch.randint(
        num_entities,
        (batch_size, updates),
        generator=_generator(seed, "base-erase-address"),
    )
    write_base = erase_base.clone()
    if "address" in required:
        offset = torch.randint(
            1,
            num_entities,
            (batch_size, updates),
            generator=_generator(seed, "base-write-offset"),
        )
        write_base = (erase_base + offset) % num_entities
    candidates_base = _normalized(
        (batch_size, updates, value_dim),
        generator=_generator(seed, "base-candidates"),
    )
    masks_base = (
        _contiguous_masks(
            batch_size=batch_size,
            updates=updates,
            value_dim=value_dim,
            seed=seed,
        )
        if "value" in required
        else torch.ones(batch_size, updates, value_dim)
    )
    visible_operations = torch.full(
        (batch_size, updates),
        3,
        dtype=torch.long,
    )
    if demand_family == "preserve":
        visible_operations.zero_()
    elif "magnitude" in required and "conditioning" not in required:
        # ADD or INVALIDATE: the event exposes which asymmetric magnitude is
        # required, exactly as the E18 oracle-demand descriptor does.
        visible_operations = 1 + torch.randint(
            2,
            (batch_size, updates),
            generator=_generator(seed, "base-magnitude-operation"),
        )

    target = initial.clone()
    affected = torch.zeros(batch_size, num_entities, dtype=torch.bool)
    erase_targets = torch.zeros(batch_size, updates)
    write_targets = torch.zeros(batch_size, updates)
    batch_index = torch.arange(batch_size)
    for update_index in range(updates):
        erase_address = erase_base[:, update_index]
        write_address = write_base[:, update_index]
        old = target[batch_index, erase_address].clone()
        candidate = candidates_base[:, update_index]
        if demand_family == "preserve":
            erase = torch.zeros(batch_size)
            write = torch.zeros(batch_size)
        elif "conditioning" in required:
            state_branch = old[:, 0] > 0.0
            if "magnitude" in required:
                erase = (~state_branch).to(torch.float32) * float(intensity)
                write = state_branch.to(torch.float32) * float(intensity)
            else:
                symmetric = state_branch.to(torch.float32) * float(intensity)
                erase = symmetric
                write = symmetric
        elif "magnitude" in required:
            operation = visible_operations[:, update_index]
            erase = (operation == 2).to(torch.float32) * float(intensity)
            write = (operation == 1).to(torch.float32) * float(intensity)
        else:
            erase = torch.full((batch_size,), float(intensity))
            write = torch.full((batch_size,), float(intensity))
        erase_targets[:, update_index] = erase
        write_targets[:, update_index] = write
        mask = masks_base[:, update_index]
        erased = old - erase[:, None] * mask * old
        next_state = target.clone()
        next_state[batch_index, erase_address] = erased
        same = erase_address == write_address
        different = ~same
        if bool(different.any()):
            next_state[batch_index[different], write_address[different]] = (
                target[batch_index[different], write_address[different]]
                + write[different, None] * mask[different] * candidate[different]
            )
        if bool(same.any()):
            next_state[batch_index[same], erase_address[same]] = (
                erased[same] + write[same, None] * mask[same] * candidate[same]
            )
        target = next_state
        affected[batch_index, erase_address] = True
        affected[batch_index, write_address] = True

    verified_features = torch.zeros(
        batch_size,
        updates,
        demand_feature_dim(value_dim),
    )
    axis_bits = torch.tensor(
        [float(axis in required) * float(intensity) for axis in CONTROLLER_AXES]
    )
    verified_features[:, :, :4] = axis_bits
    if "conditioning" not in required:
        verified_features[:, :, 4:8] = torch.nn.functional.one_hot(
            visible_operations,
            num_classes=4,
        ).to(torch.float32)
    verified_features[:, :, 8 : 8 + value_dim] = masks_base
    verified_features[:, :, -1] = 1.0

    total_events = updates + gap_events
    erase_ids = torch.empty(batch_size, total_events, dtype=torch.long)
    write_ids = torch.empty(batch_size, total_events, dtype=torch.long)
    candidates = torch.empty(batch_size, total_events, value_dim)
    features = torch.zeros(
        batch_size,
        total_events,
        demand_feature_dim(value_dim),
    )
    update_mask = torch.zeros(batch_size, total_events, dtype=torch.bool)
    positions = _verified_positions(updates=updates, gap_events=gap_events)
    erase_ids[:, positions] = erase_base
    write_ids[:, positions] = write_base
    candidates[:, positions] = candidates_base
    features[:, positions] = verified_features
    update_mask[:, positions] = True
    if gap_events:
        gap_slice = slice(1, gap_events + 1)
        erase_ids[:, gap_slice] = torch.randint(
            num_entities,
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-erase-address"),
        ).transpose(0, 1)
        write_ids[:, gap_slice] = torch.randint(
            num_entities,
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-write-address"),
        ).transpose(0, 1)
        candidates[:, gap_slice] = _normalized(
            (gap_events, batch_size, value_dim),
            generator=_generator(seed, "distractor-candidates"),
        ).transpose(0, 1)
        # Axis/intensity descriptors remain visible, but verified=0 is the
        # E18 active-path signal that must suppress every distractor update.
        features[:, gap_slice, :4] = axis_bits

    batch = SequenceControlLatticeBatch(
        inputs=SequenceControlLatticeInput(
            initial_state=initial.to(device),
            erase_entity_ids=erase_ids.to(device),
            write_entity_ids=write_ids.to(device),
            candidate_values=candidates.to(device),
            demand_features=features.to(device),
        ),
        update_mask=update_mask.to(device),
        target_state=target.to(device),
        affected_entities=affected.to(device),
        # The dataclass annotation is the E18 enum, but the runtime field is
        # metadata only and E23 has a larger registered family set.
        demand_family=demand_family,  # type: ignore[arg-type]
    )
    # Private audit fields are deliberately not attached to model inputs.
    del erase_targets, write_targets
    return batch
