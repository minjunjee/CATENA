from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import torch


class SequenceDemandFamily(StrEnum):
    MAGNITUDE = "magnitude_factorization"
    VALUE_GRANULARITY = "value_granularity"
    ADDRESS_DECOUPLING = "address_decoupling"
    STATE_CONDITIONING = "state_conditioning"


@dataclass(slots=True)
class SequenceControlLatticeInput:
    """Model-visible fields; target-only update metadata is deliberately absent."""

    initial_state: torch.Tensor
    erase_entity_ids: torch.Tensor
    write_entity_ids: torch.Tensor
    candidate_values: torch.Tensor
    demand_features: torch.Tensor


@dataclass(slots=True)
class SequenceControlLatticeBatch:
    inputs: SequenceControlLatticeInput
    update_mask: torch.Tensor
    target_state: torch.Tensor
    affected_entities: torch.Tensor
    demand_family: SequenceDemandFamily


def demand_feature_dim(value_dim: int) -> int:
    if value_dim <= 0:
        raise ValueError("value_dim must be positive")
    # family one-hot + magnitude operation one-hot + channel mask + verified bit
    return 4 + 4 + int(value_dim) + 1


def _stream_seed(seed: int, stream: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(f"{seed}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, stream: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_stream_seed(seed, stream))


def _normalized_candidates(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.randn(*shape, generator=generator)
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _family_features(
    *,
    family: SequenceDemandFamily,
    batch_size: int,
    events: int,
    value_dim: int,
    operation: torch.Tensor | None,
    channel_mask: torch.Tensor | None,
    verified: float,
) -> torch.Tensor:
    features = torch.zeros(
        batch_size,
        events,
        demand_feature_dim(value_dim),
        dtype=torch.float32,
    )
    family_index = list(SequenceDemandFamily).index(family)
    features[:, :, family_index] = 1.0
    if operation is not None:
        features[:, :, 4:8] = torch.nn.functional.one_hot(
            operation,
            num_classes=4,
        ).to(torch.float32)
    if channel_mask is not None:
        features[:, :, 8 : 8 + value_dim] = channel_mask
    features[:, :, -1] = float(verified)
    return features


def _contiguous_masks(
    *,
    batch_size: int,
    events: int,
    value_dim: int,
    width_generator: torch.Generator,
    start_generator: torch.Generator,
) -> torch.Tensor:
    if value_dim < 2:
        raise ValueError("value_dim must be at least 2 for granularity demands")
    widths = torch.randint(
        1,
        value_dim,
        (batch_size, events),
        generator=width_generator,
    )
    start_uniform = torch.rand(
        batch_size,
        events,
        generator=start_generator,
    )
    starts = (
        start_uniform * (value_dim - widths + 1).to(start_uniform.dtype)
    ).to(torch.long)
    coordinates = torch.arange(value_dim).view(1, 1, value_dim)
    return (
        (coordinates >= starts.unsqueeze(-1))
        & (coordinates < (starts + widths).unsqueeze(-1))
    ).to(torch.float32)


def _verified_positions(*, updates: int, gap_events: int) -> torch.Tensor:
    if updates == 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            torch.arange(1, updates, dtype=torch.long) + gap_events,
        ]
    )


def _validate_dimensions(
    *,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_entities < 2:
        raise ValueError("num_entities must be at least 2")
    if value_dim < 2:
        raise ValueError("value_dim must be at least 2")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if gap_events < 0:
        raise ValueError("gap_events must be non-negative")


def _apply_verified_targets(
    *,
    initial_state: torch.Tensor,
    family: SequenceDemandFamily,
    erase_entities: torch.Tensor,
    write_entities: torch.Tensor,
    candidates: torch.Tensor,
    operations: torch.Tensor | None,
    channel_masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = initial_state.clone()
    batch_size, updates = erase_entities.shape
    value_dim = initial_state.shape[-1]
    batch_index = torch.arange(batch_size)
    affected = torch.zeros(
        batch_size,
        initial_state.shape[1],
        dtype=torch.bool,
    )
    state_after_first: torch.Tensor | None = None

    for update_index in range(updates):
        erase_address = erase_entities[:, update_index]
        write_address = write_entities[:, update_index]
        old_value = target[batch_index, erase_address].clone()
        candidate = candidates[:, update_index]
        erase_mask = torch.ones(batch_size, value_dim)
        write_mask = torch.ones(batch_size, value_dim)

        if family is SequenceDemandFamily.MAGNITUDE:
            if operations is None:
                raise RuntimeError("magnitude demand requires operations")
            operation = operations[:, update_index]
            erase = ((operation == 2) | (operation == 3)).to(torch.float32)
            write = ((operation == 1) | (operation == 3)).to(torch.float32)
        elif family is SequenceDemandFamily.VALUE_GRANULARITY:
            if channel_masks is None:
                raise RuntimeError("granularity demand requires channel masks")
            erase = torch.ones(batch_size)
            write = torch.ones(batch_size)
            erase_mask = channel_masks[:, update_index]
            write_mask = channel_masks[:, update_index]
        elif family is SequenceDemandFamily.ADDRESS_DECOUPLING:
            erase = torch.ones(batch_size)
            write = torch.ones(batch_size)
        elif family is SequenceDemandFamily.STATE_CONDITIONING:
            marker = (old_value[:, 0] > 0.0).to(torch.float32)
            erase = marker
            write = 1.0 - marker
        else:  # pragma: no cover - exhaustive enum guard
            raise AssertionError(f"Unhandled family: {family}")

        erased = old_value - erase[:, None] * erase_mask * old_value
        same_address = erase_address == write_address
        next_state = target.clone()
        next_state[batch_index, erase_address] = erased
        different = ~same_address
        if different.any():
            next_state[
                batch_index[different],
                write_address[different],
            ] = (
                target[batch_index[different], write_address[different]]
                + write[different, None]
                * write_mask[different]
                * candidate[different]
            )
        if same_address.any():
            next_state[
                batch_index[same_address],
                erase_address[same_address],
            ] = (
                erased[same_address]
                + write[same_address, None]
                * write_mask[same_address]
                * candidate[same_address]
            )
        target = next_state
        affected[batch_index, erase_address] = True
        affected[batch_index, write_address] = True
        if update_index == 0:
            state_after_first = target.clone()

    if state_after_first is None:  # pragma: no cover - updates > 0 is validated
        raise RuntimeError("state after first verified update was not constructed")
    return target, affected, state_after_first


def generate_sequence_control_lattice_batch(
    *,
    family: SequenceDemandFamily,
    batch_size: int,
    num_entities: int,
    value_dim: int,
    updates: int,
    gap_events: int,
    seed: int,
    device: torch.device,
) -> SequenceControlLatticeBatch:
    """Generate paired sequence demands with a model-visible distractor block."""

    _validate_dimensions(
        batch_size=batch_size,
        num_entities=num_entities,
        value_dim=value_dim,
        updates=updates,
        gap_events=gap_events,
    )
    initial_state = (
        torch.randn(
            batch_size,
            num_entities,
            value_dim,
            generator=_generator(seed, "base-initial-state"),
        )
        * 0.15
    )
    base_erase = torch.randint(
        num_entities,
        (batch_size, updates),
        generator=_generator(seed, "base-erase-address"),
    )
    base_write = base_erase.clone()
    if family is SequenceDemandFamily.ADDRESS_DECOUPLING:
        offset = torch.randint(
            1,
            num_entities,
            (batch_size, updates),
            generator=_generator(seed, "base-write-offset"),
        )
        base_write = (base_erase + offset) % num_entities
    base_candidates = _normalized_candidates(
        (batch_size, updates, value_dim),
        generator=_generator(seed, "base-candidates"),
    )
    base_operations: torch.Tensor | None = None
    base_masks: torch.Tensor | None = None
    if family is SequenceDemandFamily.MAGNITUDE:
        base_operations = torch.randint(
            4,
            (batch_size, updates),
            generator=_generator(seed, "base-magnitude-operation"),
        )
    if family is SequenceDemandFamily.VALUE_GRANULARITY:
        base_masks = _contiguous_masks(
            batch_size=batch_size,
            events=updates,
            value_dim=value_dim,
            width_generator=_generator(seed, "base-channel-mask-width"),
            start_generator=_generator(seed, "base-channel-mask-start"),
        )
    base_features = _family_features(
        family=family,
        batch_size=batch_size,
        events=updates,
        value_dim=value_dim,
        operation=base_operations,
        channel_mask=base_masks,
        verified=1.0,
    )
    target, affected, _state_after_first = _apply_verified_targets(
        initial_state=initial_state,
        family=family,
        erase_entities=base_erase,
        write_entities=base_write,
        candidates=base_candidates,
        operations=base_operations,
        channel_masks=base_masks,
    )

    total_events = updates + gap_events
    erase_entities = torch.empty(batch_size, total_events, dtype=torch.long)
    write_entities = torch.empty(batch_size, total_events, dtype=torch.long)
    candidates = torch.empty(batch_size, total_events, value_dim)
    features = torch.empty(
        batch_size,
        total_events,
        demand_feature_dim(value_dim),
    )
    update_mask = torch.zeros(batch_size, total_events, dtype=torch.bool)
    verified_positions = _verified_positions(
        updates=updates,
        gap_events=gap_events,
    )
    erase_entities[:, verified_positions] = base_erase
    write_entities[:, verified_positions] = base_write
    candidates[:, verified_positions] = base_candidates
    features[:, verified_positions] = base_features
    update_mask[:, verified_positions] = True

    if gap_events:
        gap_slice = slice(1, gap_events + 1)
        distractor_erase = torch.randint(
            num_entities,
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-erase-address"),
        ).transpose(0, 1).contiguous()
        distractor_write = distractor_erase.clone()
        if family is SequenceDemandFamily.ADDRESS_DECOUPLING:
            distractor_offset = torch.randint(
                1,
                num_entities,
                (gap_events, batch_size),
                generator=_generator(seed, "distractor-write-offset"),
            ).transpose(0, 1).contiguous()
            distractor_write = (
                distractor_erase + distractor_offset
            ) % num_entities
        distractor_candidates = _normalized_candidates(
            (gap_events, batch_size, value_dim),
            generator=_generator(seed, "distractor-candidates"),
        ).transpose(0, 1).contiguous()
        distractor_operations: torch.Tensor | None = None
        distractor_masks: torch.Tensor | None = None
        if family is SequenceDemandFamily.MAGNITUDE:
            distractor_operations = torch.randint(
                4,
                (gap_events, batch_size),
                generator=_generator(seed, "distractor-magnitude-operation"),
            ).transpose(0, 1).contiguous()
        if family is SequenceDemandFamily.VALUE_GRANULARITY:
            # Generate in event-major order so shorter gaps are exact prefixes.
            event_major_masks = _contiguous_masks(
                batch_size=gap_events,
                events=batch_size,
                value_dim=value_dim,
                width_generator=_generator(
                    seed,
                    "distractor-channel-mask-width",
                ),
                start_generator=_generator(
                    seed,
                    "distractor-channel-mask-start",
                ),
            )
            distractor_masks = event_major_masks.transpose(0, 1).contiguous()
        distractor_features = _family_features(
            family=family,
            batch_size=batch_size,
            events=gap_events,
            value_dim=value_dim,
            operation=distractor_operations,
            channel_mask=distractor_masks,
            verified=0.0,
        )
        erase_entities[:, gap_slice] = distractor_erase
        write_entities[:, gap_slice] = distractor_write
        candidates[:, gap_slice] = distractor_candidates
        features[:, gap_slice] = distractor_features

    inputs = SequenceControlLatticeInput(
        initial_state=initial_state.to(device),
        erase_entity_ids=erase_entities.to(device),
        write_entity_ids=write_entities.to(device),
        candidate_values=candidates.to(device),
        demand_features=features.to(device),
    )
    return SequenceControlLatticeBatch(
        inputs=inputs,
        update_mask=update_mask.to(device),
        target_state=target.to(device),
        affected_entities=affected.to(device),
        demand_family=family,
    )


def sequence_control_lattice_model_input(
    batch: SequenceControlLatticeBatch,
    *,
    activate_distractor_verified: bool = False,
) -> SequenceControlLatticeInput:
    features = batch.inputs.demand_features
    if activate_distractor_verified:
        features = features.clone()
        features[:, :, -1] = torch.where(
            batch.update_mask,
            features[:, :, -1],
            torch.ones_like(features[:, :, -1]),
        )
    return SequenceControlLatticeInput(
        initial_state=batch.inputs.initial_state,
        erase_entity_ids=batch.inputs.erase_entity_ids,
        write_entity_ids=batch.inputs.write_entity_ids,
        candidate_values=batch.inputs.candidate_values,
        demand_features=features,
    )


def _hash_tensor(digest: hashlib._Hash, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())


def base_sequence_control_digest(batch: SequenceControlLatticeBatch) -> str:
    mask = batch.update_mask
    counts = mask.sum(dim=1)
    if counts.numel() == 0 or not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("each row must contain the same positive update count")
    updates = int(counts[0])
    if updates <= 0:
        raise ValueError("base digest requires at least one verified update")
    batch_size = mask.shape[0]

    def verified(value: torch.Tensor) -> torch.Tensor:
        return value[mask].reshape(batch_size, updates, *value.shape[2:])

    digest = hashlib.sha256()
    _hash_tensor(digest, "initial_state", batch.inputs.initial_state)
    _hash_tensor(
        digest,
        "erase_entity_ids",
        verified(batch.inputs.erase_entity_ids),
    )
    _hash_tensor(
        digest,
        "write_entity_ids",
        verified(batch.inputs.write_entity_ids),
    )
    _hash_tensor(
        digest,
        "candidate_values",
        verified(batch.inputs.candidate_values),
    )
    _hash_tensor(
        digest,
        "demand_features",
        verified(batch.inputs.demand_features),
    )
    _hash_tensor(digest, "target_state", batch.target_state)
    return digest.hexdigest()
