from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

import torch


class SequenceOperationV2(IntEnum):
    PRESERVE = 0
    ADD = 1
    INVALIDATE = 2
    SUPERSEDE = 3


@dataclass(slots=True)
class TransactionalSequenceInputV2:
    """Model-visible sequence fields.

    The target-only ``update_mask`` is deliberately absent.  The verification
    bit remains the sixth semantic feature and must be interpreted by the
    learned controller rather than applied as an oracle mask.
    """

    initial_state: torch.Tensor
    entity_ids: torch.Tensor
    old_value_ids: torch.Tensor
    new_value_ids: torch.Tensor
    semantic_features: torch.Tensor


@dataclass(slots=True)
class TransactionalSequenceBatchV2:
    inputs: TransactionalSequenceInputV2
    update_mask: torch.Tensor
    target_state: torch.Tensor
    affected_entities: torch.Tensor
    operations: torch.Tensor


def _stream_seed(seed: int, stream: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(f"{seed}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(seed: int, stream: str) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_stream_seed(seed, stream))


def _semantic_features(operation: torch.Tensor) -> torch.Tensor:
    """Structured relation fields without an operation one-hot."""
    batch = operation.shape[0]
    features = torch.zeros(batch, 6, dtype=torch.float32)
    preserve = operation == int(SequenceOperationV2.PRESERVE)
    add = operation == int(SequenceOperationV2.ADD)
    invalidate = operation == int(SequenceOperationV2.INVALIDATE)
    supersede = operation == int(SequenceOperationV2.SUPERSEDE)
    features[:, 0] = (preserve | add).to(torch.float32)
    features[:, 1] = (add | supersede).to(torch.float32)
    features[:, 2] = 1.0
    features[:, 3] = (invalidate | supersede).to(torch.float32)
    features[:, 4] = add.to(torch.float32)
    features[:, 5] = 1.0
    return features


def _validate_dimensions(
    *,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_entities <= 0:
        raise ValueError("num_entities must be positive")
    if value_vocab < 4:
        raise ValueError("value_vocab must be at least 4")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if gap_events < 0:
        raise ValueError("gap_events must be non-negative")


def _verified_event_indices(*, updates: int, gap_events: int) -> torch.Tensor:
    """Place one exact-length distractor block after the first update."""
    first = torch.zeros(1, dtype=torch.long)
    if updates == 1:
        return first
    remaining = torch.arange(1, updates, dtype=torch.long) + gap_events
    return torch.cat([first, remaining])


def generate_transactional_sequence_batch_v2(
    *,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    seed: int,
    device: torch.device,
) -> TransactionalSequenceBatchV2:
    """Generate paired sequences with a causally active distractor interval.

    Base-state and verified-update streams use domain-separated RNG streams
    whose draws do not depend on ``gap_events``.  Distractor fields use their
    own streams, so a shorter gap is an exact prefix of a longer gap generated
    with the same seed.
    """
    _validate_dimensions(
        batch_size=batch_size,
        num_entities=num_entities,
        value_vocab=value_vocab,
        updates=updates,
        gap_events=gap_events,
    )

    initial_values = torch.randint(
        value_vocab,
        (batch_size, num_entities),
        generator=_generator(seed, "base-initial-values"),
    )
    initial_state = torch.nn.functional.one_hot(
        initial_values, num_classes=value_vocab
    ).to(torch.float32)
    target = initial_state.clone()
    base_entities = torch.randint(
        num_entities,
        (batch_size, updates),
        generator=_generator(seed, "base-update-entities"),
    )
    base_new_ids = torch.randint(
        value_vocab,
        (batch_size, updates),
        generator=_generator(seed, "base-update-new-values"),
    )
    base_operations = torch.randint(
        len(SequenceOperationV2),
        (batch_size, updates),
        generator=_generator(seed, "base-update-operations"),
    )
    base_old_ids = torch.zeros(batch_size, updates, dtype=torch.long)
    base_semantic = torch.zeros(batch_size, updates, 6)
    affected = torch.zeros(batch_size, num_entities, dtype=torch.bool)
    batch_index = torch.arange(batch_size)
    state_after_first: torch.Tensor | None = None

    for update_index in range(updates):
        operation = base_operations[:, update_index]
        entity = base_entities[:, update_index]
        current_row = target[batch_index, entity]
        old = torch.argmax(current_row, dim=-1)
        new = base_new_ids[:, update_index].clone()
        conflict = new == old
        new[conflict] = (new[conflict] + 1) % value_vocab
        base_new_ids[:, update_index] = new
        base_old_ids[:, update_index] = old
        base_semantic[:, update_index] = _semantic_features(operation)
        affected[batch_index, entity] = True

        old_onehot = torch.nn.functional.one_hot(old, value_vocab).to(torch.float32)
        new_onehot = torch.nn.functional.one_hot(new, value_vocab).to(torch.float32)
        invalidate = operation == int(SequenceOperationV2.INVALIDATE)
        supersede = operation == int(SequenceOperationV2.SUPERSEDE)
        add = operation == int(SequenceOperationV2.ADD)
        erase = (invalidate | supersede).to(torch.float32)[:, None]
        write = (add | supersede).to(torch.float32)[:, None]
        updated = current_row - erase * old_onehot + write * new_onehot
        target[batch_index, entity] = updated.clamp(0.0, 1.0)
        if update_index == 0:
            state_after_first = target.clone()

    if state_after_first is None:  # pragma: no cover - guarded by updates > 0
        raise RuntimeError("state after first update was not constructed")

    total_events = updates + gap_events
    event_entities = torch.empty(batch_size, total_events, dtype=torch.long)
    event_old_ids = torch.empty(batch_size, total_events, dtype=torch.long)
    event_new_ids = torch.empty(batch_size, total_events, dtype=torch.long)
    event_semantic = torch.empty(batch_size, total_events, 6)
    update_mask = torch.zeros(batch_size, total_events, dtype=torch.bool)
    operations = torch.full(
        (batch_size, total_events),
        fill_value=-1,
        dtype=torch.long,
    )

    verified_indices = _verified_event_indices(
        updates=updates,
        gap_events=gap_events,
    )
    event_entities[:, verified_indices] = base_entities
    event_old_ids[:, verified_indices] = base_old_ids
    event_new_ids[:, verified_indices] = base_new_ids
    event_semantic[:, verified_indices] = base_semantic
    update_mask[:, verified_indices] = True
    operations[:, verified_indices] = base_operations

    if gap_events:
        gap_slice = slice(1, 1 + gap_events)
        distractor_entities = torch.randint(
            num_entities,
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-entities"),
        ).transpose(0, 1).contiguous()
        distractor_new_ids = torch.randint(
            value_vocab,
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-new-values"),
        ).transpose(0, 1).contiguous()
        pseudo_operations = torch.randint(
            len(SequenceOperationV2),
            (gap_events, batch_size),
            generator=_generator(seed, "distractor-pseudo-operations"),
        ).transpose(0, 1).contiguous()
        distractor_old_ids = torch.argmax(
            state_after_first[
                batch_index[:, None],
                distractor_entities,
            ],
            dim=-1,
        )
        distractor_semantic = _semantic_features(pseudo_operations.reshape(-1)).reshape(
            batch_size,
            gap_events,
            6,
        )
        distractor_semantic[:, :, 5] = 0.0
        event_entities[:, gap_slice] = distractor_entities
        event_old_ids[:, gap_slice] = distractor_old_ids
        event_new_ids[:, gap_slice] = distractor_new_ids
        event_semantic[:, gap_slice] = distractor_semantic

    inputs = TransactionalSequenceInputV2(
        initial_state=initial_state.to(device),
        entity_ids=event_entities.to(device),
        old_value_ids=event_old_ids.to(device),
        new_value_ids=event_new_ids.to(device),
        semantic_features=event_semantic.to(device),
    )
    return TransactionalSequenceBatchV2(
        inputs=inputs,
        update_mask=update_mask.to(device),
        target_state=target.to(device),
        affected_entities=affected.to(device),
        operations=operations.to(device),
    )


def sequence_model_input_v2(
    batch: TransactionalSequenceBatchV2,
    *,
    activate_distractor_verified: bool = False,
) -> TransactionalSequenceInputV2:
    """Return model-visible fields, optionally activating only distractor verification."""
    semantic = batch.inputs.semantic_features
    if activate_distractor_verified:
        semantic = semantic.clone()
        semantic[:, :, 5] = torch.where(
            batch.update_mask,
            semantic[:, :, 5],
            torch.ones_like(semantic[:, :, 5]),
        )
    return TransactionalSequenceInputV2(
        initial_state=batch.inputs.initial_state,
        entity_ids=batch.inputs.entity_ids,
        old_value_ids=batch.inputs.old_value_ids,
        new_value_ids=batch.inputs.new_value_ids,
        semantic_features=semantic,
    )


def _hash_tensor(digest: hashlib._Hash, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def base_transaction_digest_v2(batch: TransactionalSequenceBatchV2) -> str:
    """Hash only initial state, verified update stream and final target."""
    mask = batch.update_mask
    counts = mask.sum(dim=1)
    if counts.numel() == 0 or not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("every batch row must contain the same positive update count")
    updates = int(counts[0].item())
    if updates <= 0:
        raise ValueError("base transaction digest requires at least one update")
    batch_size = mask.shape[0]

    def verified_rows(value: torch.Tensor) -> torch.Tensor:
        trailing = value.shape[2:]
        return value[mask].reshape(batch_size, updates, *trailing)

    digest = hashlib.sha256()
    _hash_tensor(digest, "initial_state", batch.inputs.initial_state)
    _hash_tensor(digest, "entity_ids", verified_rows(batch.inputs.entity_ids))
    _hash_tensor(digest, "old_value_ids", verified_rows(batch.inputs.old_value_ids))
    _hash_tensor(digest, "new_value_ids", verified_rows(batch.inputs.new_value_ids))
    _hash_tensor(
        digest,
        "semantic_features",
        verified_rows(batch.inputs.semantic_features),
    )
    _hash_tensor(digest, "operations", verified_rows(batch.operations))
    _hash_tensor(digest, "target_state", batch.target_state)
    return digest.hexdigest()
