from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class SequenceOperation(IntEnum):
    PRESERVE = 0
    ADD = 1
    INVALIDATE = 2
    SUPERSEDE = 3


@dataclass(slots=True)
class TransactionalSequenceBatch:
    initial_state: torch.Tensor
    entity_ids: torch.Tensor
    old_value_ids: torch.Tensor
    new_value_ids: torch.Tensor
    semantic_features: torch.Tensor
    update_mask: torch.Tensor
    target_state: torch.Tensor
    affected_entities: torch.Tensor
    operations: torch.Tensor


def _semantic_features(operation: torch.Tensor) -> torch.Tensor:
    """Raw relation fields; no operation one-hot is exposed."""
    batch = operation.shape[0]
    features = torch.zeros(batch, 6, dtype=torch.float32)
    preserve = operation == int(SequenceOperation.PRESERVE)
    add = operation == int(SequenceOperation.ADD)
    invalidate = operation == int(SequenceOperation.INVALIDATE)
    supersede = operation == int(SequenceOperation.SUPERSEDE)
    features[:, 0] = (preserve | add).to(torch.float32)  # prior valid afterwards
    features[:, 1] = (add | supersede).to(torch.float32)  # incoming valid afterwards
    features[:, 2] = 1.0  # same semantic scope
    features[:, 3] = (invalidate | supersede).to(torch.float32)  # newer version boundary
    features[:, 4] = add.to(torch.float32)  # additive relation
    features[:, 5] = 1.0  # verified event
    return features


def generate_transactional_sequence_batch(
    *,
    batch_size: int,
    num_entities: int,
    value_vocab: int,
    updates: int,
    gap_events: int,
    generator: torch.Generator,
    device: torch.device,
) -> TransactionalSequenceBatch:
    if value_vocab < 4:
        raise ValueError("value_vocab must be at least 4")
    initial_values = torch.randint(
        value_vocab, (batch_size, num_entities), generator=generator
    )
    initial_state = torch.nn.functional.one_hot(
        initial_values, num_classes=value_vocab
    ).to(torch.float32)
    target = initial_state.clone()

    total_events = updates + gap_events
    entity_ids = torch.randint(
        num_entities, (batch_size, total_events), generator=generator
    )
    old_ids = torch.zeros(batch_size, total_events, dtype=torch.long)
    new_ids = torch.randint(
        value_vocab, (batch_size, total_events), generator=generator
    )
    semantic = torch.zeros(batch_size, total_events, 6)
    update_mask = torch.zeros(batch_size, total_events, dtype=torch.bool)
    operations = torch.full(
        (batch_size, total_events),
        fill_value=-1,
        dtype=torch.long,
    )
    affected = torch.zeros(batch_size, num_entities, dtype=torch.bool)

    for time in range(updates):
        operation = torch.randint(4, (batch_size,), generator=generator)
        entity = entity_ids[:, time]
        batch_index = torch.arange(batch_size)
        current_row = target[batch_index, entity]
        # Select one currently active old value.  The state begins one-hot but
        # ADD can make it multi-hot; choose deterministically from the first max.
        old = torch.argmax(current_row, dim=-1)
        new = new_ids[:, time]
        conflict = new == old
        new[conflict] = (new[conflict] + 1) % value_vocab
        old_ids[:, time] = old
        semantic[:, time] = _semantic_features(operation)
        update_mask[:, time] = True
        operations[:, time] = operation
        affected[batch_index, entity] = True

        old_onehot = torch.nn.functional.one_hot(old, value_vocab).to(torch.float32)
        new_onehot = torch.nn.functional.one_hot(new, value_vocab).to(torch.float32)
        preserve = operation == int(SequenceOperation.PRESERVE)
        add = operation == int(SequenceOperation.ADD)
        invalidate = operation == int(SequenceOperation.INVALIDATE)
        supersede = operation == int(SequenceOperation.SUPERSEDE)
        erase = (invalidate | supersede).to(torch.float32)[:, None]
        write = (add | supersede).to(torch.float32)[:, None]
        updated = current_row - erase * old_onehot + write * new_onehot
        target[batch_index, entity] = updated.clamp(0.0, 1.0)
        # Preserve is explicit for readability; updated already equals current.
        target[batch_index[preserve], entity[preserve]] = current_row[preserve]

    # Gap events are unverified distractors.  Their semantic fields share
    # surface statistics but the verification bit is zero.
    for time in range(updates, total_events):
        entity = entity_ids[:, time]
        batch_index = torch.arange(batch_size)
        old_ids[:, time] = torch.argmax(target[batch_index, entity], dim=-1)
        pseudo_op = torch.randint(4, (batch_size,), generator=generator)
        semantic[:, time] = _semantic_features(pseudo_op)
        semantic[:, time, 5] = 0.0

    return TransactionalSequenceBatch(
        initial_state=initial_state.to(device),
        entity_ids=entity_ids.to(device),
        old_value_ids=old_ids.to(device),
        new_value_ids=new_ids.to(device),
        semantic_features=semantic.to(device),
        update_mask=update_mask.to(device),
        target_state=target.to(device),
        affected_entities=affected.to(device),
        operations=operations.to(device),
    )
