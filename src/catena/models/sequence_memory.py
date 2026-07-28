from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from catena.data.transactional_sequence import TransactionalSequenceBatch


class SequenceControl(str, Enum):
    TIED = "tied"
    DUAL = "dual"


class TransactionalSequenceMemory(nn.Module):
    """Event-sequence bridge with a shared semantic encoder and constrained update head.

    Entity addressing and old/new candidates are still oracle structured inputs.
    The experiment therefore tests persistence, repeated updates and control
    factorization, not full language understanding or learned addressing.
    """

    def __init__(
        self,
        *,
        control: SequenceControl,
        num_entities: int,
        value_vocab: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.control = control
        self.value_vocab = int(value_vocab)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.value_embedding = nn.Embedding(value_vocab, embedding_dim)
        input_dim = 3 * embedding_dim + 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Same registered head for both conditions; tied is a projection.
        self.gate_head = nn.Linear(hidden_dim, 2)

    def forward(self, batch: TransactionalSequenceBatch) -> torch.Tensor:
        state = batch.initial_state.clone()
        batch_size, sequence_length = batch.entity_ids.shape
        batch_index = torch.arange(batch_size, device=state.device)
        for time in range(sequence_length):
            entity = batch.entity_ids[:, time]
            old_id = batch.old_value_ids[:, time]
            new_id = batch.new_value_ids[:, time]
            feature = torch.cat(
                [
                    self.entity_embedding(entity),
                    self.value_embedding(old_id),
                    self.value_embedding(new_id),
                    batch.semantic_features[:, time],
                ],
                dim=-1,
            )
            hidden = self.encoder(feature)
            raw = torch.sigmoid(self.gate_head(hidden))
            if self.control is SequenceControl.TIED:
                beta = raw.mean(dim=-1, keepdim=True)
                erase = beta
                write = beta
            else:
                erase = raw[:, 0:1]
                write = raw[:, 1:2]
            verified = batch.semantic_features[:, time, 5:6]
            erase = erase * verified
            write = write * verified
            old = torch.nn.functional.one_hot(old_id, self.value_vocab).to(state)
            new = torch.nn.functional.one_hot(new_id, self.value_vocab).to(state)
            current = state[batch_index, entity]
            updated = current - erase * old + write * new
            mask = batch.update_mask[:, time, None]
            state[batch_index, entity] = torch.where(mask, updated, current).clamp(0.0, 1.0)
        return state


def sequence_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
