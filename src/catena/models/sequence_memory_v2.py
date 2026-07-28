from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from catena.data.transactional_sequence_v2 import TransactionalSequenceInputV2


class SequenceControlV2(StrEnum):
    TIED = "tied"
    DUAL = "dual"


@dataclass(slots=True)
class TransactionalSequenceOutputV2:
    state: torch.Tensor
    erase_gates: torch.Tensor
    write_gates: torch.Tensor


class TransactionalSequenceMemoryV2(nn.Module):
    """Learned-no-op event memory without an oracle update mask.

    The sixth semantic feature contains the observed verification field.  It is
    encoded like every other field; neither it nor target metadata is applied
    as a hard gate in the update path.
    """

    def __init__(
        self,
        *,
        control: SequenceControlV2,
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
        # Both variants register the same two-output head.  Tied control is
        # implemented only as a forward projection.
        self.gate_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        inputs: TransactionalSequenceInputV2,
    ) -> TransactionalSequenceOutputV2:
        state = inputs.initial_state.clone()
        batch_size, sequence_length = inputs.entity_ids.shape
        batch_index = torch.arange(batch_size, device=state.device)
        erase_rows: list[torch.Tensor] = []
        write_rows: list[torch.Tensor] = []

        for time in range(sequence_length):
            entity = inputs.entity_ids[:, time]
            old_id = inputs.old_value_ids[:, time]
            new_id = inputs.new_value_ids[:, time]
            feature = torch.cat(
                [
                    self.entity_embedding(entity),
                    self.value_embedding(old_id),
                    self.value_embedding(new_id),
                    inputs.semantic_features[:, time],
                ],
                dim=-1,
            )
            hidden = self.encoder(feature)
            raw = torch.sigmoid(self.gate_head(hidden))
            if self.control is SequenceControlV2.TIED:
                beta = raw.mean(dim=-1, keepdim=True)
                erase = beta
                write = beta
            else:
                erase = raw[:, 0:1]
                write = raw[:, 1:2]

            old = torch.nn.functional.one_hot(old_id, self.value_vocab).to(state)
            new = torch.nn.functional.one_hot(new_id, self.value_vocab).to(state)
            current = state[batch_index, entity]
            updated = (current - erase * old + write * new).clamp(0.0, 1.0)
            next_state = state.clone()
            next_state[batch_index, entity] = updated
            state = next_state
            erase_rows.append(erase.squeeze(-1))
            write_rows.append(write.squeeze(-1))

        return TransactionalSequenceOutputV2(
            state=state,
            erase_gates=torch.stack(erase_rows, dim=1),
            write_gates=torch.stack(write_rows, dim=1),
        )


def sequence_parameter_count_v2(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
