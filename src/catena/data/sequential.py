from __future__ import annotations

from dataclasses import dataclass

import torch

from catena.core.schema import MemoryEpisode, Operation


@dataclass(slots=True)
class SequentialMemory:
    keys: torch.Tensor
    initial_values: torch.Tensor
    initial_state: torch.Tensor
    affected_index: int
    current_value: torch.Tensor
    canonical_state: torch.Tensor


def initialize_sequential_memory(
    *,
    seed: int,
    num_associations: int,
    key_dim: int,
    value_dim: int,
    key_correlation: float,
) -> SequentialMemory:
    generator = torch.Generator().manual_seed(seed)
    raw_keys = torch.randn(num_associations, key_dim, generator=generator)
    common = torch.randn(1, key_dim, generator=generator)
    keys = (1.0 - key_correlation) * raw_keys + key_correlation * common
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    values = torch.randn(num_associations, value_dim, generator=generator)
    values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    affected = int(torch.randint(num_associations, (1,), generator=generator).item())
    state = keys.transpose(0, 1) @ values
    return SequentialMemory(
        keys=keys,
        initial_values=values,
        initial_state=state.clone(),
        affected_index=affected,
        current_value=values[affected].clone(),
        canonical_state=state.clone(),
    )


def make_sequential_episode(
    *,
    memory: SequentialMemory,
    model_state: torch.Tensor,
    operation: Operation,
    new_value: torch.Tensor,
    episode_id: str,
) -> MemoryEpisode:
    key = memory.keys[memory.affected_index]
    old_value = memory.current_value.clone()
    erase_candidate = torch.outer(key, old_value)
    write_candidate = torch.outer(key, new_value)
    erase, write = operation.demand
    target = memory.canonical_state - erase * erase_candidate + write * write_candidate
    unaffected = torch.tensor(
        [idx for idx in range(memory.keys.shape[0]) if idx != memory.affected_index],
        dtype=torch.long,
    )
    features = torch.tensor(
        [
            float(operation is Operation.PRESERVE),
            float(operation is Operation.ADD),
            float(operation is Operation.INVALIDATE),
            float(operation is Operation.SUPERSEDE),
        ]
    )
    return MemoryEpisode(
        episode_id=episode_id,
        operation=operation,
        keys=memory.keys,
        values=memory.initial_values,
        state=model_state,
        target_state=target,
        affected_index=memory.affected_index,
        unaffected_indices=unaffected,
        old_value=old_value,
        new_value=new_value,
        erase_candidate=erase_candidate,
        write_candidate=write_candidate,
        operation_features=features,
        metadata={},
    )


def commit_canonical(memory: SequentialMemory, episode: MemoryEpisode) -> None:
    memory.canonical_state = episode.target_state.clone()
    if episode.operation is Operation.ADD:
        memory.current_value = memory.current_value + episode.new_value
    elif episode.operation is Operation.INVALIDATE:
        memory.current_value = torch.zeros_like(memory.current_value)
    elif episode.operation is Operation.SUPERSEDE:
        memory.current_value = episode.new_value.clone()
