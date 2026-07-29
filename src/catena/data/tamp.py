from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import torch

from catena.core.schema import CandidateMode, MemoryEpisode, Operation


@dataclass(slots=True)
class TAMPConfig:
    num_associations: int = 16
    key_dim: int = 32
    value_dim: int = 32
    key_correlation: float = 0.25
    dtype: torch.dtype = torch.float32


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _make_correlated_keys(
    num_associations: int,
    key_dim: int,
    correlation: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = torch.randn(num_associations, key_dim, generator=generator, dtype=dtype)
    common = torch.randn(1, key_dim, generator=generator, dtype=dtype)
    keys = (1.0 - correlation) * raw + correlation * common
    return _normalize_rows(keys)


def _operation_features(operation: Operation, dtype: torch.dtype) -> torch.Tensor:
    order = [Operation.PRESERVE, Operation.ADD, Operation.INVALIDATE, Operation.SUPERSEDE]
    return torch.tensor([float(operation is item) for item in order], dtype=dtype)


def build_episode(
    *,
    seed: int,
    operation: Operation,
    candidate_mode: CandidateMode,
    config: TAMPConfig,
    episode_index: int = 0,
) -> MemoryEpisode:
    generator = torch.Generator().manual_seed(seed)
    keys = _make_correlated_keys(
        config.num_associations,
        config.key_dim,
        config.key_correlation,
        generator,
        config.dtype,
    )
    values = torch.randn(
        config.num_associations,
        config.value_dim,
        generator=generator,
        dtype=config.dtype,
    )
    values = _normalize_rows(values)
    affected_index = int(torch.randint(config.num_associations, (1,), generator=generator).item())
    old_value = values[affected_index].clone()
    new_value = torch.randn(config.value_dim, generator=generator, dtype=config.dtype)
    new_value = new_value / new_value.norm().clamp_min(1e-8)
    state = keys.transpose(0, 1) @ values
    key = keys[affected_index]
    oracle_erase = torch.outer(key, old_value)
    if candidate_mode is CandidateMode.ORACLE:
        erase_candidate = oracle_erase
    else:
        recurrent_read = key @ state
        erase_candidate = torch.outer(key, recurrent_read)
    write_candidate = torch.outer(key, new_value)
    erase, write = operation.demand
    target_state = state - erase * oracle_erase + write * write_candidate
    unaffected = torch.tensor(
        [idx for idx in range(config.num_associations) if idx != affected_index],
        dtype=torch.long,
    )
    return MemoryEpisode(
        episode_id=f"{candidate_mode.value}-{seed}-{episode_index}",
        operation=operation,
        keys=keys,
        values=values,
        state=state,
        target_state=target_state,
        affected_index=affected_index,
        unaffected_indices=unaffected,
        old_value=old_value,
        new_value=new_value,
        erase_candidate=erase_candidate,
        write_candidate=write_candidate,
        operation_features=_operation_features(operation, config.dtype),
        metadata={
            "seed": seed,
            "candidate_mode": candidate_mode.value,
            "key_correlation": config.key_correlation,
        },
    )


def generate_episodes(
    *,
    count_per_operation: int,
    seed: int,
    candidate_mode: CandidateMode,
    config: TAMPConfig,
    operations: Iterable[Operation] | None = None,
) -> list[MemoryEpisode]:
    selected = list(operations or Operation)
    episodes: list[MemoryEpisode] = []
    seed_stream = itertools.count(seed)
    for operation in selected:
        for index in range(count_per_operation):
            episodes.append(
                build_episode(
                    seed=next(seed_stream),
                    operation=operation,
                    candidate_mode=candidate_mode,
                    config=config,
                    episode_index=index,
                )
            )
    return episodes


def validate_episode(episode: MemoryEpisode, atol: float = 1e-6) -> None:
    key = episode.keys[episode.affected_index]
    exact_old = torch.outer(key, episode.old_value)
    exact_new = torch.outer(key, episode.new_value)
    erase, write = episode.operation.demand
    expected = episode.state - erase * exact_old + write * exact_new
    if not torch.allclose(episode.target_state, expected, atol=atol, rtol=0.0):
        raise AssertionError("Target state does not match the operation demand.")
    if episode.affected_index in episode.unaffected_indices.tolist():
        raise AssertionError("Affected index leaked into unaffected retention set.")
