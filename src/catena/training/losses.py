from __future__ import annotations

import torch

from catena.core.schema import MemoryEpisode
from catena.models.memory import read_state


def affected_read_mse(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    key = episode.keys[episode.affected_index : episode.affected_index + 1]
    prediction = read_state(output_state, key)
    target = read_state(episode.target_state, key)
    return torch.mean((prediction - target) ** 2)


def unaffected_retention_mse(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    keys = episode.keys[episode.unaffected_indices]
    prediction = read_state(output_state, keys)
    target = read_state(episode.target_state, keys)
    return torch.mean((prediction - target) ** 2)


def target_state_mse(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    return torch.mean((output_state - episode.target_state) ** 2)


def old_association_residual(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    key = episode.keys[episode.affected_index]
    read = key @ output_state
    denominator = episode.old_value.pow(2).sum().clamp_min(1e-8)
    coefficient = torch.abs(torch.dot(read, episode.old_value) / denominator)
    return coefficient


def new_write_mse(output_state: torch.Tensor, episode: MemoryEpisode) -> torch.Tensor:
    if episode.operation.value not in {"add", "supersede"}:
        return torch.zeros((), dtype=output_state.dtype, device=output_state.device)
    key = episode.keys[episode.affected_index]
    read = key @ output_state
    target = key @ episode.target_state
    return torch.mean((read - target) ** 2)


def total_probe_loss(
    output_state: torch.Tensor,
    episode: MemoryEpisode,
    *,
    affected_weight: float,
    retention_weight: float,
    state_weight: float,
) -> torch.Tensor:
    return (
        affected_weight * affected_read_mse(output_state, episode)
        + retention_weight * unaffected_retention_mse(output_state, episode)
        + state_weight * target_state_mse(output_state, episode)
    )
