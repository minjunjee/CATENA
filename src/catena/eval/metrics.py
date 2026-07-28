from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from catena.core.schema import MemoryEpisode
from catena.training.losses import (
    affected_read_mse,
    new_write_mse,
    old_association_residual,
    target_state_mse,
    unaffected_retention_mse,
)


@dataclass(slots=True)
class EpisodeMetrics:
    episode_id: str
    operation: str
    affected_read_mse: float
    unaffected_retention_mse: float
    target_state_mse: float
    old_association_residual: float
    new_write_mse: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def evaluate_episode(output_state: torch.Tensor, episode: MemoryEpisode) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=episode.episode_id,
        operation=episode.operation.value,
        affected_read_mse=float(affected_read_mse(output_state, episode).item()),
        unaffected_retention_mse=float(unaffected_retention_mse(output_state, episode).item()),
        target_state_mse=float(target_state_mse(output_state, episode).item()),
        old_association_residual=float(old_association_residual(output_state, episode).item()),
        new_write_mse=float(new_write_mse(output_state, episode).item()),
    )


def pareto_joint(affected_mse: float, retention_mse: float) -> float:
    correction = float(torch.exp(torch.tensor(-affected_mse)).item())
    retention = float(torch.exp(torch.tensor(-retention_mse)).item())
    if correction + retention == 0:
        return 0.0
    return 2.0 * correction * retention / (correction + retention)
