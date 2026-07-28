from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from catena.core.schema import ControllerKind, MemoryEpisode
from catena.models.controllers import GateController
from catena.models.memory import apply_scalar_update, apply_vector_value_update
from catena.training.losses import total_probe_loss


@dataclass(slots=True)
class TrainConfig:
    steps: int = 500
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    affected_weight: float = 1.0
    retention_weight: float = 1.0
    state_weight: float = 0.25


@dataclass(slots=True)
class TrainTrace:
    losses: list[float]


def _apply_controller(
    model: GateController,
    episode: MemoryEpisode,
) -> torch.Tensor:
    gates = model(episode.operation_features.unsqueeze(0))
    if model.spec.kind in {ControllerKind.TIED_SCALAR, ControllerKind.DUAL_SCALAR}:
        return apply_scalar_update(episode, gates.erase.squeeze(0), gates.write.squeeze(0))
    if model.spec.kind is ControllerKind.VECTOR:
        return apply_vector_value_update(episode, gates.erase.squeeze(0), gates.write.squeeze(0))
    raise NotImplementedError(f"Training path not implemented for {model.spec.kind}")


def train_probe_controller(
    *,
    model: GateController,
    episodes: list[MemoryEpisode],
    config: TrainConfig,
    device: torch.device,
    callback: Callable[[int, float], None] | None = None,
) -> TrainTrace:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses: list[float] = []
    for step in range(config.steps):
        episode = episodes[step % len(episodes)].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = _apply_controller(model, episode)
        loss = total_probe_loss(
            output,
            episode,
            affected_weight=config.affected_weight,
            retention_weight=config.retention_weight,
            state_weight=config.state_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        value = float(loss.detach().cpu().item())
        losses.append(value)
        if callback is not None:
            callback(step, value)
    return TrainTrace(losses=losses)


def apply_trained_controller(model: GateController, episode: MemoryEpisode) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return _apply_controller(model, episode)
