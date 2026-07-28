from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from catena.core.schema import MemoryEpisode
from catena.data.geometry_sweep import controller_features
from catena.models.matched_controllers import MatchedScalarController
from catena.models.memory import apply_scalar_update
from catena.training.losses import total_probe_loss


@dataclass(slots=True)
class MatchedTrainConfig:
    steps: int
    learning_rate: float
    affected_weight: float = 1.0
    retention_weight: float = 1.0
    state_weight: float = 0.25


def train_matched_controller(
    *,
    model: MatchedScalarController,
    episodes: list[MemoryEpisode],
    config: MatchedTrainConfig,
    device: torch.device,
) -> list[float]:
    if not episodes:
        raise ValueError("Training requires at least one episode.")
    if config.steps <= 0:
        raise ValueError("Training steps must be positive.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("Learning rate must be finite and positive.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for step in range(config.steps):
        source_episode = episodes[step % len(episodes)]
        features = controller_features(source_episode).to(device).unsqueeze(0)
        episode = source_episode.to(device)
        gates = model(features)
        output = apply_scalar_update(
            episode, gates.erase.squeeze(0), gates.write.squeeze(0)
        )
        loss = total_probe_loss(
            output,
            episode,
            affected_weight=config.affected_weight,
            retention_weight=config.retention_weight,
            state_weight=config.state_weight,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu().item())
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite detached loss at step {step}.")
        losses.append(value)
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(f"Non-finite parameter after training: {name}")
    return losses


def apply_matched_controller(
    model: MatchedScalarController, episode: MemoryEpisode
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        features = controller_features(episode).to(device).unsqueeze(0)
        local = episode.to(device)
        gates = model(features)
        output = apply_scalar_update(
            local, gates.erase.squeeze(0), gates.write.squeeze(0)
        )
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError(f"Non-finite controller output for {episode.episode_id}.")
        return output
