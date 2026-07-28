from __future__ import annotations

from dataclasses import dataclass

import torch

from catena.data.semantic_transactions import SemanticTransaction
from catena.models.memory import apply_scalar_update
from catena.models.semantic_controllers import MatchedSemanticController
from catena.training.losses import total_probe_loss
from catena.training.text import hashed_bow


@dataclass(slots=True)
class SemanticProbeExample:
    transaction: SemanticTransaction
    episode: object


@dataclass(slots=True)
class SemanticProbeConfig:
    bow_dim: int
    hidden_dim: int
    include_state_read: bool
    steps: int
    learning_rate: float


def semantic_probe_features(example: SemanticProbeExample, config: SemanticProbeConfig) -> torch.Tensor:
    features = [hashed_bow(example.transaction.text, config.bow_dim)]
    if config.include_state_read:
        episode = example.episode
        key = episode.keys[episode.affected_index]
        features.append((key @ episode.state).to(torch.float32))
    return torch.cat(features)


def semantic_probe_input_dim(config: SemanticProbeConfig, value_dim: int) -> int:
    return config.bow_dim + (value_dim if config.include_state_read else 0)


def train_semantic_probe(*, model: MatchedSemanticController, examples: list[SemanticProbeExample], config: SemanticProbeConfig, device: torch.device) -> list[float]:
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate); losses = []
    for step in range(config.steps):
        example = examples[step % len(examples)]; episode = example.episode.to(device)
        gates = model(semantic_probe_features(example, config).to(device).unsqueeze(0))
        output = apply_scalar_update(episode, gates.erase.squeeze(0), gates.write.squeeze(0))
        loss = total_probe_loss(output, episode, affected_weight=1.0, retention_weight=1.0, state_weight=0.25)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def evaluate_semantic_probe(*, model: MatchedSemanticController, examples: list[SemanticProbeExample], config: SemanticProbeConfig, device: torch.device) -> list[torch.Tensor]:
    outputs = []; model.eval()
    with torch.no_grad():
        for example in examples:
            episode = example.episode.to(device)
            gates = model(semantic_probe_features(example, config).to(device).unsqueeze(0))
            outputs.append(apply_scalar_update(episode, gates.erase.squeeze(0), gates.write.squeeze(0)).cpu())
    return outputs
