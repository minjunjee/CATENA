from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch
from torch import nn

from catena.core.schema import MemoryEpisode
from catena.models.controllers import FactorizedSemanticController, SharedSemanticController
from catena.models.memory import GateOutput, apply_scalar_update
from catena.training.losses import total_probe_loss

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def hashed_bow(text: str, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    if vector.norm() > 0:
        vector /= vector.norm().clamp_min(1e-8)
    return vector


@dataclass(slots=True)
class SemanticExample:
    text: str
    episode: MemoryEpisode
    domain: str
    template_id: str


@dataclass(slots=True)
class SemanticTrainConfig:
    steps: int = 1000
    learning_rate: float = 2e-3
    bow_dim: int = 256
    hidden_dim: int = 96
    include_state_read: bool = True


def semantic_features(example: SemanticExample, config: SemanticTrainConfig) -> torch.Tensor:
    features = [hashed_bow(example.text, config.bow_dim)]
    if config.include_state_read:
        key = example.episode.keys[example.episode.affected_index]
        current_read = key @ example.episode.state
        features.append(current_read.to(torch.float32))
    return torch.cat(features, dim=0)


def semantic_input_dim(config: SemanticTrainConfig, value_dim: int) -> int:
    return config.bow_dim + (value_dim if config.include_state_read else 0)


def _forward(model: nn.Module, features: torch.Tensor) -> GateOutput:
    output = model(features)
    if not isinstance(output, GateOutput):
        raise TypeError("Semantic controller must return GateOutput")
    return output


def train_semantic_controller(
    *,
    examples: list[SemanticExample],
    factorized: bool,
    config: SemanticTrainConfig,
    value_dim: int,
    device: torch.device,
) -> nn.Module:
    input_dim = semantic_input_dim(config, value_dim)
    model: nn.Module
    if factorized:
        model = FactorizedSemanticController(input_dim, config.hidden_dim)
    else:
        model = SharedSemanticController(input_dim, config.hidden_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for step in range(config.steps):
        example = examples[step % len(examples)]
        episode = example.episode.to(device)
        features = semantic_features(example, config).to(device).unsqueeze(0)
        gates = _forward(model, features)
        output_state = apply_scalar_update(
            episode, gates.erase.squeeze(0), gates.write.squeeze(0)
        )
        loss = total_probe_loss(
            output_state,
            episode,
            affected_weight=1.0,
            retention_weight=1.0,
            state_weight=0.25,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def evaluate_semantic_controller(
    model: nn.Module,
    examples: list[SemanticExample],
    config: SemanticTrainConfig,
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            episode = example.episode.to(device)
            features = semantic_features(example, config).to(device).unsqueeze(0)
            gates = _forward(model, features)
            outputs.append(
                apply_scalar_update(
                    episode, gates.erase.squeeze(0), gates.write.squeeze(0)
                ).cpu()
            )
    return outputs
