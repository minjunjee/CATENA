from __future__ import annotations

import itertools
import math
from typing import Any

import torch

from catena.core.schema import CandidateMode, MemoryEpisode, Operation


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _paired_values(
    dim: int, old_scale: float, new_scale: float, cosine: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    old = _normalize(torch.randn(1, dim, generator=generator)).squeeze(0)
    orth = torch.randn(dim, generator=generator)
    orth = orth - torch.dot(orth, old) * old
    orth = orth / orth.norm().clamp_min(1e-8)
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    new = cosine * old + sine * orth
    return old_scale * old, new_scale * new


def build_geometry_episode(
    *,
    seed: int,
    operation: Operation,
    candidate_mode: CandidateMode,
    key_dim: int,
    value_dim: int,
    num_associations: int,
    key_correlation: float,
    old_scale: float,
    new_scale: float,
    old_new_cosine: float,
    episode_index: int,
) -> MemoryEpisode:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(num_associations, key_dim, generator=generator)
    common = torch.randn(1, key_dim, generator=generator)
    keys = _normalize((1.0 - key_correlation) * raw + key_correlation * common)
    values = _normalize(torch.randn(num_associations, value_dim, generator=generator))
    affected_index = int(torch.randint(num_associations, (1,), generator=generator).item())
    old_value, new_value = _paired_values(
        value_dim, old_scale, new_scale, old_new_cosine, generator
    )
    values[affected_index] = old_value
    state = keys.transpose(0, 1) @ values
    key = keys[affected_index]
    oracle_erase = torch.outer(key, old_value)
    recurrent_value = key @ state
    erase_candidate = (
        oracle_erase
        if candidate_mode is CandidateMode.ORACLE
        else torch.outer(key, recurrent_value)
    )
    write_candidate = torch.outer(key, new_value)
    erase, write = operation.demand
    target_state = state - erase * oracle_erase + write * write_candidate
    unaffected = torch.tensor(
        [index for index in range(num_associations) if index != affected_index],
        dtype=torch.long,
    )
    contamination = float(torch.mean((recurrent_value - old_value) ** 2).item())
    return MemoryEpisode(
        episode_id=f"geom-{candidate_mode.value}-{seed}-{episode_index}",
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
        operation_features=torch.tensor(
            [float(operation is item) for item in Operation], dtype=torch.float32
        ),
        metadata={
            "seed": seed,
            "candidate_mode": candidate_mode.value,
            "key_correlation": key_correlation,
            "state_load": float(num_associations),
            "old_scale": old_scale,
            "new_scale": new_scale,
            "old_new_cosine": old_new_cosine,
            "candidate_contamination": contamination,
        },
    )


def controller_features(episode: MemoryEpisode) -> torch.Tensor:
    scale = max(float(episode.metadata["state_load"]), 1.0)
    geometry = torch.tensor(
        [
            float(episode.metadata["old_scale"]),
            float(episode.metadata["new_scale"]),
            float(episode.metadata["old_new_cosine"]),
            float(episode.metadata["key_correlation"]),
            math.log1p(scale) / 5.0,
            float(episode.metadata["candidate_contamination"]),
        ],
        dtype=torch.float32,
    )
    return torch.cat([episode.operation_features.to(torch.float32), geometry])


def generate_geometry_grid(
    *,
    seed: int,
    candidate_mode: CandidateMode,
    grid: dict[str, Any],
    count_per_cell: int,
) -> list[MemoryEpisode]:
    names = [
        "num_associations",
        "key_correlations",
        "old_scales",
        "new_scales",
        "old_new_cosines",
    ]
    values = [grid[name] for name in names]
    episodes: list[MemoryEpisode] = []
    cursor = 0
    for cell in itertools.product(*values):
        kwargs = dict(zip(names, cell, strict=True))
        for operation in Operation:
            for repeat in range(count_per_cell):
                episodes.append(
                    build_geometry_episode(
                        seed=seed + cursor,
                        operation=operation,
                        candidate_mode=candidate_mode,
                        key_dim=int(grid["key_dim"][0]),
                        value_dim=int(grid["value_dim"][0]),
                        num_associations=int(kwargs["num_associations"]),
                        key_correlation=float(kwargs["key_correlations"]),
                        old_scale=float(kwargs["old_scales"]),
                        new_scale=float(kwargs["new_scales"]),
                        old_new_cosine=float(kwargs["old_new_cosines"]),
                        episode_index=repeat,
                    )
                )
                cursor += 1
    return episodes
