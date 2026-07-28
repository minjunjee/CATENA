from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from catena.core.schema import MemoryEpisode


def geometry_operation_key(episode: MemoryEpisode) -> tuple[object, ...]:
    """Identify a geometry cell and operation without using replicate identity."""

    metadata = episode.metadata
    return (
        episode.operation.value,
        int(metadata["state_load"]),
        float(metadata["key_correlation"]),
        float(metadata["old_scale"]),
        float(metadata["new_scale"]),
        float(metadata["old_new_cosine"]),
    )


def balanced_geometry_schedule(
    episodes: Sequence[MemoryEpisode],
    *,
    steps: int,
    seed: int,
) -> list[MemoryEpisode]:
    """Build a deterministic round-robin schedule over geometry/operation cells.

    The schedule is exactly ``steps`` long.  Every cell is visited before a
    second visit to any cell, which prevents a truncated training budget from
    silently omitting the lexicographically last geometry cells.
    """

    if not episodes:
        raise ValueError("Cannot schedule an empty episode collection.")
    if steps <= 0:
        raise ValueError("steps must be positive.")

    grouped: dict[tuple[object, ...], list[MemoryEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[geometry_operation_key(episode)].append(episode)

    rng = np.random.default_rng(seed)
    keys = sorted(grouped, key=repr)
    for values in grouped.values():
        rng.shuffle(values)

    cursors = {key: 0 for key in keys}
    schedule: list[MemoryEpisode] = []
    while len(schedule) < steps:
        round_keys = list(keys)
        rng.shuffle(round_keys)
        for key in round_keys:
            values = grouped[key]
            cursor = cursors[key]
            schedule.append(values[cursor % len(values)])
            cursors[key] = cursor + 1
            if len(schedule) == steps:
                break
    return schedule


def schedule_sha256(episodes: Sequence[MemoryEpisode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def schedule_cell_counts(
    episodes: Sequence[MemoryEpisode],
) -> dict[tuple[object, ...], int]:
    counts: dict[tuple[object, ...], int] = defaultdict(int)
    for episode in episodes:
        counts[geometry_operation_key(episode)] += 1
    return dict(counts)
