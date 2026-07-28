from __future__ import annotations

from dataclasses import dataclass

import torch

from catena.core.schema import MemoryEpisode


@dataclass(slots=True)
class GateOutput:
    erase: torch.Tensor
    write: torch.Tensor


def read_state(state: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Read value vectors from a [d_k, d_v] associative state."""
    return keys @ state


def apply_scalar_update(
    episode: MemoryEpisode,
    erase: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    erase_scalar = erase.reshape(()).to(episode.state)
    write_scalar = write.reshape(()).to(episode.state)
    return (
        episode.state
        - erase_scalar * episode.erase_candidate
        + write_scalar * episode.write_candidate
    )


def apply_vector_value_update(
    episode: MemoryEpisode,
    erase: torch.Tensor,
    write: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic diagonal control over value-side candidate coordinates.

    This is a controlled probe family, not a literal GDN2 kernel.
    """
    key = episode.keys[episode.affected_index]
    old = episode.erase_candidate.transpose(0, 1) @ key
    new = episode.new_value
    return episode.state - torch.outer(key, erase * old) + torch.outer(key, write * new)


def apply_projected_update(
    episode: MemoryEpisode,
    erase_projector: torch.Tensor,
    write_projector: torch.Tensor,
) -> torch.Tensor:
    key = episode.keys[episode.affected_index]
    old = episode.old_value
    new = episode.new_value
    erased = erase_projector @ old
    written = write_projector @ new
    return episode.state - torch.outer(key, erased) + torch.outer(key, written)
