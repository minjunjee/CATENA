from __future__ import annotations

import torch


def gdn2_reference_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    erase_gate: torch.Tensor,
    write_gate: torch.Tensor,
) -> torch.Tensor:
    """Reference GDN2-like update for numerical plumbing tests.

    State shape: [d_k, d_v]. This follows the public high-level recurrence,
    but is not a replacement for the official fused kernel.
    """
    decayed = decay[:, None] * state
    erase_vector = erase_gate * key
    erase_matrix = torch.eye(key.numel(), dtype=state.dtype, device=state.device) - torch.outer(key, erase_vector)
    write_value = write_gate * value
    return erase_matrix @ decayed + torch.outer(key, write_value)


def kda_tied_reference_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    erase = beta.expand_as(key)
    write = beta.expand_as(value)
    return gdn2_reference_update(state, key, value, decay, erase, write)
