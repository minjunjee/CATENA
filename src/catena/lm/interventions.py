from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

GateName = Literal["erase", "write"]


@dataclass(frozen=True)
class GateIntervention:
    """Inference-time intervention on recurrent gate values.

    The callback receives already-sigmoided erase/write gates with shape
    ``[batch, heads]`` and must return tensors with identical shapes. The hook
    is only applied when ``token_mask`` selects the current sequence position.
    """

    erase_scale: float = 1.0
    write_scale: float = 1.0
    force_tied: bool = False
    erase_override: torch.Tensor | None = None
    write_override: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    custom: (
        Callable[[torch.Tensor, torch.Tensor, int, int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None

    def apply(
        self,
        erase: torch.Tensor,
        write: torch.Tensor,
        *,
        layer_index: int,
        token_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_mask is not None:
            if self.token_mask.ndim != 1:
                raise ValueError("token_mask must be one-dimensional")
            if token_index >= self.token_mask.numel() or not bool(self.token_mask[token_index]):
                return erase, write
        e = erase * float(self.erase_scale)
        w = write * float(self.write_scale)
        if self.erase_override is not None:
            e = torch.broadcast_to(self.erase_override.to(e), e.shape)
        if self.write_override is not None:
            w = torch.broadcast_to(self.write_override.to(w), w.shape)
        if self.force_tied:
            tied = 0.5 * (e + w)
            e = tied
            w = tied
        if self.custom is not None:
            e, w = self.custom(e, w, layer_index, token_index)
        return e.clamp(0.0, 1.0), w.clamp(0.0, 1.0)


@dataclass(frozen=True)
class AddressIntervention:
    erase_address: torch.Tensor | None = None
    write_address: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None

    def applies(self, token_index: int) -> bool:
        if self.token_mask is None:
            return True
        return token_index < self.token_mask.numel() and bool(self.token_mask[token_index])
