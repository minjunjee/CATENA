"""Matched E26 gate-policy adapter for the pinned official GDN-2 layer.

The official layer keeps independent ``b_proj`` and ``w_proj`` modules.  This
adapter deliberately operates *after* those two projections so choosing the
Projected-Tied condition cannot remove, share, resize, or otherwise change a
projection parameter.  It changes only the mapping from the two raw logits to
the erase/write gates used by the unchanged official recurrence kernel.

The adapter is valid only for the E26 contract with ``allow_neg_eigval=False``.
The official optional ``b *= 2`` path is therefore rejected rather than
silently changing the registered gate range.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import torch
from torch import nn

E26FinalGatePolicy: TypeAlias = Literal[
    "dual_gdn2",
    "projected_tied_gdn2",
]

_DUAL_POLICY = "dual_gdn2"
_PROJECTED_TIED_POLICY = "projected_tied_gdn2"
_SUPPORTED_POLICIES = frozenset({_DUAL_POLICY, _PROJECTED_TIED_POLICY})


def _validate_contract(
    z_b: torch.Tensor,
    z_w: torch.Tensor,
    *,
    policy: str,
    allow_neg_eigval: bool,
) -> None:
    """Fail closed when inputs cannot represent the matched E26 comparison."""

    if allow_neg_eigval is not False:
        raise ValueError("E26 Final requires allow_neg_eigval=False")
    if policy not in _SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported E26 Final gate policy: {policy!r}")
    if z_b.shape != z_w.shape:
        raise ValueError(
            "E26 Final requires b_proj and w_proj logits with identical shapes; "
            f"got {tuple(z_b.shape)} and {tuple(z_w.shape)}"
        )
    if z_b.dtype != z_w.dtype:
        raise ValueError(
            "E26 Final requires b_proj and w_proj logits with identical dtypes; "
            f"got {z_b.dtype} and {z_w.dtype}"
        )
    if z_b.device != z_w.device:
        raise ValueError(
            "E26 Final requires b_proj and w_proj logits on the same device; "
            f"got {z_b.device} and {z_w.device}"
        )


def project_e26_final_gates(
    z_b: torch.Tensor,
    z_w: torch.Tensor,
    *,
    policy: E26FinalGatePolicy,
    allow_neg_eigval: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the two official projection logits to matched erase/write gates.

    ``dual_gdn2`` applies an independent sigmoid to each projection.
    ``projected_tied_gdn2`` averages the raw logits before sigmoid and
    returns that same gate on both paths.  Averaging post-sigmoid values would
    define a different intervention and is intentionally not supported.
    """

    _validate_contract(
        z_b,
        z_w,
        policy=policy,
        allow_neg_eigval=allow_neg_eigval,
    )
    if policy == _DUAL_POLICY:
        return torch.sigmoid(z_b), torch.sigmoid(z_w)

    tied = torch.sigmoid((z_b + z_w) / 2.0)
    return tied, tied


class E26FinalGatePolicyAdapter(nn.Module):
    """Parameter-free policy layer placed after official ``b_proj``/``w_proj``.

    Because the adapter accepts raw logits rather than hidden states, both
    official projection heads remain registered and trainable under either
    policy.  The contract is checked again on every call so a mutated runtime
    configuration cannot enable the negative-eigenvalue path silently.
    """

    policy: E26FinalGatePolicy
    allow_neg_eigval: bool

    def __init__(
        self,
        policy: E26FinalGatePolicy,
        *,
        allow_neg_eigval: bool = False,
    ) -> None:
        super().__init__()
        if allow_neg_eigval is not False:
            raise ValueError("E26 Final requires allow_neg_eigval=False")
        if policy not in _SUPPORTED_POLICIES:
            raise ValueError(f"Unsupported E26 Final gate policy: {policy!r}")
        self.policy = policy
        self.allow_neg_eigval = allow_neg_eigval

    def forward(
        self,
        z_b: torch.Tensor,
        z_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return project_e26_final_gates(
            z_b,
            z_w,
            policy=self.policy,
            allow_neg_eigval=self.allow_neg_eigval,
        )


__all__ = [
    "E26FinalGatePolicy",
    "E26FinalGatePolicyAdapter",
    "project_e26_final_gates",
]
