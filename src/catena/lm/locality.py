from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalityDirection:
    direction: torch.Tensor
    unit_response_error: float
    covariance_energy: float
    condition_number: float
    used_fallback: bool


def covariance_aware_direction(
    covariance: torch.Tensor,
    target_key: torch.Tensor,
    *,
    regularization: float,
    eps: float = 1.0e-8,
) -> LocalityDirection:
    """Minimum covariance-energy direction with target unit response.

    Solves ``(C + lambda I) x = k`` and normalizes by ``k^T x``. The function
    supports a single matrix/key pair; batched production code can vmap it.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if target_key.ndim != 1 or target_key.shape[0] != covariance.shape[0]:
        raise ValueError("target_key dimension does not match covariance")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    covariance = 0.5 * (covariance + covariance.T)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    system = covariance + regularization * identity
    condition = float(torch.linalg.cond(system.float()).item())
    used_fallback = False
    try:
        solution = torch.linalg.solve(system, target_key)
    except RuntimeError:
        solution = torch.linalg.pinv(system) @ target_key
        used_fallback = True
    denominator = torch.dot(target_key, solution)
    if not torch.isfinite(denominator) or abs(float(denominator.item())) <= eps:
        direction = F.normalize(target_key, dim=0)
        used_fallback = True
    else:
        direction = solution / denominator
    response = float(torch.dot(target_key, direction).item())
    energy = float((direction @ covariance @ direction).item())
    return LocalityDirection(
        direction=direction,
        unit_response_error=abs(response - 1.0),
        covariance_energy=energy,
        condition_number=condition,
        used_fallback=used_fallback,
    )


def protected_nullspace_direction(
    target_key: torch.Tensor,
    protected_keys: torch.Tensor,
    *,
    regularization: float = 1.0e-6,
) -> LocalityDirection:
    """Oracle diagnostic that removes the protected-key span."""

    if protected_keys.ndim != 2 or protected_keys.shape[1] != target_key.shape[0]:
        raise ValueError("protected_keys must have shape [n, d]")
    if protected_keys.shape[0] == 0:
        direction = target_key / torch.dot(target_key, target_key).clamp_min(regularization)
        return LocalityDirection(direction, 0.0, 0.0, 1.0, False)
    gram = protected_keys @ protected_keys.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    projection = protected_keys.T @ torch.linalg.solve(
        gram + regularization * identity, protected_keys
    )
    candidate = target_key - projection @ target_key
    denominator = torch.dot(target_key, candidate)
    used_fallback = False
    if abs(float(denominator.item())) <= regularization:
        candidate = target_key
        denominator = torch.dot(target_key, candidate)
        used_fallback = True
    direction = candidate / denominator
    response = float(torch.dot(target_key, direction).item())
    spill = float((protected_keys @ direction).float().pow(2).mean().item())
    return LocalityDirection(
        direction=direction,
        unit_response_error=abs(response - 1.0),
        covariance_energy=spill,
        condition_number=float(
            torch.linalg.cond((gram + regularization * identity).float()).item()
        ),
        used_fallback=used_fallback,
    )


def active_key_covariance(
    keys: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    normalize_keys: bool = True,
) -> torch.Tensor:
    if keys.ndim != 2:
        raise ValueError("keys must have shape [n, d]")
    values = F.normalize(keys, dim=-1) if normalize_keys else keys
    if weights is None:
        weights = torch.ones(values.shape[0], device=values.device, dtype=values.dtype)
    if weights.ndim != 1 or weights.shape[0] != values.shape[0]:
        raise ValueError("weights must have shape [n]")
    normalized_weights = weights / weights.sum().clamp_min(1.0e-12)
    return (values.T * normalized_weights) @ values


def worst_non_target_response(direction: torch.Tensor, non_target_keys: torch.Tensor) -> float:
    if non_target_keys.numel() == 0:
        return 0.0
    return float((non_target_keys @ direction).abs().max().item())
