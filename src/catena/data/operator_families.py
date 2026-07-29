from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch


class OperatorFamily(StrEnum):
    AXIS_COMMUTING = "axis_commuting"
    COMMON_ROTATED_COMMUTING = "common_rotated_commuting"
    NONCOMMUTING = "noncommuting"


@dataclass(slots=True)
class OperatorSet:
    family: OperatorFamily
    projectors: list[torch.Tensor]
    certified_shared_basis: torch.Tensor | None


def _orthogonal(
    dim: int,
    generator: torch.Generator,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    q, _ = torch.linalg.qr(
        torch.randn(dim, dim, generator=generator, dtype=dtype)
    )
    return cast(torch.Tensor, q)


def generate_operator_set(
    *,
    family: OperatorFamily,
    dim: int,
    rank: int,
    count: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> OperatorSet:
    if not isinstance(family, OperatorFamily):
        raise TypeError("family must be an OperatorFamily.")
    for name, value in (("dim", dim), ("rank", rank), ("count", count), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if dim < 2:
        raise ValueError("dim must be at least 2.")
    if not 0 < rank < dim:
        raise ValueError("rank must lie strictly between zero and dim.")
    if count < 2:
        raise ValueError("count must be at least 2.")
    if count > math.comb(dim, rank):
        raise ValueError("count exceeds the number of distinct axis-aligned masks.")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point.")

    generator = torch.Generator().manual_seed(seed)
    masks: list[torch.Tensor] = []
    seen_masks: set[tuple[int, ...]] = set()
    while len(masks) < count:
        indices = torch.randperm(dim, generator=generator)[:rank]
        signature = tuple(sorted(int(index) for index in indices.tolist()))
        if signature in seen_masks:
            continue
        seen_masks.add(signature)
        mask = torch.zeros(dim, dtype=dtype)
        mask[indices] = 1.0
        masks.append(torch.diag(mask))

    if family is OperatorFamily.AXIS_COMMUTING:
        projectors = masks
        certified_shared_basis: torch.Tensor | None = torch.eye(dim, dtype=dtype)
    elif family is OperatorFamily.COMMON_ROTATED_COMMUTING:
        q = _orthogonal(dim, generator, dtype=dtype)
        projectors = [q @ mask @ q.transpose(0, 1) for mask in masks]
        certified_shared_basis = q
    else:
        projectors = []
        for mask in masks:
            q = _orthogonal(dim, generator, dtype=dtype)
            projectors.append(q @ mask @ q.transpose(0, 1))
        certified_shared_basis = None
    return OperatorSet(
        family=family,
        projectors=projectors,
        certified_shared_basis=certified_shared_basis,
    )
