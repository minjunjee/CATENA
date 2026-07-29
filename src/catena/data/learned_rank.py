from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LowRankDemandFamily:
    """Smooth transaction-to-operator family with a known intrinsic rank.

    A descriptor z is mapped to coefficients c(z); the target update operator is
    U diag(c(z)) V^T.  The family is deliberately synthetic: it isolates whether
    a learned rank-r control surface can attain the corresponding best-rank
    reachable floor on held-out transactions.
    """

    left_basis: torch.Tensor
    right_basis: torch.Tensor
    coefficient_map: torch.Tensor
    bias: torch.Tensor

    @property
    def dimension(self) -> int:
        return int(self.left_basis.shape[0])

    @property
    def intrinsic_rank(self) -> int:
        return int(self.left_basis.shape[1])

    @property
    def descriptor_dim(self) -> int:
        return int(self.coefficient_map.shape[1])

    def operator(self, descriptor: torch.Tensor) -> torch.Tensor:
        coefficients = torch.tanh(
            descriptor @ self.coefficient_map.transpose(0, 1) + self.bias
        )
        return torch.einsum(
            "dr,...r,er->...de",
            self.left_basis,
            coefficients,
            self.right_basis,
        )


def _orthonormal_columns(
    dimension: int,
    rank: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rank > dimension:
        raise ValueError(f"rank={rank} cannot exceed dimension={dimension}")
    matrix = torch.randn(dimension, rank, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank]


def make_low_rank_family(
    *,
    dimension: int,
    descriptor_dim: int,
    intrinsic_rank: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> LowRankDemandFamily:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    left = _orthonormal_columns(
        dimension, intrinsic_rank, generator=generator, dtype=dtype
    )
    right = _orthonormal_columns(
        dimension, intrinsic_rank, generator=generator, dtype=dtype
    )
    coefficient_map = torch.randn(
        intrinsic_rank,
        descriptor_dim,
        generator=generator,
        dtype=dtype,
    ) / descriptor_dim**0.5
    bias = torch.randn(intrinsic_rank, generator=generator, dtype=dtype) * 0.1
    return LowRankDemandFamily(
        left_basis=left,
        right_basis=right,
        coefficient_map=coefficient_map,
        bias=bias,
    )


def sample_descriptors(
    *,
    count: int,
    descriptor_dim: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    descriptors = torch.randn(
        count,
        descriptor_dim,
        generator=generator,
        dtype=dtype,
    )
    return descriptors / descriptors.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def best_rank_errors(
    target: torch.Tensor,
    ranks: Iterable[int],
) -> dict[int, torch.Tensor]:
    """Per-example best-rank MSEs from one shared singular-value decomposition."""
    if target.ndim != 3:
        raise ValueError("target must have shape [batch, d, d]")
    requested = tuple(dict.fromkeys(int(rank) for rank in ranks))
    if not requested:
        raise ValueError("ranks must contain at least one value")
    if any(rank < 0 for rank in requested):
        raise ValueError("ranks must be non-negative")
    if not torch.isfinite(target).all():
        raise ValueError("target must contain only finite values")

    singular = torch.linalg.svdvals(target)
    squared_singular = singular.square()
    dimension = target.shape[-1] * target.shape[-2]
    return {
        rank: squared_singular[..., rank:].sum(dim=-1) / float(dimension)
        for rank in requested
    }


def best_rank_error(target: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-example normalized Frobenius MSE of the best rank-r approximation."""
    return best_rank_errors(target, (rank,))[int(rank)]
