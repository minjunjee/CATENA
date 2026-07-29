from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class RepresentationFamily(str, Enum):
    AXIS_COMMUTING = "axis_commuting"
    COMMON_ROTATED_COMMUTING = "common_rotated_commuting"
    NONCOMMUTING = "noncommuting"


@dataclass(slots=True)
class RepresentationDemandGenerator:
    family: RepresentationFamily
    dimension: int
    active_rank: int
    descriptor_dim: int
    coefficient_map: torch.Tensor
    shared_basis: torch.Tensor
    rotation_maps: torch.Tensor
    rotation_scale: float

    def operators(self, descriptors: torch.Tensor) -> torch.Tensor:
        coefficients = torch.tanh(
            descriptors @ self.coefficient_map.transpose(0, 1)
        )
        padded = torch.zeros(
            descriptors.shape[0],
            self.dimension,
            dtype=descriptors.dtype,
            device=descriptors.device,
        )
        padded[:, : self.active_rank] = coefficients
        diagonal = torch.diag_embed(padded)

        if self.family is RepresentationFamily.AXIS_COMMUTING:
            return diagonal
        if self.family is RepresentationFamily.COMMON_ROTATED_COMMUTING:
            basis = self.shared_basis.to(descriptors)
            return basis @ diagonal @ basis.transpose(0, 1)

        # Transaction-dependent rotations create a genuinely noncommuting family.
        flat = descriptors @ self.rotation_maps.reshape(
            self.descriptor_dim, self.dimension * self.dimension
        )
        skew = flat.view(-1, self.dimension, self.dimension)
        skew = 0.5 * (skew - skew.transpose(-1, -2))
        rotation = torch.matrix_exp(self.rotation_scale * skew)
        basis = torch.matmul(self.shared_basis.to(descriptors), rotation)
        return torch.matmul(torch.matmul(basis, diagonal), basis.transpose(-1, -2))


def _orthogonal_matrix(
    dimension: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r)).clamp(min=-1.0, max=1.0)
    signs[signs == 0] = 1
    return q * signs


def make_representation_generator(
    *,
    family: RepresentationFamily,
    dimension: int,
    active_rank: int,
    descriptor_dim: int,
    rotation_scale: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> RepresentationDemandGenerator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    coefficient_map = torch.randn(
        active_rank,
        descriptor_dim,
        generator=generator,
        dtype=dtype,
    ) / descriptor_dim**0.5
    shared_basis = (
        torch.eye(dimension, dtype=dtype)
        if family is RepresentationFamily.AXIS_COMMUTING
        else _orthogonal_matrix(dimension, generator=generator, dtype=dtype)
    )
    rotation_maps = torch.randn(
        descriptor_dim,
        dimension,
        dimension,
        generator=generator,
        dtype=dtype,
    ) / dimension**0.5
    return RepresentationDemandGenerator(
        family=family,
        dimension=dimension,
        active_rank=active_rank,
        descriptor_dim=descriptor_dim,
        coefficient_map=coefficient_map,
        shared_basis=shared_basis,
        rotation_maps=rotation_maps,
        rotation_scale=float(rotation_scale),
    )


def sample_representation_descriptors(
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
