from __future__ import annotations

import torch

from catena.data.representation_dynamics import (
    RepresentationFamily,
    make_representation_generator,
    sample_representation_descriptors,
)
from catena.models.coadaptation import LearnedBasisDiagonalController


def test_representation_family_shapes_and_commuting_axis() -> None:
    descriptor = sample_representation_descriptors(count=4, descriptor_dim=6, seed=4)
    generator = make_representation_generator(
        family=RepresentationFamily.AXIS_COMMUTING,
        dimension=12,
        active_rank=4,
        descriptor_dim=6,
        rotation_scale=0.3,
        seed=5,
    )
    operators = generator.operators(descriptor)
    assert operators.shape == (4, 12, 12)
    commutator = operators[0] @ operators[1] - operators[1] @ operators[0]
    assert float(commutator.abs().max()) < 1e-6


def test_learned_basis_controller_is_finite() -> None:
    model = LearnedBasisDiagonalController(
        descriptor_dim=6, dimension=12, hidden_dim=24
    )
    output = model(torch.randn(3, 6))
    assert output.shape == (3, 12, 12)
    assert torch.isfinite(output).all()
