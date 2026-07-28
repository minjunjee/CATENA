from __future__ import annotations

import torch

from catena.data.learned_rank import best_rank_error, make_low_rank_family, sample_descriptors
from catena.models.operator_controllers import LowRankOperatorController


def test_low_rank_family_and_oracle_floor() -> None:
    family = make_low_rank_family(
        dimension=16, descriptor_dim=5, intrinsic_rank=4, seed=1
    )
    descriptor = sample_descriptors(count=8, descriptor_dim=5, seed=2)
    target = family.operator(descriptor)
    assert target.shape == (8, 16, 16)
    floor = best_rank_error(target, 4)
    assert float(floor.max()) < 1e-10
    lower = best_rank_error(target, 2)
    assert float(lower.mean()) > float(floor.mean())


def test_low_rank_controller_shape_and_gradient() -> None:
    model = LowRankOperatorController(
        descriptor_dim=5, dimension=16, rank=4, hidden_dim=32
    )
    descriptor = torch.randn(7, 5)
    output = model(descriptor)
    assert output.shape == (7, 16, 16)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
