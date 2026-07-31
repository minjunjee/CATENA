import torch

from catena.lm.locality import (
    active_key_covariance,
    covariance_aware_direction,
    protected_nullspace_direction,
)


def test_covariance_direction_has_unit_target_response() -> None:
    torch.manual_seed(2)
    target = torch.nn.functional.normalize(torch.randn(12), dim=0)
    keys = torch.nn.functional.normalize(torch.randn(24, 12), dim=-1)
    covariance = active_key_covariance(keys)
    result = covariance_aware_direction(covariance, target, regularization=1e-2)
    assert result.unit_response_error < 1e-5
    assert torch.isfinite(result.direction).all()


def test_protected_projection_has_unit_response() -> None:
    torch.manual_seed(3)
    target = torch.randn(8)
    protected = torch.randn(3, 8)
    result = protected_nullspace_direction(target, protected)
    assert result.unit_response_error < 1e-4
