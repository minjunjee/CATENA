from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from catena.lm.e26_final_gate_policy import (
    E26FinalGatePolicy,
    E26FinalGatePolicyAdapter,
    project_e26_final_gates,
)


def test_dual_policy_applies_independent_sigmoids() -> None:
    z_b = torch.tensor([[[-2.0, 0.0, 2.0]]])
    z_w = torch.tensor([[[1.0, -1.0, 0.5]]])

    b, w = project_e26_final_gates(z_b, z_w, policy="dual_gdn2")

    torch.testing.assert_close(b, torch.sigmoid(z_b), rtol=0.0, atol=0.0)
    torch.testing.assert_close(w, torch.sigmoid(z_w), rtol=0.0, atol=0.0)


def test_projected_tied_policy_averages_logits_before_sigmoid() -> None:
    z_b = torch.tensor([[[-4.0, 1.0, 3.0]]])
    z_w = torch.tensor([[[2.0, -3.0, 1.0]]])
    expected = torch.sigmoid((z_b + z_w) / 2.0)

    b, w = project_e26_final_gates(
        z_b,
        z_w,
        policy="projected_tied_gdn2",
    )

    torch.testing.assert_close(b, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(w, expected, rtol=0.0, atol=0.0)
    assert b is w


@pytest.mark.parametrize("policy", ["dual_gdn2", "projected_tied_gdn2"])
def test_adapter_is_parameter_free_and_preserves_both_projection_heads(
    policy: E26FinalGatePolicy,
) -> None:
    b_proj = nn.Linear(7, 5, bias=False)
    w_proj = nn.Linear(7, 5, bias=False)
    b_weight_id = id(b_proj.weight)
    w_weight_id = id(w_proj.weight)
    hidden = torch.randn(2, 3, 7)

    adapter = E26FinalGatePolicyAdapter(policy)
    b, w = adapter(b_proj(hidden), w_proj(hidden))
    (b.sum() + w.sum()).backward()

    assert list(adapter.parameters()) == []
    assert id(b_proj.weight) == b_weight_id
    assert id(w_proj.weight) == w_weight_id
    assert b_proj.weight is not w_proj.weight
    assert b_proj.weight.grad is not None
    assert w_proj.weight.grad is not None
    assert b.shape == w.shape == (2, 3, 5)


def test_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        project_e26_final_gates(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 5),
            policy="dual_gdn2",
        )


def test_dtype_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="identical dtypes"):
        project_e26_final_gates(
            torch.zeros(2, 3, 4, dtype=torch.float32),
            torch.zeros(2, 3, 4, dtype=torch.float64),
            policy="dual_gdn2",
        )


def test_allow_neg_eigval_true_fails_closed() -> None:
    with pytest.raises(ValueError, match="allow_neg_eigval=False"):
        E26FinalGatePolicyAdapter("dual_gdn2", allow_neg_eigval=True)

    with pytest.raises(ValueError, match="allow_neg_eigval=False"):
        project_e26_final_gates(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            policy="dual_gdn2",
            allow_neg_eigval=True,
        )


def test_runtime_mutation_cannot_enable_negative_eigenvalue_path() -> None:
    adapter = E26FinalGatePolicyAdapter("dual_gdn2")
    adapter.allow_neg_eigval = True

    with pytest.raises(ValueError, match="allow_neg_eigval=False"):
        adapter(torch.zeros(2, 4), torch.zeros(2, 4))


def test_unknown_policy_fails_closed() -> None:
    unsupported = cast(E26FinalGatePolicy, "unknown")
    with pytest.raises(ValueError, match="Unsupported E26 Final gate policy"):
        E26FinalGatePolicyAdapter(unsupported)
