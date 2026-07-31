from __future__ import annotations

import pytest
import torch

from catena.lm.config import ModelConfig
from catena.lm.model import CatenaLM
from catena.lm.numerical_audit import (
    NumericalTolerances,
    audit_gradient_accumulation,
)


def _scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - 0.01 * step),
    )


def test_fp32_global_token_batch_is_independent_of_microbatch_partition() -> None:
    torch.manual_seed(260_201)
    model = CatenaLM(ModelConfig.tiny_reference())
    global_batch = torch.randint(0, model.config.vocab_size, (4, 16))
    rows = audit_gradient_accumulation(
        model,
        global_batch,
        accumulation_layouts=[(4,), (2, 2), (1, 1, 1, 1)],
        tolerances=NumericalTolerances(relative_l2_max=1.0e-5, max_abs_max=1.0e-5),
        autocast_dtype=None,
        scheduler_factory=_scheduler,
    )
    assert all(row.passed for row in rows)
    assert all(row.scheduler_digest_equal for row in rows)
    assert all(row.token_exposure_equal for row in rows)


def test_fp32_resource_feasible_baseline_need_not_be_one_full_microbatch() -> None:
    torch.manual_seed(260_203)
    model = CatenaLM(ModelConfig.tiny_reference())
    global_batch = torch.randint(0, model.config.vocab_size, (4, 16))
    rows = audit_gradient_accumulation(
        model,
        global_batch,
        accumulation_layouts=[(2, 2), (1, 1, 1, 1)],
        tolerances=NumericalTolerances(relative_l2_max=1.0e-5, max_abs_max=1.0e-5),
        autocast_dtype=None,
        scheduler_factory=_scheduler,
    )
    assert all(row.passed for row in rows)
    assert rows[0].microbatch_sizes == (2, 2)
    assert rows[1].microbatch_sizes == (1, 1, 1, 1)


@pytest.mark.e26_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable for E26 GPU audit")
def test_bf16_global_token_batch_uses_registered_tolerance() -> None:
    torch.manual_seed(260_202)
    mapping = ModelConfig.tiny_reference().to_dict()
    mapping.update(
        {
            "backend_id": "compiled_scan",
            "optimized_chunk_size": 8,
        }
    )
    model = CatenaLM(ModelConfig.from_mapping(mapping)).cuda()
    global_batch = torch.randint(0, model.config.vocab_size, (4, 16), device="cuda")
    rows = audit_gradient_accumulation(
        model,
        global_batch,
        accumulation_layouts=[(4,), (2, 2), (1, 1, 1, 1)],
        tolerances=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
        autocast_dtype=torch.bfloat16,
        scheduler_factory=_scheduler,
    )
    assert all(row.passed for row in rows)
