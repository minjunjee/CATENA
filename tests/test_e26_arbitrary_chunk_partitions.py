from __future__ import annotations

import pytest
import torch

from catena.lm.config import ModelConfig
from catena.lm.model import CatenaLM
from catena.lm.numerical_audit import (
    NumericalTolerances,
    audit_arbitrary_partitions,
    fixed_partition_suite,
)


def _config(*, backend_id: str = "reference_python") -> ModelConfig:
    return ModelConfig(
        vocab_size=37,
        n_layers=2,
        d_model=8,
        n_heads=2,
        ffn_multiplier=1.0,
        recurrent_layers=(0,),
        local_attention_layers=(1,),
        local_attention_window=7,
        context_length=416,
        reference_chunk_size=13,
        optimized_chunk_size=32,
        dropout=0.0,
        backend_id=backend_id,
    )


def test_mandatory_and_eight_random_partitions_cover_the_sequence() -> None:
    partitions = fixed_partition_suite(416, random_seeds=range(260_100, 260_108))
    assert partitions[:4] == (
        (416,),
        (1, 415),
        (3, 5, 7, 401),
        (31, 127, 257, 1),
    )
    assert len(partitions) == 12
    assert len(set(partitions)) == 12
    assert all(sum(partition) == 416 for partition in partitions)


def test_fp32_arbitrary_partitions_preserve_hybrid_state_and_gradients() -> None:
    torch.manual_seed(260_110)
    model = CatenaLM(_config()).eval()
    prefix = torch.randint(0, model.config.vocab_size, (1, 9))
    with torch.no_grad():
        initial = model(prefix).runtime_state.clone(detach=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 416))
    partitions = fixed_partition_suite(416, random_seeds=range(260_111, 260_119))
    report = audit_arbitrary_partitions(
        model,
        input_ids,
        partitions=partitions,
        tolerances=NumericalTolerances(relative_l2_max=1.0e-5, max_abs_max=1.0e-5),
        autocast_dtype=None,
        initial_state=initial,
    )
    assert report.passed
    assert all(row.runtime_state.position_equal for row in report.rows)
    assert all(row.runtime_state.positions_equal for row in report.rows)
    assert all(row.runtime_state.write_indices_equal for row in report.rows)
    assert all(row.gradients_finite for row in report.rows)


@pytest.mark.e26_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable for E26 GPU audit")
def test_bf16_cuda_arbitrary_partitions_use_registered_tolerance() -> None:
    torch.manual_seed(260_120)
    device = torch.device("cuda")
    model = CatenaLM(_config(backend_id="compiled_scan")).to(device).eval()
    prefix = torch.randint(0, model.config.vocab_size, (1, 9), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        initial = model(prefix).runtime_state.clone(detach=True)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 416), device=device)
    partitions = fixed_partition_suite(416, random_seeds=range(260_121, 260_129))
    report = audit_arbitrary_partitions(
        model,
        input_ids,
        partitions=partitions,
        tolerances=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
        autocast_dtype=torch.bfloat16,
        initial_state=initial,
    )
    assert report.passed
