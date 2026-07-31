import copy

import pytest
import torch

from catena.lm import ModelConfig, build_paired_models
from catena.lm.model import CatenaLM, LocalCausalSelfAttention, SwiGLU


def _hybrid_config(variant: str = "dual_delta_lm") -> ModelConfig:
    return ModelConfig(
        vocab_size=67,
        n_layers=3,
        d_model=16,
        n_heads=4,
        ffn_multiplier=2.0,
        recurrent_layers=(0, 2),
        local_attention_layers=(1,),
        local_attention_window=5,
        context_length=64,
        reference_chunk_size=3,
        variant=variant,
    )


def test_swiglu_low_precision_output_projection_preserves_input_dtype() -> None:
    config = _hybrid_config()
    module = SwiGLU(config)
    value = torch.randn(2, 7, config.d_model, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(value)
    output.square().mean().backward()

    assert output.dtype == value.dtype
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def _assert_runtime_close(left, right) -> None:
    assert left.position == right.position
    assert len(left.recurrent) == len(right.recurrent)
    assert len(left.attention) == len(right.attention)
    for left_state, right_state in zip(left.recurrent, right.recurrent, strict=True):
        torch.testing.assert_close(left_state.matrix, right_state.matrix)
    for left_state, right_state in zip(left.attention, right.attention, strict=True):
        torch.testing.assert_close(left_state.key, right_state.key)
        torch.testing.assert_close(left_state.value, right_state.value)
        assert torch.equal(left_state.positions, right_state.positions)
        assert left_state.length == right_state.length
        assert left_state.write_index == right_state.write_index


@pytest.mark.parametrize("variant", ["dual_delta_lm", "projected_tied_delta_lm"])
def test_hybrid_full_incremental_and_chunked_reference_parity(variant: str) -> None:
    torch.manual_seed(101)
    model = CatenaLM(_hybrid_config(variant)).eval()
    inputs = torch.randint(0, model.config.vocab_size, (2, 17))

    with torch.no_grad():
        full = model(inputs)
        first = model(inputs[:, :6])
        second = model(inputs[:, 6:11], first.runtime_state)
        third = model(
            inputs[:, 11:],
            second.runtime_state,
            chunked_reference=True,
        )

    incremental_logits = torch.cat((first.logits, second.logits, third.logits), dim=1)
    torch.testing.assert_close(full.logits, incremental_logits, rtol=1e-5, atol=1e-6)
    _assert_runtime_close(full.runtime_state, third.runtime_state)

    attention_state = third.runtime_state.attention[0]
    assert attention_state.length == model.config.local_attention_window
    assert torch.equal(
        torch.sort(attention_state.positions).values,
        torch.arange(12, 17),
    )
    assert attention_state.write_index == 17 % model.config.local_attention_window


def test_local_window_is_exact_across_ring_wrap_and_chunk_boundaries() -> None:
    config = ModelConfig(
        vocab_size=17,
        n_layers=1,
        d_model=4,
        n_heads=1,
        ffn_multiplier=1.0,
        recurrent_layers=(),
        local_attention_layers=(0,),
        local_attention_window=3,
        context_length=32,
    )
    attention = LocalCausalSelfAttention(config).eval()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.weight[2 * config.d_model :].copy_(torch.eye(config.d_model))
        attention.out_proj.weight.copy_(torch.eye(config.d_model))

    hidden = torch.arange(1, 29, dtype=torch.float32).view(1, 7, 4)
    expected = torch.stack(
        [
            hidden[:, max(0, index - config.local_attention_window + 1) : index + 1].mean(dim=1)
            for index in range(hidden.shape[1])
        ],
        dim=1,
    )

    with torch.no_grad():
        full, full_state = attention.forward_with_state(hidden)
        prefix, prefix_state = attention.forward_with_state(hidden[:, :2])
        middle, middle_state = attention.forward_with_state(
            hidden[:, 2:5],
            prefix_state,
            position_offset=2,
        )
        suffix, suffix_state = attention.forward_with_state(
            hidden[:, 5:],
            middle_state,
            position_offset=5,
        )

    incremental = torch.cat((prefix, middle, suffix), dim=1)
    torch.testing.assert_close(full, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(incremental, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full_state.key, suffix_state.key)
    torch.testing.assert_close(full_state.value, suffix_state.value)
    assert torch.equal(full_state.positions, suffix_state.positions)
    assert torch.equal(torch.sort(suffix_state.positions).values, torch.arange(4, 7))


def test_local_attention_complete_sequence_behavior_is_preserved() -> None:
    config = _hybrid_config()
    attention = LocalCausalSelfAttention(config).eval()
    hidden = torch.randn(2, 11, config.d_model)
    with torch.no_grad():
        q, k, value = attention.qkv(hidden).chunk(3, dim=-1)
        q = q.view(2, 11, config.n_heads, config.head_dim).transpose(1, 2)
        k = k.view(2, 11, config.n_heads, config.head_dim).transpose(1, 2)
        value = value.view(2, 11, config.n_heads, config.head_dim).transpose(1, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            value,
            attn_mask=attention._mask(11, hidden.device),
        )
        expected = expected.transpose(1, 2).contiguous().view(2, 11, config.d_model)
        expected = attention.out_proj(expected)
        actual = attention(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_hybrid_runtime_clone_detach_and_storage_coverage() -> None:
    torch.manual_seed(23)
    model = CatenaLM(_hybrid_config()).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 8))
    state = model(inputs).runtime_state
    attached = state.clone()
    detached = state.clone(detach=True)

    expected_pointer_count = len(state.recurrent) + 3 * len(state.attention)
    assert len(state.storage_ptrs()) == expected_pointer_count
    assert not (set(state.storage_ptrs()) & set(attached.storage_ptrs()))
    assert not (set(state.storage_ptrs()) & set(detached.storage_ptrs()))
    assert all(item.matrix.requires_grad for item in attached.recurrent)
    assert all(item.key.requires_grad and item.value.requires_grad for item in attached.attention)
    assert all(not item.matrix.requires_grad for item in detached.recurrent)
    assert all(
        not item.key.requires_grad and not item.value.requires_grad for item in detached.attention
    )

    original = state.clone(detach=True)
    with torch.no_grad():
        attached.recurrent[0].matrix.add_(1)
        attached.attention[0].key.add_(1)
        attached.attention[0].value.sub_(1)
        attached.attention[0].positions.fill_(-1)
    _assert_runtime_close(state, original)


def test_hybrid_branches_are_order_independent_and_do_not_mutate_prefix() -> None:
    torch.manual_seed(47)
    model = CatenaLM(_hybrid_config()).eval()
    prefix_ids = torch.randint(0, model.config.vocab_size, (1, 9))
    branch_a_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    branch_b_ids = torch.randint(0, model.config.vocab_size, (1, 6))

    with torch.no_grad():
        prefix = model(prefix_ids)
        frozen_prefix = prefix.runtime_state.clone(detach=True)

        branch_a_first = model(branch_a_ids, prefix.runtime_state)
        branch_b_second = model(branch_b_ids, prefix.runtime_state)
        branch_b_first = model(branch_b_ids, prefix.runtime_state)
        branch_a_second = model(branch_a_ids, prefix.runtime_state)

    torch.testing.assert_close(branch_a_first.logits, branch_a_second.logits)
    torch.testing.assert_close(branch_b_first.logits, branch_b_second.logits)
    _assert_runtime_close(branch_a_first.runtime_state, branch_a_second.runtime_state)
    _assert_runtime_close(branch_b_first.runtime_state, branch_b_second.runtime_state)
    _assert_runtime_close(prefix.runtime_state, frozen_prefix)


def test_hybrid_variants_keep_identical_registered_parameter_surface() -> None:
    tied, dual = build_paired_models(_hybrid_config(), seed=71)
    assert tied.parameter_signature() == dual.parameter_signature()
    assert tied.parameter_count() == dual.parameter_count()
    for tied_parameter, dual_parameter in zip(tied.parameters(), dual.parameters(), strict=True):
        assert torch.equal(tied_parameter, dual_parameter)


def test_hybrid_state_rejects_inconsistent_absolute_position_metadata() -> None:
    model = CatenaLM(_hybrid_config()).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        state = model(inputs).runtime_state
    corrupted = copy.deepcopy(state)
    corrupted.attention[0].positions[0] = 99
    continuation = torch.randint(0, model.config.vocab_size, (1, 1))
    with pytest.raises(ValueError, match="contiguous suffix"):
        model(continuation, corrupted)
