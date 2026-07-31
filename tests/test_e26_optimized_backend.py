from __future__ import annotations

import copy

import pytest
import torch

from catena.lm.config import ModelConfig
from catena.lm.interventions import AddressIntervention, GateIntervention
from catena.lm.recurrent_mixer import (
    MixerState,
    OptimizedBackendUnsupported,
    TransactionalDeltaMixer,
    optimized_backend_diagnostics,
    optimized_backend_metadata,
    reset_optimized_backend_diagnostics,
)


def _mixer(variant: str = "dual_delta_lm") -> TransactionalDeltaMixer:
    torch.manual_seed(31)
    return TransactionalDeltaMixer(ModelConfig.tiny_reference(variant), layer_index=0)


@pytest.mark.parametrize("variant", ["dual_delta_lm", "projected_tied_delta_lm"])
@pytest.mark.parametrize("sequence", [1, 7, 8, 9, 17])
def test_compiled_chunk_matches_reference(variant: str, sequence: int) -> None:
    mixer = _mixer(variant).eval()
    hidden = torch.randn(2, sequence, mixer.config.d_model)
    initial = MixerState(
        matrix=torch.randn(
            2,
            mixer.config.n_heads,
            mixer.config.head_dim,
            mixer.config.head_dim,
        )
        * 0.02
    )
    with torch.no_grad():
        expected, expected_state, expected_trace = mixer.forward_reference(
            hidden,
            initial.clone(),
            return_gate_trace=True,
        )
        actual, actual_state, actual_trace = mixer.forward_optimized(
            hidden,
            initial.clone(),
            chunk_size=8,
            compiler="eager",
            return_gate_trace=True,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_state.matrix, expected_state.matrix, rtol=1e-5, atol=1e-6)
    assert expected_trace is not None
    assert actual_trace is not None
    torch.testing.assert_close(actual_trace.erase, expected_trace.erase)
    torch.testing.assert_close(actual_trace.write, expected_trace.write)
    torch.testing.assert_close(actual_trace.decay, expected_trace.decay)


@pytest.mark.parametrize("split", [1, 5, 8])
def test_optimized_state_carry_matches_single_call(split: int) -> None:
    mixer = _mixer().eval()
    hidden = torch.randn(2, 11, mixer.config.d_model)
    initial = mixer.initial_state(2, device="cpu", dtype=hidden.dtype)
    with torch.no_grad():
        whole, whole_state, _ = mixer.forward_optimized(
            hidden,
            initial.clone(),
            chunk_size=4,
            compiler="eager",
        )
        prefix, prefix_state, _ = mixer.forward_optimized(
            hidden[:, :split],
            initial.clone(),
            chunk_size=4,
            compiler="eager",
        )
        suffix, suffix_state, _ = mixer.forward_optimized(
            hidden[:, split:],
            prefix_state,
            chunk_size=4,
            compiler="eager",
            token_offset=split,
        )
    torch.testing.assert_close(torch.cat((prefix, suffix), dim=1), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(suffix_state.matrix, whole_state.matrix, rtol=1e-5, atol=1e-6)


def test_vectorized_standard_interventions_match_reference() -> None:
    mixer = _mixer().eval()
    hidden = torch.randn(2, 9, mixer.config.d_model)
    token_mask = torch.tensor([False, True, False, True, True, False, False, True, False])
    gate = GateIntervention(
        erase_scale=0.4,
        write_scale=0.7,
        force_tied=True,
        token_mask=token_mask,
    )
    address = AddressIntervention(
        erase_address=torch.randn(mixer.config.head_dim),
        write_address=torch.randn(mixer.config.n_heads, mixer.config.head_dim),
        token_mask=token_mask,
    )
    with torch.no_grad():
        expected, expected_state, expected_trace = mixer.forward_reference(
            hidden,
            gate_intervention=gate,
            address_intervention=address,
            return_gate_trace=True,
        )
        actual, actual_state, actual_trace = mixer.forward_optimized(
            hidden,
            chunk_size=4,
            compiler="eager",
            gate_intervention=gate,
            address_intervention=address,
            return_gate_trace=True,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_state.matrix, expected_state.matrix, rtol=1e-5, atol=1e-6)
    assert expected_trace is not None
    assert actual_trace is not None
    torch.testing.assert_close(actual_trace.erase, expected_trace.erase)
    torch.testing.assert_close(actual_trace.write, expected_trace.write)


def test_reference_chunk_preserves_absolute_intervention_offset() -> None:
    mixer = _mixer().eval()
    hidden = torch.randn(1, 7, mixer.config.d_model)
    token_mask = torch.zeros(32, dtype=torch.bool)
    token_mask[12:15] = True
    intervention = GateIntervention(
        erase_scale=0.0,
        token_mask=token_mask,
    )
    with torch.no_grad():
        expected, expected_state, expected_trace = mixer.forward_reference(
            hidden,
            gate_intervention=intervention,
            return_gate_trace=True,
            token_offset=9,
        )
        actual, actual_state, actual_trace = mixer.forward_chunked_reference(
            hidden,
            chunk_size=3,
            gate_intervention=intervention,
            return_gate_trace=True,
            token_offset=9,
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_state.matrix, expected_state.matrix)
    assert expected_trace is not None
    assert actual_trace is not None
    torch.testing.assert_close(actual_trace.erase, expected_trace.erase)
    torch.testing.assert_close(actual_trace.write, expected_trace.write)


def test_custom_python_intervention_fails_closed() -> None:
    mixer = _mixer()
    hidden = torch.randn(1, 2, mixer.config.d_model)
    custom = GateIntervention(custom=lambda erase, write, _layer, _token: (erase, write))
    with pytest.raises(OptimizedBackendUnsupported, match="Custom Python"):
        mixer.forward_optimized(
            hidden,
            compiler="eager",
            gate_intervention=custom,
        )


def test_optimized_backward_is_finite_and_matches_reference() -> None:
    reference = _mixer().train()
    optimized = copy.deepcopy(reference).train()
    hidden_reference = torch.randn(2, 5, reference.config.d_model, requires_grad=True)
    hidden_optimized = hidden_reference.detach().clone().requires_grad_(True)

    expected, expected_state, _ = reference.forward_reference(hidden_reference)
    expected_loss = expected.square().mean() + expected_state.matrix.square().mean()
    torch.autograd.backward(expected_loss)

    actual, actual_state, _ = optimized.forward_optimized(
        hidden_optimized,
        chunk_size=4,
        compiler="eager",
    )
    actual_loss = actual.square().mean() + actual_state.matrix.square().mean()
    torch.autograd.backward(actual_loss)

    assert hidden_reference.grad is not None
    assert hidden_optimized.grad is not None
    assert torch.isfinite(hidden_optimized.grad).all()
    torch.testing.assert_close(hidden_optimized.grad, hidden_reference.grad, rtol=2e-5, atol=2e-7)
    for expected_parameter, actual_parameter in zip(
        reference.parameters(),
        optimized.parameters(),
        strict=True,
    ):
        assert expected_parameter.grad is not None
        assert actual_parameter.grad is not None
        assert torch.isfinite(actual_parameter.grad).all()
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=3e-5,
            atol=3e-7,
        )


def test_backend_metadata_and_diagnostics_are_fail_closed() -> None:
    reset_optimized_backend_diagnostics()
    mixer = _mixer().eval()
    with torch.no_grad():
        mixer.forward_optimized(
            torch.randn(1, 5, mixer.config.d_model),
            chunk_size=4,
            compiler="eager",
        )
    metadata = optimized_backend_metadata(device="cpu", compiler="eager", chunk_size=4)
    diagnostics = optimized_backend_diagnostics()
    assert metadata["backend_id"] == "torch_compile_fixed_chunk_scan_v1"
    assert metadata["python_token_loop_at_runtime"] is False
    assert metadata["scientific_main_capable"] is False
    assert metadata["tail_policy"] == "identity_pad_no_fallback"
    assert diagnostics["optimized_calls"] == 1
    assert diagnostics["chunks_executed"] == 2
    assert diagnostics["padded_tokens"] == 3
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["graph_break_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_inductor_fp32_and_bf16_diagnostics() -> None:
    device = torch.device("cuda")
    base = _mixer().eval()
    hidden_fp32 = torch.randn(1, 5, base.config.d_model, device=device)

    fp32 = copy.deepcopy(base).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        reference, reference_state, _ = fp32.forward_reference(hidden_fp32)
        actual, actual_state, _ = fp32.forward_optimized(
            hidden_fp32,
            chunk_size=4,
            compiler="inductor",
        )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_state.matrix, reference_state.matrix, rtol=1e-5, atol=1e-5)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bf16_output, bf16_state, _ = fp32.forward_optimized(
            hidden_fp32,
            chunk_size=4,
            compiler="inductor",
        )
    assert torch.isfinite(bf16_output).all()
    assert torch.isfinite(bf16_state.matrix).all()
    relative_l2 = (bf16_output.float() - actual.float()).norm() / actual.float().norm().clamp_min(
        torch.finfo(torch.float32).eps
    )
    # This deterministic candidate must meet the registered tolerance, but it
    # does not replace E26a's complete fail-closed parity grid.
    assert torch.isfinite(relative_l2)
    assert float(relative_l2) <= 7.0e-3

    training = copy.deepcopy(base).to(device=device, dtype=torch.float32).train()
    training_hidden = hidden_fp32.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        training_output, training_state, _ = training.forward_optimized(
            training_hidden,
            chunk_size=4,
            compiler="inductor",
        )
    torch.autograd.backward(training_output.square().mean() + training_state.matrix.square().mean())
    assert training_hidden.grad is not None
    assert torch.isfinite(training_hidden.grad).all()
    for parameter in training.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    candidate = optimized_backend_metadata(device=device, compiler="inductor", chunk_size=4)
    verified = optimized_backend_metadata(
        device=device,
        compiler="inductor",
        chunk_size=4,
        parity_verified=True,
    )
    assert candidate["candidate_codegen_capable"] is True
    assert candidate["scientific_main_capable"] is False
    assert verified["scientific_main_capable"] is True
