from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import torch

from catena.core.provenance_v61 import read_json_object_strict, sha256_canonical_json
from catena.lm.e26_final_official_adapter import (
    cache_tensor_equality_and_no_alias,
)
from tools.audit_e26_final_official_runtime import (
    DispatchCounts,
    E26FinalOfficialRuntimeError,
    KernelDispatchInstrumentation,
    _build_success_receipt,
    _configure_layers,
    audit_gpt_decode_cache_contract,
    write_runtime_receipt,
)


def _kernel_module() -> tuple[ModuleType, object, object]:
    module = ModuleType("lit_gpt.gdn2")

    def chunk_gdn2(value: torch.Tensor) -> tuple[torch.Tensor, None]:
        return value.square(), None

    def fused_recurrent_gdn2(value: torch.Tensor) -> tuple[torch.Tensor, None]:
        return value + 1, None

    module.__dict__["chunk_gdn2"] = chunk_gdn2
    module.__dict__["fused_recurrent_gdn2"] = fused_recurrent_gdn2
    return module, chunk_gdn2, fused_recurrent_gdn2


def test_dispatch_instrumentation_is_positive_phase_scoped_and_restored() -> None:
    module, original_chunk, original_fused = _kernel_module()
    value = torch.tensor([2.0], requires_grad=True)

    with KernelDispatchInstrumentation(module) as audit:
        before = audit.snapshot()
        audit.set_phase("chunk_training")
        output, _ = module.chunk_gdn2(value)
        output.sum().backward()
        chunk = audit.snapshot().subtract(before)
        assert chunk == DispatchCounts(
            chunk_attempted=1,
            chunk_completed=1,
            chunk_backward_hooks_registered=1,
            chunk_backward_hooks_completed=1,
            chunk_backward_finite=1,
            chunk_backward_nonzero=1,
        )

        before_fused = audit.snapshot()
        audit.set_phase("adapter_fused_query")
        fused, _ = module.fused_recurrent_gdn2(value.detach())
        assert torch.equal(fused, torch.tensor([3.0]))
        fused_counts = audit.snapshot().subtract(before_fused)
        assert fused_counts.fused_attempted == 1
        assert fused_counts.fused_completed == 1
        assert fused_counts.chunk_attempted == 0

    assert module.chunk_gdn2 is original_chunk
    assert module.fused_recurrent_gdn2 is original_fused


def test_dispatch_instrumentation_never_retries_or_counts_failed_call_complete() -> None:
    module, _, _ = _kernel_module()
    attempts = 0

    def failing(value: torch.Tensor) -> tuple[torch.Tensor, None]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("kernel failed")

    module.__dict__["chunk_gdn2"] = failing
    with KernelDispatchInstrumentation(module) as audit:
        audit.set_phase("chunk_training")
        with pytest.raises(RuntimeError, match="kernel failed"):
            module.chunk_gdn2(torch.ones(1))
        assert attempts == 1
        assert audit.counts.chunk_attempted == 1
        assert audit.counts.chunk_completed == 0


_DEFECTIVE_MODEL_SOURCE = """
class GPT:
    def build_kv_caches(self):
        caches = []
        caches.append(None)
        return caches

class Block:
    def __init__(self):
        self.attn = GatedDeltaNet2(hidden_size=2304)

    def forward(self, n_1):
        h, _, cache = self.attn(n_1, attention_mask=None)
        return h, cache
"""


_WIRED_MODEL_SOURCE = """
class GPT:
    def build_kv_caches(self):
        caches = []
        caches.append(Cache())
        return caches

class Block:
    def __init__(self, layer_idx):
        self.attn = GatedDeltaNet2(hidden_size=2304, layer_idx=layer_idx)

    def forward(self, n_1, cache):
        h, _, cache = self.attn(
            n_1,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )
        return h, cache
"""


def test_gpt_decode_cache_defect_is_explicitly_classified_not_silently_patched() -> None:
    report = audit_gpt_decode_cache_contract(_DEFECTIVE_MODEL_SOURCE)
    assert report == {
        "official_gpt_autoregressive_decode_supported": False,
        "constructor_layer_idx_forwarded": False,
        "block_past_key_values_and_use_cache_forwarded": False,
        "gdn2_cache_slot_allocated": False,
        "catena_adapter_required_for_gpt_cache": True,
        "disposition": "KNOWN_OFFICIAL_GPT_DECODE_CACHE_PLUMBING_DEFECT",
        "scientific_gpt_decode_eligible": False,
    }

    wired = audit_gpt_decode_cache_contract(_WIRED_MODEL_SOURCE)
    assert wired["official_gpt_autoregressive_decode_supported"] is True
    assert wired["scientific_gpt_decode_eligible"] is True
    assert wired["catena_adapter_required_for_gpt_cache"] is False


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx = None
        self.allow_neg_eigval = True
        self.mode = "fused_recurrent"


def test_layer_configuration_requires_one_explicit_policy_for_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = [_TinyLayer(), _TinyLayer()]
    monkeypatch.setattr(
        "tools.audit_e26_final_official_runtime.EXPECTED_LAYER_COUNT",
        2,
    )
    report = _configure_layers(layers, variant="projected_tied_gdn2")
    assert report["passed"] is True
    assert [layer.layer_idx for layer in layers] == [0, 1]
    assert all(layer.e26_gate_policy == "projected_tied_gdn2" for layer in layers)
    assert all(layer.allow_neg_eigval is False for layer in layers)
    assert all(layer.mode == "chunk" for layer in layers)

    with pytest.raises(E26FinalOfficialRuntimeError, match="Unknown explicit gate policy"):
        _configure_layers(layers, variant="unknown")


def test_cache_snapshot_clone_is_equal_and_storage_disjoint() -> None:
    recurrent = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    convolution = (torch.ones(1, 2), torch.zeros(1, 2))
    state: dict[str, object] = {
        "recurrent_state": recurrent,
        "conv_state": convolution,
    }
    cloned = {
        "recurrent_state": recurrent.clone(),
        "conv_state": tuple(tensor.clone() for tensor in convolution),
    }
    assert cache_tensor_equality_and_no_alias(state, cloned)["passed"] is True
    cloned_recurrent = cloned["recurrent_state"]
    original_recurrent = state["recurrent_state"]
    assert isinstance(cloned_recurrent, torch.Tensor)
    assert isinstance(original_recurrent, torch.Tensor)
    cloned_recurrent.add_(1)
    assert not torch.equal(original_recurrent, cloned_recurrent)


def _passed_section() -> dict[str, object]:
    return {"passed": True}


def test_success_receipt_keeps_operator_pass_separate_from_gpt_decode(
    tmp_path: Path,
) -> None:
    receipt = _build_success_receipt(
        variant="dual_gdn2",
        device_record={"available": True},
        source_binding=_passed_section(),
        fla_binding=_passed_section(),
        runtime_binding=_passed_section(),
        kernel_binding=_passed_section(),
        checkpoint_binding=_passed_section(),
        layer_configuration=_passed_section(),
        chunk_training=_passed_section(),
        gpt_cache_adapter=_passed_section(),
        decode_contract={
            "disposition": "KNOWN_OFFICIAL_GPT_DECODE_CACHE_PLUMBING_DEFECT",
            "scientific_gpt_decode_eligible": False,
        },
    )
    assert receipt["passed"] is True
    assert receipt["official_training_chunk_runtime_eligible"] is True
    assert receipt["official_fused_kernel_runtime_eligible"] is True
    assert receipt["upstream_official_gpt_autoregressive_decode_eligible"] is False
    assert receipt["catena_cache_adapter_gpt_autoregressive_decode_eligible"] is True
    assert receipt["e26_final_gpt_runtime_eligible"] is True
    assert receipt["scientific_e26a_started"] is False
    assert receipt["scientific_evidence"] is False
    claimed = receipt["receipt_sha256"]
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256")
    assert claimed == sha256_canonical_json(unhashed)

    output = tmp_path / "runtime.json"
    write_runtime_receipt(output, receipt)
    assert read_json_object_strict(output) == receipt
    with pytest.raises(FileExistsError):
        write_runtime_receipt(output, receipt)
