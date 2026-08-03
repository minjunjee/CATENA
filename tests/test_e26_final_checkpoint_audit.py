from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.provenance_v61 import read_json_object_strict
from tools.audit_e26_final_checkpoint import (
    CheckpointAuditSpec,
    E26FinalCheckpointAuditError,
    audit_checkpoint_payload,
    audit_strict_model_load,
    build_checkpoint_audit_receipt,
    safe_load_checkpoint,
    validate_checkpoint_audit_receipt,
    write_checkpoint_audit_receipt,
)


class _TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b_proj = torch.nn.Linear(3, 2, bias=False)
        self.w_proj = torch.nn.Linear(3, 2, bias=False)


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _TinyAttention()


class _TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(5, 3)
        self.h = torch.nn.ModuleList([_TinyBlock(), _TinyBlock()])


class _TinyGPT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyTransformer()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)


def _tiny_model() -> _TinyGPT:
    torch.manual_seed(7)
    return _TinyGPT()


def _tiny_state() -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in _tiny_model().state_dict().items()}


def _tiny_spec() -> CheckpointAuditSpec:
    state = _tiny_state()
    return CheckpointAuditSpec(
        checkpoint_bytes=0,
        checkpoint_sha256="0" * 64,
        model_key_count=len(state),
        full_model_numel=sum(value.numel() for value in state.values()),
        transformer_h_numel=sum(
            value.numel() for key, value in state.items() if key.startswith("transformer.h.")
        ),
        vocab_size=5,
        hidden_size=3,
        layer_count=2,
        projection_output_size=2,
        max_tokens=100,
        model_name="tiny-test-only",
    )


def _payload() -> dict[str, Any]:
    return {
        "model": _tiny_state(),
        "optimizer": None,
        "hparams": {"max_tokens": 100},
        "iter_num": 8,
        "step_count": 4,
    }


def _passed_file_section() -> dict[str, Any]:
    return {
        "hard_checks": {
            "regular_file_not_symlink": True,
            "checkpoint_bytes_exact": True,
            "checkpoint_sha256_exact": True,
        },
        "passed": True,
    }


def test_safe_load_uses_only_weights_only_cpu_mmap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny.pth"
    checkpoint.write_bytes(b"fixture placeholder")
    expected = _payload()
    observed: dict[str, object] = {}

    def fake_load(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
        mmap: bool,
    ) -> object:
        observed.update(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
                "mmap": mmap,
            }
        )
        return expected

    monkeypatch.setattr(torch, "load", fake_load)
    assert safe_load_checkpoint(checkpoint) is expected
    assert observed == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": True,
        "mmap": True,
    }


def test_safe_load_accepts_a_real_tiny_weights_only_mmap_fixture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save(_payload(), checkpoint)
    loaded = safe_load_checkpoint(checkpoint)
    assert set(loaded) == {"model", "optimizer", "hparams", "iter_num", "step_count"}
    assert isinstance(loaded["model"], dict)


def test_payload_audit_enforces_exact_structure_counts_shapes_and_max_tokens() -> None:
    report = audit_checkpoint_payload(_payload(), spec=_tiny_spec())
    assert report["passed"] is True
    assert all(report["hard_checks"].values())
    inventory = report["observed"]["projection_inventory"]
    assert set(inventory["b_proj"]) == {"0", "1"}
    assert set(inventory["w_proj"]) == {"0", "1"}


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("top_key", "top_level_keys_exact"),
        ("nonfinite", "all_model_tensors_finite"),
        ("projection_shape", "projection_shapes_exact"),
        ("max_tokens", "hparams_max_tokens_exact"),
    ],
)
def test_payload_audit_fails_closed_on_each_contract_class(
    mutation: str,
    failed_check: str,
) -> None:
    payload = _payload()
    if mutation == "top_key":
        payload["extra"] = 1
    elif mutation == "nonfinite":
        payload["model"]["lm_head.weight"][0, 0] = float("nan")
    elif mutation == "projection_shape":
        payload["model"]["transformer.h.0.attn.b_proj.weight"] = torch.zeros(1, 3)
    elif mutation == "max_tokens":
        payload["hparams"]["max_tokens"] = 99
    report = audit_checkpoint_payload(payload, spec=_tiny_spec())
    assert report["passed"] is False
    assert report["hard_checks"][failed_check] is False


def test_strict_model_load_receipt_checks_official_counts_keys_and_shapes() -> None:
    state = _tiny_state()
    report = audit_strict_model_load(
        state,
        model_factory=_tiny_model,
        spec=_tiny_spec(),
    )
    assert report["passed"] is True
    assert all(report["hard_checks"].values())
    assert report["load_policy"] == {
        "strict": True,
        "assign": True,
        "missing_keys_allowed": False,
        "unexpected_keys_allowed": False,
    }


def test_strict_model_load_has_no_partial_or_shape_fallback() -> None:
    state = _tiny_state()
    state.pop("lm_head.weight")
    report = audit_strict_model_load(
        state,
        model_factory=_tiny_model,
        spec=_tiny_spec(),
    )
    assert report["passed"] is False
    assert report["hard_checks"]["official_state_key_set_exact"] is False
    assert report["hard_checks"]["strict_state_dict_load_completed"] is False
    assert report["observed"]["load_error_type"] == "RuntimeError"


def test_receipt_keeps_95b_alias_warning_outside_hard_gates_and_is_immutable(
    tmp_path: Path,
) -> None:
    structure = audit_checkpoint_payload(_payload(), spec=_tiny_spec())
    official = audit_strict_model_load(
        _tiny_state(), model_factory=_tiny_model, spec=_tiny_spec()
    )
    receipt = build_checkpoint_audit_receipt(
        checkpoint_file=_passed_file_section(),
        structure=structure,
        official_load=official,
    )
    assert receipt["passed"] is True
    assert receipt["scientific_evidence"] is False
    assert "CHECKPOINT_95B_AND_100B_BYTE_IDENTICAL" not in receipt[
        "protocol_hard_checks"
    ]
    warning = receipt["warnings"][0]
    assert warning["code"] == "CHECKPOINT_95B_AND_100B_BYTE_IDENTICAL"
    assert warning["protocol_hard_gate"] is False

    destination = tmp_path / "receipt.json"
    write_checkpoint_audit_receipt(destination, receipt)
    assert read_json_object_strict(destination) == receipt
    with pytest.raises(FileExistsError):
        write_checkpoint_audit_receipt(destination, receipt)

    tampered = deepcopy(receipt)
    tampered["warnings"][0]["detail"] = "changed"
    with pytest.raises(E26FinalCheckpointAuditError, match="SHA-256"):
        validate_checkpoint_audit_receipt(tampered)


def test_file_binding_spec_can_be_derived_for_tiny_fixture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pth"
    torch.save(_payload(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec = replace(
        _tiny_spec(),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=digest,
    )
    assert spec.checkpoint_bytes > 0
    assert spec.checkpoint_sha256 == digest
