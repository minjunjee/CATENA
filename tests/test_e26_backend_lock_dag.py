from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from catena.core.provenance_v61 import sha256_canonical_json
from catena.lm.audit_contract import e26_execution_source_inventory
from catena.lm.backend_lock import (
    backend_candidate_lock_payload,
    backend_preflight_manifest,
    observed_single_visible_cuda_device,
    validate_backend_candidate_lock,
    validate_backend_preflight_manifest,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
    ).strip()


def _commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True)


def _repo(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "config", "user.email", "e26-test@example.invalid")
    _git(repo, "config", "user.name", "E26 Test")
    config = repo / "config.yaml"
    config.write_text("model_candidates: []\n", encoding="utf-8")
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _commit(repo, "executable source")
    candidates: list[dict[str, object]] = [
        {
            "id": "candidate_a",
            "vocab_size": 16_384,
            "d_model": 32,
        }
    ]
    return repo, config, candidates


def _write_receipt(path: Path, *, kind: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "catena-v8.1",
        "manifest_type": kind,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def _compiled_diagnostics(*, compiled_here: bool) -> dict[str, object]:
    return {
        "graph_compilations": 1 if compiled_here else 0,
        "graph_invocations": 2,
        "optimized_calls": 1,
        "chunks_executed": 2,
        "padded_tokens": 0,
        "fallback_count": 0,
        "graph_break_count": 0,
        "last_graph_node_count": 8 if compiled_here else 0,
        "last_graph_code_sha256": "a" * 64 if compiled_here else None,
    }


def _hardware() -> list[dict[str, object]]:
    return [
        {
            "physical_device_index": 0,
            "name": "test",
            "total_memory_bytes": 1,
            "compute_capability": "0.0",
            "driver_version": "test",
            "gpu_uuid": "GPU-test",
        }
    ]


def _execution_device() -> dict[str, object]:
    return {
        "physical_device_index": 0,
        "gpu_uuid": "GPU-test",
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": "0",
        "name": "test",
        "total_memory_bytes": 1,
        "compute_capability": "0.0",
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }


def _restart_receipt(path: Path) -> dict[str, object]:
    payload = _write_receipt(path, kind="E26_RESTART_AUDIT")
    payload["resume_cases"] = {
        "candidate_a__dual_delta_lm__general_to_transaction": {
            "physical_device_index": 0,
            "gpu_uuid": "GPU-test",
            "execution_device": _execution_device(),
            "passed": True,
        }
    }
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def test_worker_observes_the_single_visible_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _index: SimpleNamespace(
            uuid="test-uuid",
            name="test",
            total_memory=1,
            major=0,
            minor=0,
        ),
    )
    observed = observed_single_visible_cuda_device(
        expected_physical_index=3,
        expected_gpu_uuid="GPU-test-uuid",
    )
    assert observed["cuda_visible_devices"] == "3"
    assert observed["gpu_uuid"] == "GPU-test-uuid"
    assert observed["observation"] == "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED"

    with pytest.raises(RuntimeError, match="UUID differs"):
        observed_single_visible_cuda_device(
            expected_physical_index=3,
            expected_gpu_uuid="GPU-other",
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="differs from the locked physical index"):
        observed_single_visible_cuda_device(
            expected_physical_index=3,
            expected_gpu_uuid="GPU-test-uuid",
        )


def test_candidate_lock_allows_report_only_descendant_but_rejects_source_drift(
    tmp_path: Path,
) -> None:
    repo, config, candidates = _repo(tmp_path)
    lock = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config,
        candidates=candidates,
    )
    (repo / "REPORT.md").write_text("non-executable result report\n", encoding="utf-8")
    _commit(repo, "report only")
    assert (
        validate_backend_candidate_lock(
            lock,
            repo_root=repo,
            config_path=config,
            candidates=candidates,
        )["manifest_sha256"]
        == lock["manifest_sha256"]
    )

    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(repo, "executable drift")
    with pytest.raises(ValueError, match="current execution inputs"):
        validate_backend_candidate_lock(
            lock,
            repo_root=repo,
            config_path=config,
            candidates=candidates,
        )


def test_candidate_lock_rejects_misattributed_ancestor_commit(tmp_path: Path) -> None:
    repo, config, candidates = _repo(tmp_path)
    old_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(repo, "new executable bytes")
    lock = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config,
        candidates=candidates,
    )
    lock["source_commit"] = old_commit
    lock.pop("manifest_sha256")
    lock["manifest_sha256"] = sha256_canonical_json(lock)
    with pytest.raises(ValueError, match="does not contain"):
        validate_backend_candidate_lock(
            lock,
            repo_root=repo,
            config_path=config,
            candidates=candidates,
        )


def test_preflight_promotion_is_bound_and_keeps_downstream_capabilities_closed(
    tmp_path: Path,
) -> None:
    repo, config, candidates = _repo(tmp_path)
    lock = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config,
        candidates=candidates,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    numerical_path = tmp_path / "numerical.json"
    restart_path = tmp_path / "restart.json"
    numerical = _write_receipt(
        numerical_path,
        kind="E26_NUMERICAL_AUDIT",
    )
    numerical["candidate_audits"] = {
        "candidate_a": {
            "execution_device": _execution_device(),
            "variants": {
                "dual_delta_lm": {
                    "compiled_backend_diagnostics": _compiled_diagnostics(compiled_here=True)
                },
                "projected_tied_delta_lm": {
                    "compiled_backend_diagnostics": _compiled_diagnostics(compiled_here=False)
                },
            },
        }
    }
    numerical.pop("receipt_sha256")
    numerical["receipt_sha256"] = sha256_canonical_json(numerical)
    numerical_path.write_text(json.dumps(numerical, sort_keys=True), encoding="utf-8")
    restart = _restart_receipt(restart_path)
    inventory = e26_execution_source_inventory(repo)
    promotion = backend_preflight_manifest(
        candidate_lock_path=lock_path,
        candidate_lock=lock,
        numerical_receipt_path=numerical_path,
        numerical_receipt=numerical,
        restart_receipt_path=restart_path,
        restart_receipt=restart,
        hardware_inventory=_hardware(),
        source_inventory=inventory,
        source_commit=_git(repo, "rev-parse", "HEAD"),
    )
    (repo / "REPORT.md").write_text("post-preflight report\n", encoding="utf-8")
    _commit(repo, "post-preflight report")
    validated = validate_backend_preflight_manifest(
        promotion,
        repo_root=repo,
        candidate_lock_path=lock_path,
        candidate_lock=lock,
        numerical_receipt_path=numerical_path,
        numerical_receipt=numerical,
        restart_receipt_path=restart_path,
        restart_receipt=restart,
        expected_hardware_inventory=_hardware(),
    )
    assert validated["e26a_candidate_capable"] is True
    assert validated["e26a_gate_capable"] is False
    assert validated["scientific_main_capable"] is False
    assert validated["parity_verified"] is False

    different_hardware = _hardware()
    different_hardware[0]["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="differs from current expected hardware"):
        validate_backend_preflight_manifest(
            promotion,
            repo_root=repo,
            candidate_lock_path=lock_path,
            candidate_lock=lock,
            numerical_receipt_path=numerical_path,
            numerical_receipt=numerical,
            restart_receipt_path=restart_path,
            restart_receipt=restart,
            expected_hardware_inventory=different_hardware,
        )

    promotion["hardware_inventory"] = []
    promotion.pop("manifest_sha256")
    promotion["manifest_sha256"] = sha256_canonical_json(promotion)
    with pytest.raises(ValueError, match="at least one CUDA device"):
        validate_backend_preflight_manifest(
            promotion,
            repo_root=repo,
            candidate_lock_path=lock_path,
            candidate_lock=lock,
            numerical_receipt_path=numerical_path,
            numerical_receipt=numerical,
            restart_receipt_path=restart_path,
            restart_receipt=restart,
            expected_hardware_inventory=_hardware(),
        )


def test_preflight_requires_positive_compiled_execution_diagnostics(
    tmp_path: Path,
) -> None:
    repo, config, candidates = _repo(tmp_path)
    lock = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config,
        candidates=candidates,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    numerical_path = tmp_path / "numerical.json"
    restart_path = tmp_path / "restart.json"
    numerical = _write_receipt(numerical_path, kind="E26_NUMERICAL_AUDIT")
    numerical["candidate_audits"] = {
        "candidate_a": {
            "execution_device": _execution_device(),
            "variants": {
                "dual_delta_lm": {
                    "compiled_backend_diagnostics": _compiled_diagnostics(compiled_here=False)
                },
                "projected_tied_delta_lm": {
                    "compiled_backend_diagnostics": _compiled_diagnostics(compiled_here=False)
                },
            },
        }
    }
    numerical.pop("receipt_sha256")
    numerical["receipt_sha256"] = sha256_canonical_json(numerical)
    numerical_path.write_text(json.dumps(numerical, sort_keys=True), encoding="utf-8")
    restart = _restart_receipt(restart_path)
    with pytest.raises(ValueError, match="codegen capability"):
        backend_preflight_manifest(
            candidate_lock_path=lock_path,
            candidate_lock=lock,
            numerical_receipt_path=numerical_path,
            numerical_receipt=numerical,
            restart_receipt_path=restart_path,
            restart_receipt=restart,
            hardware_inventory=_hardware(),
            source_inventory=e26_execution_source_inventory(repo),
            source_commit=_git(repo, "rev-parse", "HEAD"),
        )
