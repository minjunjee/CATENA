from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from catena.core.provenance_v61 import (
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.audit_contract import E26_AUDIT_LOCKED_HASH_KEYS
from catena.lm.e26a_gate import (
    CandidateMeasurement,
    E26AGateBlocked,
    ResourcePolicy,
    bind_scientific_execution_device,
    project_candidate_resources,
    require_locked_resource_selection,
    select_candidate,
    validate_resource_preflight_receipt,
)

_RESOURCE_TOOL_SPEC = importlib.util.spec_from_file_location(
    "catena_e26_resource_preflight_tool",
    Path(__file__).resolve().parents[1] / "tools" / "run_e26_resource_preflight.py",
)
assert _RESOURCE_TOOL_SPEC is not None and _RESOURCE_TOOL_SPEC.loader is not None
_RESOURCE_TOOL = importlib.util.module_from_spec(_RESOURCE_TOOL_SPEC)
_RESOURCE_TOOL_SPEC.loader.exec_module(_RESOURCE_TOOL)
_validate_worker_result = _RESOURCE_TOOL._validate_worker_result


def _measurement(candidate_id: str = "candidate_a") -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id=candidate_id,
        parameter_count=42_000_000,
        matching_passed=True,
        numerical_passed=True,
        tokens_per_second_by_variant={
            "projected_tied_delta_lm": 2_000.0,
            "dual_delta_lm": 2_050.0,
        },
        checkpoint_bytes=100 * 1024**2,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
        p50_step_seconds=1.0,
        p95_step_seconds=1.1,
        compile_seconds=2.0,
        graph_break_count=0,
        fallback_count=0,
        context_length=4096,
        selected_microbatch_sequences=2,
        accumulation_steps=8,
        measured_optimizer_steps=100,
        descriptive_stability_steps=0,
        model_config_sha256="b" * 64,
        parameter_signature_sha256="c" * 64,
        paired_initialization_digest="d" * 64,
        token_mix_bounded_discrepancy_passed=True,
    )


def _policy() -> ResourcePolicy:
    return ResourcePolicy(
        deadline_reference_hours=240.0,
        deadline_fraction_max=0.70,
        max_main_wall_clock_hours=168.0,
        safety_time_multiplier=1.25,
        max_main_checkpoint_storage_gib=100.0,
        token_budgets=(250_000_000,),
    )


def _target_accumulation(candidate_id: str, candidate_sha: str) -> dict:
    layouts = [[16], [2] * 8, [1] * 16]
    rows = [{"microbatch_sizes": layout, "passed": True} for layout in layouts]
    variants = {
        variant: {
            "variant": variant,
            "precision": "bf16_actual_training",
            "rows": rows,
            "compiled_backend_diagnostics": {
                "fallback_count": 0,
                "graph_break_count": 0,
            },
            "passed": True,
        }
        for variant in ("projected_tied_delta_lm", "dual_delta_lm")
    }
    payload = {
        "candidate_id": candidate_id,
        "model_config_sha256": candidate_sha,
        "context_length": 4096,
        "target_global_batch_tokens": 65536,
        "global_batch_sequences": 16,
        "selected_microbatch_sequences": 2,
        "accumulation_steps": 8,
        "audited_microbatch_sequences": [16, 2, 1],
        "accumulation_layouts": layouts,
        "variants": variants,
        "passed": True,
    }
    payload["audit_sha256"] = sha256_canonical_json(payload)
    return payload


def test_locked_resource_selection_rejects_candidate_or_budget_drift() -> None:
    config = {"model_candidates": [{"id": "candidate_a"}]}
    locked = select_candidate(
        config=config,
        measurements=(_measurement(),),
        policy=_policy(),
    )
    admission = SimpleNamespace(locked_resource_selection=locked)
    require_locked_resource_selection(admission, locked)
    with pytest.raises(E26AGateBlocked, match="changed the locked selection"):
        require_locked_resource_selection(
            admission,
            replace(locked, token_budget=locked.token_budget + 1),
        )
    with pytest.raises(E26AGateBlocked, match="changed the locked selection"):
        require_locked_resource_selection(
            admission,
            replace(
                locked,
                selected_microbatch_sequences=1,
                accumulation_steps=16,
            ),
        )


def test_scientific_device_binding_requires_selected_worker_gpu_uuid() -> None:
    config = {"model_candidates": [{"id": "candidate_a"}]}
    selection = select_candidate(
        config=config,
        measurements=(_measurement(),),
        policy=_policy(),
    )
    resource_preflight = {
        "candidates": [
            {
                "candidate_id": "candidate_a",
                "execution_device": {
                    "physical_device_index": 2,
                    "gpu_uuid": "GPU-test",
                    "worker_visible_cuda_index": 0,
                    "cuda_visible_devices": "2",
                },
            }
        ]
    }
    inventory = ({"logical_index": 1, "uuid": "GPU-test"},)
    binding = bind_scientific_execution_device(
        requested_device="cuda:1",
        resource_preflight=resource_preflight,
        selection=selection,
        gpu_inventory=inventory,
    )
    assert binding.logical_device_index == 1
    assert binding.physical_device_index == 2
    assert binding.gpu_uuid == "GPU-test"

    with pytest.raises(E26AGateBlocked, match="UUID differs"):
        bind_scientific_execution_device(
            requested_device="cuda:1",
            resource_preflight=resource_preflight,
            selection=selection,
            gpu_inventory=({"logical_index": 1, "uuid": "GPU-other"},),
        )
    with pytest.raises(E26AGateBlocked, match="explicit CUDA index"):
        bind_scientific_execution_device(
            requested_device="cuda",
            resource_preflight=resource_preflight,
            selection=selection,
            gpu_inventory=inventory,
        )


def _worker_fixture(tmp_path: Path) -> tuple[dict, dict, Path, dict]:
    output_root = tmp_path.resolve()
    run_dir = output_root / "e26_resource_candidate_a" / "run"
    run_dir.mkdir(parents=True)
    device = {
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test",
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": "2",
        "name": "test-gpu",
        "total_memory_bytes": 123,
        "compute_capability": "9.0",
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }
    measurement = asdict(_measurement())
    target_accumulation = _target_accumulation("candidate_a", "b" * 64)
    input_hashes = {"config_sha256": "e" * 64}
    spec = {
        "candidate_id": "candidate_a",
        "candidate_config_sha256": "b" * 64,
        "source_commit": "a" * 40,
        "source_inventory": {"source_tree_sha256": "f" * 64},
        "input_hashes": input_hashes,
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test",
    }
    spec["spec_sha256"] = sha256_canonical_json(spec)
    report = {
        "candidate_id": "candidate_a",
        "candidate_config_sha256": "b" * 64,
        "measurement": measurement,
        "target_gradient_accumulation": target_accumulation,
        "worker_spec_sha256": spec["spec_sha256"],
        "source_commit": "a" * 40,
        "source_tree_sha256": "f" * 64,
        "input_hashes": input_hashes,
        "execution_device": device,
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    report_path = run_dir / "report.json"
    write_json_strict(report_path, report)
    report_sha = sha256_file(report_path)
    write_json_strict(
        run_dir / "run_manifest.json",
        {
            "report_sha256": report_sha,
            "visible_devices": "2",
            "source_fingerprint_verified_at_completion": True,
            "git": {
                "head": "a" * 40,
                "status_porcelain": "",
            },
        },
    )
    receipt = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESOURCE_WORKER_RECEIPT",
        "candidate_id": "candidate_a",
        "candidate_config_sha256": "b" * 64,
        "measurement": measurement,
        "target_gradient_accumulation": target_accumulation,
        "run_dir": str(run_dir),
        "report_sha256": report_sha,
        "worker_spec_sha256": spec["spec_sha256"],
        "source_commit": "a" * 40,
        "source_tree_sha256": "f" * 64,
        "input_hashes": input_hashes,
        "execution_device": device,
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    hardware = {
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test",
        "name": "test-gpu",
        "total_memory_bytes": 123,
        "compute_capability": "9.0",
    }
    return receipt, spec, output_root, hardware


def test_parent_validates_worker_receipt_report_and_observed_device(
    tmp_path: Path,
) -> None:
    receipt, spec, output_root, hardware = _worker_fixture(tmp_path)
    assert (
        _validate_worker_result(
            payload=receipt,
            spec=spec,
            output_root=output_root,
            expected_hardware=hardware,
        )["candidate_id"]
        == "candidate_a"
    )
    tampered = json.loads(json.dumps(receipt))
    tampered["execution_device"]["gpu_uuid"] = "GPU-other"
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = sha256_canonical_json(tampered)
    with pytest.raises(ValueError, match="observed device"):
        _validate_worker_result(
            payload=tampered,
            spec=spec,
            output_root=output_root,
            expected_hardware=hardware,
        )


def test_resource_receipt_binds_upstreams_hardware_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {"id": "candidate_a"}
    candidate_sha = sha256_canonical_json(candidate)
    measurement = replace(_measurement(), model_config_sha256=candidate_sha)
    policy = _policy()
    config = {
        "model_candidates": [candidate],
        "variants": ["projected_tied_delta_lm", "dual_delta_lm"],
        "throughput": {
            "target_global_batch_tokens": 65536,
            "microbatch_size_candidates": [1, 2, 4, 8],
        },
    }
    selection = select_candidate(
        config=config,
        measurements=(measurement,),
        policy=policy,
    )
    files = {}
    for name in ("backend_manifest", "numerical_audit", "restart_audit"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path
    paths = SimpleNamespace(**files)
    source_inventory = {
        "algorithm": "fixture",
        "suffixes": [],
        "files": 0,
        "rows": [],
        "source_tree_sha256": "f" * 64,
    }
    hardware = [
        {
            "physical_device_index": 2,
            "name": "test-gpu",
            "total_memory_bytes": 123,
            "compute_capability": "9.0",
            "driver_version": "test",
            "gpu_uuid": "GPU-test",
        }
    ]
    monkeypatch.setattr(
        "catena.lm.e26a_gate.e26_execution_source_inventory",
        lambda _: source_inventory,
    )
    monkeypatch.setattr(
        "catena.lm.e26a_gate.cuda_hardware_inventory",
        lambda _: hardware,
    )
    monkeypatch.setattr("catena.lm.e26a_gate._git", lambda *_: "ok")
    locked = {key: "1" * 64 for key in E26_AUDIT_LOCKED_HASH_KEYS}
    locked["source_tree_sha256"] = "f" * 64
    input_hashes = dict(locked)
    backend = {"manifest_sha256": "2" * 64}
    numerical = {"receipt_sha256": "3" * 64}
    restart = {"receipt_sha256": "4" * 64}
    row = {
        **asdict(measurement),
        "candidate_config_sha256": candidate_sha,
        "resource_projections": list(project_candidate_resources(measurement, policy)),
        "worker_run_dir": "/tmp/non-evidence",
        "worker_report_sha256": "5" * 64,
        "worker_spec_sha256": "6" * 64,
        "worker_receipt_sha256": "7" * 64,
        "execution_device": {
            "physical_device_index": 2,
            "gpu_uuid": "GPU-test",
            "worker_visible_cuda_index": 0,
            "cuda_visible_devices": "2",
            "name": "test-gpu",
            "total_memory_bytes": 123,
            "compute_capability": "9.0",
            "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
        },
        "target_gradient_accumulation": _target_accumulation(
            "candidate_a",
            candidate_sha,
        ),
    }
    payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESOURCE_PREFLIGHT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "source_commit": "a" * 40,
        "source_inventory": source_inventory,
        "locked_hashes": locked,
        "upstream_receipts": {
            "backend_manifest": {
                "sha256": sha256_file(files["backend_manifest"]),
                "manifest_sha256": "2" * 64,
            },
            "numerical_audit": {
                "sha256": sha256_file(files["numerical_audit"]),
                "receipt_sha256": "3" * 64,
            },
            "restart_audit": {
                "sha256": sha256_file(files["restart_audit"]),
                "receipt_sha256": "4" * 64,
            },
        },
        "hardware_inventory": hardware,
        "candidates": [row],
        "selection": selection.as_dict(),
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "canonical_e26_artifact_created": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    _, validated_selection = validate_resource_preflight_receipt(
        payload,
        repo_root=tmp_path,
        paths=paths,
        config=config,
        policy=policy,
        input_hashes=input_hashes,
        backend_preflight=backend,
        numerical_audit=numerical,
        restart_audit=restart,
    )
    assert validated_selection.candidate_id == "candidate_a"
    tampered_device = json.loads(json.dumps(payload))
    tampered_device["candidates"][0]["execution_device"]["name"] = "other-gpu"
    tampered_device.pop("receipt_sha256")
    tampered_device["receipt_sha256"] = sha256_canonical_json(tampered_device)
    with pytest.raises(E26AGateBlocked, match="worker device"):
        validate_resource_preflight_receipt(
            tampered_device,
            repo_root=tmp_path,
            paths=paths,
            config=config,
            policy=policy,
            input_hashes=input_hashes,
            backend_preflight=backend,
            numerical_audit=numerical,
            restart_audit=restart,
        )
    tampered = json.loads(json.dumps(payload))
    tampered["selection"]["token_budget"] += 1
    tampered["selection"].pop("selection_sha256")
    tampered["selection"]["selection_sha256"] = sha256_canonical_json(tampered["selection"])
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = sha256_canonical_json(tampered)
    with pytest.raises(E26AGateBlocked, match="deterministic selection"):
        validate_resource_preflight_receipt(
            tampered,
            repo_root=tmp_path,
            paths=paths,
            config=config,
            policy=policy,
            input_hashes=input_hashes,
            backend_preflight=backend,
            numerical_audit=numerical,
            restart_audit=restart,
        )
