from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)
from catena.lm.e26a_executor import (
    E26AExecutionResult,
    RealE26AExecutionBackend,
    bounded_80_20_nonpadding_token_mix,
    run_scientific_e26a,
)
from catena.lm.e26a_gate import (
    CandidateMeasurement,
    E26AGateBlocked,
    ResourcePolicy,
    ScientificExecutionDeviceBinding,
    select_candidate,
)


def _measurement() -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id="candidate_a",
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
        selected_microbatch_sequences=1,
        accumulation_steps=16,
        token_mix_bounded_discrepancy_passed=True,
    )


def _admission(tmp_path: Path) -> Any:
    root = Path(__file__).resolve().parents[1]
    paths = SimpleNamespace(
        as_dict=lambda: {"fixture": "injected"},
        backend_manifest=root / "tests" / "fixtures" / "unused.json",
    )
    policy = ResourcePolicy(
        deadline_reference_hours=240.0,
        deadline_fraction_max=0.70,
        max_main_wall_clock_hours=168.0,
        safety_time_multiplier=1.25,
        max_main_checkpoint_storage_gib=100.0,
        token_budgets=(250_000_000,),
    )
    config = {"model_candidates": [{"id": "candidate_a"}]}
    admission = SimpleNamespace(
        repo_root=root,
        artifact_root=tmp_path,
        protocol={"schema_version": "catena-v8.1", "locked": True},
        paths=paths,
        input_hashes={
            "fixture_sha256": "a" * 64,
            "validation_population_lock_sha256": "e" * 64,
        },
        backend_candidate_lock={"schema_version": "catena-v8.1"},
        validation_population_lock={
            "schema_version": "catena-e26a-validation-population-v1",
            "manifest_type": "E26A_VALIDATION_POPULATION_LOCK",
            "split": "validation",
            "records_sha256": "f" * 64,
            "episode_count": 500,
        },
        data_readiness={"readiness_sha256": "b" * 64},
        numerical_audit={"receipt_sha256": "c" * 64},
        restart_audit={"receipt_sha256": "d" * 64},
        resource_preflight={"receipt_sha256": "e" * 64},
        gpu_inventory=(
            {
                "logical_index": 0,
                "uuid": "GPU-test",
            },
        ),
        config=config,
        resource_policy=policy,
        locked_resource_selection=(
            selection := select_candidate(
                config=config,
                measurements=(_measurement(),),
                policy=policy,
            )
        ),
        execution_device_binding=ScientificExecutionDeviceBinding(
            cli_device="cuda:0",
            logical_device_index=0,
            physical_device_index=2,
            gpu_uuid="GPU-test",
            resource_worker_visible_cuda_index=0,
            resource_worker_cuda_visible_devices="2",
        ),
    )
    admission.resource_preflight.update(
        {
            "candidates": [
                {
                    "candidate_id": selection.candidate_id,
                    "execution_device": {
                        "physical_device_index": 2,
                        "gpu_uuid": "GPU-test",
                        "worker_visible_cuda_index": 0,
                        "cuda_visible_devices": "2",
                    },
                }
            ]
        }
    )
    return admission


class _PassingBackend:
    def execute(self, admission: Any, run: Any, *, device: torch.device) -> E26AExecutionResult:
        assert device == torch.device("cuda:0")
        return E26AExecutionResult(
            measurements=(_measurement(),),
            gates=(
                {
                    "name": "injected_tiny_executor",
                    "passed": True,
                    "observed": "PASS",
                    "criterion": "PASS",
                },
            ),
            model_manifest={"schema_version": "catena-v8.1", "fixture": True},
            backend_manifest={"schema_version": "catena-v8.1", "fixture": True},
            data_manifest={
                "schema_version": "catena-v8.1",
                "main_test_opened": False,
            },
            pilot_summary={
                "schema_version": "catena-v8.1",
                "main_test_opened": False,
            },
        )


class _FailingBackend:
    def execute(self, admission: Any, run: Any, *, device: torch.device) -> E26AExecutionResult:
        raise RuntimeError("injected operational failure")


class _ScientificGateFailingBackend:
    def execute(
        self,
        admission: Any,
        run: Any,
        *,
        device: torch.device,
    ) -> E26AExecutionResult:
        result = _PassingBackend().execute(admission, run, device=device)
        return E26AExecutionResult(
            measurements=result.measurements,
            gates=(
                {
                    "name": "injected_floor_gate",
                    "passed": False,
                    "observed": 0.0,
                    "criterion": ">=0.1",
                },
            ),
            model_manifest=result.model_manifest,
            backend_manifest=result.backend_manifest,
            data_manifest=result.data_manifest,
            pilot_summary=result.pilot_summary,
        )


def test_sequence_count_ratio_cannot_masquerade_as_token_ratio() -> None:
    old_padded_schedule = {
        "nonpadding_input_tokens": 1_000 * (4 * 4096 + 273),
        "general_nonpadding_input_tokens": 1_000 * 4 * 4096,
        "transaction_nonpadding_input_tokens": 1_000 * 273,
    }
    exact_token_schedule = {
        "nonpadding_input_tokens": 20_000,
        "general_nonpadding_input_tokens": 16_000,
        "transaction_nonpadding_input_tokens": 4_000,
    }
    indivisible_complete_example_schedule = {
        "nonpadding_input_tokens": 1_000_000,
        "general_nonpadding_input_tokens": 800_840,
        "transaction_nonpadding_input_tokens": 199_160,
    }
    assert not bounded_80_20_nonpadding_token_mix(
        old_padded_schedule,
        context_length=4096,
    )
    assert bounded_80_20_nonpadding_token_mix(
        exact_token_schedule,
        context_length=4096,
    )
    assert bounded_80_20_nonpadding_token_mix(
        indivisible_complete_example_schedule,
        context_length=4096,
    )


def test_candidate_measurements_are_preserved_as_scientific_gate_failures() -> None:
    measurement = _measurement()
    object.__setattr__(
        measurement,
        "token_mix_bounded_discrepancy_passed",
        False,
    )
    admission = SimpleNamespace(
        config={
            "candidate_selection": {
                "parameter_count_min": 35_000_000,
                "parameter_count_max": 50_000_000,
            }
        },
        resource_policy=_admission(Path("/tmp")).resource_policy,
    )
    gates = RealE26AExecutionBackend._selection_prerequisite_gates(
        admission,
        (measurement,),
    )
    by_name = {gate["name"]: gate for gate in gates}
    assert by_name["candidate_numerical_parity_restart_contract_available"]["passed"]
    assert not by_name["candidate_token_mix_data_contract_available"]["passed"]
    assert "candidate_compiled_backend_without_graph_break_or_fallback_available" not in by_name
    assert "candidate_throughput_deadline_storage_budget_available" not in by_name


def test_injected_executor_finalizes_evidence_bounded_gate_only(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "catena.lm.e26a_gate._runtime_cuda_device",
        lambda _: {"logical_index": 0, "gpu_uuid": "GPU-test"},
    )
    run_dir, report = run_scientific_e26a(
        _admission(tmp_path),
        device="cuda:0",
        backend=_PassingBackend(),
    )
    assert report["disposition"] == "GO_E26B"
    assert report["scientific_evidence"] is False
    assert report["main_test_opened"] is False
    assert report["e26b_started"] is False
    assert report["e26c_started"] is False
    assert (run_dir / "candidate_selection_lock.json").is_file()
    device_binding = read_json_object_strict(run_dir / "scientific_execution_device_binding.json")
    assert device_binding == _admission(tmp_path).execution_device_binding.as_dict()
    assert report["scientific_execution_device_binding"]["gpu_uuid"] == "GPU-test"
    assert report["scientific_execution_device_binding"]["artifact_sha256"] == sha256_file(
        run_dir / "scientific_execution_device_binding.json"
    )
    assert not any("e26b" in path.name or "e26c" in path.name for path in run_dir.iterdir())
    backend = read_json_object_strict(run_dir / "backend_manifest.json")
    backend_hash = backend.pop("manifest_sha256")
    assert backend_hash == sha256_canonical_json(backend)


def test_post_admission_executor_failure_is_preserved_and_fail_closed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "catena.lm.e26a_gate._runtime_cuda_device",
        lambda _: {"logical_index": 0, "gpu_uuid": "GPU-test"},
    )
    run_dir, report = run_scientific_e26a(
        _admission(tmp_path),
        device="cuda:0",
        backend=_FailingBackend(),
    )
    assert report["disposition"] == "BLOCKED_DEPENDENCY"
    assert report["status"] == "FAIL"
    assert report["main_test_opened"] is False
    assert "injected operational failure" in (run_dir / "pilot_summary.json").read_text(
        encoding="utf-8"
    )


def test_resource_only_candidate_selection_is_preserved_when_pilot_gate_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "catena.lm.e26a_gate._runtime_cuda_device",
        lambda _: {"logical_index": 0, "gpu_uuid": "GPU-test"},
    )
    run_dir, report = run_scientific_e26a(
        _admission(tmp_path),
        device="cuda:0",
        backend=_ScientificGateFailingBackend(),
    )
    assert report["status"] == "FAIL"
    assert report["candidate_selection"]["candidate_id"] == "candidate_a"
    selection = (run_dir / "candidate_selection_lock.json").read_text(encoding="utf-8")
    assert '"candidate_id": "candidate_a"' in selection


def test_arbitrary_cli_device_is_rejected_before_artifact_creation(tmp_path: Path) -> None:
    with pytest.raises(E26AGateBlocked, match="admitted resource device"):
        run_scientific_e26a(
            _admission(tmp_path),
            device="cuda:1",
            backend=_PassingBackend(),
        )
    assert not any(tmp_path.iterdir())


def test_runtime_uuid_drift_is_rejected_before_artifact_creation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "catena.lm.e26a_gate._runtime_cuda_device",
        lambda _: {"logical_index": 0, "gpu_uuid": "GPU-other"},
    )
    with pytest.raises(E26AGateBlocked, match="Runtime CUDA UUID"):
        run_scientific_e26a(
            _admission(tmp_path),
            device="cuda:0",
            backend=_PassingBackend(),
        )
    assert not any(tmp_path.iterdir())
