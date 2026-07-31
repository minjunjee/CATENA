from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from catena.lm.artifacts import ArtifactContractError, ArtifactRun
from catena.lm.audit_contract import E26_AUDIT_LOCKED_HASH_KEYS
from catena.lm.e26a_gate import (
    CandidateMeasurement,
    E26AGateBlocked,
    E26AGateInputPaths,
    ResourcePolicy,
    _failure_disposition,
    _require_audit_locked_hashes,
    _require_hash_binding,
    _require_receipt_pass,
    candidate_numerical_coverage,
    exclusive_scientific_gate_lock,
    project_candidate_resources,
    select_candidate,
    validate_scientific_gate_admission,
)


def _dummy_paths(tmp_path: Path) -> E26AGateInputPaths:
    missing = tmp_path / "missing"
    return E26AGateInputPaths(
        config=missing,
        calibration_config=missing,
        protocol_lock=missing,
        backend_candidate_lock=missing,
        backend_manifest=missing,
        tokenizer_manifest=missing,
        corpus_manifest=missing,
        data_lock=missing,
        data_readiness=missing,
        transaction_manifest=missing,
        validation_population_lock=missing,
        schedule_manifest=missing,
        numerical_audit=missing,
        restart_audit=missing,
        frozen_tree_receipt=missing,
        resource_preflight=missing,
    )


def _measurement(candidate_id: str, *, tokens_per_second: float = 2_000.0) -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id=candidate_id,
        parameter_count=42_025_616,
        matching_passed=True,
        numerical_passed=True,
        tokens_per_second_by_variant={
            "projected_tied_delta_lm": tokens_per_second,
            "dual_delta_lm": tokens_per_second * 1.01,
        },
        checkpoint_bytes=512 * 1024**2,
        peak_allocated_bytes=8 * 1024**3,
        peak_reserved_bytes=9 * 1024**3,
        p50_step_seconds=1.0,
        p95_step_seconds=1.1,
        compile_seconds=30.0,
        graph_break_count=0,
        fallback_count=0,
        context_length=4096,
        selected_microbatch_sequences=1,
        accumulation_steps=16,
        token_mix_bounded_discrepancy_passed=True,
    )


def _policy() -> ResourcePolicy:
    return ResourcePolicy(
        deadline_reference_hours=240.0,
        deadline_fraction_max=0.70,
        max_main_wall_clock_hours=168.0,
        safety_time_multiplier=1.25,
        max_main_checkpoint_storage_gib=100.0,
        token_budgets=(250_000_000, 300_000_000, 400_000_000, 500_000_000),
    )


def test_scientific_gate_rejects_missing_explicit_ack_before_reading_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(E26AGateBlocked, match="authorization"):
        validate_scientific_gate_admission(
            repo_root=tmp_path,
            artifact_root=tmp_path,
            execution_ack="",
            paths=_dummy_paths(tmp_path),
            expected_resource_preflight_sha256="0" * 64,
            execution_device="cuda:0",
            require_gpu_inventory=False,
        )


def test_scientific_gate_requires_explicit_resource_receipt_file_hash(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "resource_preflight.json"
    resource.write_text("{}\n", encoding="utf-8")
    paths = replace(_dummy_paths(tmp_path), resource_preflight=resource)
    with pytest.raises(E26AGateBlocked, match="explicitly approved SHA-256"):
        validate_scientific_gate_admission(
            repo_root=tmp_path,
            artifact_root=tmp_path,
            execution_ack="E26A_SCIENTIFIC_GATE_AUTHORIZED",
            paths=paths,
            expected_resource_preflight_sha256="0" * 64,
            execution_device="cuda:0",
            require_gpu_inventory=False,
        )


def test_receipt_with_main_test_access_is_rejected() -> None:
    with pytest.raises(E26AGateBlocked, match="main-test access"):
        _require_receipt_pass(
            {
                "schema_version": "catena-v8.1",
                "scientific_evidence": False,
                "passed": True,
                "main_test_opened": True,
                "main_test_access_count": 1,
            },
            "fixture",
        )


def test_actual_token_mix_failure_is_classified_as_data_no_go() -> None:
    assert (
        _failure_disposition(
            [
                {
                    "name": "candidate_token_mix_data_contract_available",
                    "passed": False,
                },
                {
                    "name": "throughput_deadline_storage_candidate_selection",
                    "passed": False,
                },
            ]
        )
        == "NO_GO_DATA"
    )


def test_protocol_lock_dag_allows_downstream_audits_but_not_missing_upstream() -> None:
    observed = {
        "config_sha256": "a" * 64,
        "data_lock_sha256": "b" * 64,
        "protocol_lock_sha256": "c" * 64,
        "numerical_audit_sha256": "d" * 64,
        "restart_audit_sha256": "e" * 64,
        "backend_candidate_lock_sha256": "1" * 64,
        "backend_manifest_sha256": "2" * 64,
    }
    protocol = {
        "execution_inputs": {
            "config_sha256": observed["config_sha256"],
            "data_lock_sha256": observed["data_lock_sha256"],
            "backend_candidate_lock_sha256": observed["backend_candidate_lock_sha256"],
        }
    }
    _require_hash_binding(protocol, observed)
    protocol["execution_inputs"].pop("backend_candidate_lock_sha256")
    with pytest.raises(E26AGateBlocked, match="does not bind"):
        _require_hash_binding(protocol, observed)


def test_audit_receipt_protocol_hash_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "algorithm": "fixture",
        "suffixes": [],
        "files": 0,
        "rows": [],
        "source_tree_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        "catena.lm.e26a_gate.e26_execution_source_inventory",
        lambda _: inventory,
    )
    input_hashes = {
        key: f"{index + 1:064x}"
        for index, key in enumerate(sorted(E26_AUDIT_LOCKED_HASH_KEYS - {"source_tree_sha256"}))
    }
    locked = {**input_hashes, "source_tree_sha256": "f" * 64}
    locked["protocol_lock_sha256"] = "0" * 64
    with pytest.raises(E26AGateBlocked, match="protocol_lock_sha256"):
        _require_audit_locked_hashes(
            {
                "locked_hashes": locked,
                "source_inventory": inventory,
            },
            label="fixture",
            repo_root=tmp_path,
            input_hashes=input_hashes,
        )


def test_candidate_selection_is_config_order_not_measurement_order() -> None:
    config = {
        "model_candidates": [
            {"id": "d512_ctx4096"},
            {"id": "d512_ctx2048"},
        ]
    }
    selection = select_candidate(
        config=config,
        measurements=(
            _measurement("d512_ctx2048"),
            _measurement("d512_ctx4096"),
        ),
        policy=_policy(),
    )
    assert selection.candidate_id == "d512_ctx4096"
    assert selection.candidate_config_index == 0
    assert selection.token_budget == 300_000_000
    assert selection.safety_adjusted_wall_hours <= 168.0
    assert selection.deadline_fraction <= 0.70


def test_resource_projection_uses_conservative_variant_and_storage() -> None:
    measurement = _measurement("d512_ctx4096", tokens_per_second=2_000.0)
    projection = project_candidate_resources(measurement, _policy())
    by_budget = {row["token_budget"]: row for row in projection}
    assert by_budget[300_000_000]["eligible"]
    assert not by_budget[400_000_000]["eligible"]
    assert by_budget[300_000_000]["wave_count"] == 3
    assert by_budget[300_000_000]["checkpoint_count_per_run"] == 13
    assert by_budget[300_000_000]["checkpoint_storage_gib"] == pytest.approx(65.0)


def test_resource_projection_rejects_missing_paired_throughput() -> None:
    measurement = _measurement("d512_ctx4096")
    object.__setattr__(measurement, "tokens_per_second_by_variant", {})
    with pytest.raises(E26AGateBlocked, match="throughput"):
        project_candidate_resources(measurement, _policy())


def test_candidate_numerical_coverage_cannot_reuse_one_candidate_audit() -> None:
    config = {
        "model_candidates": [
            {"id": "a", "d_model": 512},
            {"id": "b", "d_model": 448},
        ],
        "variants": ["projected_tied_delta_lm", "dual_delta_lm"],
    }
    receipt = {
        "candidate_audits": {
            "a": {
                "model_config_sha256": "0" * 64,
                "all_passed": True,
                "variants": {},
            }
        }
    }
    with pytest.raises(E26AGateBlocked, match="candidate IDs"):
        candidate_numerical_coverage(config, receipt)


def test_scientific_gate_execution_lock_is_nonblocking(tmp_path: Path) -> None:
    with (
        exclusive_scientific_gate_lock(tmp_path),
        pytest.raises(E26AGateBlocked, match="holds the execution lock"),
        exclusive_scientific_gate_lock(tmp_path),
    ):
        pass


def test_non_dry_artifact_requires_explicit_evidence_boundary(tmp_path: Path) -> None:
    with pytest.raises(ArtifactContractError, match="explicitly declare"):
        ArtifactRun(
            experiment="e26a_operator_data_gate",
            artifact_root=tmp_path,
            canonical_artifact_root=tmp_path,
            run_mode="MAIN",
            dry_run=False,
        )

    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=tmp_path,
        canonical_artifact_root=tmp_path,
        run_mode="MAIN",
        dry_run=False,
        scientific_evidence=False,
        evidence_tier="SCIENTIFIC_PROTOCOL_GATE",
        claim_ceiling="PROTOCOL_IDENTIFIABILITY_ONLY",
    )
    with pytest.raises(ArtifactContractError, match="evidence boundary"):
        run.finalize(
            {
                "schema_version": "catena-v8.1",
                "experiment": "e26a_operator_data_gate",
                "run_id": run.run_id,
                "run_mode": "MAIN",
                "status": "PASS",
                "scientific_evidence": True,
                "evidence_tier": "SCIENTIFIC_EVIDENCE",
                "claim_ceiling": "LM_CLAIM",
                "disposition": "GO_E26B",
                "allowed_claim": "",
                "forbidden_claims": [],
                "gates": [],
                "artifacts": {},
            },
            "# invalid",
        )


def test_scientific_launcher_contains_no_follow_on_experiment_invocation() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_e26a_scientific_gate.sh").read_text(encoding="utf-8")
    command_lines = [
        line
        for line in source.splitlines()
        if "experiments/" in line and not line.lstrip().startswith("#")
    ]
    assert command_lines
    assert all("e26a_operator_data_gate.py" in line for line in command_lines)
    assert "launch_e26c" not in source
    assert ': "${E26A_DEVICE:?' in source
    assert '--device "$E26A_DEVICE"' in source
    assert "--device cuda:0" not in source


def test_scientific_driver_has_callable_executor_not_an_admission_placeholder() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/catena/lm/experiment_driver.py").read_text(encoding="utf-8")
    assert "run_scientific_e26a(" in source
    assert "execution_device=args.device" in source
    assert "device=admission.execution_device_binding.cli_device" in source
    assert "measurement executor has not been attached" not in source
