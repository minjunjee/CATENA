from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file
from catena.lm.e26a_gate import CandidateMeasurement

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_e26_stage3d_resource_preflight.py"
_SPEC = importlib.util.spec_from_file_location("e26_stage3d_resource_tool", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def _config() -> dict:
    return {
        "model_candidates": [
            {"id": "d512_ctx4096", "context_length": 4096},
            {"id": "d512_ctx2048", "context_length": 2048},
            {"id": "d448_ctx4096", "context_length": 4096},
        ],
        "throughput": {"target_global_batch_tokens": 65_536},
    }


def _comparison() -> dict:
    return {"passed": True, "relative_l2": 0.0}


def _go_receipt() -> dict:
    payload = {
        "schema_version": "catena-e26-stage3d-fixed-layout-receipt-v1",
        "manifest_type": "E26_STAGE3D_FIXED_LAYOUT_RECEIPT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "disposition": "STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE",
        "passed": True,
        "source": {
            "git_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "protocol_lock": {
            "path": "/tmp/protocol.json",
            "sha256": "c" * 64,
            "protocol_sha256": "d" * 64,
        },
        "input_hashes": {
            "config_sha256": "e" * 64,
            "tokenizer_manifest_sha256": "f" * 64,
            "corpus_manifest_sha256": "1" * 64,
        },
        "fixed_layouts": _TOOL._expected_layouts(_config()),
        "g3_cases": [
            {
                "candidate_id": f"candidate_{index // 4}",
                "passed": True,
                "comparisons": {
                    "compiled_bf16_vs_reference_python_bf16": _comparison(),
                    "reference_python_bf16_vs_reference_python_fp32": _comparison(),
                },
            }
            for index in range(12)
        ],
        "g4_replays": [{"replay_id": index, "passed": True} for index in range(6)],
        "gate_summary": {
            **{f"g{index}_passed": True for index in range(7)},
            "g3_pass_count": 12,
            "g3_required_count": 12,
            "g4_pass_count": 6,
            "g4_required_count": 6,
        },
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def _rehash(payload: dict) -> dict:
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def _measurement() -> CandidateMeasurement:
    return CandidateMeasurement(
        candidate_id="d512_ctx4096",
        parameter_count=42_000_000,
        matching_passed=True,
        numerical_passed=True,
        tokens_per_second_by_variant={
            "projected_tied_delta_lm": 2_000.0,
            "dual_delta_lm": 2_100.0,
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
        measured_optimizer_steps=100,
        model_config_sha256="2" * 64,
        parameter_signature_sha256="3" * 64,
        paired_initialization_digest="4" * 64,
        token_mix_bounded_discrepancy_passed=True,
    )


def test_stage3d_go_receipt_and_fixed_layout_are_exact() -> None:
    receipt = _TOOL.validate_stage3d_go_receipt(
        _go_receipt(),
        config=_config(),
        config_sha256="e" * 64,
        tokenizer_manifest_sha256="f" * 64,
        corpus_manifest_sha256="1" * 64,
        verify_canonical_contract=False,
    )
    assert receipt["passed"] is True
    assert receipt["fixed_layouts"] == [
        {
            "candidate_id": "d512_ctx4096",
            "context_length": 4096,
            "microbatch_sequences": 1,
            "target_global_input_tokens": 65_536,
            "global_batch_sequences": 16,
            "accumulation_steps": 16,
        },
        {
            "candidate_id": "d512_ctx2048",
            "context_length": 2048,
            "microbatch_sequences": 1,
            "target_global_input_tokens": 65_536,
            "global_batch_sequences": 32,
            "accumulation_steps": 32,
        },
        {
            "candidate_id": "d448_ctx4096",
            "context_length": 4096,
            "microbatch_sequences": 1,
            "target_global_input_tokens": 65_536,
            "global_batch_sequences": 16,
            "accumulation_steps": 16,
        },
    ]


def test_default_admission_requires_the_bound_canonical_protocol() -> None:
    with pytest.raises(FileNotFoundError):
        _TOOL.validate_stage3d_go_receipt(
            _go_receipt(),
            config=_config(),
            config_sha256="e" * 64,
            tokenizer_manifest_sha256="f" * 64,
            corpus_manifest_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda row: row.update(
                disposition="STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY"
            ),
            "GO receipt",
        ),
        (lambda row: row["fixed_layouts"][0].update(microbatch_sequences=2), "fixed layouts"),
        (lambda row: row["gate_summary"].update(g3_pass_count=11), "required-case counts"),
        (lambda row: row["g3_cases"][0].update(passed=False), "G3 case"),
        (lambda row: row["g4_replays"][0].update(passed=False), "same-layout replay"),
        (lambda row: row["input_hashes"].update(config_sha256="9" * 64), "config_sha256"),
    ],
)
def test_stage3d_resource_admission_fails_closed(mutation, match: str) -> None:
    receipt = _go_receipt()
    mutation(receipt)
    _rehash(receipt)
    with pytest.raises(ValueError, match=match):
        _TOOL.validate_stage3d_go_receipt(
            receipt,
            config=_config(),
            config_sha256="e" * 64,
            tokenizer_manifest_sha256="f" * 64,
            corpus_manifest_sha256="1" * 64,
            verify_canonical_contract=False,
        )


def test_stage3d_resource_policy_projects_only_registered_budgets() -> None:
    policy = _TOOL.fixed_resource_policy()
    assert policy.token_budgets == (250_000_000, 375_000_000, 500_000_000)
    assert policy.max_main_wall_clock_hours == 168.0
    assert policy.safety_time_multiplier == 1.25
    assert policy.max_main_checkpoint_storage_gib == 100.0
    assert policy.main_runs == 10
    assert policy.gpu_lanes == 4
    rows = _TOOL.resource_projections_with_gpu_hours(_measurement(), policy)
    assert [row["token_budget"] for row in rows] == [
        250_000_000,
        375_000_000,
        500_000_000,
    ]
    assert all(row["wave_count"] == 3 for row in rows)
    assert rows[0]["safety_adjusted_wall_hours"] == pytest.approx(
        (250_000_000 / 2_000.0 / 3600.0) * 3 * 1.25
    )
    assert rows[0]["total_gpu_hours"] == pytest.approx((250_000_000 / 2_000.0 / 3600.0) * 10)
    assert rows[0]["safety_adjusted_total_gpu_hours"] == pytest.approx(
        (250_000_000 / 2_000.0 / 3600.0) * 10 * 1.25
    )


def test_cuda_inventory_uses_canonical_gpu_uuid_field() -> None:
    rows = [
        {
            "physical_device_index": 2,
            "gpu_uuid": "GPU-canonical-2",
            "name": "mock-gpu",
        }
    ]
    indexed = _TOOL._hardware_by_physical_index(rows)
    assert indexed[2]["gpu_uuid"] == "GPU-canonical-2"


@pytest.mark.parametrize(
    "rows",
    [
        [{"physical_device_index": 0, "uuid": "GPU-wrong-field"}],
        [{"physical_device_index": 0, "gpu_uuid": ""}],
        [
            {"physical_device_index": 0, "gpu_uuid": "GPU-a"},
            {"physical_device_index": 0, "gpu_uuid": "GPU-b"},
        ],
    ],
)
def test_cuda_inventory_rejects_missing_or_ambiguous_gpu_uuid(rows: list[dict]) -> None:
    with pytest.raises(ValueError, match="gpu_uuid|repeats"):
        _TOOL._hardware_by_physical_index(rows)


def test_resource_inputs_require_exact_stage3c_paths_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "e26a.yaml"
    tokenizer = tmp_path / "tokenizer.json"
    corpus = tmp_path / "corpus.json"
    config.write_text("model_candidates: []\n", encoding="utf-8")
    tokenizer.write_text("{}\n", encoding="utf-8")
    corpus.write_text("{}\n", encoding="utf-8")
    stage3c = tmp_path / "stage3c.json"
    stage3c.write_text(
        json.dumps(
            {
                "execution_input_paths": {
                    "config": str(config),
                    "tokenizer_manifest": str(tokenizer),
                    "corpus_manifest": str(corpus),
                },
                "execution_inputs": {
                    "config_sha256": sha256_file(config),
                    "tokenizer_manifest_sha256": sha256_file(tokenizer),
                    "corpus_manifest_sha256": sha256_file(corpus),
                },
            }
        ),
        encoding="utf-8",
    )
    stage3d = tmp_path / "stage3d.yaml"
    stage3d.write_text(
        "stage3c:\n"
        "  protocol:\n"
        f"    path: {stage3c}\n"
        f"    sha256: {sha256_file(stage3c)}\n",
        encoding="utf-8",
    )
    receipt = {
        "protocol_lock": {
            "path": str(stage3d),
            "sha256": sha256_file(stage3d),
        }
    }
    monkeypatch.setattr(_TOOL, "validate_stage3d_protocol_lock", lambda payload: payload)
    exact = {
        "config": config.resolve(),
        "tokenizer_manifest": tokenizer.resolve(),
        "corpus_manifest": corpus.resolve(),
    }
    _TOOL._require_exact_bound_input_paths(exact, stage3d_receipt=receipt)

    substitute = tmp_path / "same_bytes_elsewhere.yaml"
    substitute.write_bytes(config.read_bytes())
    with pytest.raises(ValueError, match="path differs"):
        _TOOL._require_exact_bound_input_paths(
            {**exact, "config": substitute.resolve()},
            stage3d_receipt=receipt,
        )


def test_resource_output_is_a_fresh_canonical_run_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    namespace = artifact_root / "e26_stage3d_resource_preflight"
    monkeypatch.setattr(_TOOL, "CANONICAL_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(_TOOL, "RESOURCE_NAMESPACE", namespace)
    created = _TOOL._fresh_output_root(namespace / "20260802T120000Z")
    assert created.parent == namespace.resolve()
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        _TOOL._fresh_output_root(created)
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match="canonical"):
        _TOOL._fresh_output_root(tmp_path / "other" / "run")


def test_post_namespace_execution_error_is_durable_and_not_infeasible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    namespace = artifact_root / "e26_stage3d_resource_preflight"
    monkeypatch.setattr(_TOOL, "CANONICAL_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(_TOOL, "RESOURCE_NAMESPACE", namespace)
    run_dir = _TOOL._fresh_output_root(namespace / "run-error")
    (run_dir / "worker.log").write_text("failure\n", encoding="utf-8")
    _TOOL._write_resource_execution_error(run_dir, RuntimeError("worker crashed"))
    error = json.loads((run_dir / "execution_error.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    latest = json.loads((namespace / "latest.json").read_text(encoding="utf-8"))
    assert error["disposition"] == (
        "RESOURCE_PREFLIGHT_NOT_EVALUABLE_DEPENDENCY_OR_EXECUTION_ERROR"
    )
    assert status["resource_feasibility_evaluated"] is False
    assert latest["disposition"] == error["disposition"]
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "artifact_audit.json").is_file()
    summary = (run_dir / "RESULTS_SUMMARY_KO.md").read_text(encoding="utf-8")
    assert "Scientific E26a started: `false`" in summary
    assert "worker crashed" in summary
    assert error["scientific_e26a_started"] is False


def test_worker_receipt_rejects_layout_or_variant_recipe_drift() -> None:
    measurement = asdict(_measurement())
    layout = _TOOL._expected_layouts(_config())[0]
    spec = {
        "candidate": _config()["model_candidates"][0],
        "fixed_layout": layout,
        "spec_sha256": "5" * 64,
        "source": {"git_commit": "a" * 40, "source_tree_sha256": "b" * 64},
        "input_hashes": {"config_sha256": "e" * 64},
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test-2",
    }
    identity = {
        "same_source_token_accounting": True,
        "same_parameter_signature": True,
        "same_initialization_digest": True,
        "same_optimizer_signature": True,
        "variant_specific_layout": False,
        "variant_specific_precision": False,
        "oom_layout_fallback": False,
        "alternative_layout_audit": False,
    }
    receipt = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_RESOURCE_WORKER_RECEIPT",
        "candidate_id": "d512_ctx4096",
        "candidate_config_sha256": sha256_canonical_json(spec["candidate"]),
        "fixed_layout": layout,
        "measurement": measurement,
        "paired_recipe_identity": identity,
        "run_dir": "/tmp/run",
        "report_sha256": "6" * 64,
        "worker_spec_sha256": "5" * 64,
        "source": spec["source"],
        "input_hashes": spec["input_hashes"],
        "execution_device": {
            "physical_device_index": 2,
            "gpu_uuid": "GPU-test-2",
            "worker_visible_cuda_index": 0,
            "cuda_visible_devices": "2",
            "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
        },
        "scientific_evidence": False,
        "scientific_e26a_started": False,
    }
    _rehash(receipt)
    assert (
        _TOOL._validate_worker_receipt(receipt, spec=spec, verify_artifacts=False)[
            "measurement"
        ]
        == measurement
    )

    drifted = dict(receipt)
    drifted["measurement"] = dict(measurement, selected_microbatch_sequences=2)
    _rehash(drifted)
    with pytest.raises(ValueError, match="fixed-layout contract"):
        _TOOL._validate_worker_receipt(drifted, spec=spec, verify_artifacts=False)

    fallback = dict(receipt)
    fallback["paired_recipe_identity"] = dict(identity, oom_layout_fallback=True)
    _rehash(fallback)
    with pytest.raises(ValueError, match="prohibited layout"):
        _TOOL._validate_worker_receipt(fallback, spec=spec, verify_artifacts=False)

    wrong_device = dict(receipt)
    wrong_device["execution_device"] = dict(receipt["execution_device"], gpu_uuid="GPU-other")
    _rehash(wrong_device)
    with pytest.raises(ValueError, match="device binding"):
        _TOOL._validate_worker_receipt(
            wrong_device,
            spec=spec,
            verify_artifacts=False,
        )


def test_worker_receipt_rehashes_canonical_report_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(_TOOL, "CANONICAL_ARTIFACT_ROOT", artifact_root)
    candidate = _config()["model_candidates"][0]
    layout = _TOOL._expected_layouts(_config())[0]
    experiment = "e26_stage3d_resource_worker_d512_ctx4096"
    run_dir = artifact_root / experiment / "run-1"
    run_dir.mkdir(parents=True)
    device = {
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test-2",
        "worker_visible_cuda_index": 0,
        "cuda_visible_devices": "2",
        "name": "test-gpu",
        "total_memory_bytes": 100,
        "compute_capability": "12.0",
        "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
    }
    spec = {
        "candidate": candidate,
        "fixed_layout": layout,
        "spec_sha256": "5" * 64,
        "source": {"git_commit": "a" * 40, "source_tree_sha256": "b" * 64},
        "input_hashes": {"config_sha256": "e" * 64},
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test-2",
    }
    report = {
        "run_id": "run-1",
        "experiment": experiment,
        "run_mode": "MAIN",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "candidate_id": candidate["id"],
        "worker_spec_sha256": spec["spec_sha256"],
        "source": spec["source"],
        "input_hashes": spec["input_hashes"],
        "measurement": asdict(_measurement()),
        "execution_device": device,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    identity = {
        "same_source_token_accounting": True,
        "same_parameter_signature": True,
        "same_initialization_digest": True,
        "same_optimizer_signature": True,
        "variant_specific_layout": False,
        "variant_specific_precision": False,
        "oom_layout_fallback": False,
        "alternative_layout_audit": False,
    }
    receipt = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_RESOURCE_WORKER_RECEIPT",
        "candidate_id": candidate["id"],
        "candidate_config_sha256": sha256_canonical_json(candidate),
        "fixed_layout": layout,
        "measurement": asdict(_measurement()),
        "paired_recipe_identity": identity,
        "run_dir": str(run_dir),
        "report_sha256": sha256_file(report_path),
        "worker_spec_sha256": spec["spec_sha256"],
        "source": spec["source"],
        "input_hashes": spec["input_hashes"],
        "execution_device": device,
        "scientific_evidence": False,
        "scientific_e26a_started": False,
    }
    _rehash(receipt)
    hardware = {
        "physical_device_index": 2,
        "gpu_uuid": "GPU-test-2",
        "name": "test-gpu",
        "total_memory_bytes": 100,
        "compute_capability": "12.0",
    }
    _TOOL._validate_worker_receipt(
        receipt,
        spec=spec,
        expected_hardware=hardware,
    )
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="report SHA-256 changed"):
        _TOOL._validate_worker_receipt(
            receipt,
            spec=spec,
            expected_hardware=hardware,
        )


def test_tool_contains_no_alternative_layout_or_scientific_launcher() -> None:
    source = _TOOL_PATH.read_text(encoding="utf-8")
    assert "audit_target_gradient_accumulation" not in source
    assert "microbatch_candidates=(FIXED_MICROBATCH_SEQUENCES,)" in source
    assert '"scientific_e26a_started": False' in source
    assert "E26A_SCIENTIFIC_GATE_AUTHORIZED" not in source

    launcher = (
        Path(__file__).resolve().parents[1] / "scripts/run_e26_stage3d_fixed_layout.sh"
    ).read_text(encoding="utf-8")
    assert 'if [[ "$RUNNER_EXIT" -ne 0 ]]' in launcher
    assert "STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE" in launcher
    assert "run_e26_stage3d_resource_preflight.py" in launcher
    assert "run_e26a_scientific_gate.sh" not in launcher


def test_resource_schema_and_tool_policy_are_synchronized() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "v8_1"
        / "e26_stage3d_resource_preflight_receipt.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["schema_version"]["const"] == _TOOL.RESOURCE_RECEIPT_VERSION
    policy = properties["resource_policy"]["properties"]
    assert policy["token_budgets"]["const"] == list(_TOOL.TOKEN_BUDGETS)
    assert policy["max_main_wall_clock_hours"]["const"] == 168
    assert policy["safety_time_multiplier"]["const"] == 1.25
    assert policy["max_main_checkpoint_storage_gib"]["const"] == 100


@pytest.mark.parametrize("passed", [True, False])
def test_resource_schema_accepts_feasible_and_infeasible_terminal_receipts(
    passed: bool,
) -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "v8_1"
        / "e26_stage3d_resource_preflight_receipt.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    file_binding = {"path": "/locked/input", "sha256": "a" * 64}
    layouts = _TOOL._expected_layouts(_config())
    candidates = []
    for candidate, layout in zip(_config()["model_candidates"], layouts, strict=True):
        candidates.append(
            {
                "candidate_id": candidate["id"],
                "candidate_config_sha256": sha256_canonical_json(candidate),
                "fixed_layout": layout,
                "measurement": asdict(_measurement()),
                "resource_projections": [{"eligible": passed} for _ in range(3)],
                "paired_recipe_identity": {},
                "execution_device": {},
                "worker_run_dir": "/data/worker",
                "worker_report_sha256": "b" * 64,
                "worker_receipt_sha256": "c" * 64,
                "worker_report": {"path": "/data/worker/report.json", "sha256": "b" * 64},
                "worker_receipt": {
                    "path": "/data/aggregate/worker.json",
                    "sha256": "d" * 64,
                    "receipt_sha256": "c" * 64,
                },
            }
        )
    payload = {
        "schema_version": _TOOL.RESOURCE_RECEIPT_VERSION,
        "manifest_type": _TOOL.RESOURCE_MANIFEST_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "run_dir": "/data/aggregate/run",
        "disposition": (
            "RESOURCE_PREFLIGHT_FEASIBLE"
            if passed
            else "RESOURCE_PREFLIGHT_INFEASIBLE"
        ),
        "inputs": {
            "config": file_binding,
            "tokenizer_manifest": file_binding,
            "corpus_manifest": file_binding,
            "stage3d_receipt": {**file_binding, "receipt_sha256": "e" * 64},
        },
        "stage3d_receipt": {**file_binding, "receipt_sha256": "e" * 64},
        "source": {
            "git_commit": "f" * 40,
            "source_tree_sha256": "1" * 64,
            "source_file_count": 1,
        },
        "fixed_layouts": layouts,
        "resource_policy": {
            "token_budgets": [250_000_000, 375_000_000, 500_000_000],
            "deadline_reference_hours": 240,
            "deadline_fraction_max": 0.70,
            "max_main_wall_clock_hours": 168,
            "safety_time_multiplier": 1.25,
            "max_main_checkpoint_storage_gib": 100,
            "main_runs": 10,
            "gpu_lanes": 4,
            "save_every_tokens": 25_000_000,
        },
        "hardware_inventory": [{}],
        "candidates": candidates,
        "selection": {} if passed else None,
        "selection_error": None if passed else "No eligible candidate",
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "canonical_e26_artifact_created": False,
        "passed": passed,
        "receipt_sha256": "2" * 64,
    }
    jsonschema.Draft202012Validator(schema).validate(payload)
