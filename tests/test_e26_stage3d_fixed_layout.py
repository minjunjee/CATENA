from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest
from helpers.e26_stage3d_contract_fixture import stage3c_fixture_inputs

from catena.core.provenance_v61 import sha256_canonical_json, write_json_strict
from catena.lm.stage3d_fixed_layout import (
    STAGE3D_BLOCKED,
    STAGE3D_GO,
    STAGE3D_NOT_EVALUABLE,
    Stage3DContractError,
    _validate_stage3c_fp32_report,
    build_stage3d_admissibility_receipt,
    build_stage3d_protocol_lock,
    validate_stage3d_admissibility_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e26_stage3d_fixed_layout_bf16_admissibility.yaml"
VARIANTS = ("projected_tied_delta_lm", "dual_delta_lm")
CONTEXTS = ("zero_state", "prefilled_state")
FRESH_G3_BOUNDARY_RUN = Path(
    "/data/minjun_dev/CATENA/artifacts/e26_stage3d_fixed_layout_bf16_admissibility/"
    "20260802T144040.692630Z"
)


def _protocol(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    paths = stage3c_fixture_inputs(tmp_path)
    lock = build_stage3d_protocol_lock(
        config_path=CONFIG_PATH,
        stage3c_result_path=paths["result"],
        stage3c_protocol_path=paths["protocol"],
        stage3c_artifact_manifest_path=paths["artifact"],
        frozen_e00_e25_receipt_path=paths["frozen"],
        source_commit="3" * 40,
        source_inventory={"source_tree_sha256": "2" * 64, "files": 600},
    )
    lock_path = tmp_path / "stage3d_protocol.json"
    write_json_strict(lock_path, lock)
    return lock, lock_path


def _error(value: float = 0.0) -> dict[str, float]:
    return {"relative_l2": value, "max_abs": value}


def _comparison(value: float = 0.0, *, passed: bool = True) -> dict[str, Any]:
    return {
        "logits": _error(value),
        "runtime_state": _error(value),
        "gradients": _error(value),
        "passed": passed,
    }


def _g3_comparison(value: float = 0.0, *, passed: bool = True) -> dict[str, Any]:
    return {
        "logits": _error(value),
        "runtime_state": _error(value),
        "runtime_state_components": {
            "recurrent": _error(value),
            "attention_key": _error(value),
            "attention_value": _error(value),
        },
        "state_metadata_exact": True,
        "gradients": _error(value),
        "gradients_worst_leaf": _error(value),
        "tolerance": {"relative_l2_max": 0.007, "max_abs_max": None},
        "passed": passed,
    }


def _g3(lock: dict[str, Any]) -> list[dict[str, Any]]:
    layouts = {row["candidate_id"]: row for row in lock["fixed_layouts"]}
    rows = []
    for candidate, layout in layouts.items():
        for variant in VARIANTS:
            for context in CONTEXTS:
                layout_identity = {
                    "physical_microbatch_sequences": int(layout["microbatch_sequences"]),
                    "sequence_length": int(layout["context_length"]),
                    "accumulation_steps": int(layout["accumulation_steps"]),
                    "target_global_input_tokens": int(layout["target_global_input_tokens"]),
                    "loss_denominator": (
                        "TOTAL_VALID_NEXT_TOKEN_PREDICTIONS_ACROSS_FIXED_LAYOUT"
                    ),
                    "optimizer_update_boundary": "AFTER_EXACT_ACCUMULATION_STEPS",
                    "autocast_scope": "CUDA_BF16_FORWARD_LOSS_ONLY",
                    "gradient_clipping_order": "AFTER_ACCUMULATION_BEFORE_ADAMW",
                    "initialization_matched": True,
                    "parameter_surface_matched": True,
                    "optimizer_state_shape_matched": True,
                    "shape_contract_valid": True,
                    "passed": True,
                }
                rows.append(
                    {
                        "candidate_id": candidate,
                        "variant": variant,
                        "state_context": context,
                        "fixed_layout": layout,
                        "initialization_digest": "4" * 64,
                        "parameter_signature_sha256": "5" * 64,
                        "optimizer_state_signature_sha256": "6" * 64,
                        "token_ids_sha256": "7" * 64,
                        "data_cursor_sha256": "8" * 64,
                        "layout_identity": layout_identity,
                        "layout_identity_passed": True,
                        "comparisons": {
                            "compiled_bf16_vs_reference_python_bf16": _g3_comparison(),
                            "reference_python_bf16_vs_reference_python_fp32": (
                                _g3_comparison()
                            ),
                        },
                        "gradient_finite": True,
                        "gradient_norms": {
                            "compiled_bf16": 1.0,
                            "reference_python_bf16": 1.0,
                            "reference_python_fp32": 1.0,
                        },
                        "gradient_norm_in_range": True,
                        "state_metadata_exact": True,
                        "clone_no_alias": True,
                        "graph_break_count": 0,
                        "fallback_count": 0,
                        "variant_specific_fp32_path_count": 0,
                        "variant_specific_padding_count": 0,
                        "backend_integrity": {
                            "optimized_backend_id": "compiled_scan",
                            "registered_backend_id": "torch_compile_fixed_chunk_scan_v1",
                            "runtime_backend_alias": "compiled_scan",
                            "backend_alias_matches_registration": True,
                            "reference_backend_id": "reference_python",
                            "strict_reference_python": True,
                            "positive_compiled_execution": True,
                            "python_token_loop_at_scientific_runtime": False,
                            "graph_break_count": 0,
                            "fallback_count": 0,
                            "variant_specific_fp32_path_count": 0,
                            "variant_specific_padding_count": 0,
                            "passed": True,
                        },
                        "passed": True,
                    }
                )
    return rows


def _g4(lock: dict[str, Any]) -> list[dict[str, Any]]:
    layouts = {row["candidate_id"]: row for row in lock["fixed_layouts"]}
    rows = []
    for candidate, layout in layouts.items():
        valid_tokens = (
            int(layout["microbatch_sequences"])
            * int(layout["accumulation_steps"])
            * (int(layout["context_length"]) - 1)
        )
        for variant in VARIANTS:
            rows.append(
                {
                    "candidate_id": candidate,
                    "variant": variant,
                    "fixed_layout": layout,
                    "checkpoint_sha256": "9" * 64,
                    "checkpoint_semantic_sha256": "0" * 64,
                    "rng_state_sha256": "a" * 64,
                    "data_ids_sha256": "b" * 64,
                    "data_cursor_sha256": "1" * 64,
                    "backend_graph_sha256": ("c" if variant == VARIANTS[0] else "e") * 64,
                    "backend_recipe_sha256": "2" * 64,
                    "optimizer_input_sha256": "d" * 64,
                    "initialization_digest": "3" * 64,
                    "parameter_signature_sha256": "4" * 64,
                    "initial_optimizer_state_signature_sha256": "5" * 64,
                    "comparison": _comparison(),
                    "optimizer_state": _error(),
                    "optimizer_state_structure_equal": True,
                    "scheduler_state_equal": True,
                    "optimizer_step_integrity": {
                        "global_token_normalization_identity": True,
                        "accumulation_buffer_reset_once": True,
                        "gradient_clipping_after_accumulation": True,
                        "adamw_step_and_bias_correction_identity": True,
                        "weight_decay_order_and_value_identity": True,
                        "skipped_optimizer_steps_zero": True,
                        "all_gradients_finite": True,
                        "valid_prediction_tokens": valid_tokens,
                        "expected_valid_prediction_tokens": valid_tokens,
                        "exposed_input_tokens": int(layout["target_global_input_tokens"]),
                        "expected_input_tokens": int(layout["target_global_input_tokens"]),
                        "microbatch_count": int(layout["accumulation_steps"]),
                        "expected_microbatch_count": int(layout["accumulation_steps"]),
                        "execution_events": [
                            "zero_grad",
                            "gradient_clip",
                            "adamw_step",
                            "scheduler_step",
                        ],
                        "adamw_state_steps": [1],
                        "passed": True,
                    },
                    "gradients_finite": True,
                    "state_metadata_exact": True,
                    "clone_no_alias": True,
                    "optimizer_integrity_passed": True,
                    "graph_break_count": 0,
                    "fallback_count": 0,
                    "passed": True,
                }
            )
    return rows


def _fp32_reference(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "reuse_policy": "EXACT_STAGE3C_HASH_BINDING_ONLY",
        "stage3c_result_sha256": lock["stage3c"]["result"]["sha256"],
        "stage3c_artifact_manifest_sha256": lock["stage3c"]["artifact_hash_manifest"]["sha256"],
        "stage3c_artifact_aggregate_sha256": lock["stage3c"][
            "artifact_manifest_rehash_aggregate_sha256"
        ],
        "fp32_arbitrary_partition_reports_passed": 12,
        "fp32_arbitrary_partition_reports_required": 12,
        "fp32_nontrivial_rows_passed": 132,
        "fp32_nontrivial_rows_required": 132,
        "thresholds": {"relative_l2_max": 1.0e-5, "max_abs_max": 1.0e-5},
    }


def test_fixed_layout_receipt_go_requires_exact_12_plus_6(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=_g4(lock),
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["disposition"] == STAGE3D_GO
    assert receipt["gate_summary"]["g3_pass_count"] == 12
    assert receipt["gate_summary"]["g4_pass_count"] == 6
    assert validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock) == receipt
    schema = json.loads(
        (REPO_ROOT / "schemas/v8_1/e26_stage3d_fixed_layout_receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)


def test_realistic_failed_g3_diagnostics_are_evaluable_and_schema_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The frozen Stage-3C manifest intentionally records absolute predecessor
    # anchors from the frozen live repository. Resolve its relative semantic
    # checks in that registered context when regression runs from a later
    # additive worktree.
    monkeypatch.chdir("/home/minjun_dev/CATENA")
    original_report_path = FRESH_G3_BOUNDARY_RUN / "report.json"
    protocol_path = FRESH_G3_BOUNDARY_RUN / "protocol_lock.json"
    original_report = json.loads(original_report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amended = build_stage3d_admissibility_receipt(
        protocol_lock_path=protocol_path,
        g3_cases=original_report["g3_cases"],
        g4_replays=original_report["g4_replays"],
        fp32_reference_binding=original_report["fp32_reference"],
    )
    assert amended["execution_status"] == "COMPLETED_NUMERICAL_EVALUATION"
    assert amended["disposition"] == STAGE3D_BLOCKED
    assert amended["gate_summary"]["g3_pass_count"] == 0
    assert validate_stage3d_admissibility_receipt(amended, protocol_lock=protocol) == amended
    schema = json.loads(
        (REPO_ROOT / "schemas/v8_1/e26_stage3d_fixed_layout_receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(amended)


def test_g3_diagnostic_tampering_is_not_numerically_dispositioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir("/home/minjun_dev/CATENA")
    original_report = json.loads(
        (FRESH_G3_BOUNDARY_RUN / "report.json").read_text(encoding="utf-8")
    )
    protocol_path = FRESH_G3_BOUNDARY_RUN / "protocol_lock.json"
    cases = deepcopy(original_report["g3_cases"])
    comparison = cases[0]["comparisons"]["compiled_bf16_vs_reference_python_bf16"]
    comparison["runtime_state_components"]["recurrent"]["relative_l2"] = 0.0
    amended = build_stage3d_admissibility_receipt(
        protocol_lock_path=protocol_path,
        g3_cases=cases,
        g4_replays=original_report["g4_replays"],
        fp32_reference_binding=original_report["fp32_reference"],
    )
    assert amended["execution_status"] == "FAILED_IMPLEMENTATION_OR_EXECUTION"
    assert amended["disposition"] == STAGE3D_NOT_EVALUABLE


def test_g4_cross_variant_identity_is_recomputed_but_graph_may_differ(
    tmp_path: Path,
) -> None:
    lock, lock_path = _protocol(tmp_path)
    replay = _g4(lock)
    assert replay[0]["backend_graph_sha256"] != replay[1]["backend_graph_sha256"]
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["gate_summary"]["g4_passed"] is True
    assert receipt["disposition"] == STAGE3D_GO

    replay[1]["checkpoint_semantic_sha256"] = "f" * 64
    blocked = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert blocked["execution_status"] == "COMPLETED_NUMERICAL_EVALUATION"
    assert blocked["gate_summary"]["g4_passed"] is False
    assert blocked["disposition"] == STAGE3D_BLOCKED
    validate_stage3d_admissibility_receipt(blocked, protocol_lock=lock)


def test_rehashed_g4_cross_variant_drift_cannot_be_forged_into_go(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    replay = _g4(lock)
    replay[1]["backend_recipe_sha256"] = "f" * 64
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    forged = deepcopy(receipt)
    forged["gate_summary"]["g4_passed"] = True
    forged["passed"] = True
    forged["disposition"] = STAGE3D_GO
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = sha256_canonical_json(forged)
    with pytest.raises(Stage3DContractError, match="G4|summary"):
        validate_stage3d_admissibility_receipt(forged, protocol_lock=lock)


def test_blocked_g6_backend_row_is_schema_valid_and_not_promotable(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    cases = _g3(lock)
    cases[0]["graph_break_count"] = 1
    cases[0]["backend_integrity"].update(
        {
            "optimized_backend_id": "reference_python",
            "positive_compiled_execution": False,
            "graph_break_count": 1,
            "passed": False,
        }
    )
    cases[0]["passed"] = False
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=cases,
        g4_replays=_g4(lock),
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["gate_summary"]["g6_passed"] is False
    assert receipt["disposition"] == STAGE3D_BLOCKED
    assert validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock) == receipt
    schema = json.loads(
        (REPO_ROOT / "schemas/v8_1/e26_stage3d_fixed_layout_receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)


def test_parent_rederives_g2_before_namespace_or_gpu_work() -> None:
    import importlib.util

    path = REPO_ROOT / "tools/run_e26_stage3d_preflight.py"
    spec = importlib.util.spec_from_file_location("catena_stage3d_g2_order_tool", path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source = inspect.getsource(tool._main_parent_run)
    derive_index = source.index("derive_stage3c_fp32_reference_binding(protocol)")
    assert derive_index < source.index("_fresh_output_root")
    assert derive_index < source.index("cuda_hardware_inventory")
    assert derive_index < source.index("ThreadPoolExecutor")


def test_g5_optimizer_integrity_failure_blocks_without_relabelling_stage3c(
    tmp_path: Path,
) -> None:
    lock, lock_path = _protocol(tmp_path)
    replay = _g4(lock)
    replay[0]["optimizer_integrity_passed"] = False
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["disposition"] == STAGE3D_BLOCKED
    assert receipt["gate_summary"]["g5_passed"] is False
    assert receipt["stage3c"]["disposition"].startswith("BLOCKED_NUMERICAL")
    validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock)


def test_g5_forged_boolean_cannot_hide_optimizer_trace_drift(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    replay = _g4(lock)
    replay[0]["optimizer_step_integrity"]["execution_events"] = [
        "zero_grad",
        "adamw_step",
        "gradient_clip",
        "scheduler_step",
    ]
    replay[0]["optimizer_step_integrity"]["passed"] = True
    replay[0]["optimizer_integrity_passed"] = True
    replay[0]["passed"] = True
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock),
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["gate_summary"]["g5_passed"] is False
    assert receipt["disposition"] == STAGE3D_BLOCKED
    validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock)
    schema = json.loads(
        (REPO_ROOT / "schemas/v8_1/e26_stage3d_fixed_layout_receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(receipt)

    forged = deepcopy(receipt)
    forged["gate_summary"]["g5_passed"] = True
    forged["passed"] = True
    forged["disposition"] = STAGE3D_GO
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = sha256_canonical_json(forged)
    with pytest.raises(Stage3DContractError, match="G5|summary"):
        validate_stage3d_admissibility_receipt(forged, protocol_lock=lock)


def test_fixed_probe_reference_mismatch_is_separate_from_layout_sensitivity(
    tmp_path: Path,
) -> None:
    lock, lock_path = _protocol(tmp_path)
    cases = _g3(lock)
    cases[0]["comparisons"]["compiled_bf16_vs_reference_python_bf16"] = _g3_comparison(
        0.02, passed=False
    )
    cases[0]["passed"] = False
    replay = [
        {
            "candidate_id": candidate,
            "variant": variant,
            "fixed_layout": next(
                row for row in lock["fixed_layouts"] if row["candidate_id"] == candidate
            ),
            "optimizer_integrity_passed": False,
            "graph_break_count": 0,
            "fallback_count": 0,
            "status": "NOT_RUN_BLOCKED_UPSTREAM",
            "passed": False,
        }
        for candidate in ("d512_ctx4096", "d512_ctx2048", "d448_ctx4096")
        for variant in VARIANTS
    ]
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=cases,
        g4_replays=replay,
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["diagnostic_disposition"] == ("KNOWN_BF16_AND_OPTIMIZER_LAYOUT_SENSITIVITY")
    assert receipt["fixed_probe_reference_mismatch"] == "OBSERVED_IN_STAGE3D_G3"
    assert receipt["disposition"] == STAGE3D_BLOCKED
    validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock)


def test_rehashed_forged_go_is_rejected(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    cases = _g3(lock)
    cases[0]["passed"] = False
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=cases,
        g4_replays=_g4(lock),
        fp32_reference_binding=_fp32_reference(lock),
    )
    forged = deepcopy(receipt)
    forged["passed"] = True
    forged["disposition"] = STAGE3D_GO
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = sha256_canonical_json(forged)
    with pytest.raises(Stage3DContractError):
        validate_stage3d_admissibility_receipt(forged, protocol_lock=lock)


def test_stage3c_fp32_raw_metrics_are_recomputed_not_trusted() -> None:
    numerical = json.loads(
        (
            Path(
                "/data/minjun_dev/CATENA/artifacts/e26_stage3c_numerical_preflight/"
                "20260802T060323Z/d512_ctx4096_numerical.json"
            )
        ).read_text(encoding="utf-8")
    )
    report = deepcopy(
        numerical["variants"]["projected_tied_delta_lm"]["arbitrary_partitions"]["zero_state"][
            "fp32"
        ]
    )
    assert report["passed"] is True
    assert report["rows"][1]["passed"] is True
    report["rows"][1]["gradients"]["relative_l2"] = 0.5
    with pytest.raises(Stage3DContractError, match="exceeds inherited"):
        _validate_stage3c_fp32_report(
            report,
            label="tampered_raw",
            partition_length=int(numerical["partition_length"]),
            relative_max=1.0e-5,
            max_abs_max=1.0e-5,
        )


def test_g6_reference_fallback_cannot_be_rehashed_into_go(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    cases = _g3(lock)
    cases[0]["backend_integrity"]["optimized_backend_id"] = "reference_python"
    cases[0]["backend_integrity"]["positive_compiled_execution"] = False
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=cases,
        g4_replays=_g4(lock),
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["disposition"] == STAGE3D_BLOCKED
    assert receipt["gate_summary"]["g6_passed"] is False
    forged = deepcopy(receipt)
    forged["gate_summary"]["g6_passed"] = True
    forged["disposition"] = STAGE3D_GO
    forged["passed"] = True
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = sha256_canonical_json(forged)
    with pytest.raises(Stage3DContractError):
        validate_stage3d_admissibility_receipt(forged, protocol_lock=lock)


def test_missing_g3_raw_receipt_is_not_a_numerical_failure(tmp_path: Path) -> None:
    lock, lock_path = _protocol(tmp_path)
    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=lock_path,
        g3_cases=_g3(lock)[:-1],
        g4_replays=[],
        fp32_reference_binding=_fp32_reference(lock),
    )
    assert receipt["execution_status"] == "FAILED_IMPLEMENTATION_OR_EXECUTION"
    assert receipt["disposition"] == STAGE3D_NOT_EVALUABLE
    assert receipt["passed"] is False
    assert validate_stage3d_admissibility_receipt(receipt, protocol_lock=lock) == receipt
