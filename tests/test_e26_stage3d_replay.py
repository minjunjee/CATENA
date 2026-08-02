from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict
from catena.lm.config import ModelConfig
from catena.lm.model import CatenaLM
from catena.lm.numerical_audit import NumericalTolerances

_SPEC = importlib.util.spec_from_file_location(
    "catena_e26_stage3d_preflight_tool",
    Path(__file__).resolve().parents[1] / "tools" / "run_e26_stage3d_preflight.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def _layout() -> dict[str, int | str]:
    return {
        "candidate_id": "d512_ctx4096",
        "context_length": 4096,
        "microbatch_sequences": 1,
        "target_global_input_tokens": 65536,
        "global_batch_sequences": 16,
        "accumulation_steps": 16,
    }


def _diagnostics() -> dict[str, int | str]:
    return {
        "graph_invocations": 2,
        "optimized_calls": 1,
        "chunks_executed": 128,
        "padded_tokens": 0,
        "fallback_count": 0,
        "graph_break_count": 0,
        "last_graph_code_sha256": "a" * 64,
    }


def _row(replay_id: str) -> dict[str, Any]:
    return {
        "candidate_id": "d512_ctx4096",
        "variant": "dual_delta_lm",
        "replay_id": replay_id,
        "fixed_layout": _layout(),
        "checkpoint_input_sha256": "b" * 64,
        "checkpoint_semantic_sha256": "a" * 64,
        "rng_input_sha256": "c" * 64,
        "data_ids_sha256": "d" * 64,
        "data_cursor_sha256": "7" * 64,
        "backend_input_sha256": "e" * 64,
        "backend_recipe_sha256": "6" * 64,
        "optimizer_input_sha256": "f" * 64,
        "initial_parameter_digest": "1" * 64,
        "parameter_signature_sha256": "2" * 64,
        "initial_optimizer_state_signature_sha256": "3" * 64,
        "scheduler_sha256": "4" * 64,
        "gradients_finite": True,
        "state_metadata": {
            "position": 8,
            "attention_lengths": [],
            "attention_write_indices": [],
        },
        "clone_no_alias": True,
        "backend_diagnostics": _diagnostics(),
        "optimizer_step_integrity": {
            "global_token_normalization_identity": True,
            "accumulation_buffer_reset_once": True,
            "gradient_clipping_after_accumulation": True,
            "adamw_step_and_bias_correction_identity": True,
            "weight_decay_order_and_value_identity": True,
            "skipped_optimizer_steps_zero": True,
            "all_gradients_finite": True,
            "passed": True,
        },
    }


def _tensors() -> dict[str, Any]:
    torch.manual_seed(260_990)
    model = CatenaLM(ModelConfig.tiny_reference())
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        output = model(ids)
    return {
        "logits": output.logits.detach().clone(),
        "runtime_state": output.runtime_state.clone(detach=True),
        "gradients": {"weight": torch.ones(3)},
        "model_state": {"weight": torch.arange(3, dtype=torch.float32)},
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.ones(3)}},
            "param_groups": [{"lr": 3.0e-4}],
        },
    }


def test_same_fixed_layout_replay_passes_and_emits_contract_names() -> None:
    left = _row("A")
    right = _row("B")
    tensors = _tensors()
    result = _TOOL._compare_replay_rows(
        left,
        right,
        left_tensors=tensors,
        right_tensors=copy.deepcopy(tensors),
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    assert result["passed"] is True
    assert result["checkpoint_sha256"] == "b" * 64
    assert result["checkpoint_semantic_sha256"] == "a" * 64
    assert result["rng_state_sha256"] == "c" * 64
    assert result["backend_graph_sha256"] == "a" * 64
    assert result["optimizer_integrity_passed"] is True
    assert set(result["comparison"]) == {
        "logits",
        "runtime_state",
        "gradients",
        "passed",
    }


def test_replay_fails_closed_on_layout_data_or_backend_graph_drift() -> None:
    left = _row("A")
    right = _row("B")
    right["fixed_layout"] = {**_layout(), "accumulation_steps": 15}
    right["data_ids_sha256"] = "9" * 64
    right["backend_diagnostics"] = {**_diagnostics(), "last_graph_code_sha256": "8" * 64}
    tensors = _tensors()
    result = _TOOL._compare_replay_rows(
        left,
        right,
        left_tensors=tensors,
        right_tensors=copy.deepcopy(tensors),
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    assert result["passed"] is False
    assert set(result["identity_mismatches"]) >= {
        "fixed_layout",
        "data_ids_sha256",
        "backend_graph_sha256",
    }


def test_replay_fails_closed_on_numerical_or_optimizer_drift() -> None:
    left = _row("A")
    right = _row("B")
    left_tensors = _tensors()
    right_tensors = copy.deepcopy(left_tensors)
    right_tensors["gradients"]["weight"].mul_(2.0)
    right_tensors["optimizer_state"]["state"][0]["exp_avg"].mul_(2.0)
    result = _TOOL._compare_replay_rows(
        left,
        right,
        left_tensors=left_tensors,
        right_tensors=right_tensors,
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    assert result["passed"] is False
    assert result["comparison"]["passed"] is False
    assert result["optimizer_integrity_passed"] is False


def test_replay_fails_closed_on_each_registered_optimizer_integrity_drift() -> None:
    fields = (
        "global_token_normalization_identity",
        "accumulation_buffer_reset_once",
        "gradient_clipping_after_accumulation",
        "adamw_step_and_bias_correction_identity",
        "weight_decay_order_and_value_identity",
        "skipped_optimizer_steps_zero",
        "all_gradients_finite",
    )
    tensors = _tensors()
    for field in fields:
        left = _row("A")
        right = _row("B")
        right["optimizer_step_integrity"][field] = False
        right["optimizer_step_integrity"]["passed"] = False
        result = _TOOL._compare_replay_rows(
            left,
            right,
            left_tensors=tensors,
            right_tensors=copy.deepcopy(tensors),
            tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
        )
        assert result["passed"] is False, field
        assert result["optimizer_integrity_passed"] is False, field


def test_g3_failure_creates_six_explicit_not_run_replays() -> None:
    candidates = {
        candidate: {"id": candidate}
        for candidate in ("d512_ctx4096", "d512_ctx2048", "d448_ctx4096")
    }
    layouts = {
        "d512_ctx4096": _layout(),
        "d512_ctx2048": {
            **_layout(),
            "candidate_id": "d512_ctx2048",
            "context_length": 2048,
            "global_batch_sequences": 32,
            "accumulation_steps": 32,
        },
        "d448_ctx4096": {**_layout(), "candidate_id": "d448_ctx4096"},
    }
    rows = _TOOL._blocked_g4_dependency_rows(
        candidate_by_id=candidates,
        layouts=layouts,
        reason="G3_FAILED",
    )
    assert len(rows) == 6
    assert {(row["candidate_id"], row["variant"]) for row in rows} == {
        (candidate, variant)
        for candidate in candidates
        for variant in ("projected_tied_delta_lm", "dual_delta_lm")
    }
    assert all(row["execution_status"] == "NOT_RUN_BLOCKED_DEPENDENCY" for row in rows)
    assert all(row["passed"] is False for row in rows)


def test_cross_variant_g4_binds_common_checkpoint_data_and_recipe() -> None:
    tied = _TOOL._compare_replay_rows(
        {**_row("A"), "variant": "projected_tied_delta_lm"},
        {**_row("B"), "variant": "projected_tied_delta_lm"},
        left_tensors=_tensors(),
        right_tensors=copy.deepcopy(_tensors()),
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    dual = _TOOL._compare_replay_rows(
        _row("A"),
        _row("B"),
        left_tensors=_tensors(),
        right_tensors=copy.deepcopy(_tensors()),
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    rows = _TOOL._apply_cross_variant_g4([tied, dual])
    assert all(row["cross_variant_identity"]["passed"] is True for row in rows)

    drifted = copy.deepcopy(dual)
    drifted["data_ids_sha256"] = "9" * 64
    rows = _TOOL._apply_cross_variant_g4([tied, drifted])
    assert all(row["passed"] is False for row in rows)
    assert all("data_ids_sha256" in row["cross_variant_identity"]["mismatches"] for row in rows)


def test_common_checkpoint_receipt_detects_byte_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "common.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    backend_recipe = {"backend_id": "compiled_scan", "compiler": "inductor"}
    spec: dict[str, Any] = {
        "candidate": {"id": "d512_ctx4096"},
        "fixed_layout": _layout(),
        "checkpoint_locked_hashes": {"source_tree_sha256": "a" * 64},
        "backend_recipe": backend_recipe,
        "spec_sha256": "7" * 64,
    }
    row: dict[str, Any] = {
        "candidate_id": "d512_ctx4096",
        "fixed_layout": _layout(),
        "worker_spec_sha256": "7" * 64,
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "checkpoint_locked_hashes": spec["checkpoint_locked_hashes"],
        "backend_manifest": backend_recipe,
        "rng_state_sha256": "1" * 64,
        "data_cursor_sha256": "2" * 64,
        "initialization_digest": "3" * 64,
        "parameter_signature_sha256": "4" * 64,
        "optimizer_state_signature_sha256": "5" * 64,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
    }
    row["receipt_sha256"] = sha256_canonical_json(row)
    assert _TOOL._validate_checkpoint_row(row, spec=spec)["candidate_id"] == "d512_ctx4096"
    checkpoint.write_bytes(b"checkpoint-v2-mutated")
    with pytest.raises(ValueError, match="checkpoint_bytes"):
        _TOOL._validate_checkpoint_row(row, spec=spec)


def test_execution_error_terminal_is_not_a_numerical_disposition(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_json_strict(run / "protocol_lock.json", {"locked": True})
    write_json_strict(run / "layout_manifest.json", {"locked": True})
    _TOOL._write_execution_error_terminal(
        output_root=run,
        stage="G4_SAME_LAYOUT_REPLAY",
        failures=["worker crashed"],
        completed_g3=12,
        completed_g4=2,
    )
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["execution_status"] == "EXECUTION_ERROR"
    assert report["disposition"] == _TOOL.STAGE3D_NOT_EVALUABLE
    assert report["disposition"] != _TOOL.STAGE3D_BLOCKED


def test_checkpoint_cursor_is_authoritative_and_rejects_data_drift() -> None:
    expected = {
        "data_seed": 260801,
        "candidate_id": "d512_ctx4096",
        "next_microbatch_index": 0,
    }
    cursor = {**expected, "snapshot_sha256": sha256_canonical_json(expected)}
    assert (
        _TOOL._validated_checkpoint_cursor(
            cursor,
            candidate_id="d512_ctx4096",
            data_seed=260801,
        )["data_seed"]
        == 260801
    )
    with pytest.raises(RuntimeError, match="data cursor changed"):
        _TOOL._validated_checkpoint_cursor(
            {**cursor, "data_seed": 260802},
            candidate_id="d512_ctx4096",
            data_seed=260801,
        )


def test_candidate_lane_creates_one_checkpoint_then_four_fresh_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_dir = tmp_path / "g4"
    (replay_dir / "checkpoints").mkdir(parents=True)
    calls: list[str] = []

    def fake_run_worker(**kwargs: Any) -> int:
        mode = str(kwargs["mode"])
        calls.append(mode)
        spec = json.loads(Path(kwargs["spec_path"]).read_text(encoding="utf-8"))
        output = Path(kwargs["output_path"])
        if mode == "--worker-checkpoint":
            checkpoint_path = Path(kwargs["checkpoint_path"])
            checkpoint_path.write_bytes(b"one-common-checkpoint")
            checkpoint = {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
                "semantic_payload_sha256": "a" * 64,
            }
            row: dict[str, Any] = {
                "candidate_id": "d512_ctx4096",
                "fixed_layout": _layout(),
                "worker_spec_sha256": spec["spec_sha256"],
                "scientific_evidence": False,
                "main_test_opened": False,
                "scientific_e26a_started": False,
                "checkpoint_locked_hashes": spec["checkpoint_locked_hashes"],
                "backend_manifest": spec["backend_recipe"],
                "checkpoint": checkpoint,
                "rng_state_sha256": "c" * 64,
                "data_cursor_sha256": "7" * 64,
                "initialization_digest": "1" * 64,
                "parameter_signature_sha256": "2" * 64,
                "optimizer_state_signature_sha256": "3" * 64,
            }
        else:
            tensor_path = Path(kwargs["tensor_path"])
            torch.save(_tensors(), tensor_path)
            row = {
                **_row(str(spec["replay_id"])),
                "candidate_id": "d512_ctx4096",
                "variant": spec["variant"],
                "fixed_layout": _layout(),
                "worker_spec_sha256": spec["spec_sha256"],
                "scientific_evidence": False,
                "main_test_opened": False,
                "scientific_e26a_started": False,
                "checkpoint_input_sha256": spec["checkpoint"]["sha256"],
                "checkpoint_semantic_sha256": spec["checkpoint"]["semantic_payload_sha256"],
                "rng_input_sha256": "c" * 64,
                "initial_parameter_digest": "1" * 64,
                "parameter_signature_sha256": "2" * 64,
                "initial_optimizer_state_signature_sha256": "3" * 64,
                "backend_recipe_sha256": sha256_canonical_json(spec["backend_recipe"]),
                "tensor_payload_path": str(tensor_path),
                "tensor_payload_sha256": sha256_file(tensor_path),
            }
        row["receipt_sha256"] = sha256_canonical_json(row)
        write_json_strict(output, row)
        return 0

    monkeypatch.setattr(_TOOL, "_run_worker", fake_run_worker)
    common = {
        "schema_version": "catena-v8.1",
        "protocol_sha256": "4" * 64,
        "source_tree_sha256": "5" * 64,
        "initialization_seed": 260301,
        "data_seed": 260801,
        "checkpoint_locked_hashes": {"source_tree_sha256": "5" * 64},
        "backend_binding": {
            "registered_backend_id": "torch_compile_fixed_chunk_scan_v1",
            "runtime_backend_alias": "compiled_scan",
            "strict_reference_backend_id": "reference_python",
        },
    }
    rows, failures = _TOOL._run_g4_candidate_lane(
        repo=tmp_path,
        tool=tmp_path / "tool.py",
        replay_dir=replay_dir,
        candidate={
            "id": "d512_ctx4096",
            "context_length": 4096,
            "d_model": 512,
            "n_layers": 8,
            "n_heads": 8,
            "vocab_size": 16384,
            "ffn_multiplier": 4.0,
            "local_attention_window": 256,
            "recurrent_layers": [0, 1, 2, 4, 5, 6],
            "local_attention_layers": [3, 7],
        },
        layout=_layout(),
        device="0",
        hardware={"gpu_uuid": "GPU-test", "name": "test-gpu"},
        common_spec=common,
        tolerance=NumericalTolerances(relative_l2_max=7.0e-3, max_abs_max=None),
    )
    assert failures == []
    assert len(rows) == 2
    assert calls == ["--worker-checkpoint", *(["--worker-replay"] * 4)]
    checkpoint_shas = {row["checkpoint_sha256"] for row in rows}
    assert len(checkpoint_shas) == 1


def test_parent_exception_after_namespace_is_terminal_not_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "catena_e26_stage3d_parent_failure"

    def fail_after_namespace(_args: argparse.Namespace) -> int:
        output.mkdir()
        write_json_strict(output / "protocol_lock.json", {"locked": True})
        write_json_strict(output / "layout_manifest.json", {"locked": True})
        raise RuntimeError("future exploded")

    monkeypatch.setattr(_TOOL, "_main_parent_run", fail_after_namespace)
    code = _TOOL._main_parent(argparse.Namespace(output_root=output))
    assert code == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert report["disposition"] == _TOOL.STAGE3D_NOT_EVALUABLE
    assert status["disposition"] == _TOOL.STAGE3D_NOT_EVALUABLE
    assert (output / "unexpected_execution_error.json").is_file()
    summary_text = (output / "RESULTS_SUMMARY_KO.md").read_text(encoding="utf-8")
    assert len(summary_text.splitlines()) <= 60
    for gate in range(7):
        assert f"G{gate}: NOT_EVALUABLE" in summary_text
    assert "KNOWN_BF16_AND_OPTIMIZER_LAYOUT_SENSITIVITY" in summary_text
    assert "resource preflight started: `false`" in summary_text
    assert "Scientific E26a started: `false`" in summary_text
    assert "허용 claim:" in summary_text
    assert "금지 claim:" in summary_text
    audit = json.loads((output / "artifact_audit.json").read_text(encoding="utf-8"))
    assert "RESULTS_SUMMARY_KO.md" in {row["path"] for row in audit["files"]}


def test_admission_failure_does_not_create_run_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "catena_e26_stage3d_not_admitted"

    def fail_before_namespace(_args: argparse.Namespace) -> int:
        raise ValueError("G0/G2 admission failed")

    monkeypatch.setattr(_TOOL, "_main_parent_run", fail_before_namespace)
    with pytest.raises(ValueError, match="admission failed"):
        _TOOL._main_parent(argparse.Namespace(output_root=output))
    assert not output.exists()


def test_parent_exception_preserves_existing_terminal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "catena_e26_stage3d_preserve_terminal"
    original_report = {"disposition": "PREEXISTING_REPORT", "gate_summary": {}}
    original_status = {"disposition": "PREEXISTING_STATUS"}

    def fail_after_partial_terminal(_args: argparse.Namespace) -> int:
        output.mkdir()
        write_json_strict(output / "protocol_lock.json", {"locked": True})
        write_json_strict(output / "layout_manifest.json", {"locked": True})
        write_json_strict(output / "report.json", original_report)
        write_json_strict(output / "status.json", original_status)
        raise RuntimeError("post-report failure")

    monkeypatch.setattr(_TOOL, "_main_parent_run", fail_after_partial_terminal)
    assert _TOOL._main_parent(argparse.Namespace(output_root=output)) == 2
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == original_report
    assert json.loads((output / "status.json").read_text(encoding="utf-8")) == original_status
    assert (output / "unexpected_execution_error.json").is_file()
    assert (output / "artifact_audit.json").is_file()
