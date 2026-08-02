#!/usr/bin/env python3
"""Run the prospective E26 Stage-3D fixed-layout numerical preflight.

Stage-3D is deliberately not a repair or a reinterpretation of Stage-3C.  It
keeps the failed counterfactual-layout audit immutable and asks the narrower
question needed by E26: whether one preregistered physical BF16 layout is
stable, matched across variants, and free from optimized-backend fallbacks.

The parent process is fail-closed.  Every G3 case and every G4 replay is run in
a fresh, single-GPU subprocess.  G4 is never opened when any G3 case fails and
this tool never launches Scientific E26a.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.audit_contract import (
    e26_execution_source_inventory,
    validate_e26_audit_locked_hashes,
)
from catena.lm.backend_lock import (
    cuda_hardware_inventory,
    observed_single_visible_cuda_device,
)
from catena.lm.checkpointing import (
    RNGSnapshot,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from catena.lm.frozen_invariance import (
    validate_historical_frozen_invariance_receipt,
)
from catena.lm.hashing import (
    optimizer_state_signature,
    parameter_signature_hash,
    state_dict_digest,
    tensor_tree_digest,
)
from catena.lm.model import CatenaLM, RuntimeState, cross_entropy_loss
from catena.lm.numerical_audit import (
    NumericalTolerances,
    _gradient_errors,
    _state_tree_tensor_errors,
    _tensor_error,
    runtime_state_error,
)
from catena.lm.preflight_audit import model_config_for_candidate
from catena.lm.recurrent_mixer import (
    optimized_backend_diagnostics,
    optimized_backend_metadata,
    reset_optimized_backend_diagnostics,
)
from catena.lm.stage3d_fixed_layout import (
    KNOWN_LAYOUT_SENSITIVITY,
    REGISTERED_COMPILED_BACKEND_ID,
    RUNTIME_COMPILED_BACKEND_ALIAS,
    STAGE3D_BLOCKED,
    STAGE3D_GO,
    STAGE3D_NOT_EVALUABLE,
    STRICT_REFERENCE_BACKEND_ID,
    build_stage3d_admissibility_receipt,
    build_stage3d_protocol_lock,
    derive_stage3c_fp32_reference_binding,
    fixed_layouts_from_config,
    load_stage3d_config,
    validate_stage3d_admissibility_receipt,
    validate_stage3d_protocol_lock,
)
from catena.lm.trainer import make_optimizer, optimizer_step_microbatches

_VARIANTS = ("projected_tied_delta_lm", "dual_delta_lm")
_STATE_CONTEXTS = ("zero_state", "prefilled_state")
_REFERENCE_BACKEND = STRICT_REFERENCE_BACKEND_ID
_OPTIMIZED_BACKEND = RUNTIME_COMPILED_BACKEND_ALIAS


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _canonical_hashed(payload: Mapping[str, Any], *, field: str) -> bool:
    observed = payload.get(field)
    if not isinstance(observed, str):
        return False
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return bool(observed == sha256_canonical_json(unhashed))


def _validate_worker_spec_common(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not _canonical_hashed(spec, field="spec_sha256"):
        raise ValueError("Stage-3D worker spec canonical hash mismatch")
    repo = Path(str(spec["repo_root"])).expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Stage-3D worker requires a clean committed source tree")
    if _git(repo, "rev-parse", "HEAD") != spec.get("source_commit"):
        raise ValueError("Stage-3D worker source commit changed")
    inventory = e26_execution_source_inventory(repo)
    if inventory.get("source_tree_sha256") != spec.get("source_tree_sha256"):
        raise ValueError("Stage-3D worker source inventory changed")
    protocol_path = Path(str(spec["protocol_lock_path"])).resolve(strict=True)
    if sha256_file(protocol_path) != spec.get("protocol_lock_file_sha256"):
        raise ValueError("Stage-3D worker protocol-lock bytes changed")
    protocol = validate_stage3d_protocol_lock(read_json_object_strict(protocol_path))
    if protocol.get("protocol_sha256") != spec.get("protocol_sha256"):
        raise ValueError("Stage-3D worker protocol canonical hash changed")
    if spec.get("backend_binding") != protocol.get("backend_binding"):
        raise ValueError("Stage-3D worker registered/runtime backend binding changed")
    layout_manifest_path = Path(str(spec["layout_manifest_path"])).resolve(strict=True)
    if sha256_file(layout_manifest_path) != spec.get("layout_manifest_file_sha256"):
        raise ValueError("Stage-3D fixed-layout manifest bytes changed")
    layout_manifest = read_json_object_strict(layout_manifest_path)
    if not _canonical_hashed(layout_manifest, field="manifest_sha256") or layout_manifest.get(
        "manifest_sha256"
    ) != spec.get("layout_manifest_sha256"):
        raise ValueError("Stage-3D fixed-layout manifest canonical hash changed")
    input_paths = spec.get("input_paths")
    input_hashes = spec.get("input_hashes")
    if not isinstance(input_paths, Mapping) or not isinstance(input_hashes, Mapping):
        raise ValueError("Stage-3D worker lacks input path/hash bindings")
    observed = {
        f"{name}_sha256": sha256_file(Path(str(path)).resolve(strict=True))
        for name, path in input_paths.items()
    }
    if observed != dict(input_hashes):
        raise ValueError("Stage-3D worker input hashes changed")
    return protocol


def _fresh_output_root(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(f"Stage-3D output root must be fresh: {unresolved}")
    parent = unresolved.parent.resolve(strict=True)
    resolved = parent / unresolved.name
    below_tmp = parent == Path("/tmp") or Path("/tmp") in parent.parents
    durable_parent = Path(
        "/data/minjun_dev/CATENA/artifacts/e26_stage3d_fixed_layout_bf16_admissibility"
    )
    is_durable = parent == durable_parent
    if not below_tmp and not is_durable:
        raise ValueError("Stage-3D output must be below /tmp or the canonical artifact namespace")
    if below_tmp and not unresolved.name.startswith("catena_e26_stage3d_"):
        raise ValueError("Temporary Stage-3D output must start with catena_e26_stage3d_")
    if is_durable and re.fullmatch(r"\d{8}T\d{6}(?:\.\d{6})?Z", unresolved.name) is None:
        raise ValueError("Durable Stage-3D output directory must be a UTC run id")
    resolved.mkdir(mode=0o700)
    return resolved


def _write_hashed_json(path: Path, payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized[field] = sha256_canonical_json(normalized)
    write_json_strict(path, normalized)
    return normalized


def _yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a mapping: {path}")
    return payload


def _resolve_stage3c_execution_paths(
    *, protocol_path: Path, execution_paths: Mapping[str, Any]
) -> dict[str, str]:
    """Resolve every Stage-3C input, including lock-bundle-relative bindings."""

    resolved: dict[str, str] = {}
    for name, raw in execution_paths.items():
        if not isinstance(raw, str):
            raise ValueError(f"Stage-3C execution input {name} is not a path")
        if raw.startswith("BUNDLE_RELATIVE:"):
            relative = raw.removeprefix("BUNDLE_RELATIVE:")
            path = (protocol_path.parent / relative).resolve(strict=True)
        else:
            path = Path(raw).expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Stage-3C execution input is not a regular file: {name}")
        resolved[str(name)] = str(path)
    return resolved


def _checkpoint_locked_hashes(
    *,
    stage3c_artifact_root: Path,
    source_inventory: Mapping[str, Any],
    stage3c_execution_paths: Mapping[str, str],
) -> dict[str, str]:
    """Bind G4 checkpoints to the exact Stage-3C E26 inputs and current source."""

    source_lock = read_json_object_strict(stage3c_artifact_root / "source_lock.json")
    raw = source_lock.get("locked_hashes")
    if not isinstance(raw, Mapping):
        raise ValueError("Stage-3C source lock lacks E26 locked hashes")
    locked = dict(raw)
    locked["source_tree_sha256"] = str(source_inventory["source_tree_sha256"])
    normalized = validate_e26_audit_locked_hashes(locked)
    for name, path in stage3c_execution_paths.items():
        key = f"{name}_sha256"
        if key in normalized and sha256_file(path) != normalized[key]:
            raise ValueError(f"Stage-3C checkpoint input hash changed: {name}")
    return normalized


def _row_aggregate(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def _verify_stage3c_artifact_manifest(
    *,
    manifest_path: Path,
    artifact_root: Path,
    expected_predecessor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rehash every Stage-3C artifact row, not merely the manifest bytes."""

    manifest = read_json_object_strict(manifest_path)
    if not _canonical_hashed(manifest, field="manifest_sha256"):
        raise ValueError("Stage-3C artifact manifest canonical hash mismatch")
    if manifest.get("file_count") != 11:
        raise ValueError("Stage-3C artifact manifest must bind exactly eleven raw files")
    registered = manifest.get("registered_predecessor")
    if not isinstance(registered, Mapping):
        raise ValueError("Stage-3C artifact manifest lacks registered predecessor anchors")
    if expected_predecessor is not None:
        expected = {
            "result_sha256": expected_predecessor.get("result_sha256"),
            "status_sha256": expected_predecessor.get("status_sha256"),
            "raw_aggregate_sha256": expected_predecessor.get("raw_registered_aggregate_sha256"),
            "failure_status_sha256": expected_predecessor.get("failure_status_sha256"),
            "disposition": expected_predecessor.get("required_disposition"),
        }
        observed = {
            "result_sha256": (
                registered.get("result", {}).get("sha256")
                if isinstance(registered.get("result"), Mapping)
                else None
            ),
            "status_sha256": (
                registered.get("status", {}).get("sha256")
                if isinstance(registered.get("status"), Mapping)
                else None
            ),
            "raw_aggregate_sha256": registered.get("raw_run_aggregate_sha256"),
            "failure_status_sha256": registered.get("failure_status_sha256"),
            "disposition": registered.get("disposition"),
        }
        if observed != expected:
            raise ValueError("Stage-3C registered predecessor anchors changed")
    rows_raw = manifest.get("files")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("Stage-3C artifact manifest must contain non-empty files rows")
    root = artifact_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Stage-3C artifact root must be a real directory")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Stage-3C artifact row {index} is not a mapping")
        relative = str(raw.get("path", ""))
        if not relative or relative in seen or Path(relative).is_absolute():
            raise ValueError("Stage-3C artifact paths must be unique relative paths")
        seen.add(relative)
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"Stage-3C artifact row escapes/is not regular: {relative}")
        row = {
            "path": relative,
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        if row != dict(raw):
            raise ValueError(f"Stage-3C artifact changed: {relative}")
        rows.append(row)
    if manifest.get("file_count") != len(rows):
        raise ValueError("Stage-3C artifact manifest file_count mismatch")
    aggregate = _row_aggregate(rows)
    if manifest.get("aggregate_sha256") != aggregate:
        raise ValueError("Stage-3C artifact aggregate SHA-256 mismatch")
    failure_rows = [row for row in rows if row["path"] == "failure_status.json"]
    if len(failure_rows) != 1 or failure_rows[0]["sha256"] != registered.get(
        "failure_status_sha256"
    ):
        raise ValueError("Stage-3C failure status artifact binding changed")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_root": str(root),
        "file_count": len(rows),
        "aggregate_sha256": aggregate,
        "passed": True,
    }


def _verify_g0_frozen_inputs(
    *,
    stage3c_protocol_path: Path,
    stage3c_artifact_manifest_path: Path,
    stage3c_artifact_root: Path,
    frozen_receipt_path: Path,
    expected_predecessor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = _yaml_mapping(stage3c_protocol_path, label="Stage-3C protocol")
    execution_paths = protocol.get("execution_input_paths")
    if not isinstance(execution_paths, Mapping):
        raise ValueError("Stage-3C protocol lacks execution_input_paths")
    execution_inputs = protocol.get("execution_inputs")
    if not isinstance(execution_inputs, Mapping):
        raise ValueError("Stage-3C protocol lacks execution_inputs")
    data_lock_raw = execution_paths.get("data_lock")
    if not isinstance(data_lock_raw, str):
        raise ValueError("Stage-3C protocol lacks its bound data_lock path")
    data_lock_path = Path(data_lock_raw).expanduser().resolve(strict=True)
    registered_data_lock_sha256 = execution_inputs.get("data_lock_sha256")
    if (
        not isinstance(registered_data_lock_sha256, str)
        or sha256_file(data_lock_path) != registered_data_lock_sha256
    ):
        raise ValueError("Stage-3C bound data_lock SHA-256 changed")
    frozen_receipt_raw = execution_paths.get("frozen_tree_receipt")
    if not isinstance(frozen_receipt_raw, str):
        raise ValueError("Stage-3C protocol lacks its bound frozen receipt path")
    registered_frozen_path = Path(frozen_receipt_raw).expanduser().resolve(strict=True)
    if frozen_receipt_path != registered_frozen_path:
        raise ValueError("Supplied frozen receipt differs from the Stage-3C binding")
    registered_frozen_sha256 = execution_inputs.get("frozen_tree_receipt_sha256")
    historical_file_sha256 = sha256_file(frozen_receipt_path)
    if (
        not isinstance(registered_frozen_sha256, str)
        or historical_file_sha256 != registered_frozen_sha256
    ):
        raise ValueError("Stage-3C bound frozen receipt SHA-256 changed")
    data_lock = read_json_object_strict(data_lock_path)
    frozen = read_json_object_strict(frozen_receipt_path)
    refreshed = validate_historical_frozen_invariance_receipt(
        frozen,
        data_lock=data_lock,
    )
    artifact_audit = _verify_stage3c_artifact_manifest(
        manifest_path=stage3c_artifact_manifest_path,
        artifact_root=stage3c_artifact_root,
        expected_predecessor=expected_predecessor,
    )
    return {
        "stage3c_artifacts": artifact_audit,
        "e00_e25_frozen_receipt_path": str(frozen_receipt_path),
        "e00_e25_frozen_receipt_sha256": historical_file_sha256,
        "stage3c_registered_frozen_receipt_sha256": registered_frozen_sha256,
        "stage3c_registered_data_lock_sha256": registered_data_lock_sha256,
        "historical_observed_head": frozen["live_repository"]["observed_head"],
        "live_observed_head": refreshed["live_repository"]["observed_head"],
        "dynamic_head_change_allowed": (
            frozen["live_repository"]["observed_head"]
            != refreshed["live_repository"]["observed_head"]
        ),
        "e00_e25_live_reaudit_receipt_sha256": refreshed["receipt_sha256"],
        "e00_e25_live_reaudit_passed": refreshed["passed"] is True,
        "passed": artifact_audit["passed"] is True and refreshed["passed"] is True,
    }


def _error_dict(error: Any) -> dict[str, float]:
    return {
        "relative_l2": float(error.relative_l2),
        "max_abs": float(error.max_abs),
    }


def _combined_runtime_state_error(observed: RuntimeState, expected: RuntimeState) -> dict[str, Any]:
    error = runtime_state_error(observed, expected)
    floating = [
        error.recurrent,
        error.attention_key,
        error.attention_value,
    ]
    aggregate = {
        "relative_l2": max(value.relative_l2 for value in floating),
        "max_abs": max(value.max_abs for value in floating),
    }
    metadata_exact = (
        error.positions_equal
        and error.lengths_equal
        and error.write_indices_equal
        and error.position_equal
    )
    return {
        **aggregate,
        "metadata_exact": metadata_exact,
        "components": {
            "recurrent": _error_dict(error.recurrent),
            "attention_key": _error_dict(error.attention_key),
            "attention_value": _error_dict(error.attention_value),
        },
    }


def _passes_error(error: Mapping[str, Any], tolerance: NumericalTolerances) -> bool:
    return float(error["relative_l2"]) <= tolerance.relative_l2_max and (
        tolerance.max_abs_max is None or float(error["max_abs"]) <= tolerance.max_abs_max
    )


def _comparison(
    *,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: NumericalTolerances,
) -> dict[str, Any]:
    logits = _error_dict(_tensor_error(observed["logits"], expected["logits"]))
    runtime_detail = _combined_runtime_state_error(
        cast(RuntimeState, observed["runtime_state"]),
        cast(RuntimeState, expected["runtime_state"]),
    )
    runtime_state = {
        "relative_l2": runtime_detail["relative_l2"],
        "max_abs": runtime_detail["max_abs"],
    }
    gradient, gradient_worst = _gradient_errors(
        cast(Mapping[str, torch.Tensor], observed["gradients"]),
        cast(Mapping[str, torch.Tensor], expected["gradients"]),
    )
    gradients = _error_dict(gradient)
    passed = (
        _passes_error(logits, tolerance)
        and _passes_error(runtime_state, tolerance)
        and bool(runtime_detail["metadata_exact"])
        and _passes_error(gradients, tolerance)
        and bool(observed["gradients_finite"])
        and bool(expected["gradients_finite"])
    )
    return {
        "logits": logits,
        "runtime_state": runtime_state,
        "state_metadata_exact": runtime_detail["metadata_exact"],
        "runtime_state_components": runtime_detail["components"],
        "gradients": gradients,
        "gradients_worst_leaf": _error_dict(gradient_worst),
        "tolerance": {
            "relative_l2_max": tolerance.relative_l2_max,
            "max_abs_max": tolerance.max_abs_max,
        },
        "passed": passed,
    }


def _runtime_state_to_cpu(state: RuntimeState) -> RuntimeState:
    # RuntimeState intentionally has no broad ``to`` method because its
    # metadata are integral.  Convert its typed children explicitly.
    return RuntimeState(
        recurrent=[value.to(device="cpu") for value in state.recurrent],
        attention=[value.to(device="cpu") for value in state.attention],
        position=state.position,
    )


def _set_reproducible_seed(seed: int, *, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _run_model_path(
    *,
    candidate: Mapping[str, Any],
    variant: str,
    backend_id: str,
    input_ids: torch.Tensor,
    prefix_ids: torch.Tensor | None,
    seed: int,
    autocast_dtype: torch.dtype | None,
    capture_backend: bool,
) -> dict[str, Any]:
    device = input_ids.device
    _set_reproducible_seed(seed, device=device)
    config = model_config_for_candidate(candidate, variant=variant)
    mapping = config.to_dict()
    mapping["backend_id"] = backend_id
    mapping["backend_scientific_main_capable"] = False
    config = type(config).from_mapping(mapping)
    if backend_id == _REFERENCE_BACKEND and config.backend_id != _REFERENCE_BACKEND:
        raise RuntimeError("G3 reference path did not instantiate reference_python")
    if backend_id == _OPTIMIZED_BACKEND and config.backend_id != _OPTIMIZED_BACKEND:
        raise RuntimeError("G3 optimized path did not instantiate compiled_scan")
    model = CatenaLM(config).to(device)
    optimizer = make_optimizer(model)
    initialization_digest = state_dict_digest(model)
    parameter_signature = parameter_signature_hash(model)
    optimizer_signature = optimizer_state_signature(optimizer)
    reset_optimized_backend_diagnostics()
    context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    initial_state: RuntimeState | None = None
    if prefix_ids is not None:
        model.eval()
        with torch.no_grad(), context:
            initial_state = model(prefix_ids).runtime_state.clone(detach=True)
    initial_clone_no_alias = True
    if initial_state is not None:
        initial_clone = initial_state.clone(detach=True)
        initial_clone_no_alias = not bool(
            set(initial_state.storage_ptrs()) & set(initial_clone.storage_ptrs())
        )
    model.zero_grad(set_to_none=True)
    model.train()
    with context:
        output = model(input_ids, initial_state)
        loss = cross_entropy_loss(output.logits, input_ids)
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    gradients_finite = len(gradients) == sum(1 for _ in model.parameters()) and all(
        bool(torch.isfinite(value).all().item()) for value in gradients.values()
    )
    gradient_norm = float(
        torch.sqrt(
            torch.stack([value.float().square().sum() for value in gradients.values()]).sum()
        ).item()
    )
    output_clone = output.runtime_state.clone(detach=True)
    output_clone_no_alias = not bool(
        set(output.runtime_state.storage_ptrs()) & set(output_clone.storage_ptrs())
    )
    diagnostics = optimized_backend_diagnostics()
    result = {
        "backend_id": config.backend_id,
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
        "initialization_digest": initialization_digest,
        "parameter_signature_sha256": parameter_signature,
        "optimizer_state_signature_sha256": optimizer_signature,
        "optimized_chunk_size": config.optimized_chunk_size,
        "loss": float(loss.detach().float().item()),
        "logits": output.logits.detach().cpu().clone(),
        "runtime_state": _runtime_state_to_cpu(output.runtime_state),
        "gradients": gradients,
        "gradients_finite": gradients_finite,
        "gradient_norm": gradient_norm,
        "state_metadata": {
            "position": output.runtime_state.position,
            "attention_lengths": [value.length for value in output.runtime_state.attention],
            "attention_write_indices": [
                value.write_index for value in output.runtime_state.attention
            ],
        },
        "clone_no_alias": initial_clone_no_alias and output_clone_no_alias,
        "backend_diagnostics": diagnostics if capture_backend else None,
    }
    del optimizer, model, output, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _fixed_layout_identity(
    *,
    candidate: Mapping[str, Any],
    layout: Mapping[str, Any],
    paths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context_length = int(candidate["context_length"])
    microbatch = int(layout["microbatch_sequences"])
    accumulation = int(layout["accumulation_steps"])
    target_tokens = int(layout["target_global_input_tokens"])
    initialization = {str(row["initialization_digest"]) for row in paths}
    parameters = {str(row["parameter_signature_sha256"]) for row in paths}
    optimizers = {str(row["optimizer_state_signature_sha256"]) for row in paths}
    return {
        "physical_microbatch_sequences": microbatch,
        "sequence_length": context_length,
        "accumulation_steps": accumulation,
        "target_global_input_tokens": target_tokens,
        "loss_denominator": "TOTAL_VALID_NEXT_TOKEN_PREDICTIONS_ACROSS_FIXED_LAYOUT",
        "optimizer_update_boundary": "AFTER_EXACT_ACCUMULATION_STEPS",
        "autocast_scope": "CUDA_BF16_FORWARD_LOSS_ONLY",
        "gradient_clipping_order": "AFTER_ACCUMULATION_BEFORE_ADAMW",
        "initialization_matched": len(initialization) == 1,
        "parameter_surface_matched": len(parameters) == 1,
        "optimizer_state_shape_matched": len(optimizers) == 1,
        "shape_contract_valid": (
            microbatch == 1 and target_tokens == microbatch * context_length * accumulation
        ),
        "passed": (
            len(initialization) == 1
            and len(parameters) == 1
            and len(optimizers) == 1
            and microbatch == 1
            and target_tokens == microbatch * context_length * accumulation
        ),
    }


def _backend_integrity_from_diagnostics(
    *,
    optimized: Mapping[str, Any],
    reference_backend_id: str,
    variant_specific_fp32_path_count: int = 0,
    variant_specific_padding_count: int = 0,
) -> dict[str, Any]:
    diagnostics = optimized.get("backend_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    static = optimized.get("backend_metadata")
    if not isinstance(static, Mapping):
        static = {}
    graph_breaks = int(diagnostics.get("graph_break_count", -1))
    fallbacks = int(diagnostics.get("fallback_count", -1))
    positive_execution = all(
        isinstance(diagnostics.get(field), int)
        and not isinstance(diagnostics.get(field), bool)
        and int(diagnostics[field]) > 0
        for field in ("graph_invocations", "optimized_calls", "chunks_executed")
    )
    strict_reference = reference_backend_id == _REFERENCE_BACKEND
    backend_alias_matches_registration = optimized.get("backend_id") == _OPTIMIZED_BACKEND
    no_python_scientific_path = static.get("python_token_loop_at_runtime") is False
    precision_policy_sha256 = sha256_canonical_json(
        {
            "autocast_dtype": optimized.get("autocast_dtype"),
            "accumulation_policy": static.get("accumulation_policy"),
            "backend_id": optimized.get("backend_id"),
        }
    )
    passed = (
        graph_breaks == 0
        and fallbacks == 0
        and positive_execution
        and strict_reference
        and backend_alias_matches_registration
        and no_python_scientific_path
        and variant_specific_fp32_path_count == 0
        and variant_specific_padding_count == 0
    )
    return {
        "optimized_backend_id": optimized.get("backend_id"),
        "registered_backend_id": REGISTERED_COMPILED_BACKEND_ID,
        "runtime_backend_alias": RUNTIME_COMPILED_BACKEND_ALIAS,
        "backend_alias_matches_registration": backend_alias_matches_registration,
        "reference_backend_id": reference_backend_id,
        "strict_reference_python": strict_reference,
        "positive_compiled_execution": positive_execution,
        "python_token_loop_at_scientific_runtime": not no_python_scientific_path,
        "graph_break_count": graph_breaks,
        "fallback_count": fallbacks,
        "variant_specific_fp32_path_count": int(variant_specific_fp32_path_count),
        "variant_specific_padding_count": int(variant_specific_padding_count),
        "observed_padded_tokens": diagnostics.get("padded_tokens"),
        "precision_policy_sha256": precision_policy_sha256,
        "passed": passed,
    }


def _g3_worker(args: argparse.Namespace) -> int:
    spec = read_json_object_strict(Path(args.worker_spec).resolve(strict=True))
    _validate_worker_spec_common(spec)
    output_path = Path(args.worker_output).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite G3 output: {output_path}")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    device = torch.device("cuda:0")
    candidate = cast(Mapping[str, Any], spec["candidate"])
    variant = str(spec["variant"])
    state_context = str(spec["state_context"])
    layout = cast(Mapping[str, Any], spec["fixed_layout"])
    seed = int(spec["initialization_seed"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec["data_seed"]))
    prefix_generator = torch.Generator(device="cpu")
    prefix_generator.manual_seed(int(spec["prefill_seed"]))
    context_length = int(layout["context_length"])
    input_cpu = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (int(layout["microbatch_sequences"]), context_length),
        generator=generator,
    )
    prefix_cpu = torch.randint(
        0,
        int(candidate["vocab_size"]),
        (int(layout["microbatch_sequences"]), int(spec["prefix_length"])),
        generator=prefix_generator,
    )
    input_ids = input_cpu.to(device)
    prefix_ids = None if state_context == "zero_state" else prefix_cpu.to(device)
    if state_context not in _STATE_CONTEXTS:
        raise ValueError(f"Unknown G3 state context: {state_context}")

    optimized_bf16 = _run_model_path(
        candidate=candidate,
        variant=variant,
        backend_id=_OPTIMIZED_BACKEND,
        input_ids=input_ids,
        prefix_ids=prefix_ids,
        seed=seed,
        autocast_dtype=torch.bfloat16,
        capture_backend=True,
    )
    optimized_bf16["backend_metadata"] = optimized_backend_metadata(
        device=device,
        compiler="inductor",
        chunk_size=int(optimized_bf16["optimized_chunk_size"]),
        parity_verified=False,
    )
    reference_bf16 = _run_model_path(
        candidate=candidate,
        variant=variant,
        backend_id=_REFERENCE_BACKEND,
        input_ids=input_ids,
        prefix_ids=prefix_ids,
        seed=seed,
        autocast_dtype=torch.bfloat16,
        capture_backend=False,
    )
    reference_fp32 = _run_model_path(
        candidate=candidate,
        variant=variant,
        backend_id=_REFERENCE_BACKEND,
        input_ids=input_ids,
        prefix_ids=prefix_ids,
        seed=seed,
        autocast_dtype=None,
        capture_backend=False,
    )
    bf16_tolerance = NumericalTolerances(
        relative_l2_max=float(spec["bf16_relative_l2_max"]),
        max_abs_max=None,
    )
    compiled_vs_reference = _comparison(
        observed=optimized_bf16,
        expected=reference_bf16,
        tolerance=bf16_tolerance,
    )
    bf16_vs_fp32 = _comparison(
        observed=reference_bf16,
        expected=reference_fp32,
        tolerance=bf16_tolerance,
    )
    layout_identity = _fixed_layout_identity(
        candidate=candidate,
        layout=layout,
        paths=(optimized_bf16, reference_bf16, reference_fp32),
    )
    backend = _backend_integrity_from_diagnostics(
        optimized=optimized_bf16,
        reference_backend_id=str(reference_bf16["backend_id"]),
    )
    state_metadata_exact = bool(
        compiled_vs_reference["state_metadata_exact"] and bf16_vs_fp32["state_metadata_exact"]
    )
    gradient_finite = all(
        bool(row["gradients_finite"]) for row in (optimized_bf16, reference_bf16, reference_fp32)
    )
    gradient_norms = {
        "compiled_bf16": float(optimized_bf16["gradient_norm"]),
        "reference_python_bf16": float(reference_bf16["gradient_norm"]),
        "reference_python_fp32": float(reference_fp32["gradient_norm"]),
    }
    gradient_norm_in_range = all(
        float(spec["gradient_norm_min"]) <= value <= float(spec["gradient_norm_max"])
        for value in gradient_norms.values()
    )
    clone_no_alias = all(
        bool(row["clone_no_alias"]) for row in (optimized_bf16, reference_bf16, reference_fp32)
    )
    passed = (
        bool(layout_identity["passed"])
        and bool(compiled_vs_reference["passed"])
        and bool(bf16_vs_fp32["passed"])
        and gradient_finite
        and gradient_norm_in_range
        and state_metadata_exact
        and clone_no_alias
        and bool(backend["passed"])
    )
    row = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_G3_CASE",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "candidate_id": str(candidate["id"]),
        "variant": variant,
        "state_context": state_context,
        "fixed_layout": dict(layout),
        "initialization_digest": optimized_bf16["initialization_digest"],
        "parameter_signature_sha256": optimized_bf16["parameter_signature_sha256"],
        "optimizer_state_signature_sha256": optimized_bf16["optimizer_state_signature_sha256"],
        "token_ids_sha256": tensor_tree_digest({"input_ids": input_cpu, "prefill_ids": prefix_cpu}),
        "data_cursor_sha256": sha256_canonical_json(
            {
                "seed": int(spec["data_seed"]),
                "candidate_id": str(candidate["id"]),
                "state_context": state_context,
            }
        ),
        "layout_identity": layout_identity,
        "layout_identity_passed": bool(layout_identity["passed"]),
        "comparisons": {
            "compiled_bf16_vs_reference_python_bf16": compiled_vs_reference,
            "reference_python_bf16_vs_reference_python_fp32": bf16_vs_fp32,
        },
        "gradient_finite": gradient_finite,
        "gradient_norms": gradient_norms,
        "gradient_norm_in_range": gradient_norm_in_range,
        "state_metadata_exact": state_metadata_exact,
        "clone_no_alias": clone_no_alias,
        "graph_break_count": backend["graph_break_count"],
        "fallback_count": backend["fallback_count"],
        "variant_specific_fp32_path_count": 0,
        "variant_specific_padding_count": 0,
        "backend_integrity": backend,
        "execution_device": execution_device,
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "passed": passed,
    }
    _write_hashed_json(output_path, row, field="receipt_sha256")
    # A completed, hash-valid numerical row is successful execution even when
    # its threshold disposition is negative.  Only operational exceptions may
    # produce a non-zero worker exit code.
    return 0


def _build_backend_recipe(
    *,
    candidate: Mapping[str, Any],
    layout: Mapping[str, Any],
    hardware: Mapping[str, Any],
    protocol_sha256: str,
    source_tree_sha256: str,
    backend_binding: Mapping[str, Any],
) -> dict[str, Any]:
    config = model_config_for_candidate(candidate, variant=_VARIANTS[0])
    return {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_G4_CHECKPOINT_BACKEND_RECIPE",
        "backend_id": _OPTIMIZED_BACKEND,
        "registered_backend_id": str(backend_binding["registered_backend_id"]),
        "runtime_backend_alias": str(backend_binding["runtime_backend_alias"]),
        "strict_reference_backend_id": str(backend_binding["strict_reference_backend_id"]),
        "compiler": "inductor",
        "autocast_precision": "torch.bfloat16",
        "candidate_id": str(candidate["id"]),
        "optimized_chunk_size": int(config.optimized_chunk_size),
        "fixed_layout": dict(layout),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_uuid": hardware.get("gpu_uuid"),
        "gpu_name": hardware.get("name"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "protocol_sha256": protocol_sha256,
        "source_tree_sha256": source_tree_sha256,
    }


def _checkpoint_backend_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = spec.get("backend_recipe")
    if not isinstance(raw, Mapping):
        raise ValueError("G4 worker spec lacks backend recipe")
    return dict(raw)


def _g4_checkpoint_worker(args: argparse.Namespace) -> int:
    """Create one variant-neutral initial checkpoint per candidate in a fresh process."""

    spec = read_json_object_strict(Path(args.worker_spec).resolve(strict=True))
    _validate_worker_spec_common(spec)
    output_path = Path(args.worker_output).expanduser()
    checkpoint_path = Path(args.worker_checkpoint_output).expanduser()
    if any(path.exists() or path.is_symlink() for path in (output_path, checkpoint_path)):
        raise FileExistsError("Refusing to overwrite G4 checkpoint output")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    device = torch.device("cuda:0")
    candidate = cast(Mapping[str, Any], spec["candidate"])
    _set_reproducible_seed(int(spec["initialization_seed"]), device=device)
    # The parameter surface is identical by construction.  The common bytes
    # are created once from the projected-tied configuration, then strictly
    # loaded into both configurations in their independent replay processes.
    config = model_config_for_candidate(candidate, variant=_VARIANTS[0])
    if config.backend_id != _OPTIMIZED_BACKEND:
        raise RuntimeError("G4 common checkpoint requires compiled_scan")
    model = CatenaLM(config).to(device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    progress = TrainingProgress(
        optimizer_step=0,
        tokens_seen=0,
        general_sequences_seen=0,
        transaction_sequences_seen=0,
        document_index=0,
        episode_index=0,
        cursor_snapshot={
            "snapshot_sha256": sha256_canonical_json(
                {
                    "data_seed": int(spec["data_seed"]),
                    "candidate_id": str(candidate["id"]),
                    "next_microbatch_index": 0,
                }
            ),
            "data_seed": int(spec["data_seed"]),
            "next_microbatch_index": 0,
        },
        last_source_type=None,
    )
    backend_manifest = _checkpoint_backend_manifest(spec)
    rng_snapshot = RNGSnapshot.capture()
    receipt = save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        locked_hashes=cast(Mapping[str, str], spec["checkpoint_locked_hashes"]),
        amp_policy={"dtype": "bfloat16", "grad_scaler": None},
        backend_manifest=backend_manifest,
        rng_snapshot=rng_snapshot,
    )
    row = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_G4_COMMON_CHECKPOINT",
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "candidate_id": str(candidate["id"]),
        "variant_surface_source": _VARIANTS[0],
        "fixed_layout": dict(spec["fixed_layout"]),
        "checkpoint": receipt.as_dict(),
        "checkpoint_locked_hashes": dict(spec["checkpoint_locked_hashes"]),
        "amp_policy": {"dtype": "bfloat16", "grad_scaler": None},
        "backend_manifest": backend_manifest,
        "initialization_digest": state_dict_digest(model),
        "parameter_signature_sha256": parameter_signature_hash(model),
        "optimizer_state_signature_sha256": optimizer_state_signature(optimizer),
        "rng_state_sha256": tensor_tree_digest(rng_snapshot.as_payload()),
        "data_cursor_sha256": str(progress.cursor_snapshot["snapshot_sha256"]),
        "execution_device": execution_device,
        "worker_spec_sha256": str(spec["spec_sha256"]),
    }
    _write_hashed_json(output_path, row, field="receipt_sha256")
    return 0


def _validated_checkpoint_cursor(
    cursor: Mapping[str, Any], *, candidate_id: str, data_seed: int
) -> dict[str, Any]:
    expected = {
        "data_seed": data_seed,
        "candidate_id": candidate_id,
        "next_microbatch_index": 0,
    }
    if (
        cursor.get("snapshot_sha256") != sha256_canonical_json(expected)
        or cursor.get("data_seed") != data_seed
        or cursor.get("next_microbatch_index") != 0
    ):
        raise RuntimeError("G4 checkpoint data cursor changed")
    return dict(cursor)


def _run_fixed_layout_optimizer_step(spec: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    candidate = cast(Mapping[str, Any], spec["candidate"])
    layout = cast(Mapping[str, Any], spec["fixed_layout"])
    variant = str(spec["variant"])
    seed = int(spec["initialization_seed"])
    _set_reproducible_seed(seed, device=device)
    config = model_config_for_candidate(candidate, variant=variant)
    if config.backend_id != _OPTIMIZED_BACKEND:
        raise RuntimeError("G4 replay must use the optimized fixed-layout backend")
    model = CatenaLM(config).to(device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    checkpoint = cast(Mapping[str, Any], spec["checkpoint"])
    backend_manifest = _checkpoint_backend_manifest(spec)
    loaded = load_training_checkpoint(
        str(checkpoint["path"]),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_locked_hashes=cast(Mapping[str, str], spec["checkpoint_locked_hashes"]),
        expected_file_sha256=str(checkpoint["sha256"]),
        map_location="cpu",
        restore_rng=True,
        expected_amp_policy={"dtype": "bfloat16", "grad_scaler": None},
        expected_backend_manifest=backend_manifest,
    )
    if loaded.progress.optimizer_step != 0 or loaded.progress.tokens_seen != 0:
        raise RuntimeError("G4 common checkpoint is not an initial optimizer boundary")
    cursor = _validated_checkpoint_cursor(
        loaded.progress.cursor_snapshot,
        candidate_id=str(candidate["id"]),
        data_seed=int(spec["data_seed"]),
    )
    initial_parameter_digest = state_dict_digest(model)
    parameter_signature = parameter_signature_hash(model)
    optimizer_signature = optimizer_state_signature(optimizer)
    optimizer_recipe = {
        "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_group_count": len(optimizer.param_groups),
        "parameter_groups": [
            {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in group.items()
                if key
                in {
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "amsgrad",
                    "maximize",
                    "foreach",
                    "capturable",
                    "differentiable",
                    "fused",
                }
            }
            for group in optimizer.param_groups
        ],
        "gradient_clip_norm": 1.0,
        "scheduler_class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
    }
    microbatch = int(layout["microbatch_sequences"])
    context_length = int(layout["context_length"])
    accumulation_steps = int(layout["accumulation_steps"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cursor["data_seed"]))
    batches_cpu = [
        torch.randint(
            0,
            int(candidate["vocab_size"]),
            (microbatch, context_length),
            generator=generator,
        )
        for _ in range(accumulation_steps)
    ]
    data_ids_sha256 = tensor_tree_digest(batches_cpu)
    batches = [value.to(device) for value in batches_cpu]
    rng_input_sha256 = tensor_tree_digest(loaded.rng_snapshot.as_payload())
    reset_optimized_backend_diagnostics()
    execution_events: list[str] = []
    original_zero_grad = optimizer.zero_grad
    original_optimizer_step = optimizer.step
    original_scheduler_step = scheduler.step
    original_clip = torch.nn.utils.clip_grad_norm_

    def observed_zero_grad(*call_args: Any, **call_kwargs: Any) -> Any:
        execution_events.append("zero_grad")
        return original_zero_grad(*call_args, **call_kwargs)

    def observed_clip(*call_args: Any, **call_kwargs: Any) -> Any:
        execution_events.append("gradient_clip")
        return original_clip(*call_args, **call_kwargs)

    def observed_optimizer_step(*call_args: Any, **call_kwargs: Any) -> Any:
        execution_events.append("adamw_step")
        return original_optimizer_step(*call_args, **call_kwargs)

    def observed_scheduler_step(*call_args: Any, **call_kwargs: Any) -> Any:
        execution_events.append("scheduler_step")
        return original_scheduler_step(*call_args, **call_kwargs)

    optimizer.zero_grad = observed_zero_grad  # type: ignore[method-assign]
    optimizer.step = observed_optimizer_step  # type: ignore[method-assign]
    scheduler.step = observed_scheduler_step  # type: ignore[method-assign]
    torch.nn.utils.clip_grad_norm_ = observed_clip
    try:
        step = optimizer_step_microbatches(
            model,
            batches,
            optimizer=optimizer,
            scheduler=scheduler,
            autocast_dtype=torch.bfloat16,
            capture_gradients=True,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        optimizer.zero_grad = original_zero_grad  # type: ignore[method-assign]
        optimizer.step = original_optimizer_step  # type: ignore[method-assign]
        scheduler.step = original_scheduler_step  # type: ignore[method-assign]
    diagnostics = optimized_backend_diagnostics()
    if step.gradients_before_clip is None:
        raise RuntimeError("G4 replay did not capture gradients")
    gradients_finite = all(
        bool(torch.isfinite(value).all().item()) for value in step.gradients_before_clip.values()
    )
    probe = batches[0]
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(probe)
    clone = output.runtime_state.clone(detach=True)
    clone_no_alias = not bool(set(output.runtime_state.storage_ptrs()) & set(clone.storage_ptrs()))
    adamw_steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if torch.is_tensor(value):
            adamw_steps.append(int(value.detach().cpu().item()))
        elif value is not None:
            adamw_steps.append(int(value))
    expected_valid_tokens = microbatch * accumulation_steps * (context_length - 1)
    expected_input_tokens = int(layout["target_global_input_tokens"])
    optimizer_integrity = {
        "global_token_normalization_identity": (
            step.valid_prediction_tokens == expected_valid_tokens
        ),
        "accumulation_buffer_reset_once": execution_events.count("zero_grad") == 1,
        "gradient_clipping_after_accumulation": (
            execution_events == ["zero_grad", "gradient_clip", "adamw_step", "scheduler_step"]
        ),
        "adamw_step_and_bias_correction_identity": (
            execution_events.count("adamw_step") == 1
            and bool(adamw_steps)
            and set(adamw_steps) == {1}
        ),
        "weight_decay_order_and_value_identity": (
            all(float(group["weight_decay"]) == 0.1 for group in optimizer.param_groups)
            and all(tuple(group["betas"]) == (0.9, 0.95) for group in optimizer.param_groups)
        ),
        "skipped_optimizer_steps_zero": execution_events.count("adamw_step") == 1,
        "all_gradients_finite": gradients_finite,
        "valid_prediction_tokens": step.valid_prediction_tokens,
        "expected_valid_prediction_tokens": expected_valid_tokens,
        "exposed_input_tokens": step.exposed_input_tokens,
        "expected_input_tokens": expected_input_tokens,
        "microbatch_count": step.microbatch_count,
        "expected_microbatch_count": accumulation_steps,
        "execution_events": execution_events,
        "adamw_state_steps": sorted(set(adamw_steps)),
    }
    optimizer_integrity["passed"] = (
        all(
            optimizer_integrity[field] is True
            for field in (
                "global_token_normalization_identity",
                "accumulation_buffer_reset_once",
                "gradient_clipping_after_accumulation",
                "adamw_step_and_bias_correction_identity",
                "weight_decay_order_and_value_identity",
                "skipped_optimizer_steps_zero",
                "all_gradients_finite",
            )
        )
        and step.exposed_input_tokens == expected_input_tokens
        and step.microbatch_count == accumulation_steps
    )
    return {
        "initial_parameter_digest": initial_parameter_digest,
        "parameter_signature_sha256": parameter_signature,
        "initial_optimizer_state_signature_sha256": optimizer_signature,
        "rng_input_sha256": rng_input_sha256,
        "data_ids_sha256": data_ids_sha256,
        "data_cursor_sha256": str(cursor["snapshot_sha256"]),
        "backend_input_sha256": sha256_canonical_json(
            {
                "backend_id": config.backend_id,
                "compiler": "inductor",
                "autocast": "torch.bfloat16",
                "layout": dict(layout),
            }
        ),
        "backend_recipe_sha256": sha256_canonical_json(backend_manifest),
        "optimizer_input_sha256": sha256_canonical_json(optimizer_recipe),
        "optimizer_recipe": optimizer_recipe,
        "checkpoint_input_sha256": str(checkpoint["sha256"]),
        "checkpoint_semantic_sha256": str(checkpoint["semantic_payload_sha256"]),
        "step": step.to_dict(),
        "gradients": {key: value.cpu() for key, value in step.gradients_before_clip.items()},
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": copy.deepcopy(scheduler.state_dict()),
        "logits": output.logits.detach().cpu(),
        "runtime_state": _runtime_state_to_cpu(output.runtime_state),
        "gradients_finite": gradients_finite,
        "state_metadata": {
            "position": output.runtime_state.position,
            "attention_lengths": [value.length for value in output.runtime_state.attention],
            "attention_write_indices": [
                value.write_index for value in output.runtime_state.attention
            ],
        },
        "clone_no_alias": clone_no_alias,
        "backend_diagnostics": diagnostics,
        "optimizer_step_integrity": optimizer_integrity,
        "model_sha256": state_dict_digest(model),
        "optimizer_sha256": tensor_tree_digest(optimizer.state_dict()),
        "scheduler_sha256": tensor_tree_digest(scheduler.state_dict()),
    }


def _g4_replay_worker(args: argparse.Namespace) -> int:
    spec = read_json_object_strict(Path(args.worker_spec).resolve(strict=True))
    _validate_worker_spec_common(spec)
    output_path = Path(args.worker_output).expanduser()
    tensor_path = Path(args.worker_tensor_output).expanduser()
    if any(path.exists() or path.is_symlink() for path in (output_path, tensor_path)):
        raise FileExistsError("Refusing to overwrite G4 replay output")
    execution_device = observed_single_visible_cuda_device(
        expected_physical_index=int(spec["physical_device_index"]),
        expected_gpu_uuid=str(spec["gpu_uuid"]),
    )
    result = _run_fixed_layout_optimizer_step(spec)
    tensors = {
        key: result.pop(key)
        for key in ("gradients", "model_state", "optimizer_state", "logits", "runtime_state")
    }
    torch.save(tensors, tensor_path)
    row = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_G4_REPLAY_RUN",
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "candidate_id": str(spec["candidate"]["id"]),
        "variant": str(spec["variant"]),
        "replay_id": str(spec["replay_id"]),
        "fixed_layout": dict(spec["fixed_layout"]),
        **result,
        "tensor_payload_path": str(tensor_path.resolve()),
        "tensor_payload_sha256": sha256_file(tensor_path),
        "execution_device": execution_device,
        "worker_spec_sha256": str(spec["spec_sha256"]),
    }
    _write_hashed_json(output_path, row, field="receipt_sha256")
    return 0


def _compare_replay_rows(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_tensors: Mapping[str, Any],
    right_tensors: Mapping[str, Any],
    tolerance: NumericalTolerances,
) -> dict[str, Any]:
    identity_fields = (
        "candidate_id",
        "variant",
        "fixed_layout",
        "checkpoint_input_sha256",
        "checkpoint_semantic_sha256",
        "rng_input_sha256",
        "data_ids_sha256",
        "data_cursor_sha256",
        "backend_input_sha256",
        "backend_recipe_sha256",
        "optimizer_input_sha256",
        "initial_parameter_digest",
        "parameter_signature_sha256",
        "initial_optimizer_state_signature_sha256",
    )
    identity_mismatches = [
        field for field in identity_fields if left.get(field) != right.get(field)
    ]
    logits = _error_dict(_tensor_error(left_tensors["logits"], right_tensors["logits"]))
    runtime_detail = _combined_runtime_state_error(
        cast(RuntimeState, left_tensors["runtime_state"]),
        cast(RuntimeState, right_tensors["runtime_state"]),
    )
    runtime_state = {
        "relative_l2": runtime_detail["relative_l2"],
        "max_abs": runtime_detail["max_abs"],
    }
    gradient, gradient_worst, gradient_structure = _state_tree_tensor_errors(
        left_tensors["gradients"], right_tensors["gradients"]
    )
    model, model_worst, model_structure = _state_tree_tensor_errors(
        left_tensors["model_state"], right_tensors["model_state"]
    )
    optimizer, optimizer_worst, optimizer_structure = _state_tree_tensor_errors(
        left_tensors["optimizer_state"], right_tensors["optimizer_state"]
    )
    scheduler_equal = left.get("scheduler_sha256") == right.get("scheduler_sha256")
    left_integrity = left.get("optimizer_step_integrity")
    right_integrity = right.get("optimizer_step_integrity")
    required_integrity_fields = (
        "global_token_normalization_identity",
        "accumulation_buffer_reset_once",
        "gradient_clipping_after_accumulation",
        "adamw_step_and_bias_correction_identity",
        "weight_decay_order_and_value_identity",
        "skipped_optimizer_steps_zero",
        "all_gradients_finite",
    )
    actual_optimizer_integrity = (
        isinstance(left_integrity, Mapping)
        and isinstance(right_integrity, Mapping)
        and left_integrity == right_integrity
        and left_integrity.get("passed") is True
        and all(left_integrity.get(field) is True for field in required_integrity_fields)
    )
    state_metadata_exact = bool(runtime_detail["metadata_exact"]) and left.get(
        "state_metadata"
    ) == right.get("state_metadata")
    gradients_finite = bool(left.get("gradients_finite")) and bool(right.get("gradients_finite"))
    clone_no_alias = bool(left.get("clone_no_alias")) and bool(right.get("clone_no_alias"))
    comparison = {
        "logits": logits,
        "runtime_state": runtime_state,
        "gradients": _error_dict(gradient),
        "gradients_worst_leaf": _error_dict(gradient_worst),
        "model_state": _error_dict(model),
        "model_state_worst_leaf": _error_dict(model_worst),
        "optimizer_state": _error_dict(optimizer),
        "optimizer_state_worst_leaf": _error_dict(optimizer_worst),
    }
    numerical_pass = all(
        _passes_error(comparison[name], tolerance)
        for name in ("logits", "runtime_state", "gradients", "model_state", "optimizer_state")
    )
    diagnostics = [left.get("backend_diagnostics"), right.get("backend_diagnostics")]
    graph_hashes = [
        value.get("last_graph_code_sha256") if isinstance(value, Mapping) else None
        for value in diagnostics
    ]
    if graph_hashes[0] != graph_hashes[1] or not isinstance(graph_hashes[0], str):
        identity_mismatches.append("backend_graph_sha256")
    backend_pass = all(
        isinstance(value, Mapping)
        and value.get("graph_break_count") == 0
        and value.get("fallback_count") == 0
        and isinstance(value.get("optimized_calls"), int)
        and int(value["optimized_calls"]) > 0
        for value in diagnostics
    )
    passed = (
        not identity_mismatches
        and numerical_pass
        and gradient_structure
        and model_structure
        and optimizer_structure
        and scheduler_equal
        and actual_optimizer_integrity
        and gradients_finite
        and state_metadata_exact
        and clone_no_alias
        and backend_pass
    )
    return {
        "candidate_id": left.get("candidate_id"),
        "variant": left.get("variant"),
        "fixed_layout": left.get("fixed_layout"),
        "checkpoint_sha256": left.get("checkpoint_input_sha256"),
        "checkpoint_semantic_sha256": left.get("checkpoint_semantic_sha256"),
        "rng_state_sha256": left.get("rng_input_sha256"),
        "data_ids_sha256": left.get("data_ids_sha256"),
        "data_cursor_sha256": left.get("data_cursor_sha256"),
        "backend_graph_sha256": graph_hashes[0],
        "backend_recipe_sha256": left.get("backend_recipe_sha256"),
        "optimizer_input_sha256": left.get("optimizer_input_sha256"),
        "initialization_digest": left.get("initial_parameter_digest"),
        "parameter_signature_sha256": left.get("parameter_signature_sha256"),
        "initial_optimizer_state_signature_sha256": left.get(
            "initial_optimizer_state_signature_sha256"
        ),
        "identity_mismatches": identity_mismatches,
        "comparison": {
            "logits": comparison["logits"],
            "runtime_state": comparison["runtime_state"],
            "gradients": comparison["gradients"],
            "passed": (
                _passes_error(comparison["logits"], tolerance)
                and _passes_error(comparison["runtime_state"], tolerance)
                and _passes_error(comparison["gradients"], tolerance)
            ),
        },
        "optimizer_state": comparison["optimizer_state"],
        "optimizer_state_structure_equal": optimizer_structure,
        "scheduler_state_equal": scheduler_equal,
        "optimizer_integrity_passed": (
            optimizer_structure
            and scheduler_equal
            and _passes_error(comparison["optimizer_state"], tolerance)
            and actual_optimizer_integrity
        ),
        "optimizer_step_integrity": left_integrity,
        "gradients_finite": gradients_finite,
        "state_metadata_exact": state_metadata_exact,
        "clone_no_alias": clone_no_alias,
        "graph_break_count": sum(
            int(value.get("graph_break_count", -1))
            for value in diagnostics
            if isinstance(value, Mapping)
        ),
        "fallback_count": sum(
            int(value.get("fallback_count", -1))
            for value in diagnostics
            if isinstance(value, Mapping)
        ),
        "passed": passed,
    }


def _apply_cross_variant_g4(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Enforce one checkpoint/data/layout/optimizer/backend recipe per pair."""

    normalized = [copy.deepcopy(dict(row)) for row in rows]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_candidate.setdefault(str(row.get("candidate_id")), []).append(row)
    fields = (
        "fixed_layout",
        "checkpoint_sha256",
        "checkpoint_semantic_sha256",
        "rng_state_sha256",
        "data_ids_sha256",
        "data_cursor_sha256",
        "backend_recipe_sha256",
        "optimizer_input_sha256",
        "initialization_digest",
        "parameter_signature_sha256",
        "initial_optimizer_state_signature_sha256",
    )
    for candidate_id, pair in by_candidate.items():
        complete = len(pair) == 2 and {row.get("variant") for row in pair} == set(_VARIANTS)
        mismatches = [] if complete else ["variant_coverage"]
        if complete:
            for field in fields:
                if pair[0].get(field) != pair[1].get(field):
                    mismatches.append(field)
        for row in pair:
            row["cross_variant_identity"] = {
                "candidate_id": candidate_id,
                "fields": list(fields),
                "mismatches": mismatches,
                "passed": not mismatches,
            }
            row["passed"] = bool(row.get("passed") is True and not mismatches)
    return normalized


def _validate_g3_row(payload: Mapping[str, Any], *, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not _canonical_hashed(payload, field="receipt_sha256"):
        raise ValueError("G3 receipt canonical hash mismatch")
    bindings = {
        "candidate_id": str(spec["candidate"]["id"]),
        "variant": str(spec["variant"]),
        "state_context": str(spec["state_context"]),
        "fixed_layout": dict(spec["fixed_layout"]),
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
    }
    mismatches = [key for key, expected in bindings.items() if payload.get(key) != expected]
    if mismatches:
        raise ValueError(f"G3 receipt binding mismatch: {mismatches}")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "compiled_bf16_vs_reference_python_bf16",
        "reference_python_bf16_vs_reference_python_fp32",
    }:
        raise ValueError("G3 receipt comparison set is incomplete")
    return dict(payload)


def _apply_cross_variant_g1_and_g6(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bind paired variants to identical data/layout and observed precision policy."""

    normalized = [copy.deepcopy(dict(row)) for row in rows]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in normalized:
        key = (str(row.get("candidate_id")), str(row.get("state_context")))
        by_key.setdefault(key, []).append(row)
    for key, pair in by_key.items():
        if {str(row.get("variant")) for row in pair} != set(_VARIANTS) or len(pair) != 2:
            for row in pair:
                row["layout_identity_passed"] = False
                row["passed"] = False
            continue
        identity_fields = (
            "fixed_layout",
            "initialization_digest",
            "parameter_signature_sha256",
            "optimizer_state_signature_sha256",
            "token_ids_sha256",
            "data_cursor_sha256",
        )
        paired_identity = all(pair[0].get(field) == pair[1].get(field) for field in identity_fields)
        padded = [
            int(cast(Mapping[str, Any], row["backend_integrity"]).get("observed_padded_tokens", -1))
            for row in pair
        ]
        policies = [
            cast(Mapping[str, Any], row["backend_integrity"]).get("precision_policy_sha256")
            for row in pair
        ]
        padding_difference = 0 if padded[0] == padded[1] and padded[0] >= 0 else 1
        precision_difference = 0 if policies[0] == policies[1] and policies[0] else 1
        for row in pair:
            row["layout_identity_passed"] = bool(
                row.get("layout_identity_passed") is True and paired_identity
            )
            row["variant_specific_padding_count"] = padding_difference
            row["variant_specific_fp32_path_count"] = precision_difference
            backend = cast(dict[str, Any], row["backend_integrity"])
            backend["variant_specific_padding_count"] = padding_difference
            backend["variant_specific_fp32_path_count"] = precision_difference
            backend["passed"] = bool(
                backend.get("passed") is True
                and padding_difference == 0
                and precision_difference == 0
            )
            row["passed"] = bool(
                row.get("passed") is True and row["layout_identity_passed"] and backend["passed"]
            )
            row["cross_variant_identity"] = {
                "candidate_id": key[0],
                "state_context": key[1],
                "fields": list(identity_fields),
                "passed": paired_identity,
            }
    return normalized


def _blocked_g3_coverage_rows(
    specs: Sequence[tuple[dict[str, Any], Path, Path, Path, str]],
    observed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row.get("candidate_id")),
            str(row.get("variant")),
            str(row.get("state_context")),
        ): copy.deepcopy(dict(row))
        for row in observed
    }
    rows: list[dict[str, Any]] = []
    for spec, _spec_path, _output, _log, _device in specs:
        key = (
            str(spec["candidate"]["id"]),
            str(spec["variant"]),
            str(spec["state_context"]),
        )
        row = by_key.get(key)
        if row is None:
            row = {
                "candidate_id": key[0],
                "variant": key[1],
                "state_context": key[2],
                "fixed_layout": dict(spec["fixed_layout"]),
                "execution_status": "FAILED_WORKER_NO_RECEIPT",
                "layout_identity_passed": False,
                "passed": False,
            }
        rows.append(row)
    return rows


def _blocked_g4_dependency_rows(
    *,
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    layouts: Mapping[str, Mapping[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id,
            "variant": variant,
            "fixed_layout": dict(layouts[candidate_id]),
            "execution_status": "NOT_RUN_BLOCKED_DEPENDENCY",
            "blocked_reason": reason,
            "optimizer_integrity_passed": False,
            "graph_break_count": None,
            "fallback_count": None,
            "passed": False,
        }
        for candidate_id in candidate_by_id
        for variant in _VARIANTS
    ]


def _worker_command(
    tool: Path,
    *,
    mode: str,
    spec: Path,
    output: Path,
    tensor_output: Path | None = None,
    checkpoint_output: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(tool),
        mode,
        "--worker-spec",
        str(spec),
        "--worker-output",
        str(output),
    ]
    if tensor_output is not None:
        command.extend(("--worker-tensor-output", str(tensor_output)))
    if checkpoint_output is not None:
        command.extend(("--worker-checkpoint-output", str(checkpoint_output)))
    return command


def _run_worker(
    *,
    repo: Path,
    tool: Path,
    device: str,
    mode: str,
    spec_path: Path,
    output_path: Path,
    log_path: Path,
    tensor_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    with log_path.open("x", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            _worker_command(
                tool,
                mode=mode,
                spec=spec_path,
                output=output_path,
                tensor_output=tensor_path,
                checkpoint_output=checkpoint_path,
            ),
            cwd=repo,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _run_g3_lane(
    *,
    repo: Path,
    tool: Path,
    tasks: Sequence[tuple[dict[str, Any], Path, Path, Path, str]],
) -> dict[str, int]:
    """Run one candidate's fresh case subprocesses serially on one GPU lane."""

    failures: dict[str, int] = {}
    for _spec, spec_path, output_path, log_path, device in tasks:
        code = _run_worker(
            repo=repo,
            tool=tool,
            device=device,
            mode="--worker-g3",
            spec_path=spec_path,
            output_path=output_path,
            log_path=log_path,
        )
        if code != 0:
            failures[output_path.stem] = code
    return failures


def _validate_checkpoint_row(
    payload: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> dict[str, Any]:
    if not _canonical_hashed(payload, field="receipt_sha256"):
        raise ValueError("G4 common checkpoint receipt canonical hash mismatch")
    expected = {
        "candidate_id": str(spec["candidate"]["id"]),
        "fixed_layout": dict(spec["fixed_layout"]),
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        mismatches.append("checkpoint")
    else:
        checkpoint_path = Path(str(checkpoint.get("path")))
        if (
            checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != checkpoint.get("sha256")
            or checkpoint.get("bytes") != checkpoint_path.stat().st_size
        ):
            mismatches.append("checkpoint_bytes")
    if payload.get("checkpoint_locked_hashes") != spec.get("checkpoint_locked_hashes"):
        mismatches.append("checkpoint_locked_hashes")
    if payload.get("backend_manifest") != _checkpoint_backend_manifest(spec):
        mismatches.append("backend_manifest")
    for field in (
        "rng_state_sha256",
        "data_cursor_sha256",
        "initialization_digest",
        "parameter_signature_sha256",
        "optimizer_state_signature_sha256",
    ):
        value = payload.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            mismatches.append(field)
    if mismatches:
        raise ValueError(f"G4 common checkpoint receipt binding mismatch: {mismatches}")
    return dict(payload)


def _validate_g4_run_row(payload: Mapping[str, Any], *, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not _canonical_hashed(payload, field="receipt_sha256"):
        raise ValueError("G4 replay receipt canonical hash mismatch")
    expected = {
        "candidate_id": str(spec["candidate"]["id"]),
        "variant": str(spec["variant"]),
        "replay_id": str(spec["replay_id"]),
        "fixed_layout": dict(spec["fixed_layout"]),
        "worker_spec_sha256": str(spec["spec_sha256"]),
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    tensor_path = payload.get("tensor_payload_path")
    resolved_tensor = Path(tensor_path) if isinstance(tensor_path, str) else None
    if (
        resolved_tensor is None
        or resolved_tensor.is_symlink()
        or not resolved_tensor.is_file()
        or sha256_file(resolved_tensor) != payload.get("tensor_payload_sha256")
    ):
        mismatches.append("tensor_payload")
    if mismatches:
        raise ValueError(f"G4 replay receipt binding mismatch: {mismatches}")
    return dict(payload)


def _run_g4_candidate_lane(
    *,
    repo: Path,
    tool: Path,
    replay_dir: Path,
    candidate: Mapping[str, Any],
    layout: Mapping[str, Any],
    device: str,
    hardware: Mapping[str, Any],
    common_spec: Mapping[str, Any],
    tolerance: NumericalTolerances,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run checkpoint plus tied/dual A/B replays serially on one GPU lane."""

    candidate_id = str(candidate["id"])
    failures: list[str] = []
    checkpoint_spec: dict[str, Any] = {
        **common_spec,
        "manifest_type": "E26_STAGE3D_G4_CHECKPOINT_WORKER_SPEC",
        "candidate": dict(candidate),
        "fixed_layout": dict(layout),
        "physical_device_index": int(device),
        "gpu_uuid": str(hardware["gpu_uuid"]),
        "backend_recipe": _build_backend_recipe(
            candidate=candidate,
            layout=layout,
            hardware=hardware,
            protocol_sha256=str(common_spec["protocol_sha256"]),
            source_tree_sha256=str(common_spec["source_tree_sha256"]),
            backend_binding=cast(Mapping[str, Any], common_spec["backend_binding"]),
        ),
    }
    checkpoint_spec["spec_sha256"] = sha256_canonical_json(checkpoint_spec)
    checkpoint_prefix = replay_dir / "checkpoints" / candidate_id
    checkpoint_spec_path = checkpoint_prefix.with_suffix(".spec.json")
    checkpoint_receipt_path = checkpoint_prefix.with_suffix(".receipt.json")
    checkpoint_path = checkpoint_prefix.with_suffix(".pt")
    checkpoint_log_path = checkpoint_prefix.with_suffix(".log")
    write_json_strict(checkpoint_spec_path, checkpoint_spec)
    code = _run_worker(
        repo=repo,
        tool=tool,
        device=device,
        mode="--worker-checkpoint",
        spec_path=checkpoint_spec_path,
        output_path=checkpoint_receipt_path,
        log_path=checkpoint_log_path,
        checkpoint_path=checkpoint_path,
    )
    if code != 0 or not checkpoint_receipt_path.is_file() or not checkpoint_path.is_file():
        return [], [f"checkpoint_worker:{candidate_id}:exit={code}"]
    try:
        checkpoint_row = _validate_checkpoint_row(
            read_json_object_strict(checkpoint_receipt_path), spec=checkpoint_spec
        )
    except Exception as error:  # noqa: BLE001 - converted to fail-closed receipt
        return [], [f"checkpoint_receipt:{candidate_id}:{type(error).__name__}:{error}"]
    checkpoint = cast(Mapping[str, Any], checkpoint_row["checkpoint"])
    rows: list[dict[str, Any]] = []
    for variant in _VARIANTS:
        base_spec: dict[str, Any] = {
            **common_spec,
            "manifest_type": "E26_STAGE3D_G4_WORKER_SPEC",
            "candidate": dict(candidate),
            "variant": variant,
            "fixed_layout": dict(layout),
            "physical_device_index": int(device),
            "gpu_uuid": str(hardware["gpu_uuid"]),
            "checkpoint": dict(checkpoint),
            "checkpoint_receipt_sha256": checkpoint_row["receipt_sha256"],
            "backend_recipe": dict(checkpoint_spec["backend_recipe"]),
        }
        runs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for replay_id in ("A", "B"):
            spec = {**base_spec, "replay_id": replay_id}
            spec["spec_sha256"] = sha256_canonical_json(spec)
            prefix = replay_dir / f"{candidate_id}__{variant}__{replay_id}"
            spec_path = prefix.with_suffix(".spec.json")
            output_path = prefix.with_suffix(".json")
            tensor_path = prefix.with_suffix(".tensors.pt")
            log_path = prefix.with_suffix(".log")
            write_json_strict(spec_path, spec)
            code = _run_worker(
                repo=repo,
                tool=tool,
                device=device,
                mode="--worker-replay",
                spec_path=spec_path,
                output_path=output_path,
                log_path=log_path,
                tensor_path=tensor_path,
            )
            if code != 0 or not output_path.is_file() or not tensor_path.is_file():
                failures.append(f"replay_worker:{candidate_id}:{variant}:{replay_id}:exit={code}")
                break
            try:
                run_row = _validate_g4_run_row(read_json_object_strict(output_path), spec=spec)
                checkpoint_bindings = {
                    "checkpoint_input_sha256": checkpoint.get("sha256"),
                    "checkpoint_semantic_sha256": checkpoint.get("semantic_payload_sha256"),
                    "rng_input_sha256": checkpoint_row.get("rng_state_sha256"),
                    "data_cursor_sha256": checkpoint_row.get("data_cursor_sha256"),
                    "initial_parameter_digest": checkpoint_row.get("initialization_digest"),
                    "parameter_signature_sha256": checkpoint_row.get("parameter_signature_sha256"),
                    "initial_optimizer_state_signature_sha256": checkpoint_row.get(
                        "optimizer_state_signature_sha256"
                    ),
                }
                drift = [
                    name
                    for name, expected in checkpoint_bindings.items()
                    if run_row.get(name) != expected
                ]
                if drift:
                    raise ValueError(f"G4 replay diverged from common checkpoint: {drift}")
                tensors = torch.load(tensor_path, map_location="cpu", weights_only=False)
                if not isinstance(tensors, Mapping):
                    raise TypeError("G4 tensor payload is not a mapping")
                runs.append((run_row, dict(tensors)))
            except Exception as error:  # noqa: BLE001 - converted to fail-closed receipt
                failures.append(
                    f"replay_receipt:{candidate_id}:{variant}:{replay_id}:"
                    f"{type(error).__name__}:{error}"
                )
                break
        if failures:
            break
        comparison = _compare_replay_rows(
            runs[0][0],
            runs[1][0],
            left_tensors=runs[0][1],
            right_tensors=runs[1][1],
            tolerance=tolerance,
        )
        rows.append(comparison)
        _write_hashed_json(
            replay_dir / f"{candidate_id}__{variant}__comparison.json",
            comparison,
            field="receipt_sha256",
        )
    return rows, failures


def _write_execution_error_terminal(
    *,
    output_root: Path,
    stage: str,
    failures: Sequence[str],
    completed_g3: int,
    completed_g4: int,
) -> None:
    """Preserve operational failures without issuing a numerical disposition."""

    report = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_EXECUTION_ERROR_REPORT",
        "execution_status": "EXECUTION_ERROR",
        "disposition": STAGE3D_NOT_EVALUABLE,
        "diagnostic_disposition": KNOWN_LAYOUT_SENSITIVITY,
        "failure_stage": stage,
        "failures": list(failures),
        "g3_completed_cases": completed_g3,
        "g4_completed_replay_pairs": completed_g4,
        "resource_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_evidence": False,
    }
    _write_hashed_json(output_root / "report.json", report, field="receipt_sha256")
    status = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_STATUS",
        "execution_status": "EXECUTION_ERROR",
        "disposition": STAGE3D_NOT_EVALUABLE,
        "failure_stage": stage,
        "resource_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_evidence": False,
        "report_sha256": sha256_file(output_root / "report.json"),
    }
    _write_hashed_json(output_root / "status.json", status, field="receipt_sha256")
    _finalize_terminal_artifacts(output_root)


def _write_results_summary(output_root: Path) -> Path:
    """Generate the compact Korean terminal summary before artifact hashing."""

    destination = output_root / "RESULTS_SUMMARY_KO.md"
    if destination.exists() or destination.is_symlink():
        return destination
    report = read_json_object_strict(output_root / "report.json")
    status = read_json_object_strict(output_root / "status.json")
    summary = report.get("gate_summary")
    gates = summary if isinstance(summary, Mapping) else {}
    disposition = status.get("disposition", report.get("disposition", "UNKNOWN"))
    execution_status = status.get("execution_status", report.get("execution_status", "UNKNOWN"))
    go = disposition == STAGE3D_GO
    gate_lines = []
    for index in range(7):
        gate_value = gates.get(f"g{index}_passed")
        gate_label = (
            "PASS" if gate_value is True else "FAIL" if gate_value is False else "NOT_EVALUABLE"
        )
        gate_lines.append(f"- G{index}: {gate_label}")
    g3_count = gates.get("g3_pass_count", report.get("g3_completed_cases", 0))
    g4_count = gates.get("g4_pass_count", report.get("g4_completed_replay_pairs", 0))
    g3_expected = status.get("g3_expected_cases", report.get("g3_expected_cases", "NOT_EVALUABLE"))
    g4_expected = status.get(
        "g4_expected_replay_pairs",
        report.get("g4_expected_replay_pairs", "NOT_EVALUABLE"),
    )
    text = "\n".join(
        [
            "# E26 Stage-3D 결과 요약",
            "",
            f"- execution_status: `{execution_status}`",
            f"- disposition: `{disposition}`",
            f"- diagnostic: `{KNOWN_LAYOUT_SENSITIVITY}`",
            f"- G3 passed/completed: `{g3_count}/{g3_expected}`",
            f"- G4 passed/completed: `{g4_count}/{g4_expected}`",
            *gate_lines,
            f"- resource preflight eligible: `{str(go).lower()}`",
            "- resource preflight started: `false`",
            "- Scientific E26a started: `false`",
            "",
            (
                "허용 claim: Stage-3D가 GO인 경우에만 사전 고정한 단일 "
                "physical layout의 BF16 numerical admissibility를 주장할 수 있다."
            ),
            "",
            (
                "금지 claim: arbitrary batching-layout invariance, Scientific E26a 성능, "
                "official GDN2/KDA 대응, 언어모델 우월성은 주장할 수 없다."
            ),
            "",
        ]
    )
    destination.write_text(text, encoding="utf-8")
    return destination


def _publish_terminal_latest(output_root: Path) -> None:
    canonical_namespace = Path(
        "/data/minjun_dev/CATENA/artifacts/e26_stage3d_fixed_layout_bf16_admissibility"
    )
    if output_root.parent != canonical_namespace:
        return
    status = read_json_object_strict(output_root / "status.json")
    latest = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_LATEST_POINTER",
        "run_dir": str(output_root),
        "disposition": status.get("disposition"),
        "report_sha256": sha256_file(output_root / "report.json"),
        "status_sha256": sha256_file(output_root / "status.json"),
        "artifact_audit_sha256": sha256_file(output_root / "artifact_audit.json"),
        "results_summary_sha256": sha256_file(output_root / "RESULTS_SUMMARY_KO.md"),
        "scientific_e26a_started": False,
    }
    latest["pointer_sha256"] = sha256_canonical_json(latest)
    temporary = canonical_namespace / f".latest.{os.getpid()}.json"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing to reuse latest temp path: {temporary}")
    write_json_strict(temporary, latest)
    os.replace(temporary, canonical_namespace / "latest.json")


def _finalize_terminal_artifacts(output_root: Path) -> None:
    """Write a terminal file inventory and then atomically publish durable latest."""

    required = ("report.json", "status.json")
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise RuntimeError(f"Stage-3D terminal artifact set is incomplete: {missing}")
    _write_results_summary(output_root)
    paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "artifact_audit.json"
    )
    rows = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    audit = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_TERMINAL_ARTIFACT_AUDIT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "run_dir": str(output_root),
        "file_count_excluding_self": len(rows),
        "aggregate_sha256_excluding_self": _row_aggregate(rows),
        "files": rows,
        "protocol_lock_present": (output_root / "protocol_lock.json").is_file(),
        "layout_manifest_present": (output_root / "layout_manifest.json").is_file(),
        "passed": True,
    }
    _write_hashed_json(output_root / "artifact_audit.json", audit, field="receipt_sha256")
    _publish_terminal_latest(output_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E26 Stage-3D fixed-physical-layout BF16 preflight (non-evidence only)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/home/minjun_dev/CATENA"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage3c-result", type=Path)
    parser.add_argument("--stage3c-protocol-lock", type=Path)
    parser.add_argument("--stage3c-artifact-manifest", type=Path)
    parser.add_argument("--stage3c-artifact-root", type=Path)
    parser.add_argument("--e00-e25-manifest", type=Path)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--worker-g3", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-replay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-tensor-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_g3 or args.worker_checkpoint or args.worker_replay:
        if args.worker_spec is None or args.worker_output is None:
            parser.error("worker mode requires --worker-spec and --worker-output")
        if args.worker_replay and args.worker_tensor_output is None:
            parser.error("replay worker requires --worker-tensor-output")
        if args.worker_checkpoint and args.worker_checkpoint_output is None:
            parser.error("checkpoint worker requires --worker-checkpoint-output")
        return args
    required = (
        "output_root",
        "config",
        "stage3c_result",
        "stage3c_protocol_lock",
        "stage3c_artifact_manifest",
        "stage3c_artifact_root",
        "e00_e25_manifest",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def _main_parent_run(args: argparse.Namespace) -> int:
    repo = args.repo_root.expanduser().resolve(strict=True)
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Stage-3D requires a clean committed worktree")
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_stage3d_config(config_path)
    predecessor = config.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise ValueError("Stage-3D config lacks predecessor anchors")
    stage3c_protocol_path = args.stage3c_protocol_lock.expanduser().resolve(strict=True)
    stage3c_artifact_manifest_path = args.stage3c_artifact_manifest.expanduser().resolve(
        strict=True
    )
    stage3c_artifact_root = args.stage3c_artifact_root.expanduser().resolve(strict=True)
    frozen_receipt_path = args.e00_e25_manifest.expanduser().resolve(strict=True)
    g0_audit = _verify_g0_frozen_inputs(
        stage3c_protocol_path=stage3c_protocol_path,
        stage3c_artifact_manifest_path=stage3c_artifact_manifest_path,
        stage3c_artifact_root=stage3c_artifact_root,
        frozen_receipt_path=frozen_receipt_path,
        expected_predecessor=predecessor,
    )
    if g0_audit["passed"] is not True:
        raise RuntimeError("Stage-3D G0 frozen-evidence re-audit failed")
    source_commit = _git(repo, "rev-parse", "HEAD")
    source_inventory = e26_execution_source_inventory(repo)
    protocol = build_stage3d_protocol_lock(
        config_path=config_path,
        stage3c_result_path=args.stage3c_result.expanduser().resolve(strict=True),
        stage3c_protocol_path=stage3c_protocol_path,
        stage3c_artifact_manifest_path=stage3c_artifact_manifest_path,
        frozen_e00_e25_receipt_path=frozen_receipt_path,
        source_commit=source_commit,
        source_inventory=source_inventory,
    )
    protocol = validate_stage3d_protocol_lock(protocol, config_path=config_path)
    # G2 is re-derived from the immutable Stage-3C raw 12-report/132-row
    # population before a run namespace, hardware inventory, or GPU worker is
    # opened.  A stale/forged summary therefore cannot authorize GPU work.
    fp32_binding = derive_stage3c_fp32_reference_binding(protocol)
    layout_rows = fixed_layouts_from_config(config)
    layouts = {str(row["candidate_id"]): dict(row) for row in layout_rows}
    stage3c_protocol = _yaml_mapping(stage3c_protocol_path, label="Stage-3C protocol")
    stage3c_snapshot = stage3c_protocol.get("full_config_snapshot")
    if not isinstance(stage3c_snapshot, Mapping):
        raise ValueError("Stage-3C protocol lacks full_config_snapshot")
    candidates = stage3c_snapshot.get("model_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Stage-3C protocol model_candidates must be a list")
    candidate_by_id = {str(row["id"]): row for row in candidates}
    if set(candidate_by_id) != set(layouts):
        raise ValueError("Stage-3D fixed layouts do not cover the candidate set exactly")
    devices = [value.strip() for value in str(args.devices).split(",") if value.strip()]
    if len(devices) < len(candidates) or len(set(devices)) != len(devices):
        raise ValueError("Supply one distinct CUDA device per Stage-3D candidate")
    hardware = cuda_hardware_inventory(devices)
    hardware_by_index = {int(row["physical_device_index"]): row for row in hardware}
    stage3c_execution_paths = stage3c_protocol.get("execution_input_paths")
    if not isinstance(stage3c_execution_paths, Mapping):
        raise ValueError("Stage-3C protocol lacks execution input paths")
    resolved_stage3c_paths = _resolve_stage3c_execution_paths(
        protocol_path=stage3c_protocol_path,
        execution_paths=stage3c_execution_paths,
    )
    input_paths: dict[str, str] = {
        "stage3d_config": str(config_path),
        "stage3c_result": str(args.stage3c_result.expanduser().resolve(strict=True)),
        "stage3c_protocol": str(stage3c_protocol_path),
        "stage3c_artifact_manifest": str(stage3c_artifact_manifest_path),
        "e00_e25_manifest": str(frozen_receipt_path),
        **{f"stage3c_input_{name}": path for name, path in resolved_stage3c_paths.items()},
    }
    input_hashes = {f"{name}_sha256": sha256_file(path) for name, path in input_paths.items()}
    checkpoint_locked_hashes = _checkpoint_locked_hashes(
        stage3c_artifact_root=stage3c_artifact_root,
        source_inventory=source_inventory,
        stage3c_execution_paths=resolved_stage3c_paths,
    )
    layout_manifest = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_FIXED_LAYOUT_MANIFEST",
        "scientific_evidence": False,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": source_commit,
        "source_tree_sha256": source_inventory["source_tree_sha256"],
        "candidate_order": list(candidate_by_id),
        "variants": list(_VARIANTS),
        "backend_binding": protocol["backend_binding"],
        "layouts": layout_rows,
        "input_hashes": input_hashes,
        "input_paths": input_paths,
        "checkpoint_locked_hashes": checkpoint_locked_hashes,
        "g0_frozen_input_audit": g0_audit,
    }
    # The canonical namespace is created only after G0/G2, source/data/input
    # admission, candidate/layout validation, and CUDA hardware admission have
    # all completed.  Any failure above leaves no misleading partial run.
    output_root = _fresh_output_root(args.output_root)
    protocol_lock_path = output_root / "protocol_lock.json"
    write_json_strict(protocol_lock_path, protocol)
    layout_manifest = _write_hashed_json(
        output_root / "layout_manifest.json",
        layout_manifest,
        field="manifest_sha256",
    )
    tool = Path(__file__).resolve()
    g3_dir = output_root / "g3_cases"
    g3_dir.mkdir()
    g3_specs: list[tuple[dict[str, Any], Path, Path, Path, str]] = []
    determinism = cast(Mapping[str, Any], config["determinism"])
    thresholds = cast(Mapping[str, Any], config["thresholds"])
    for candidate_index, (candidate_id, candidate) in enumerate(candidate_by_id.items()):
        device = devices[candidate_index]
        hardware_row = hardware_by_index[int(device)]
        for variant in _VARIANTS:
            for state_context in _STATE_CONTEXTS:
                case_id = f"{candidate_id}__{variant}__{state_context}"
                spec: dict[str, Any] = {
                    "schema_version": "catena-v8.1",
                    "manifest_type": "E26_STAGE3D_G3_WORKER_SPEC",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "protocol_lock_path": str(protocol_lock_path),
                    "protocol_lock_file_sha256": sha256_file(protocol_lock_path),
                    "layout_manifest_sha256": layout_manifest["manifest_sha256"],
                    "layout_manifest_path": str(output_root / "layout_manifest.json"),
                    "layout_manifest_file_sha256": sha256_file(
                        output_root / "layout_manifest.json"
                    ),
                    "repo_root": str(repo),
                    "source_commit": source_commit,
                    "source_tree_sha256": source_inventory["source_tree_sha256"],
                    "candidate": dict(candidate),
                    "variant": variant,
                    "state_context": state_context,
                    "fixed_layout": dict(layouts[candidate_id]),
                    "initialization_seed": int(determinism["initialization_seed"]),
                    "data_seed": int(determinism["g3_data_seed"]),
                    "prefill_seed": int(determinism["prefill_seed"]),
                    "prefix_length": int(determinism["prefill_length"]),
                    "bf16_relative_l2_max": float(thresholds["bf16_relative_l2_max"]),
                    "gradient_norm_min": float(thresholds["gradient_norm_min"]),
                    "gradient_norm_max": float(thresholds["gradient_norm_max"]),
                    "physical_device_index": int(device),
                    "gpu_uuid": hardware_row["gpu_uuid"],
                    "input_hashes": input_hashes,
                    "input_paths": input_paths,
                    "backend_binding": protocol["backend_binding"],
                }
                spec["spec_sha256"] = sha256_canonical_json(spec)
                spec_path = g3_dir / f"{case_id}.spec.json"
                output_path = g3_dir / f"{case_id}.json"
                log_path = g3_dir / f"{case_id}.log"
                write_json_strict(spec_path, spec)
                g3_specs.append((spec, spec_path, output_path, log_path, device))

    # Preserve every case independently.  The three candidates are sharded
    # outcome-neutrally across three GPUs; each candidate's four fresh
    # subprocesses remain serial on its lane to avoid memory/cache interference.
    g3_failures: dict[str, int] = {}
    lanes = {
        device: [task for task in g3_specs if task[4] == device]
        for device in devices[: len(candidates)]
    }
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [
            executor.submit(_run_g3_lane, repo=repo, tool=tool, tasks=tasks)
            for tasks in lanes.values()
        ]
        for future in futures:
            g3_failures.update(future.result())
    g3_execution_errors = [
        f"g3_worker:{case_id}:exit={code}" for case_id, code in sorted(g3_failures.items())
    ]
    raw_g3_rows: list[dict[str, Any]] = []
    for spec, _spec_path, output, _log, _device in g3_specs:
        if not output.is_file():
            g3_execution_errors.append(f"g3_missing_receipt:{output.stem}")
            continue
        try:
            raw_g3_rows.append(_validate_g3_row(read_json_object_strict(output), spec=spec))
        except Exception as error:  # noqa: BLE001 - terminal fail-closed conversion
            g3_execution_errors.append(
                f"g3_invalid_receipt:{output.stem}:{type(error).__name__}:{error}"
            )
    if g3_execution_errors or len(raw_g3_rows) != 12:
        _write_execution_error_terminal(
            output_root=output_root,
            stage="G3_FIXED_LAYOUT_BF16_ADMISSIBILITY",
            failures=g3_execution_errors,
            completed_g3=len(raw_g3_rows),
            completed_g4=0,
        )
        return 2
    g3_rows = _apply_cross_variant_g1_and_g6(raw_g3_rows)
    if not all(row.get("passed") is True for row in g3_rows):
        g3_coverage = _blocked_g3_coverage_rows(g3_specs, g3_rows)
        blocked_replays = _blocked_g4_dependency_rows(
            candidate_by_id=candidate_by_id,
            layouts=layouts,
            reason="G3_FIXED_LAYOUT_BF16_ADMISSIBILITY_FAILED",
        )
        receipt = build_stage3d_admissibility_receipt(
            protocol_lock_path=protocol_lock_path,
            g3_cases=g3_coverage,
            g4_replays=blocked_replays,
            fp32_reference_binding=fp32_binding,
        )
        receipt = validate_stage3d_admissibility_receipt(receipt, protocol_lock=protocol)
        write_json_strict(output_root / "report.json", receipt)
        status = {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_STAGE3D_STATUS",
            "disposition": STAGE3D_BLOCKED,
            "diagnostic_disposition": KNOWN_LAYOUT_SENSITIVITY,
            "g3_expected_cases": 12,
            "g3_completed_cases": len(g3_rows),
            "g3_numerical_failures": sum(row.get("passed") is not True for row in g3_rows),
            "g4_started": False,
            "g4_disposition": "NOT_RUN_BLOCKED_G3_DEPENDENCY",
            "resource_preflight_started": False,
            "scientific_e26a_started": False,
            "scientific_evidence": False,
            "report_sha256": sha256_file(output_root / "report.json"),
        }
        _write_hashed_json(output_root / "status.json", status, field="receipt_sha256")
        _finalize_terminal_artifacts(output_root)
        return 1

    replay_dir = output_root / "g4_replays"
    replay_dir.mkdir()
    (replay_dir / "checkpoints").mkdir()
    replay_rows: list[dict[str, Any]] = []
    g4_execution_errors: list[str] = []
    g4_common = {
        "schema_version": "catena-v8.1",
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_lock_path": str(protocol_lock_path),
        "protocol_lock_file_sha256": sha256_file(protocol_lock_path),
        "layout_manifest_sha256": layout_manifest["manifest_sha256"],
        "layout_manifest_path": str(output_root / "layout_manifest.json"),
        "layout_manifest_file_sha256": sha256_file(output_root / "layout_manifest.json"),
        "repo_root": str(repo),
        "source_commit": source_commit,
        "source_tree_sha256": source_inventory["source_tree_sha256"],
        "initialization_seed": int(determinism["initialization_seed"]),
        "data_seed": int(determinism["g4_data_seed"]),
        "checkpoint_locked_hashes": checkpoint_locked_hashes,
        "input_hashes": input_hashes,
        "input_paths": input_paths,
        "backend_binding": protocol["backend_binding"],
    }
    tolerance = NumericalTolerances(
        relative_l2_max=float(thresholds["bf16_relative_l2_max"]),
        max_abs_max=None,
    )
    with ThreadPoolExecutor(max_workers=len(candidate_by_id)) as executor:
        g4_futures = []
        for candidate_index, (candidate_id, candidate) in enumerate(candidate_by_id.items()):
            device = devices[candidate_index]
            hardware_row = hardware_by_index[int(device)]
            g4_futures.append(
                executor.submit(
                    _run_g4_candidate_lane,
                    repo=repo,
                    tool=tool,
                    replay_dir=replay_dir,
                    candidate=candidate,
                    layout=layouts[candidate_id],
                    device=device,
                    hardware=hardware_row,
                    common_spec=g4_common,
                    tolerance=tolerance,
                )
            )
        for g4_future in g4_futures:
            lane_rows, lane_failures = g4_future.result()
            replay_rows.extend(lane_rows)
            g4_execution_errors.extend(lane_failures)
    if g4_execution_errors or len(replay_rows) != 6:
        _write_execution_error_terminal(
            output_root=output_root,
            stage="G4_SAME_LAYOUT_REPLAY",
            failures=g4_execution_errors,
            completed_g3=len(g3_rows),
            completed_g4=len(replay_rows),
        )
        return 2
    replay_rows = _apply_cross_variant_g4(replay_rows)

    receipt = build_stage3d_admissibility_receipt(
        protocol_lock_path=protocol_lock_path,
        g3_cases=g3_rows,
        g4_replays=replay_rows,
        fp32_reference_binding=fp32_binding,
    )
    receipt = validate_stage3d_admissibility_receipt(receipt, protocol_lock=protocol)
    write_json_strict(output_root / "report.json", receipt)
    passed = receipt.get("disposition") == STAGE3D_GO
    status = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_STATUS",
        "disposition": receipt.get("disposition"),
        "diagnostic_disposition": KNOWN_LAYOUT_SENSITIVITY,
        "g3_expected_cases": 12,
        "g3_completed_cases": len(g3_rows),
        "g4_expected_replay_pairs": 6,
        "g4_completed_replay_pairs": len(replay_rows),
        "resource_preflight_eligible": passed,
        "resource_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_evidence": False,
        "report_sha256": sha256_file(output_root / "report.json"),
        "input_hashes": input_hashes,
    }
    _write_hashed_json(output_root / "status.json", status, field="receipt_sha256")
    _finalize_terminal_artifacts(output_root)
    return 0 if passed else 1


def _recover_unexpected_parent_exception(*, output_root: Path, error: Exception) -> None:
    """Finish a created namespace without rewriting any terminal artifact."""

    failure = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_STAGE3D_UNEXPECTED_PARENT_EXCEPTION",
        "execution_status": "EXECUTION_ERROR",
        "disposition": STAGE3D_NOT_EVALUABLE,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "resource_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_evidence": False,
    }
    failure_path = output_root / "unexpected_execution_error.json"
    if not failure_path.exists() and not failure_path.is_symlink():
        _write_hashed_json(failure_path, failure, field="receipt_sha256")

    report_path = output_root / "report.json"
    if not report_path.exists() and not report_path.is_symlink():
        _write_hashed_json(report_path, failure, field="receipt_sha256")
    status_path = output_root / "status.json"
    if not status_path.exists() and not status_path.is_symlink():
        status = {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_STAGE3D_STATUS",
            "execution_status": "EXECUTION_ERROR",
            "disposition": STAGE3D_NOT_EVALUABLE,
            "recovered_after_unexpected_parent_exception": True,
            "resource_preflight_started": False,
            "scientific_e26a_started": False,
            "scientific_evidence": False,
            "report_sha256": sha256_file(report_path),
        }
        _write_hashed_json(status_path, status, field="receipt_sha256")

    # Do not alter a previously completed audit.  Otherwise inventory every
    # preserved partial file plus the explicit exception receipt.
    audit_path = output_root / "artifact_audit.json"
    if not audit_path.exists() and not audit_path.is_symlink():
        _finalize_terminal_artifacts(output_root)
    elif (output_root / "RESULTS_SUMMARY_KO.md").is_file():
        _publish_terminal_latest(output_root)


def _main_parent(args: argparse.Namespace) -> int:
    """Guard all post-admission parent/future failures with a terminal receipt."""

    try:
        return _main_parent_run(args)
    except Exception as error:  # noqa: BLE001 - explicit terminal fail-closed boundary
        output_raw = getattr(args, "output_root", None)
        if output_raw is None:
            raise
        output_root = Path(output_raw).expanduser()
        if not output_root.is_dir() or output_root.is_symlink():
            # Admission failed before a canonical namespace was created.
            raise
        _recover_unexpected_parent_exception(output_root=output_root.resolve(), error=error)
        return 2


def main() -> int:
    args = _parse_args()
    if args.worker_g3:
        return _g3_worker(args)
    if args.worker_checkpoint:
        return _g4_checkpoint_worker(args)
    if args.worker_replay:
        return _g4_replay_worker(args)
    return _main_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
