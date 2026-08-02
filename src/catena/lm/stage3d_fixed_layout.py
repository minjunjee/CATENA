"""Strict contracts for the prospective E26 Stage-3D numerical gate.

Stage-3D does not repair or supersede Stage-3C.  It binds the failed Stage-3C
result and tests admissibility under one fixed physical training layout.  This
module intentionally contains only protocol/receipt construction and
validation helpers; execution lives in the Stage-3D tools.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file


class Stage3DContractError(RuntimeError):
    """Raised when a Stage-3D lock or receipt violates the frozen contract."""


STAGE3D_GO = "STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE"
STAGE3D_BLOCKED = "STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY"
STAGE3D_NOT_EVALUABLE = "STAGE3D_NOT_EVALUABLE_IMPLEMENTATION_OR_EXECUTION_ERROR"
KNOWN_LAYOUT_SENSITIVITY = "KNOWN_BF16_AND_OPTIMIZER_LAYOUT_SENSITIVITY"
STAGE3C_DISPOSITION = "BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE"
REGISTERED_COMPILED_BACKEND_ID = "torch_compile_fixed_chunk_scan_v1"
RUNTIME_COMPILED_BACKEND_ALIAS = "compiled_scan"
STRICT_REFERENCE_BACKEND_ID = "reference_python"

_STAGE3C_RESULT_SHA256 = "83fab26e7936654b664653776d501c3fdee6cb7f0ffd78c3d9682ed41d319b56"
_STAGE3C_STATUS_SHA256 = "15b896a33e0fe286c80f2c204b7be2be0fbe6aaf8cdc512fafbd31040f8aabda"
_STAGE3C_RAW_AGGREGATE_SHA256 = "296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc"
_STAGE3C_FAILURE_STATUS_SHA256 = "dc7ed1837ccf022fe5110fdb44907c5e340391f0bcc5c92b7d5e26dcf2a95616"
_STAGE3C_RAW_FILE_COUNT = 11
_FP32_NONTRIVIAL_DEFINITION = (
    "PARTITION_VECTOR_NOT_EXACT_SINGLETON_PARTITION_LENGTH;PRIMARY_LOGITS_RUNTIME_"
    "STATE_GLOBAL_GRADIENT_REFERENCE_RECOMPUTED;WORST_LEAF_FINITE_DIAGNOSTIC_ONLY"
)

_PROTOCOL_VERSION = "catena-e26-stage3d-protocol-v1"
_RECEIPT_VERSION = "catena-e26-stage3d-fixed-layout-receipt-v1"
_BACKEND_VERSION = "catena-e26-stage3d-backend-manifest-v1"
_RESOURCE_VERSION = "catena-e26-stage3d-resource-preflight-v1"
_MANIFEST_TYPE = "E26_STAGE3D_FIXED_LAYOUT_RECEIPT"
_CANDIDATE_ORDER = ("d512_ctx4096", "d512_ctx2048", "d448_ctx4096")
_VARIANTS = ("projected_tied_delta_lm", "dual_delta_lm")
_STATE_CONTEXTS = ("zero_state", "prefilled_state")
_COMPARISON_NAMES = (
    "compiled_bf16_vs_reference_python_bf16",
    "reference_python_bf16_vs_reference_python_fp32",
)
_SHA_FIELDS = (
    "initialization_digest",
    "parameter_signature_sha256",
    "optimizer_state_signature_sha256",
    "token_ids_sha256",
    "data_cursor_sha256",
)
_G4_SHA_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_semantic_sha256",
    "rng_state_sha256",
    "data_ids_sha256",
    "data_cursor_sha256",
    "backend_graph_sha256",
    "backend_recipe_sha256",
    "optimizer_input_sha256",
    "initialization_digest",
    "parameter_signature_sha256",
    "initial_optimizer_state_signature_sha256",
)
_G4_CROSS_VARIANT_IDENTITY_FIELDS = (
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
_OPTIMIZER_TRUE_FIELDS = (
    "global_token_normalization_identity",
    "accumulation_buffer_reset_once",
    "gradient_clipping_after_accumulation",
    "adamw_step_and_bias_correction_identity",
    "weight_decay_order_and_value_identity",
    "skipped_optimizer_steps_zero",
    "all_gradients_finite",
)
_OPTIMIZER_TRACE_FIELDS = {
    *_OPTIMIZER_TRUE_FIELDS,
    "valid_prediction_tokens",
    "expected_valid_prediction_tokens",
    "exposed_input_tokens",
    "expected_input_tokens",
    "microbatch_count",
    "expected_microbatch_count",
    "execution_events",
    "adamw_state_steps",
    "passed",
}
_OPTIMIZER_EXECUTION_EVENTS = ("zero_grad", "gradient_clip", "adamw_step", "scheduler_step")
_STAGE3C_RAW_FILES = (
    "d448_ctx4096.log",
    "d448_ctx4096_numerical.json",
    "d448_ctx4096_worker_spec.json",
    "d512_ctx2048.log",
    "d512_ctx2048_numerical.json",
    "d512_ctx2048_worker_spec.json",
    "d512_ctx4096.log",
    "d512_ctx4096_numerical.json",
    "d512_ctx4096_worker_spec.json",
    "failure_status.json",
    "source_lock.json",
)
_STAGE3C_NUMERICAL_FILES = {
    "d448_ctx4096": "d448_ctx4096_numerical.json",
    "d512_ctx2048": "d512_ctx2048_numerical.json",
    "d512_ctx4096": "d512_ctx4096_numerical.json",
}
_STAGE3C_RAW_SHA256 = {
    "d448_ctx4096.log": "cd6bfb5bacc6808f99be1bbed00d1c043df9667cbfd0a5537febb5965a3c78ba",
    "d448_ctx4096_numerical.json": (
        "d6232e02dd52d6fe96220ec8f846ae2e710992bea40b1139e254046c6fb2c4de"
    ),
    "d448_ctx4096_worker_spec.json": (
        "d453dd4c53aae8396cd9f09c9c42ffd48a486baae5fd2c97711707fbbdf8a4a2"
    ),
    "d512_ctx2048.log": "cd6bfb5bacc6808f99be1bbed00d1c043df9667cbfd0a5537febb5965a3c78ba",
    "d512_ctx2048_numerical.json": (
        "63d6c364c475c8351ed24df4b331a5bcd84cabdd1e48895e3c52e22545c4e8e2"
    ),
    "d512_ctx2048_worker_spec.json": (
        "259af9c632779493ff308cc14c213056f983fdb964ed7633aef6b786ee5adddc"
    ),
    "d512_ctx4096.log": "cd6bfb5bacc6808f99be1bbed00d1c043df9667cbfd0a5537febb5965a3c78ba",
    "d512_ctx4096_numerical.json": (
        "b9ce170be1766909cfe0624116c1305b7be345ca72676445045bbe7a85e3f55d"
    ),
    "d512_ctx4096_worker_spec.json": (
        "6a60adc88607e515a36796661d60855f35c4834fc76edd91963840cc2e9277f8"
    ),
    "failure_status.json": _STAGE3C_FAILURE_STATUS_SHA256,
    "source_lock.json": "c0107b3e836b3579dd89e747711da61fdd0a406dcc96a468f649b830fada6f32",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise Stage3DContractError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _require_git_commit(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise Stage3DContractError(f"{label} must be a full lowercase 40-character Git SHA")
    return value


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise Stage3DContractError(f"{label} must be {expected}")


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage3DContractError(f"{label} must be an integer >= {minimum}")
    return value


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Stage3DContractError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage3DContractError(f"{label} does not exist: {candidate}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise Stage3DContractError(f"{label} must be a regular file: {resolved}")
    return resolved


def bind_file(path: str | Path, label: str = "input") -> dict[str, str]:
    """Return a path+byte-SHA binding for an immutable Stage-3D input."""

    resolved = _regular_file(path, label)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_binding(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise Stage3DContractError(f"{label} must be an exact path/SHA binding")
    observed = bind_file(str(value["path"]), label)
    if observed != dict(value):
        raise Stage3DContractError(f"{label} bytes or path changed")
    return observed


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise Stage3DContractError(f"Cannot parse {label}: {source}") from error
    if not isinstance(payload, dict):
        raise Stage3DContractError(f"{label} must contain a mapping")
    return payload


def _artifact_row_aggregate(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def _validate_registered_stage3c_artifact_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = _regular_file(path, "Stage-3C artifact hash manifest")
    payload = _load_mapping(manifest_path, "Stage-3C artifact hash manifest")
    claimed = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    if claimed != sha256_canonical_json(unhashed):
        raise Stage3DContractError("Stage-3C artifact manifest canonical hash changed")
    if (
        payload.get("schema_version") != "catena-e26-stage3c-artifact-hash-manifest-v1"
        or payload.get("manifest_type") != "E26_STAGE3C_ARTIFACT_HASH_MANIFEST"
        or payload.get("scientific_evidence") is not False
        or payload.get("predecessor_disposition") != STAGE3C_DISPOSITION
        or payload.get("predecessor_mutated") is not False
        or payload.get("aggregate_algorithm") != "path_nul_bytes_nul_sha256_newline_v1"
    ):
        raise Stage3DContractError("Stage-3C artifact manifest contract changed")
    registered = payload.get("registered_predecessor")
    expected_registered = {
        "result": {
            "path": str(Path("docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_RESULT_KO.md").resolve()),
            "sha256": _STAGE3C_RESULT_SHA256,
        },
        "status": {
            "path": str(Path("docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_STATUS.json").resolve()),
            "sha256": _STAGE3C_STATUS_SHA256,
        },
        "raw_run_aggregate_sha256": _STAGE3C_RAW_AGGREGATE_SHA256,
        "failure_status_sha256": _STAGE3C_FAILURE_STATUS_SHA256,
        "disposition": STAGE3C_DISPOSITION,
    }
    if registered != expected_registered:
        raise Stage3DContractError("Stage-3C registered predecessor anchors changed")
    _verify_binding(registered["result"], "registered Stage-3C result")
    status_binding = _verify_binding(registered["status"], "registered Stage-3C status")
    status = _load_mapping(status_binding["path"], "registered Stage-3C status")
    if (
        status.get("stage3c_disposition") != STAGE3C_DISPOSITION
        or status.get("raw_run_aggregate_sha256") != _STAGE3C_RAW_AGGREGATE_SHA256
        or status.get("failure_status_sha256") != _STAGE3C_FAILURE_STATUS_SHA256
        or status.get("scientific_e26a_started") is not False
        or status.get("restart_audit_started") is not False
        or status.get("resource_preflight_started") is not False
    ):
        raise Stage3DContractError("Registered Stage-3C status contents changed")

    root_raw = payload.get("artifact_root")
    if not isinstance(root_raw, str):
        raise Stage3DContractError("Stage-3C artifact manifest lacks artifact_root")
    root = Path(root_raw).expanduser()
    if root.is_symlink():
        raise Stage3DContractError("Stage-3C artifact root must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise Stage3DContractError("Stage-3C artifact root is not a directory")
    raw_rows = payload.get("files")
    if not isinstance(raw_rows, list) or payload.get("file_count") != _STAGE3C_RAW_FILE_COUNT:
        raise Stage3DContractError("Stage-3C raw artifact count changed")
    if len(raw_rows) != _STAGE3C_RAW_FILE_COUNT:
        raise Stage3DContractError("Stage-3C raw artifact row count changed")
    rows: list[dict[str, Any]] = []
    bound_paths: dict[str, Path] = {}
    for expected_name, raw in zip(_STAGE3C_RAW_FILES, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("path") != expected_name:
            raise Stage3DContractError("Stage-3C raw artifact order/path changed")
        candidate = (root / expected_name).resolve(strict=True)
        if not candidate.is_relative_to(root) or candidate.is_symlink() or not candidate.is_file():
            raise Stage3DContractError(f"Invalid Stage-3C raw artifact: {expected_name}")
        observed = {
            "path": expected_name,
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        if observed != dict(raw):
            raise Stage3DContractError(f"Stage-3C raw artifact changed: {expected_name}")
        if observed["sha256"] != _STAGE3C_RAW_SHA256[expected_name]:
            raise Stage3DContractError(
                f"Stage-3C raw artifact differs from registered bytes: {expected_name}"
            )
        rows.append(observed)
        bound_paths[expected_name] = candidate
    aggregate = _artifact_row_aggregate(rows)
    if (
        payload.get("aggregate_sha256") != aggregate
        or bound_paths["failure_status.json"] is None
        or sha256_file(bound_paths["failure_status.json"]) != _STAGE3C_FAILURE_STATUS_SHA256
    ):
        raise Stage3DContractError("Stage-3C raw artifact rehash changed")
    return payload, bound_paths


def _expected_layouts() -> list[dict[str, int | str]]:
    return [
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


def load_stage3d_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the prospective Stage-3D YAML."""

    payload = _load_mapping(path, "Stage-3D config")
    required_scalars = {
        "schema_version": "catena-v8.1",
        "stage_id": "E26_STAGE3D",
        "name": "FIXED_PHYSICAL_LAYOUT_BF16_ADMISSIBILITY",
        "status": "PROSPECTIVE_DRAFT",
        "run_mode": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "FIXED_LAYOUT_NUMERICAL_ADMISSIBILITY_ONLY",
    }
    for key, expected in required_scalars.items():
        if payload.get(key) != expected:
            raise Stage3DContractError(f"Stage-3D config changed registered {key}")

    predecessor = payload.get("predecessor")
    if not isinstance(predecessor, Mapping):
        raise Stage3DContractError("Stage-3D config lacks predecessor contract")
    if (
        predecessor.get("stage_id") != "E26_STAGE3C"
        or predecessor.get("required_disposition") != STAGE3C_DISPOSITION
        or predecessor.get("diagnostic_disposition") != KNOWN_LAYOUT_SENSITIVITY
        or predecessor.get("immutable") is not True
        or predecessor.get("result_sha256") != _STAGE3C_RESULT_SHA256
        or predecessor.get("status_sha256") != _STAGE3C_STATUS_SHA256
        or predecessor.get("raw_registered_file_count") != _STAGE3C_RAW_FILE_COUNT
        or predecessor.get("raw_registered_aggregate_sha256") != _STAGE3C_RAW_AGGREGATE_SHA256
        or predecessor.get("failure_status_sha256") != _STAGE3C_FAILURE_STATUS_SHA256
        or predecessor.get("raw_registered_files_sha256") != _STAGE3C_RAW_SHA256
    ):
        raise Stage3DContractError("Stage-3C predecessor disposition was changed")

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise Stage3DContractError("Stage-3D config lacks inherited thresholds")
    expected_thresholds: dict[str, float | str] = {
        "source": "INHERIT_EXACTLY_FROM_STAGE3C_PROTOCOL_LOCK",
        "fp32_relative_l2_max": 1.0e-5,
        "fp32_max_abs_max": 1.0e-5,
        "bf16_relative_l2_max": 7.0e-3,
        "gradient_norm_min": 1.0e-8,
        "gradient_norm_max": 1.0e3,
        "nontrivial_row_definition": "INHERIT_EXACTLY_FROM_STAGE3C_PROTOCOL_LOCK",
    }
    for key, expected in expected_thresholds.items():
        observed = thresholds.get(key)
        if isinstance(expected, float):
            if isinstance(observed, (int, float, str)) and not isinstance(observed, bool):
                equal = float(observed) == expected
            else:
                equal = False
        else:
            equal = observed == expected
        if not equal:
            raise Stage3DContractError(f"Stage-3D threshold {key} was changed")

    if tuple(payload.get("variants", ())) != _VARIANTS:
        raise Stage3DContractError("Stage-3D variant order or coverage changed")
    expected_determinism = {
        "initialization_seed": 260_301,
        "g3_data_seed": 260_701,
        "g4_data_seed": 260_801,
        "prefill_seed": 260_901,
        "prefill_length": 17,
    }
    if payload.get("determinism") != expected_determinism:
        raise Stage3DContractError("Stage-3D deterministic probe contract changed")
    fixed = payload.get("fixed_physical_layout")
    if not isinstance(fixed, Mapping):
        raise Stage3DContractError("Stage-3D fixed physical layout is missing")
    if (
        fixed.get("target_global_input_tokens") != 65_536
        or fixed.get("microbatch_sequences") != 1
        or fixed.get("autocast_precision") != "bf16"
        or fixed.get("optimizer_precision") != "fp32"
        or fixed.get("metrics_precision") != "fp32"
        or fixed.get("semantic_padding") is not False
        or fixed.get("variant_specific_layout_allowed") is not False
        or fixed.get("variant_specific_precision_allowed") is not False
    ):
        raise Stage3DContractError("Stage-3D physical-layout identity was changed")
    expected_candidates = [
        {
            key: value
            for key, value in layout.items()
            if key
            not in {
                "microbatch_sequences",
                "target_global_input_tokens",
            }
        }
        for layout in _expected_layouts()
    ]
    if fixed.get("candidates") != expected_candidates:
        raise Stage3DContractError("Stage-3D candidate layouts changed")

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise Stage3DContractError("Stage-3D gates are missing")
    g3 = gates.get("g3_fixed_layout_bf16_admissibility")
    g4 = gates.get("g4_same_layout_replay")
    if not isinstance(g3, Mapping) or not isinstance(g4, Mapping):
        raise Stage3DContractError("Stage-3D G3/G4 contracts are missing")
    if (
        tuple(g3.get("state_contexts", ())) != _STATE_CONTEXTS
        or g3.get("compiled_backend") != REGISTERED_COMPILED_BACKEND_ID
        or g3.get("strict_reference_backend") != STRICT_REFERENCE_BACKEND_ID
        or g3.get("required_case_count") != 12
        or tuple(g3.get("comparisons_per_case", ())) != _COMPARISON_NAMES
        or g4.get("required_case_count") != 6
        or g4.get("replay_count_per_case") != 2
    ):
        raise Stage3DContractError("Stage-3D G3/G4 coverage changed")
    dispositions = payload.get("registered_dispositions")
    if dispositions != {
        "go": STAGE3D_GO,
        "blocked": STAGE3D_BLOCKED,
        "not_evaluable": STAGE3D_NOT_EVALUABLE,
    }:
        raise Stage3DContractError("Stage-3D dispositions changed")
    stop = payload.get("stop_policy")
    if not isinstance(stop, Mapping) or any(
        stop.get(key) is not False
        for key in ("execute_scientific_e26a", "execute_e26b", "execute_e26c_or_later")
    ):
        raise Stage3DContractError("Stage-3D stop policy opens a scientific run")
    return deepcopy(payload)


def fixed_layouts_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical candidate layouts after validating the config."""

    # Validate from bytes when possible; otherwise require the same invariants
    # on the supplied mapping by a compact in-memory check below.
    if config.get("stage_id") != "E26_STAGE3D":
        raise Stage3DContractError("Not a Stage-3D config")
    fixed = config.get("fixed_physical_layout")
    if not isinstance(fixed, Mapping):
        raise Stage3DContractError("Stage-3D fixed physical layout is missing")
    layouts: list[dict[str, Any]] = []
    target = _require_int(
        fixed.get("target_global_input_tokens"),
        "target_global_input_tokens",
        minimum=1,
    )
    microbatch = _require_int(
        fixed.get("microbatch_sequences"),
        "microbatch_sequences",
        minimum=1,
    )
    for candidate in fixed.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            raise Stage3DContractError("Candidate layout must be a mapping")
        row = dict(candidate)
        row["microbatch_sequences"] = microbatch
        row["target_global_input_tokens"] = target
        layouts.append(row)
    if layouts != _expected_layouts():
        raise Stage3DContractError("Stage-3D fixed layouts are not canonical")
    return layouts


def _stage3c_thresholds(stage3c_protocol: Mapping[str, Any]) -> dict[str, float]:
    snapshot = stage3c_protocol.get("full_config_snapshot")
    if not isinstance(snapshot, Mapping):
        raise Stage3DContractError("Stage-3C protocol lacks full_config_snapshot")
    gates = snapshot.get("backend_gates")
    if not isinstance(gates, Mapping):
        raise Stage3DContractError("Stage-3C protocol lacks backend_gates")
    result = {
        "fp32_relative_l2_max": float(gates["fp32_full_chunk_relative_l2_max"]),
        "fp32_max_abs_max": float(gates["fp32_full_chunk_max_abs_max"]),
        "bf16_relative_l2_max": float(gates["bf16_fp32_relative_l2_max"]),
        "gradient_norm_min": float(gates["gradient_norm_min"]),
        "gradient_norm_max": float(gates["gradient_norm_max"]),
    }
    expected = {
        "fp32_relative_l2_max": 1.0e-5,
        "fp32_max_abs_max": 1.0e-5,
        "bf16_relative_l2_max": 7.0e-3,
        "gradient_norm_min": 1.0e-8,
        "gradient_norm_max": 1.0e3,
    }
    if result != expected:
        raise Stage3DContractError("Stage-3C numerical tolerances differ from frozen values")
    return result


def build_stage3d_protocol_lock(
    *,
    config_path: str | Path,
    stage3c_result_path: str | Path,
    stage3c_protocol_path: str | Path,
    stage3c_artifact_manifest_path: str | Path,
    frozen_e00_e25_receipt_path: str | Path,
    source_commit: str,
    source_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a prospective lock which binds, but never reclassifies, Stage-3C."""

    config = load_stage3d_config(config_path)
    result_file = _regular_file(stage3c_result_path, "Stage-3C result")
    if sha256_file(result_file) != _STAGE3C_RESULT_SHA256:
        raise Stage3DContractError("Registered Stage-3C result bytes changed")
    result_text = result_file.read_text(encoding="utf-8")
    for required in (
        STAGE3C_DISPOSITION,
        "scientific_e26a_started: false",
        "restart_audit_started: false",
        "resource_preflight_started: false",
    ):
        if required not in result_text:
            raise Stage3DContractError(f"Stage-3C result lacks immutable marker: {required}")
    stage3c_protocol = _load_mapping(stage3c_protocol_path, "Stage-3C protocol lock")
    thresholds = _stage3c_thresholds(stage3c_protocol)
    config_thresholds = config["thresholds"]
    if any(float(config_thresholds[key]) != value for key, value in thresholds.items()):
        raise Stage3DContractError("Stage-3D does not exactly inherit Stage-3C tolerances")
    artifact_manifest, artifact_paths = _validate_registered_stage3c_artifact_manifest(
        stage3c_artifact_manifest_path
    )
    registered = artifact_manifest["registered_predecessor"]
    if registered["result"]["path"] != str(result_file):
        raise Stage3DContractError("Supplied Stage-3C result differs from registered anchor")

    frozen = _load_mapping(frozen_e00_e25_receipt_path, "E00-E25 frozen receipt")
    if (
        frozen.get("manifest_type") != "E26_FROZEN_INVARIANCE_RECEIPT"
        or frozen.get("passed") is not True
        or frozen.get("scientific_evidence") is not False
    ):
        raise Stage3DContractError("E00-E25 frozen invariance is not valid")
    _require_git_commit(source_commit, "Stage-3D source commit")
    source_tree_sha = _require_sha256(
        source_inventory.get("source_tree_sha256"),
        "Stage-3D source inventory SHA",
    )

    payload: dict[str, Any] = {
        "schema_version": _PROTOCOL_VERSION,
        "manifest_type": "E26_STAGE3D_PROTOCOL_LOCK",
        "stage_id": "E26_STAGE3D",
        "status": "PROSPECTIVE_LOCKED",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "FIXED_LAYOUT_NUMERICAL_ADMISSIBILITY_ONLY",
        "config": bind_file(config_path, "Stage-3D config"),
        "source": {
            "git_commit": source_commit,
            "source_tree_sha256": source_tree_sha,
            "source_file_count": _require_int(
                source_inventory.get("files"), "source inventory file count", minimum=1
            ),
        },
        "stage3c": {
            "stage_id": "E26_STAGE3C",
            "disposition": STAGE3C_DISPOSITION,
            "immutable": True,
            "result": bind_file(result_file, "Stage-3C result"),
            "status": deepcopy(registered["status"]),
            "protocol": bind_file(stage3c_protocol_path, "Stage-3C protocol lock"),
            "artifact_hash_manifest": bind_file(
                stage3c_artifact_manifest_path, "Stage-3C artifact hash manifest"
            ),
            "artifact_manifest_internal_sha256": artifact_manifest["manifest_sha256"],
            "artifact_root": artifact_manifest["artifact_root"],
            "raw_registered_file_count": _STAGE3C_RAW_FILE_COUNT,
            "raw_registered_aggregate_sha256": _STAGE3C_RAW_AGGREGATE_SHA256,
            "raw_registered_files_sha256": deepcopy(_STAGE3C_RAW_SHA256),
            "artifact_manifest_rehash_aggregate_sha256": artifact_manifest["aggregate_sha256"],
            "failure_status": bind_file(
                artifact_paths["failure_status.json"], "Stage-3C failure status"
            ),
            "diagnostic_disposition": KNOWN_LAYOUT_SENSITIVITY,
            "fixed_probe_reference_mismatch": "SEPARATE_STAGE3D_G3_HARD_GATE",
        },
        "frozen_e00_e25": bind_file(frozen_e00_e25_receipt_path, "E00-E25 frozen receipt"),
        "inherited_thresholds": thresholds,
        "fp32_nontrivial_row_contract": {
            "definition": _FP32_NONTRIVIAL_DEFINITION,
            "reports_required": 12,
            "rows_per_report": 12,
            "nontrivial_rows_per_report": 11,
            "nontrivial_rows_required": 132,
            "derive_from_raw_artifacts": True,
            "trust_precomputed_pass_flag_only": False,
            "gated_tensor_fields": [
                "logits",
                "runtime_state.recurrent",
                "runtime_state.attention_key",
                "runtime_state.attention_value",
                "gradients_global_tree",
                "reference_comparisons",
            ],
            "diagnostic_only_fields": ["gradients_worst_leaf"],
        },
        "fixed_layouts": fixed_layouts_from_config(config),
        "variants": list(_VARIANTS),
        "state_contexts": list(_STATE_CONTEXTS),
        "determinism": deepcopy(config["determinism"]),
        "g3_required_case_count": 12,
        "g4_required_replay_count": 6,
        "backend_binding": {
            "registered_backend_id": REGISTERED_COMPILED_BACKEND_ID,
            "runtime_backend_alias": RUNTIME_COMPILED_BACKEND_ALIAS,
            "strict_reference_backend_id": STRICT_REFERENCE_BACKEND_ID,
        },
        "strict_reference_backend": STRICT_REFERENCE_BACKEND_ID,
        "research_contract": deepcopy(config["research_contract"]),
        "stop_policy": deepcopy(config["stop_policy"]),
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
    }
    payload["protocol_sha256"] = sha256_canonical_json(payload)
    return payload


def _validate_protocol_semantics(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != _PROTOCOL_VERSION
        or payload.get("manifest_type") != "E26_STAGE3D_PROTOCOL_LOCK"
        or payload.get("stage_id") != "E26_STAGE3D"
        or payload.get("status") != "PROSPECTIVE_LOCKED"
    ):
        raise Stage3DContractError("Not a locked Stage-3D protocol")
    _require_bool(payload.get("scientific_evidence"), False, "scientific_evidence")
    for key in ("main_test_opened", "scientific_e26a_started", "scientific_main_started"):
        _require_bool(payload.get(key), False, key)
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise Stage3DContractError("Stage-3D protocol lacks source binding")
    _require_git_commit(source.get("git_commit"), "source git commit")
    _require_sha256(source.get("source_tree_sha256"), "source tree SHA")
    _require_int(source.get("source_file_count"), "source file count", minimum=1)
    stage3c = payload.get("stage3c")
    if not isinstance(stage3c, Mapping):
        raise Stage3DContractError("Stage-3D protocol lacks Stage-3C binding")
    if (
        stage3c.get("disposition") != STAGE3C_DISPOSITION
        or stage3c.get("immutable") is not True
        or stage3c.get("diagnostic_disposition") != KNOWN_LAYOUT_SENSITIVITY
        or stage3c.get("fixed_probe_reference_mismatch") != "SEPARATE_STAGE3D_G3_HARD_GATE"
    ):
        raise Stage3DContractError("Stage-3C was relabelled or merged with Stage-3D G3")
    for field in ("result", "status", "protocol", "artifact_hash_manifest", "failure_status"):
        _verify_binding(stage3c.get(field), f"Stage-3C {field}")
    if (
        stage3c["result"]["sha256"] != _STAGE3C_RESULT_SHA256
        or stage3c["status"]["sha256"] != _STAGE3C_STATUS_SHA256
        or stage3c["failure_status"]["sha256"] != _STAGE3C_FAILURE_STATUS_SHA256
        or stage3c.get("raw_registered_file_count") != _STAGE3C_RAW_FILE_COUNT
        or stage3c.get("raw_registered_aggregate_sha256") != _STAGE3C_RAW_AGGREGATE_SHA256
        or stage3c.get("raw_registered_files_sha256") != _STAGE3C_RAW_SHA256
    ):
        raise Stage3DContractError("Stage-3C registered hash anchors changed")
    manifest, _paths = _validate_registered_stage3c_artifact_manifest(
        stage3c["artifact_hash_manifest"]["path"]
    )
    if (
        stage3c.get("artifact_manifest_internal_sha256") != manifest["manifest_sha256"]
        or stage3c.get("artifact_root") != manifest["artifact_root"]
        or stage3c.get("artifact_manifest_rehash_aggregate_sha256") != manifest["aggregate_sha256"]
    ):
        raise Stage3DContractError("Stage-3C artifact manifest internal binding changed")
    _verify_binding(payload.get("frozen_e00_e25"), "E00-E25 frozen receipt")
    config_binding = _verify_binding(payload.get("config"), "Stage-3D config")
    config = load_stage3d_config(config_binding["path"])
    if payload.get("fixed_layouts") != fixed_layouts_from_config(config):
        raise Stage3DContractError("Stage-3D protocol fixed layouts changed")
    if tuple(payload.get("variants", ())) != _VARIANTS:
        raise Stage3DContractError("Stage-3D protocol variants changed")
    if tuple(payload.get("state_contexts", ())) != _STATE_CONTEXTS:
        raise Stage3DContractError("Stage-3D state contexts changed")
    if payload.get("determinism") != {
        "initialization_seed": 260_301,
        "g3_data_seed": 260_701,
        "g4_data_seed": 260_801,
        "prefill_seed": 260_901,
        "prefill_length": 17,
    }:
        raise Stage3DContractError("Stage-3D deterministic probe contract changed")
    if (
        payload.get("g3_required_case_count") != 12
        or payload.get("g4_required_replay_count") != 6
        or payload.get("strict_reference_backend") != STRICT_REFERENCE_BACKEND_ID
    ):
        raise Stage3DContractError("Stage-3D coverage/reference contract changed")
    backend_binding = payload.get("backend_binding")
    expected_backend_binding = {
        "registered_backend_id": REGISTERED_COMPILED_BACKEND_ID,
        "runtime_backend_alias": RUNTIME_COMPILED_BACKEND_ALIAS,
        "strict_reference_backend_id": STRICT_REFERENCE_BACKEND_ID,
    }
    if backend_binding != expected_backend_binding:
        raise Stage3DContractError("Stage-3D registered/runtime backend binding changed")
    config_gates = config.get("gates")
    if not isinstance(config_gates, Mapping):
        raise Stage3DContractError("Stage-3D config gates must be a mapping")
    configured_g3 = config_gates.get("g3_fixed_layout_bf16_admissibility")
    if not isinstance(configured_g3, Mapping) or (
        configured_g3.get("compiled_backend") != backend_binding["registered_backend_id"]
        or configured_g3.get("strict_reference_backend")
        != backend_binding["strict_reference_backend_id"]
    ):
        raise Stage3DContractError("Stage-3D backend binding differs from config")
    if payload.get("inherited_thresholds") != {
        "fp32_relative_l2_max": 1.0e-5,
        "fp32_max_abs_max": 1.0e-5,
        "bf16_relative_l2_max": 7.0e-3,
        "gradient_norm_min": 1.0e-8,
        "gradient_norm_max": 1.0e3,
    }:
        raise Stage3DContractError("Stage-3D inherited thresholds changed")
    if payload.get("fp32_nontrivial_row_contract") != {
        "definition": _FP32_NONTRIVIAL_DEFINITION,
        "reports_required": 12,
        "rows_per_report": 12,
        "nontrivial_rows_per_report": 11,
        "nontrivial_rows_required": 132,
        "derive_from_raw_artifacts": True,
        "trust_precomputed_pass_flag_only": False,
        "gated_tensor_fields": [
            "logits",
            "runtime_state.recurrent",
            "runtime_state.attention_key",
            "runtime_state.attention_value",
            "gradients_global_tree",
            "reference_comparisons",
        ],
        "diagnostic_only_fields": ["gradients_worst_leaf"],
    }:
        raise Stage3DContractError("Stage-3D FP32 nontrivial-row contract changed")
    stop = payload.get("stop_policy")
    if not isinstance(stop, Mapping) or any(
        stop.get(key) is not False
        for key in ("execute_scientific_e26a", "execute_e26b", "execute_e26c_or_later")
    ):
        raise Stage3DContractError("Stage-3D protocol opens scientific execution")


def validate_stage3d_protocol_lock(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate protocol self-hash, immutable bindings, and exact semantics."""

    normalized = deepcopy(dict(payload))
    observed = normalized.pop("protocol_sha256", None)
    if observed != sha256_canonical_json(normalized):
        raise Stage3DContractError("Stage-3D protocol canonical hash changed")
    normalized["protocol_sha256"] = observed
    _validate_protocol_semantics(normalized)
    if (
        config_path is not None
        and bind_file(config_path, "supplied Stage-3D config") != normalized["config"]
    ):
        raise Stage3DContractError("Validator was supplied a different Stage-3D config")
    return normalized


def _layout_by_candidate(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    layouts = protocol.get("fixed_layouts")
    if not isinstance(layouts, list):
        raise Stage3DContractError("Stage-3D protocol lacks fixed layouts")
    return {str(row["candidate_id"]): dict(row) for row in layouts}


def _validate_tensor_error(
    value: Any,
    label: str,
    *,
    relative_max: float,
    max_abs_max: float | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"relative_l2", "max_abs"}:
        raise Stage3DContractError(f"{label} must contain relative_l2 and max_abs")
    try:
        relative = float(value["relative_l2"])
        maximum = float(value["max_abs"])
    except (TypeError, ValueError) as error:
        raise Stage3DContractError(f"{label} contains a non-numeric error") from error
    if relative < 0.0 or maximum < 0.0 or not math.isfinite(relative) or not math.isfinite(maximum):
        raise Stage3DContractError(f"{label} contains a non-finite/negative error")
    if relative > relative_max:
        raise Stage3DContractError(f"{label} exceeds inherited BF16 tolerance")
    if max_abs_max is not None and maximum > max_abs_max:
        raise Stage3DContractError(f"{label} exceeds inherited maximum-absolute tolerance")


def _validate_tensor_diagnostic(value: Any, label: str) -> None:
    """Validate an ungated diagnostic tensor error without inventing a threshold."""

    if not isinstance(value, Mapping) or set(value) != {"relative_l2", "max_abs"}:
        raise Stage3DContractError(f"{label} must contain relative_l2 and max_abs")
    try:
        relative = float(value["relative_l2"])
        maximum = float(value["max_abs"])
    except (TypeError, ValueError) as error:
        raise Stage3DContractError(f"{label} contains a non-numeric diagnostic") from error
    if relative < 0.0 or maximum < 0.0 or not math.isfinite(relative) or not math.isfinite(maximum):
        raise Stage3DContractError(f"{label} contains a non-finite/negative diagnostic")


def _validate_comparison(value: Any, label: str, *, relative_max: float) -> None:
    if not isinstance(value, Mapping):
        raise Stage3DContractError(f"{label} must be a mapping")
    for field in ("logits", "runtime_state", "gradients"):
        _validate_tensor_error(value.get(field), f"{label}.{field}", relative_max=relative_max)
    _require_bool(value.get("passed"), True, f"{label}.passed")


def _validate_fixed_layout(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise Stage3DContractError(f"{label} differs from the prospective fixed layout")


def _validate_g3_cases(cases: Any, protocol: Mapping[str, Any]) -> None:
    if not isinstance(cases, list) or len(cases) != 12:
        raise Stage3DContractError("Stage-3D requires exactly 12 G3 cases")
    layouts = _layout_by_candidate(protocol)
    expected_keys = {
        (candidate, variant, context)
        for candidate in _CANDIDATE_ORDER
        for variant in _VARIANTS
        for context in _STATE_CONTEXTS
    }
    observed_keys: set[tuple[str, str, str]] = set()
    tolerance = float(protocol["inherited_thresholds"]["bf16_relative_l2_max"])
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            raise Stage3DContractError(f"G3 case {index} must be a mapping")
        key = (str(raw.get("candidate_id")), str(raw.get("variant")), str(raw.get("state_context")))
        if key in observed_keys:
            raise Stage3DContractError(f"Duplicate G3 case: {key}")
        observed_keys.add(key)
        if key[0] not in layouts:
            raise Stage3DContractError(f"Unknown G3 candidate: {key[0]}")
        _validate_fixed_layout(raw.get("fixed_layout"), layouts[key[0]], f"G3 case {key}")
        for field in _SHA_FIELDS:
            _require_sha256(raw.get(field), f"G3 case {key}.{field}")
        _require_bool(raw.get("layout_identity_passed"), True, f"G3 case {key}.layout")
        comparisons = raw.get("comparisons")
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(_COMPARISON_NAMES):
            raise Stage3DContractError(f"G3 case {key} comparisons changed")
        for name in _COMPARISON_NAMES:
            _validate_comparison(comparisons[name], f"G3 case {key}.{name}", relative_max=tolerance)
        for field in ("gradient_finite", "state_metadata_exact", "clone_no_alias"):
            _require_bool(raw.get(field), True, f"G3 case {key}.{field}")
        for field in (
            "graph_break_count",
            "fallback_count",
            "variant_specific_fp32_path_count",
            "variant_specific_padding_count",
        ):
            if _require_int(raw.get(field), f"G3 case {key}.{field}") != 0:
                raise Stage3DContractError(f"G3 case {key}.{field} must be zero")
        backend = raw.get("backend_integrity")
        if not isinstance(backend, Mapping):
            raise Stage3DContractError(f"G3 case {key} lacks backend integrity")
        if (
            backend.get("optimized_backend_id") != RUNTIME_COMPILED_BACKEND_ALIAS
            or backend.get("registered_backend_id") != REGISTERED_COMPILED_BACKEND_ID
            or backend.get("runtime_backend_alias") != RUNTIME_COMPILED_BACKEND_ALIAS
            or backend.get("backend_alias_matches_registration") is not True
            or backend.get("reference_backend_id") != STRICT_REFERENCE_BACKEND_ID
            or backend.get("strict_reference_python") is not True
            or backend.get("positive_compiled_execution") is not True
            or backend.get("python_token_loop_at_scientific_runtime") is not False
            or backend.get("graph_break_count") != 0
            or backend.get("fallback_count") != 0
            or backend.get("variant_specific_fp32_path_count") != 0
            or backend.get("variant_specific_padding_count") != 0
            or backend.get("passed") is not True
        ):
            raise Stage3DContractError(f"G3 case {key} backend integrity failed")
        _require_bool(raw.get("passed"), True, f"G3 case {key}.passed")
    if observed_keys != expected_keys:
        raise Stage3DContractError("G3 candidate/variant/state coverage is incomplete")


def _validate_g4_replays(rows: Any, protocol: Mapping[str, Any]) -> None:
    if not isinstance(rows, list) or len(rows) != 6:
        raise Stage3DContractError("Stage-3D requires exactly six G4 replay cases")
    layouts = _layout_by_candidate(protocol)
    expected_keys = {
        (candidate, variant) for candidate in _CANDIDATE_ORDER for variant in _VARIANTS
    }
    observed_keys: set[tuple[str, str]] = set()
    tolerance = float(protocol["inherited_thresholds"]["bf16_relative_l2_max"])
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise Stage3DContractError(f"G4 replay {index} must be a mapping")
        key = (str(raw.get("candidate_id")), str(raw.get("variant")))
        if key in observed_keys:
            raise Stage3DContractError(f"Duplicate G4 replay: {key}")
        observed_keys.add(key)
        if key[0] not in layouts:
            raise Stage3DContractError(f"Unknown G4 candidate: {key[0]}")
        _validate_fixed_layout(raw.get("fixed_layout"), layouts[key[0]], f"G4 replay {key}")
        for field in _G4_SHA_FIELDS:
            _require_sha256(raw.get(field), f"G4 replay {key}.{field}")
        comparison = raw.get("comparison")
        _validate_comparison(comparison, f"G4 replay {key}.comparison", relative_max=tolerance)
        optimizer = raw.get("optimizer_state")
        _validate_tensor_error(
            optimizer, f"G4 replay {key}.optimizer_state", relative_max=tolerance
        )
        for field in (
            "gradients_finite",
            "state_metadata_exact",
            "clone_no_alias",
            "optimizer_integrity_passed",
        ):
            _require_bool(raw.get(field), True, f"G4 replay {key}.{field}")
        for field in ("graph_break_count", "fallback_count"):
            if _require_int(raw.get(field), f"G4 replay {key}.{field}") != 0:
                raise Stage3DContractError(f"G4 replay {key}.{field} must be zero")
        _require_bool(raw.get("passed"), True, f"G4 replay {key}.passed")
    if observed_keys != expected_keys:
        raise Stage3DContractError("G4 candidate/variant coverage is incomplete")
    if not _g4_cross_variant_identity_passed(rows):
        raise Stage3DContractError(
            "G4 tied/dual rows do not share the registered physical replay inputs"
        )
    if not _g5_optimizer_integrity_passed(rows, protocol):
        raise Stage3DContractError("G5 optimizer trace/state integrity failed")


def _g1_variant_identity_passed(cases: Sequence[Mapping[str, Any]]) -> bool:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in cases:
        key = (str(row.get("candidate_id")), str(row.get("state_context")))
        by_key.setdefault(key, []).append(row)
    if len(by_key) != len(_CANDIDATE_ORDER) * len(_STATE_CONTEXTS):
        return False
    for rows in by_key.values():
        if {str(row.get("variant")) for row in rows} != set(_VARIANTS):
            return False
        if not all(row.get("layout_identity_passed") is True for row in rows):
            return False
        for field in (*_SHA_FIELDS, "fixed_layout"):
            if len({sha256_canonical_json(row.get(field)) for row in rows}) != 1:
                return False
    return True


def _g6_backend_integrity_passed(
    cases: Sequence[Mapping[str, Any]], replays: Sequence[Mapping[str, Any]]
) -> bool:
    base = all(
        row.get("graph_break_count") == 0 and row.get("fallback_count") == 0
        for row in (*cases, *replays)
    )
    if not base:
        return False
    for row in cases:
        backend = row.get("backend_integrity")
        if not isinstance(backend, Mapping):
            return False
        if not (
            row.get("variant_specific_fp32_path_count") == 0
            and row.get("variant_specific_padding_count") == 0
            and backend.get("optimized_backend_id") == RUNTIME_COMPILED_BACKEND_ALIAS
            and backend.get("registered_backend_id") == REGISTERED_COMPILED_BACKEND_ID
            and backend.get("runtime_backend_alias") == RUNTIME_COMPILED_BACKEND_ALIAS
            and backend.get("backend_alias_matches_registration") is True
            and backend.get("reference_backend_id") == STRICT_REFERENCE_BACKEND_ID
            and backend.get("strict_reference_python") is True
            and backend.get("positive_compiled_execution") is True
            and backend.get("python_token_loop_at_scientific_runtime") is False
            and backend.get("graph_break_count") == 0
            and backend.get("fallback_count") == 0
            and backend.get("variant_specific_fp32_path_count") == 0
            and backend.get("variant_specific_padding_count") == 0
            and backend.get("passed") is True
        ):
            return False
    return True


def _fixed_probe_reference_mismatch(cases: Sequence[Mapping[str, Any]]) -> bool:
    for row in cases:
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, Mapping):
            continue
        if any(
            not isinstance(comparisons.get(name), Mapping)
            or comparisons[name].get("passed") is not True
            for name in _COMPARISON_NAMES
        ):
            return True
    return False


def _exact_g3_coverage(cases: Sequence[Mapping[str, Any]]) -> bool:
    expected = {
        (candidate, variant, state)
        for candidate in _CANDIDATE_ORDER
        for variant in _VARIANTS
        for state in _STATE_CONTEXTS
    }
    observed = {
        (str(row.get("candidate_id")), str(row.get("variant")), str(row.get("state_context")))
        for row in cases
    }
    return len(cases) == 12 and observed == expected


def _partial_g3_coverage_valid(cases: Sequence[Mapping[str, Any]]) -> bool:
    expected = {
        (candidate, variant, state)
        for candidate in _CANDIDATE_ORDER
        for variant in _VARIANTS
        for state in _STATE_CONTEXTS
    }
    observed = [
        (str(row.get("candidate_id")), str(row.get("variant")), str(row.get("state_context")))
        for row in cases
    ]
    return len(observed) <= 12 and len(set(observed)) == len(observed) and set(observed) <= expected


def _tensor_error_observed(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"relative_l2", "max_abs"}:
        return False
    if not all(
        isinstance(value.get(field), (int, float)) and not isinstance(value.get(field), bool)
        for field in ("relative_l2", "max_abs")
    ):
        return False
    relative = float(value["relative_l2"])
    maximum = float(value["max_abs"])
    return (
        relative >= 0.0
        and maximum >= 0.0
        and math.isfinite(relative)
        and math.isfinite(maximum)
    )


def _tensor_error_within(value: Any, relative_max: float) -> bool:
    return bool(
        _tensor_error_observed(value)
        and isinstance(value, Mapping)
        and float(value["relative_l2"]) <= relative_max
    )


def _comparison_observed(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"logits", "runtime_state", "gradients", "passed"}
        and all(
            _tensor_error_observed(value.get(field))
            for field in ("logits", "runtime_state", "gradients")
        )
        and isinstance(value.get("passed"), bool)
    )


def _g3_comparison_observed(value: Any, *, relative_max: float) -> bool:
    """Validate the complete G3 diagnostic row without requiring it to pass."""

    expected_fields = {
        "logits",
        "runtime_state",
        "runtime_state_components",
        "state_metadata_exact",
        "gradients",
        "gradients_worst_leaf",
        "tolerance",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        return False
    primary_fields = ("logits", "runtime_state", "gradients")
    if not all(_tensor_error_observed(value.get(field)) for field in primary_fields):
        return False
    if not _tensor_error_observed(value.get("gradients_worst_leaf")):
        return False
    components = value.get("runtime_state_components")
    if not isinstance(components, Mapping) or set(components) != {
        "recurrent",
        "attention_key",
        "attention_value",
    }:
        return False
    if not all(_tensor_error_observed(component) for component in components.values()):
        return False
    runtime = value["runtime_state"]
    if not isinstance(runtime, Mapping):
        return False
    if float(runtime["relative_l2"]) != max(
        float(component["relative_l2"])
        for component in components.values()
        if isinstance(component, Mapping)
    ):
        return False
    if float(runtime["max_abs"]) != max(
        float(component["max_abs"])
        for component in components.values()
        if isinstance(component, Mapping)
    ):
        return False
    if value.get("tolerance") != {"relative_l2_max": relative_max, "max_abs_max": None}:
        return False
    if not isinstance(value.get("state_metadata_exact"), bool) or not isinstance(
        value.get("passed"), bool
    ):
        return False
    metric_pass = bool(
        value.get("state_metadata_exact") is True
        and all(
            isinstance(value[field], Mapping)
            and float(value[field]["relative_l2"]) <= relative_max
            for field in primary_fields
        )
    )
    # A metric failure must never be labelled PASS. A metric-valid comparison
    # may still fail because per-path gradient finiteness is intentionally not
    # duplicated inside this diagnostic mapping.
    return value.get("passed") is not True or metric_pass


def _g3_layout_identity_observed(
    value: Any, *, fixed_layout: Mapping[str, Any]
) -> bool:
    expected_fields = {
        "physical_microbatch_sequences",
        "sequence_length",
        "accumulation_steps",
        "target_global_input_tokens",
        "loss_denominator",
        "optimizer_update_boundary",
        "autocast_scope",
        "gradient_clipping_order",
        "initialization_matched",
        "parameter_surface_matched",
        "optimizer_state_shape_matched",
        "shape_contract_valid",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        return False
    expected_shape = bool(
        fixed_layout.get("microbatch_sequences") == 1
        and fixed_layout.get("target_global_input_tokens")
        == int(fixed_layout["microbatch_sequences"])
        * int(fixed_layout["context_length"])
        * int(fixed_layout["accumulation_steps"])
    )
    identity_bools = (
        "initialization_matched",
        "parameter_surface_matched",
        "optimizer_state_shape_matched",
    )
    expected_pass = expected_shape and all(value.get(field) is True for field in identity_bools)
    return bool(
        value.get("physical_microbatch_sequences") == fixed_layout.get("microbatch_sequences")
        and value.get("sequence_length") == fixed_layout.get("context_length")
        and value.get("accumulation_steps") == fixed_layout.get("accumulation_steps")
        and value.get("target_global_input_tokens")
        == fixed_layout.get("target_global_input_tokens")
        and value.get("loss_denominator")
        == "TOTAL_VALID_NEXT_TOKEN_PREDICTIONS_ACROSS_FIXED_LAYOUT"
        and value.get("optimizer_update_boundary") == "AFTER_EXACT_ACCUMULATION_STEPS"
        and value.get("autocast_scope") == "CUDA_BF16_FORWARD_LOSS_ONLY"
        and value.get("gradient_clipping_order") == "AFTER_ACCUMULATION_BEFORE_ADAMW"
        and all(isinstance(value.get(field), bool) for field in (*identity_bools, "passed"))
        and value.get("shape_contract_valid") is expected_shape
        and value.get("passed") is expected_pass
    )


def _g3_case_observed(
    row: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> bool:
    layouts = _layout_by_candidate(protocol)
    candidate = str(row.get("candidate_id"))
    if candidate not in layouts or row.get("fixed_layout") != layouts[candidate]:
        return False
    comparisons = row.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(_COMPARISON_NAMES):
        return False
    thresholds = protocol["inherited_thresholds"]
    relative_max = float(thresholds["bf16_relative_l2_max"])
    if any(
        not _g3_comparison_observed(comparisons.get(name), relative_max=relative_max)
        for name in _COMPARISON_NAMES
    ):
        return False
    if not _g3_layout_identity_observed(
        row.get("layout_identity"), fixed_layout=layouts[candidate]
    ):
        return False
    layout_identity = row["layout_identity"]
    if not isinstance(layout_identity, Mapping) or row.get("layout_identity_passed") is not bool(
        layout_identity.get("passed")
    ):
        return False
    norms = row.get("gradient_norms")
    if not isinstance(norms, Mapping) or set(norms) != {
        "compiled_bf16",
        "reference_python_bf16",
        "reference_python_fp32",
    }:
        return False
    if not all(
        isinstance(norms.get(name), (int, float)) and not isinstance(norms.get(name), bool)
        for name in norms
    ):
        return False
    try:
        norm_values = [float(norms[name]) for name in sorted(norms)]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and value >= 0.0 for value in norm_values):
        return False
    expected_norm_range = all(
        float(thresholds["gradient_norm_min"])
        <= value
        <= float(thresholds["gradient_norm_max"])
        for value in norm_values
    )
    if row.get("gradient_norm_in_range") is not expected_norm_range:
        return False
    if not all(
        isinstance(row.get(field), bool)
        for field in (
            "gradient_finite",
            "state_metadata_exact",
            "clone_no_alias",
            "passed",
        )
    ):
        return False
    comparison_metadata = all(
        isinstance(comparisons[name], Mapping)
        and comparisons[name].get("state_metadata_exact") is True
        for name in _COMPARISON_NAMES
    )
    if row.get("state_metadata_exact") is not comparison_metadata:
        return False
    if not _backend_integrity_observed(row.get("backend_integrity")):
        return False
    backend = row["backend_integrity"]
    if not isinstance(backend, Mapping):
        return False
    count_fields = (
        "graph_break_count",
        "fallback_count",
        "variant_specific_fp32_path_count",
        "variant_specific_padding_count",
    )
    if any(row.get(field) != backend.get(field) for field in count_fields):
        return False
    expected_pass = bool(
        row.get("layout_identity_passed") is True
        and all(
            isinstance(comparisons[name], Mapping) and comparisons[name].get("passed") is True
            for name in _COMPARISON_NAMES
        )
        and row.get("gradient_finite") is True
        and expected_norm_range
        and comparison_metadata
        and row.get("clone_no_alias") is True
        and backend.get("passed") is True
    )
    return row.get("passed") is expected_pass


def _comparison_passed_within(value: Any, relative_max: float) -> bool:
    return bool(
        _comparison_observed(value)
        and isinstance(value, Mapping)
        and value.get("passed") is True
        and all(
            _tensor_error_within(value.get(field), relative_max)
            for field in ("logits", "runtime_state", "gradients")
        )
    )


def _backend_integrity_observed(value: Any) -> bool:
    required = {
        "optimized_backend_id",
        "registered_backend_id",
        "runtime_backend_alias",
        "backend_alias_matches_registration",
        "reference_backend_id",
        "strict_reference_python",
        "positive_compiled_execution",
        "python_token_loop_at_scientific_runtime",
        "graph_break_count",
        "fallback_count",
        "variant_specific_fp32_path_count",
        "variant_specific_padding_count",
        "passed",
    }
    optional = {"observed_padded_tokens", "precision_policy_sha256"}
    if not isinstance(value, Mapping) or not required <= set(value) <= required | optional:
        return False
    if not (
        all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in (
                "optimized_backend_id",
                "registered_backend_id",
                "runtime_backend_alias",
                "reference_backend_id",
            )
        )
        and all(
            isinstance(value.get(field), bool)
            for field in (
                "backend_alias_matches_registration",
                "strict_reference_python",
                "positive_compiled_execution",
                "python_token_loop_at_scientific_runtime",
                "passed",
            )
        )
        and all(
            isinstance(value.get(field), int)
            and not isinstance(value.get(field), bool)
            and int(value[field]) >= 0
            for field in (
                "graph_break_count",
                "fallback_count",
                "variant_specific_fp32_path_count",
                "variant_specific_padding_count",
            )
        )
    ):
        return False
    if "observed_padded_tokens" in value and not (
        isinstance(value["observed_padded_tokens"], int)
        and not isinstance(value["observed_padded_tokens"], bool)
        and int(value["observed_padded_tokens"]) >= 0
    ):
        return False
    return "precision_policy_sha256" not in value or _is_sha256(value["precision_policy_sha256"])


def _optimizer_trace_observed(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != _OPTIMIZER_TRACE_FIELDS:
        return False
    if not all(isinstance(value.get(field), bool) for field in (*_OPTIMIZER_TRUE_FIELDS, "passed")):
        return False
    integer_fields = (
        "valid_prediction_tokens",
        "expected_valid_prediction_tokens",
        "exposed_input_tokens",
        "expected_input_tokens",
        "microbatch_count",
        "expected_microbatch_count",
    )
    if not all(
        isinstance(value.get(field), int)
        and not isinstance(value.get(field), bool)
        and int(value[field]) >= 0
        for field in integer_fields
    ):
        return False
    events = value.get("execution_events")
    steps = value.get("adamw_state_steps")
    return bool(
        isinstance(events, list)
        and all(isinstance(event, str) for event in events)
        and isinstance(steps, list)
        and all(isinstance(step, int) and not isinstance(step, bool) for step in steps)
    )


def _optimizer_trace_passed(row: Mapping[str, Any], relative_max: float) -> bool:
    trace = row.get("optimizer_step_integrity")
    layout = row.get("fixed_layout")
    if not _optimizer_trace_observed(trace) or not isinstance(trace, Mapping):
        return False
    if not isinstance(layout, Mapping):
        return False
    try:
        microbatch = int(layout["microbatch_sequences"])
        accumulation = int(layout["accumulation_steps"])
        context = int(layout["context_length"])
        target_input_tokens = int(layout["target_global_input_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    if min(microbatch, accumulation, context, target_input_tokens) <= 0:
        return False
    expected_valid_tokens = microbatch * accumulation * (context - 1)
    return bool(
        all(trace.get(field) is True for field in _OPTIMIZER_TRUE_FIELDS)
        and trace.get("passed") is True
        and trace.get("execution_events") == list(_OPTIMIZER_EXECUTION_EVENTS)
        and trace.get("adamw_state_steps") == [1]
        and trace.get("valid_prediction_tokens") == expected_valid_tokens
        and trace.get("expected_valid_prediction_tokens") == expected_valid_tokens
        and trace.get("exposed_input_tokens") == target_input_tokens
        and trace.get("expected_input_tokens") == target_input_tokens
        and trace.get("microbatch_count") == accumulation
        and trace.get("expected_microbatch_count") == accumulation
        and row.get("optimizer_integrity_passed") is True
        and row.get("optimizer_state_structure_equal") is True
        and row.get("scheduler_state_equal") is True
        and row.get("gradients_finite") is True
        and row.get("state_metadata_exact") is True
        and row.get("clone_no_alias") is True
        and row.get("graph_break_count") == 0
        and row.get("fallback_count") == 0
        and _comparison_passed_within(row.get("comparison"), relative_max)
        and _tensor_error_within(row.get("optimizer_state"), relative_max)
    )


def _g5_optimizer_integrity_passed(
    replays: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> bool:
    if not _exact_g4_coverage(replays):
        return False
    relative_max = float(protocol["inherited_thresholds"]["bf16_relative_l2_max"])
    return all(_optimizer_trace_passed(row, relative_max) for row in replays)


def _g3_raw_evaluable(
    cases: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> bool:
    if not _exact_g3_coverage(cases):
        return False
    for row in cases:
        if row.get("execution_status") in {
            "FAILED_WORKER_NO_RECEIPT",
            "NOT_RUN_BLOCKED_DEPENDENCY",
            "EXECUTION_ERROR",
        }:
            return False
        if not all(_is_sha256(row.get(field)) for field in _SHA_FIELDS):
            return False
        if not _g3_case_observed(row, protocol=protocol):
            return False
        if not all(
            isinstance(row.get(field), int)
            and not isinstance(row.get(field), bool)
            and int(row[field]) >= 0
            for field in (
                "graph_break_count",
                "fallback_count",
                "variant_specific_fp32_path_count",
                "variant_specific_padding_count",
            )
        ):
            return False
    return True


def _exact_g4_coverage(replays: Sequence[Mapping[str, Any]]) -> bool:
    expected = {(candidate, variant) for candidate in _CANDIDATE_ORDER for variant in _VARIANTS}
    observed = {(str(row.get("candidate_id")), str(row.get("variant"))) for row in replays}
    return len(replays) == 6 and observed == expected


def _g4_cross_variant_identity_passed(replays: Sequence[Mapping[str, Any]]) -> bool:
    """Recompute paired replay identity without trusting runner annotations.

    The compiled graph digest is intentionally excluded across variants: the
    tied projection may produce a different graph while every physical input
    to the replay must remain identical.  It is still required and compared
    between replay A/B inside each variant.
    """

    if not _exact_g4_coverage(replays):
        return False
    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in replays:
        by_candidate.setdefault(str(row.get("candidate_id")), []).append(row)
    for candidate in _CANDIDATE_ORDER:
        pair = by_candidate.get(candidate, [])
        if len(pair) != 2 or {str(row.get("variant")) for row in pair} != set(_VARIANTS):
            return False
        for field in _G4_CROSS_VARIANT_IDENTITY_FIELDS:
            if sha256_canonical_json(pair[0].get(field)) != sha256_canonical_json(
                pair[1].get(field)
            ):
                return False
    return True


def _partial_g4_coverage_valid(replays: Sequence[Mapping[str, Any]]) -> bool:
    expected = {(candidate, variant) for candidate in _CANDIDATE_ORDER for variant in _VARIANTS}
    observed = [(str(row.get("candidate_id")), str(row.get("variant"))) for row in replays]
    return len(observed) <= 6 and len(set(observed)) == len(observed) and set(observed) <= expected


def _g4_raw_evaluable(replays: Sequence[Mapping[str, Any]]) -> bool:
    if not _exact_g4_coverage(replays):
        return False
    required = (
        *_G4_SHA_FIELDS,
        "comparison",
        "optimizer_state",
        "optimizer_state_structure_equal",
        "scheduler_state_equal",
        "optimizer_step_integrity",
        "optimizer_integrity_passed",
        "gradients_finite",
        "state_metadata_exact",
        "clone_no_alias",
        "graph_break_count",
        "fallback_count",
        "passed",
    )
    return all(
        row.get("execution_status")
        not in {"NOT_RUN_BLOCKED_DEPENDENCY", "FAILED_WORKER_NO_RECEIPT", "EXECUTION_ERROR"}
        and all(field in row for field in required)
        and all(_is_sha256(row.get(field)) for field in _G4_SHA_FIELDS)
        and _comparison_observed(row.get("comparison"))
        and _tensor_error_observed(row.get("optimizer_state"))
        and _optimizer_trace_observed(row.get("optimizer_step_integrity"))
        and all(
            isinstance(row.get(field), bool)
            for field in (
                "gradients_finite",
                "state_metadata_exact",
                "clone_no_alias",
                "optimizer_integrity_passed",
                "optimizer_state_structure_equal",
                "scheduler_state_equal",
                "passed",
            )
        )
        and all(
            isinstance(row.get(field), int)
            and not isinstance(row.get(field), bool)
            and int(row[field]) >= 0
            for field in ("graph_break_count", "fallback_count")
        )
        for row in replays
    )


def _validate_fp32_runtime_state(
    value: Any,
    label: str,
    *,
    relative_max: float,
    max_abs_max: float,
) -> None:
    if not isinstance(value, Mapping):
        raise Stage3DContractError(f"{label} must be a runtime-state error mapping")
    for field in ("attention_key", "attention_value", "recurrent"):
        _validate_tensor_error(
            value.get(field),
            f"{label}.{field}",
            relative_max=relative_max,
            max_abs_max=max_abs_max,
        )
    for field in (
        "lengths_equal",
        "position_equal",
        "positions_equal",
        "write_indices_equal",
    ):
        _require_bool(value.get(field), True, f"{label}.{field}")


def _validate_stage3c_fp32_report(
    report: Any,
    *,
    label: str,
    partition_length: int,
    relative_max: float,
    max_abs_max: float,
) -> int:
    if not isinstance(report, Mapping) or report.get("precision") != "fp32":
        raise Stage3DContractError(f"{label} is not a raw FP32 report")
    _require_bool(report.get("passed"), True, f"{label}.passed")
    _require_bool(report.get("reference_gradients_finite"), True, f"{label}.reference finite")
    _validate_tensor_error(
        report.get("reference_logits"),
        f"{label}.reference_logits",
        relative_max=relative_max,
        max_abs_max=max_abs_max,
    )
    _validate_tensor_error(
        report.get("reference_gradients"),
        f"{label}.reference_gradients",
        relative_max=relative_max,
        max_abs_max=max_abs_max,
    )
    _validate_tensor_diagnostic(
        report.get("reference_gradients_worst_leaf"),
        f"{label}.reference_gradients_worst_leaf",
    )
    _validate_fp32_runtime_state(
        report.get("reference_runtime_state"),
        f"{label}.reference_runtime_state",
        relative_max=relative_max,
        max_abs_max=max_abs_max,
    )
    rows = report.get("rows")
    partitions = report.get("partitions")
    if not isinstance(rows, list) or not isinstance(partitions, list) or len(rows) != 12:
        raise Stage3DContractError(f"{label} raw partition coverage changed")
    if len(partitions) != len(rows):
        raise Stage3DContractError(f"{label} row/partition coverage differs")
    nontrivial = 0
    monolithic = [partition_length]
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise Stage3DContractError(f"{label}.rows[{index}] is not a mapping")
        partition = raw.get("partition")
        if partition != partitions[index] or not isinstance(partition, list):
            raise Stage3DContractError(f"{label}.rows[{index}] partition binding changed")
        if any(
            isinstance(piece, bool) or not isinstance(piece, int) or piece <= 0
            for piece in partition
        ):
            raise Stage3DContractError(f"{label}.rows[{index}] has an invalid partition")
        if sum(partition) != partition_length:
            raise Stage3DContractError(f"{label}.rows[{index}] changes sequence length")
        _require_bool(raw.get("passed"), True, f"{label}.rows[{index}].passed")
        _require_bool(raw.get("gradients_finite"), True, f"{label}.rows[{index}].gradients_finite")
        for field in ("logits", "gradients"):
            _validate_tensor_error(
                raw.get(field),
                f"{label}.rows[{index}].{field}",
                relative_max=relative_max,
                max_abs_max=max_abs_max,
            )
        _validate_tensor_diagnostic(
            raw.get("gradients_worst_leaf"),
            f"{label}.rows[{index}].gradients_worst_leaf",
        )
        _validate_fp32_runtime_state(
            raw.get("runtime_state"),
            f"{label}.rows[{index}].runtime_state",
            relative_max=relative_max,
            max_abs_max=max_abs_max,
        )
        if partition != monolithic:
            nontrivial += 1
    if rows[0].get("partition") != monolithic or nontrivial != 11:
        raise Stage3DContractError(f"{label} nontrivial-row definition/coverage changed")
    return nontrivial


def _derive_fp32_reference_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    stage3c = protocol.get("stage3c")
    if not isinstance(stage3c, Mapping):
        raise Stage3DContractError("Protocol lacks Stage-3C raw bindings")
    manifest_binding = stage3c.get("artifact_hash_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise Stage3DContractError("Protocol lacks Stage-3C artifact manifest binding")
    manifest, paths = _validate_registered_stage3c_artifact_manifest(str(manifest_binding["path"]))
    thresholds = protocol["inherited_thresholds"]
    relative_max = float(thresholds["fp32_relative_l2_max"])
    max_abs_max = float(thresholds["fp32_max_abs_max"])
    report_passes = 0
    nontrivial_passes = 0
    for candidate_id in _CANDIDATE_ORDER:
        numerical = _load_mapping(
            paths[_STAGE3C_NUMERICAL_FILES[candidate_id]],
            f"Stage-3C raw numerical report {candidate_id}",
        )
        if (
            numerical.get("candidate_id") != candidate_id
            or numerical.get("initialization_matched_across_variants") is not True
        ):
            raise Stage3DContractError("Stage-3C numerical candidate binding changed")
        partition_length = _require_int(
            numerical.get("partition_length"),
            f"{candidate_id} partition_length",
            minimum=1,
        )
        variants = numerical.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(_VARIANTS):
            raise Stage3DContractError("Stage-3C numerical variant coverage changed")
        for variant in _VARIANTS:
            variant_row = variants[variant]
            if not isinstance(variant_row, Mapping):
                raise Stage3DContractError("Stage-3C raw variant row is invalid")
            states = variant_row.get("arbitrary_partitions")
            if not isinstance(states, Mapping) or set(states) != set(_STATE_CONTEXTS):
                raise Stage3DContractError("Stage-3C FP32 state-context coverage changed")
            for state_context in _STATE_CONTEXTS:
                precision_rows = states[state_context]
                if not isinstance(precision_rows, Mapping) or "fp32" not in precision_rows:
                    raise Stage3DContractError("Stage-3C raw state lacks FP32 report")
                nontrivial_passes += _validate_stage3c_fp32_report(
                    precision_rows["fp32"],
                    label=f"{candidate_id}.{variant}.{state_context}.fp32",
                    partition_length=partition_length,
                    relative_max=relative_max,
                    max_abs_max=max_abs_max,
                )
                report_passes += 1
    if report_passes != 12 or nontrivial_passes != 132:
        raise Stage3DContractError("Stage-3C raw FP32 recomputation coverage changed")
    return {
        "passed": True,
        "reuse_policy": "EXACT_STAGE3C_HASH_BINDING_ONLY",
        "stage3c_result_sha256": _STAGE3C_RESULT_SHA256,
        "stage3c_artifact_manifest_sha256": sha256_file(str(manifest_binding["path"])),
        "stage3c_artifact_aggregate_sha256": manifest["aggregate_sha256"],
        "fp32_arbitrary_partition_reports_passed": report_passes,
        "fp32_arbitrary_partition_reports_required": 12,
        "fp32_nontrivial_rows_passed": nontrivial_passes,
        "fp32_nontrivial_rows_required": 132,
        "thresholds": {
            "relative_l2_max": relative_max,
            "max_abs_max": max_abs_max,
        },
    }


def derive_stage3c_fp32_reference_binding(
    protocol_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute G2 directly from the immutable Stage-3C raw artifacts."""

    protocol = validate_stage3d_protocol_lock(protocol_lock)
    return _derive_fp32_reference_binding(protocol)


def _validate_fp32_reference_binding(value: Any, protocol: Mapping[str, Any]) -> None:
    expected = _derive_fp32_reference_binding(protocol)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise Stage3DContractError(
            "G2 FP32 binding differs from independent raw-artifact recomputation"
        )


def build_stage3d_admissibility_receipt(
    *,
    protocol_lock_path: str | Path,
    g3_cases: Sequence[Mapping[str, Any]],
    g4_replays: Sequence[Mapping[str, Any]],
    fp32_reference_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fixed-layout receipt; any failed hard case yields BLOCKED."""

    protocol_path = _regular_file(protocol_lock_path, "Stage-3D protocol lock")
    protocol = _load_mapping(protocol_path, "Stage-3D protocol lock")
    protocol = validate_stage3d_protocol_lock(protocol)
    _validate_fp32_reference_binding(fp32_reference_binding, protocol)
    # Builders accept failed rows so executors can emit a terminal receipt. The
    # strict validator below only accepts internally consistent GO/BLOCKED
    # dispositions; failed rows are counted here without rewriting metrics.
    cases = deepcopy([dict(row) for row in g3_cases])
    replays = deepcopy([dict(row) for row in g4_replays])
    g3_passes = sum(row.get("passed") is True for row in cases)
    g4_passes = sum(row.get("passed") is True for row in replays)
    g0_pass = True
    g1_pass = _g1_variant_identity_passed(cases)
    g2_pass = True
    g3_pass = g3_passes == 12
    g4_pass = g4_passes == 6 and _g4_cross_variant_identity_passed(replays)
    g5_pass = _g5_optimizer_integrity_passed(replays, protocol)
    g6_pass = _g6_backend_integrity_passed(cases, replays)
    reference_mismatch = _fixed_probe_reference_mismatch(cases)
    g3_evaluable = _g3_raw_evaluable(cases, protocol)
    g4_required = g3_evaluable and g3_pass
    g4_evaluable = _g4_raw_evaluable(replays) if g4_required else True
    execution_evaluable = g3_evaluable and g4_evaluable
    hard_pass = execution_evaluable and all(
        (g0_pass, g1_pass, g2_pass, g3_pass, g4_pass, g5_pass, g6_pass)
    )
    if not execution_evaluable:
        execution_status = "FAILED_IMPLEMENTATION_OR_EXECUTION"
        disposition = STAGE3D_NOT_EVALUABLE
    else:
        execution_status = "COMPLETED_NUMERICAL_EVALUATION"
        disposition = STAGE3D_GO if hard_pass else STAGE3D_BLOCKED
    payload: dict[str, Any] = {
        "schema_version": _RECEIPT_VERSION,
        "manifest_type": _MANIFEST_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "FIXED_LAYOUT_NUMERICAL_ADMISSIBILITY_ONLY",
        "protocol_lock": {
            **bind_file(protocol_path, "Stage-3D protocol lock"),
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "source": deepcopy(protocol["source"]),
        "stage3c": deepcopy(protocol["stage3c"]),
        "diagnostic_disposition": KNOWN_LAYOUT_SENSITIVITY,
        "fixed_probe_reference_mismatch": (
            "OBSERVED_IN_STAGE3D_G3" if reference_mismatch else "NONE"
        ),
        "inherited_thresholds": deepcopy(protocol["inherited_thresholds"]),
        "fixed_layouts": deepcopy(protocol["fixed_layouts"]),
        "determinism": deepcopy(protocol["determinism"]),
        "fp32_reference": deepcopy(dict(fp32_reference_binding)),
        "g3_cases": cases,
        "g4_replays": replays,
        "gate_summary": {
            "g0_passed": g0_pass,
            "g1_passed": g1_pass,
            "g2_passed": g2_pass,
            "g3_passed": g3_pass,
            "g4_passed": g4_pass,
            "g5_passed": g5_pass,
            "g6_passed": g6_pass,
            "g3_pass_count": g3_passes,
            "g3_required_count": 12,
            "g4_pass_count": g4_passes,
            "g4_required_count": 6,
        },
        "execution_status": execution_status,
        "disposition": disposition,
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "passed": hard_pass,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_stage3d_admissibility_receipt(
    payload: Mapping[str, Any],
    *,
    protocol_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact Stage-3D coverage and reject a hand-edited PASS."""

    protocol = validate_stage3d_protocol_lock(protocol_lock)
    normalized = deepcopy(dict(payload))
    observed_hash = normalized.pop("receipt_sha256", None)
    if observed_hash != sha256_canonical_json(normalized):
        raise Stage3DContractError("Stage-3D fixed-layout receipt hash changed")
    normalized["receipt_sha256"] = observed_hash
    if (
        normalized.get("schema_version") != _RECEIPT_VERSION
        or normalized.get("manifest_type") != _MANIFEST_TYPE
    ):
        raise Stage3DContractError("Not a Stage-3D fixed-layout receipt")
    _require_bool(normalized.get("scientific_evidence"), False, "scientific_evidence")
    for key in ("main_test_opened", "scientific_e26a_started", "scientific_main_started"):
        _require_bool(normalized.get(key), False, key)
    binding = normalized.get("protocol_lock")
    if not isinstance(binding, Mapping):
        raise Stage3DContractError("Receipt lacks Stage-3D protocol binding")
    file_binding = {"path": binding.get("path"), "sha256": binding.get("sha256")}
    _verify_binding(file_binding, "Stage-3D protocol lock")
    if binding.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise Stage3DContractError("Receipt binds a different Stage-3D protocol hash")
    if normalized.get("source") != protocol["source"]:
        raise Stage3DContractError("Receipt source differs from the Stage-3D protocol")
    if normalized.get("stage3c") != protocol["stage3c"]:
        raise Stage3DContractError("Receipt changed Stage-3C evidence")
    if normalized.get("diagnostic_disposition") != KNOWN_LAYOUT_SENSITIVITY:
        raise Stage3DContractError("Receipt changed the Stage-3C diagnostic disposition")
    if normalized.get("inherited_thresholds") != protocol["inherited_thresholds"]:
        raise Stage3DContractError("Receipt changed inherited numerical thresholds")
    if normalized.get("fixed_layouts") != protocol["fixed_layouts"]:
        raise Stage3DContractError("Receipt changed the fixed physical layouts")
    if normalized.get("determinism") != protocol["determinism"]:
        raise Stage3DContractError("Receipt changed the deterministic probe contract")
    _validate_fp32_reference_binding(normalized.get("fp32_reference"), protocol)

    cases = normalized.get("g3_cases")
    replays = normalized.get("g4_replays")
    summary = normalized.get("gate_summary")
    if (
        not isinstance(cases, list)
        or not isinstance(replays, list)
        or not isinstance(summary, Mapping)
    ):
        raise Stage3DContractError("Receipt lacks G3/G4/summary records")
    g3_passes = sum(isinstance(row, Mapping) and row.get("passed") is True for row in cases)
    g4_passes = sum(isinstance(row, Mapping) and row.get("passed") is True for row in replays)
    claimed_go = bool(normalized.get("passed") is True)
    if claimed_go:
        _validate_g3_cases(cases, protocol)
        _validate_g4_replays(replays, protocol)
    elif not _partial_g3_coverage_valid(cases) or not _partial_g4_coverage_valid(replays):
        raise Stage3DContractError("Non-GO receipt has invalid or duplicate raw coverage")
    expected_summary = {
        "g0_passed": True,
        "g1_passed": _g1_variant_identity_passed(cases),
        "g2_passed": bool(
            isinstance(normalized.get("fp32_reference"), Mapping)
            and normalized["fp32_reference"].get("passed") is True
        ),
        "g3_passed": g3_passes == 12,
        "g4_passed": g4_passes == 6 and _g4_cross_variant_identity_passed(replays),
        "g5_passed": _g5_optimizer_integrity_passed(replays, protocol),
        "g6_passed": _g6_backend_integrity_passed(cases, replays),
        "g3_pass_count": g3_passes,
        "g3_required_count": 12,
        "g4_pass_count": g4_passes,
        "g4_required_count": 6,
    }
    if dict(summary) != expected_summary:
        raise Stage3DContractError("Stage-3D gate summary is inconsistent with raw cells")
    all_gates_pass = all(bool(expected_summary[f"g{index}_passed"]) for index in range(7))
    g3_evaluable = _g3_raw_evaluable(cases, protocol)
    g4_required = g3_evaluable and expected_summary["g3_passed"]
    g4_evaluable = _g4_raw_evaluable(replays) if g4_required else True
    execution_evaluable = g3_evaluable and g4_evaluable
    if execution_evaluable and not _exact_g3_coverage(cases):
        raise Stage3DContractError("Completed receipt lacks exact G3 coverage")
    if execution_evaluable and g4_required and not _exact_g4_coverage(replays):
        raise Stage3DContractError("Completed receipt lacks exact G4 coverage")
    expected_execution = (
        "COMPLETED_NUMERICAL_EVALUATION"
        if execution_evaluable
        else "FAILED_IMPLEMENTATION_OR_EXECUTION"
    )
    if normalized.get("execution_status") != expected_execution:
        raise Stage3DContractError("Stage-3D execution status is inconsistent with raw coverage")
    expected_disposition = (
        STAGE3D_GO
        if execution_evaluable and all_gates_pass
        else STAGE3D_BLOCKED
        if execution_evaluable
        else STAGE3D_NOT_EVALUABLE
    )
    if normalized.get("disposition") != expected_disposition:
        raise Stage3DContractError("Stage-3D disposition is inconsistent with raw cells")
    expected_pass = execution_evaluable and all_gates_pass
    if normalized.get("passed") is not expected_pass:
        raise Stage3DContractError("Stage-3D pass flag is inconsistent with raw cells")
    expected_mismatch = (
        "OBSERVED_IN_STAGE3D_G3" if _fixed_probe_reference_mismatch(cases) else "NONE"
    )
    if normalized.get("fixed_probe_reference_mismatch") != expected_mismatch:
        raise Stage3DContractError("G3 reference mismatch was merged with layout sensitivity")
    return normalized


def build_stage3d_backend_manifest(
    *,
    protocol_lock: Mapping[str, Any],
    fixed_layout_receipt_path: str | Path,
    backend_id: str,
) -> dict[str, Any]:
    """Build the fail-closed backend promotion record after a Stage-3D GO."""

    protocol = validate_stage3d_protocol_lock(protocol_lock)
    receipt_path = _regular_file(fixed_layout_receipt_path, "Stage-3D receipt")
    receipt = validate_stage3d_admissibility_receipt(
        _load_mapping(receipt_path, "Stage-3D receipt"), protocol_lock=protocol
    )
    if receipt["disposition"] != STAGE3D_GO:
        raise Stage3DContractError("A blocked Stage-3D receipt cannot promote a backend")
    if backend_id == "reference_python":
        raise Stage3DContractError("Reference Python cannot become the scientific backend")
    payload: dict[str, Any] = {
        "schema_version": _BACKEND_VERSION,
        "manifest_type": "E26_STAGE3D_BACKEND_MANIFEST",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_NUMERICAL_PREFLIGHT",
        "claim_ceiling": "FIXED_LAYOUT_BACKEND_ADMISSIBILITY_ONLY",
        "protocol_lock": deepcopy(receipt["protocol_lock"]),
        "stage3d_receipt": {
            **bind_file(receipt_path, "Stage-3D receipt"),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "source": deepcopy(protocol["source"]),
        "backend_id": backend_id,
        "strict_reference_backend": "reference_python",
        "fixed_layouts": deepcopy(protocol["fixed_layouts"]),
        "fallback_count": 0,
        "graph_break_count": 0,
        "variant_specific_fp32_path_count": 0,
        "variant_specific_padding_count": 0,
        "reference_python_scientific_path": False,
        "fixed_layout_admissible": True,
        "batching_layout_invariance_claim_eligible": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
        "passed": True,
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    return payload


def validate_stage3d_backend_manifest(
    payload: Mapping[str, Any],
    *,
    protocol_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Stage-3D backend manifest and its GO receipt binding."""

    protocol = validate_stage3d_protocol_lock(protocol_lock)
    normalized = deepcopy(dict(payload))
    observed = normalized.pop("manifest_sha256", None)
    if observed != sha256_canonical_json(normalized):
        raise Stage3DContractError("Stage-3D backend manifest hash changed")
    normalized["manifest_sha256"] = observed
    if (
        normalized.get("schema_version") != _BACKEND_VERSION
        or normalized.get("manifest_type") != "E26_STAGE3D_BACKEND_MANIFEST"
        or normalized.get("scientific_evidence") is not False
        or normalized.get("passed") is not True
    ):
        raise Stage3DContractError("Invalid Stage-3D backend manifest")
    if normalized.get("source") != protocol["source"]:
        raise Stage3DContractError("Backend manifest source changed")
    if normalized.get("fixed_layouts") != protocol["fixed_layouts"]:
        raise Stage3DContractError("Backend manifest layouts changed")
    if normalized.get("backend_id") == "reference_python":
        raise Stage3DContractError("Reference Python cannot be the scientific backend")
    for field in (
        "fallback_count",
        "graph_break_count",
        "variant_specific_fp32_path_count",
        "variant_specific_padding_count",
    ):
        if normalized.get(field) != 0:
            raise Stage3DContractError(f"Backend manifest {field} must be zero")
    for field in (
        "reference_python_scientific_path",
        "scientific_e26a_started",
        "scientific_main_started",
    ):
        _require_bool(normalized.get(field), False, field)
    _require_bool(normalized.get("fixed_layout_admissible"), True, "fixed_layout_admissible")
    _require_bool(
        normalized.get("batching_layout_invariance_claim_eligible"),
        False,
        "batching_layout_invariance_claim_eligible",
    )
    receipt_binding = normalized.get("stage3d_receipt")
    if not isinstance(receipt_binding, Mapping):
        raise Stage3DContractError("Backend manifest lacks Stage-3D receipt binding")
    bound = _verify_binding(
        {"path": receipt_binding.get("path"), "sha256": receipt_binding.get("sha256")},
        "Stage-3D fixed-layout receipt",
    )
    receipt = validate_stage3d_admissibility_receipt(
        _load_mapping(bound["path"], "Stage-3D fixed-layout receipt"),
        protocol_lock=protocol,
    )
    if (
        receipt_binding.get("receipt_sha256") != receipt["receipt_sha256"]
        or receipt["disposition"] != STAGE3D_GO
    ):
        raise Stage3DContractError("Backend manifest is not bound to a Stage-3D GO")
    return normalized


def validate_stage3d_resource_preflight_receipt(
    payload: Mapping[str, Any],
    *,
    protocol_lock: Mapping[str, Any],
    stage3d_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the post-GO, non-evidence resource-preflight receipt."""

    protocol = validate_stage3d_protocol_lock(protocol_lock)
    stage3d = validate_stage3d_admissibility_receipt(stage3d_receipt, protocol_lock=protocol)
    if stage3d["disposition"] != STAGE3D_GO:
        raise Stage3DContractError("Resource preflight requires a Stage-3D GO")
    normalized = deepcopy(dict(payload))
    observed = normalized.pop("receipt_sha256", None)
    if observed != sha256_canonical_json(normalized):
        raise Stage3DContractError("Stage-3D resource receipt hash changed")
    normalized["receipt_sha256"] = observed
    if (
        normalized.get("schema_version") != _RESOURCE_VERSION
        or normalized.get("manifest_type") != "E26_STAGE3D_RESOURCE_PREFLIGHT_RECEIPT"
        or normalized.get("scientific_evidence") is not False
        or normalized.get("passed") is not True
    ):
        raise Stage3DContractError("Invalid Stage-3D resource receipt")
    if normalized.get("source") != protocol["source"]:
        raise Stage3DContractError("Resource receipt source changed")
    if normalized.get("fixed_layouts") != protocol["fixed_layouts"]:
        raise Stage3DContractError("Resource receipt fixed layouts changed")
    resource_policy = normalized.get("resource_policy")
    expected_policy = {
        "token_budgets": [250_000_000, 375_000_000, 500_000_000],
        "deadline_reference_hours": 240,
        "deadline_fraction_max": 0.70,
        "max_main_wall_clock_hours": 168,
        "safety_time_multiplier": 1.25,
        "max_main_checkpoint_storage_gib": 100,
        "main_runs": 10,
        "gpu_lanes": 4,
        "save_every_tokens": 25_000_000,
    }
    if resource_policy != expected_policy:
        raise Stage3DContractError("Resource policy differs from Stage-3D registration")
    receipt_binding = normalized.get("stage3d_receipt")
    if not isinstance(receipt_binding, Mapping):
        raise Stage3DContractError("Resource receipt lacks Stage-3D receipt binding")
    if receipt_binding.get("receipt_sha256") != stage3d["receipt_sha256"]:
        raise Stage3DContractError("Resource receipt binds a different Stage-3D result")
    _verify_binding(
        {"path": receipt_binding.get("path"), "sha256": receipt_binding.get("sha256")},
        "Stage-3D fixed-layout receipt",
    )
    for field in (
        "main_test_opened",
        "scientific_e26a_started",
        "scientific_e26b_started",
        "scientific_main_started",
        "canonical_e26_artifact_created",
    ):
        _require_bool(normalized.get(field), False, field)
    candidates = normalized.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise Stage3DContractError("Resource receipt must cover all three candidates")
    if [row.get("candidate_id") for row in candidates if isinstance(row, Mapping)] != list(
        _CANDIDATE_ORDER
    ):
        raise Stage3DContractError("Resource candidate order/coverage changed")
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise Stage3DContractError(f"Resource candidate {index} must be a mapping")
        _require_sha256(row.get("candidate_config_sha256"), "candidate config SHA")
        _require_sha256(row.get("worker_report_sha256"), "worker report SHA")
        _require_sha256(row.get("worker_receipt_sha256"), "worker receipt SHA")
        for field in (
            "fixed_layout",
            "measurement",
            "paired_recipe_identity",
            "execution_device",
        ):
            if not isinstance(row.get(field), Mapping):
                raise Stage3DContractError(f"Resource candidate {index} lacks {field}")
        if not isinstance(row.get("resource_projections"), list):
            raise Stage3DContractError(
                f"Resource candidate {index} lacks resource_projections rows"
            )
    return normalized
