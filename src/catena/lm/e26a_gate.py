from __future__ import annotations

import fcntl
import math
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

from .artifacts import ArtifactContractError, ArtifactRun
from .audit_contract import (
    E26_AUDIT_LOCKED_HASH_KEYS,
    e26_execution_source_inventory,
    validate_e26_audit_locked_hashes,
)
from .backend_lock import (
    cuda_hardware_inventory,
    validate_backend_candidate_lock,
    validate_backend_preflight_manifest,
)
from .checkpointing import validate_restart_audit_coverage
from .data_readiness_v2 import (
    Stage2DataReadinessError,
    validate_stage2_data_bundle,
)
from .e26a_population_lock import (
    E26AValidationPopulationError,
    validate_e26a_validation_population_lock,
)
from .frozen_invariance import (
    FrozenInvarianceError,
    validate_frozen_invariance_receipt,
)
from .readiness import E26AReadiness, validate_e26a_readiness
from .transactional_stream import TransactionEpisode

E26A_EXECUTION_ACK = "E26A_SCIENTIFIC_GATE_AUTHORIZED"
E26A_EVIDENCE_TIER = "SCIENTIFIC_PROTOCOL_GATE"
E26A_CLAIM_CEILING = "PROTOCOL_IDENTIFIABILITY_ONLY"
PRIMARY_OPERATIONS = ("ADD", "INVALIDATE")
SATURATION_LOW = 0.01
SATURATION_HIGH = 0.99
DEADLINE_REFERENCE_HOURS = 240.0
DEFAULT_MAIN_RUNS = 10
DEFAULT_GPU_LANES = 4
DEFAULT_SAVE_EVERY_TOKENS = 25_000_000
GIB = 1024**3


class E26AGateBlocked(RuntimeError):
    """Raised before a canonical run is created when gate admission is invalid."""


@dataclass(frozen=True, slots=True)
class E26AGateInputPaths:
    config: Path
    calibration_config: Path
    protocol_lock: Path
    backend_candidate_lock: Path
    backend_manifest: Path
    tokenizer_manifest: Path
    corpus_manifest: Path
    data_lock: Path
    data_readiness: Path
    transaction_manifest: Path
    validation_population_lock: Path
    schedule_manifest: Path
    numerical_audit: Path
    restart_audit: Path
    frozen_tree_receipt: Path
    resource_preflight: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def hashes(self) -> dict[str, str]:
        return {f"{key}_sha256": sha256_file(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    deadline_reference_hours: float
    deadline_fraction_max: float
    max_main_wall_clock_hours: float
    safety_time_multiplier: float
    max_main_checkpoint_storage_gib: float
    token_budgets: tuple[int, ...]
    main_runs: int = DEFAULT_MAIN_RUNS
    gpu_lanes: int = DEFAULT_GPU_LANES
    save_every_tokens: int = DEFAULT_SAVE_EVERY_TOKENS

    @property
    def effective_wall_cap_hours(self) -> float:
        return min(
            self.max_main_wall_clock_hours,
            self.deadline_reference_hours * self.deadline_fraction_max,
        )


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    candidate_id: str
    parameter_count: int
    matching_passed: bool
    numerical_passed: bool
    tokens_per_second_by_variant: dict[str, float]
    checkpoint_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    p50_step_seconds: float
    p95_step_seconds: float
    compile_seconds: float
    graph_break_count: int
    fallback_count: int
    context_length: int = 0
    selected_microbatch_sequences: int = 0
    accumulation_steps: int = 0
    measured_optimizer_steps: int = 0
    descriptive_stability_steps: int = 0
    model_config_sha256: str = ""
    parameter_signature_sha256: str = ""
    paired_initialization_digest: str = ""
    token_mix_bounded_discrepancy_passed: bool = False

    @property
    def conservative_tokens_per_second(self) -> float:
        values = tuple(float(value) for value in self.tokens_per_second_by_variant.values())
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise E26AGateBlocked(f"{self.candidate_id}: missing/non-positive paired throughput")
        return min(values)


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    candidate_id: str
    token_budget: int
    candidate_config_index: int
    parameter_count: int
    context_length: int
    target_global_batch_tokens: int
    selected_microbatch_sequences: int
    accumulation_steps: int
    conservative_tokens_per_second: float
    projected_single_run_hours: float
    projected_wave_count: int
    projected_wall_hours: float
    safety_adjusted_wall_hours: float
    projected_checkpoint_storage_gib: float
    deadline_fraction: float
    selection_rule: str
    projections: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["projections"] = list(self.projections)
        payload["selection_sha256"] = sha256_canonical_json(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ScientificExecutionDeviceBinding:
    """Exact canonical CUDA device authorized by the resource receipt."""

    cli_device: str
    logical_device_index: int
    physical_device_index: int
    gpu_uuid: str
    resource_worker_visible_cuda_index: int
    resource_worker_cuda_visible_devices: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class E26AGateAdmission:
    repo_root: Path
    artifact_root: Path
    config: dict[str, Any]
    calibration_config: dict[str, Any]
    protocol: dict[str, Any]
    readiness: E26AReadiness
    paths: E26AGateInputPaths
    input_hashes: dict[str, str]
    backend_candidate_lock: dict[str, Any]
    data_lock: dict[str, Any]
    data_readiness: dict[str, Any]
    transaction_manifest: dict[str, Any]
    validation_population_lock: dict[str, Any]
    validation_episodes: tuple[TransactionEpisode, ...]
    schedule_manifest: dict[str, Any]
    numerical_audit: dict[str, Any]
    restart_audit: dict[str, Any]
    frozen_tree_receipt: dict[str, Any]
    resource_preflight: dict[str, Any]
    locked_resource_selection: CandidateSelection
    resource_policy: ResourcePolicy
    gpu_inventory: tuple[dict[str, Any], ...]
    execution_device_binding: ScientificExecutionDeviceBinding


def _strict_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E26AGateBlocked(f"Expected a YAML mapping: {path}")
    return payload


def _resolve_file(value: str | Path, field: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise E26AGateBlocked(f"{field} is missing: {value}") from error
    if not path.is_file() or path.is_symlink():
        raise E26AGateBlocked(f"{field} must be a regular non-symlink file: {path}")
    return path


def gate_input_paths(
    *,
    config: str | Path,
    calibration_config: str | Path | None = None,
    protocol_lock: str | Path,
    backend_candidate_lock: str | Path,
    backend_manifest: str | Path,
    tokenizer_manifest: str | Path,
    corpus_manifest: str | Path,
    data_lock: str | Path,
    data_readiness: str | Path,
    transaction_manifest: str | Path,
    validation_population_lock: str | Path,
    schedule_manifest: str | Path,
    numerical_audit: str | Path,
    restart_audit: str | Path,
    frozen_tree_receipt: str | Path,
    resource_preflight: str | Path,
) -> E26AGateInputPaths:
    resolved_config = _resolve_file(config, "config")
    calibration_value = (
        calibration_config
        if calibration_config is not None
        else resolved_config.parent / "e26b_calibration_lock.yaml"
    )
    raw = {
        "config": resolved_config,
        "calibration_config": calibration_value,
        "protocol_lock": protocol_lock,
        "backend_candidate_lock": backend_candidate_lock,
        "backend_manifest": backend_manifest,
        "tokenizer_manifest": tokenizer_manifest,
        "corpus_manifest": corpus_manifest,
        "data_lock": data_lock,
        "data_readiness": data_readiness,
        "transaction_manifest": transaction_manifest,
        "validation_population_lock": validation_population_lock,
        "schedule_manifest": schedule_manifest,
        "numerical_audit": numerical_audit,
        "restart_audit": restart_audit,
        "frozen_tree_receipt": frozen_tree_receipt,
        "resource_preflight": resource_preflight,
    }
    return E26AGateInputPaths(**{key: _resolve_file(value, key) for key, value in raw.items()})


def _readiness_input_path(
    readiness: Mapping[str, Any],
    field: str,
) -> Path:
    value = readiness.get(field)
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise E26AGateBlocked(f"Data readiness lacks {field}.path")
    return _resolve_file(value, f"data_readiness.{field}")


def _revalidate_stage2_data_readiness(
    *,
    paths: E26AGateInputPaths,
    recorded: dict[str, Any],
) -> None:
    """Re-run the complete raw-input audit instead of trusting an aggregate bit."""

    try:
        observed = validate_stage2_data_bundle(
            data_lock_path=paths.data_lock,
            construction_receipt_path=_readiness_input_path(recorded, "construction_source"),
            source_inventory_path=_readiness_input_path(recorded, "source_inventory"),
            source_metadata_path=_readiness_input_path(recorded, "source_metadata"),
            download_receipt_path=_readiness_input_path(recorded, "download_receipt"),
            tokenizer_manifest_path=paths.tokenizer_manifest,
            tokenizer_replay_path=_readiness_input_path(recorded, "tokenizer_replay"),
            dedup_receipt_path=_readiness_input_path(recorded, "dedup_receipt"),
            near_duplicate_audit_path=_readiness_input_path(recorded, "near_duplicate_audit"),
            memmap_receipt_path=_readiness_input_path(recorded, "general_memmap_receipt"),
            transaction_manifest_path=paths.transaction_manifest,
            schedule_manifest_path=paths.schedule_manifest,
        ).as_dict()
    except (Stage2DataReadinessError, OSError, ValueError) as error:
        raise E26AGateBlocked(f"Raw Stage-2 data bundle failed revalidation: {error}") from error
    if observed != recorded:
        raise E26AGateBlocked("Data-readiness receipt differs from a fresh raw-input revalidation")


def _deep_values(payload: Any, key: str) -> Iterator[Any]:
    if isinstance(payload, Mapping):
        for name, value in payload.items():
            if name == key:
                yield value
            yield from _deep_values(value, key)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            yield from _deep_values(value, key)


def _require_receipt_pass(payload: dict[str, Any], label: str) -> None:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not (
        schema_version == "catena-v8.1" or schema_version.startswith("catena-e26-")
    ):
        raise E26AGateBlocked(f"{label} has an unsupported schema_version")
    if payload.get("scientific_evidence") is not False:
        raise E26AGateBlocked(f"{label} must remain scientific_evidence=false")
    pass_values = [
        payload.get("passed"),
        payload.get("all_passed"),
        payload.get("status") == "PASS",
        payload.get("scientific_main_input_eligible"),
        payload.get("replay_identical"),
    ]
    if not any(value is True for value in pass_values):
        raise E26AGateBlocked(f"{label} does not record a passing audit")
    if any(value is True for value in _deep_values(payload, "main_test_opened")):
        raise E26AGateBlocked(f"{label} records main-test access")
    access_counts = list(_deep_values(payload, "main_test_access_count"))
    if any(value not in (None, 0) for value in access_counts):
        raise E26AGateBlocked(f"{label} records nonzero main-test access")
    if label == "transaction_manifest":
        if payload.get("replay_identical") is not True:
            raise E26AGateBlocked("Transaction manifest lacks deterministic replay")
        if payload.get("visible_operation_gate_address_future_query_leakage") != 0:
            raise E26AGateBlocked("Transaction manifest records visible-input leakage")
        split_audit = payload.get("split_audit")
        if not isinstance(split_audit, dict) or split_audit.get("disjoint") is not True:
            raise E26AGateBlocked("Transaction split audit is not disjoint")
        if split_audit.get("duplicates") or split_audit.get("validation_errors"):
            raise E26AGateBlocked("Transaction split audit contains duplicates/errors")


def _require_embedded_canonical_hash(
    payload: dict[str, Any],
    *,
    field: str,
    label: str,
) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or not SHA256_PATTERN.fullmatch(claimed):
        raise E26AGateBlocked(f"{label} lacks a valid {field}")
    unhashed = dict(payload)
    unhashed.pop(field)
    if sha256_canonical_json(unhashed) != claimed:
        raise E26AGateBlocked(f"{label}.{field} does not match its payload")


def _require_audit_locked_hashes(
    receipt: dict[str, Any],
    *,
    label: str,
    repo_root: Path,
    input_hashes: Mapping[str, str],
) -> None:
    locked = receipt.get("locked_hashes")
    if not isinstance(locked, dict):
        raise E26AGateBlocked(f"{label}.locked_hashes must be a mapping")
    try:
        normalized_locked = validate_e26_audit_locked_hashes(locked)
    except ValueError as error:
        raise E26AGateBlocked(f"{label}.locked_hashes is invalid: {error}") from error
    source_inventory = e26_execution_source_inventory(repo_root)
    observed = {
        key: value for key, value in input_hashes.items() if key in E26_AUDIT_LOCKED_HASH_KEYS
    }
    observed["source_tree_sha256"] = str(source_inventory["source_tree_sha256"])
    if normalized_locked != dict(sorted(observed.items())):
        mismatched = sorted(
            key
            for key in E26_AUDIT_LOCKED_HASH_KEYS
            if normalized_locked.get(key) != observed.get(key)
        )
        raise E26AGateBlocked(f"{label}.locked_hashes changed for {mismatched}")
    if receipt.get("source_inventory") != source_inventory:
        raise E26AGateBlocked(f"{label}.source_inventory differs from current execution sources")


def _require_backend_preflight_promotion(
    *,
    payload: dict[str, Any],
    candidate_lock: dict[str, Any],
    repo_root: Path,
    paths: E26AGateInputPaths,
    numerical_audit: Mapping[str, Any],
    restart_audit: Mapping[str, Any],
) -> None:
    recorded_hardware = payload.get("hardware_inventory")
    if not isinstance(recorded_hardware, list) or not recorded_hardware:
        raise E26AGateBlocked("Backend preflight manifest lacks a physical hardware inventory")
    physical_indices: list[str] = []
    for row in recorded_hardware:
        if not isinstance(row, Mapping):
            raise E26AGateBlocked("Backend hardware inventory row is malformed")
        index = row.get("physical_device_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise E26AGateBlocked("Backend hardware inventory has an invalid physical index")
        physical_indices.append(str(index))
    try:
        current_hardware = cuda_hardware_inventory(physical_indices)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise E26AGateBlocked(
            f"Cannot independently re-inventory locked CUDA devices: {error}"
        ) from error
    try:
        validated = validate_backend_preflight_manifest(
            payload,
            repo_root=repo_root,
            candidate_lock_path=paths.backend_candidate_lock,
            candidate_lock=candidate_lock,
            numerical_receipt_path=paths.numerical_audit,
            numerical_receipt=numerical_audit,
            restart_receipt_path=paths.restart_audit,
            restart_receipt=restart_audit,
            expected_hardware_inventory=current_hardware,
        )
    except (OSError, ValueError) as error:
        raise E26AGateBlocked(f"Backend preflight manifest is invalid: {error}") from error
    required_values = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_BACKEND_PREFLIGHT_MANIFEST",
        "backend_type": "TORCH_COMPILED",
        "fallback_count": 0,
        "graph_break_count": 0,
        "e26a_candidate_capable": True,
    }
    mismatched = [
        name for name, expected in required_values.items() if validated.get(name) != expected
    ]
    if mismatched:
        raise E26AGateBlocked(
            f"Backend preflight manifest capability boundary failed: {mismatched}"
        )


def candidate_numerical_coverage(
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate exact per-candidate, per-variant numerical coverage."""

    candidates = config.get("model_candidates")
    rows = receipt.get("candidate_audits")
    variants = config.get("variants")
    if not isinstance(candidates, list) or not isinstance(rows, Mapping):
        raise E26AGateBlocked("Numerical receipt lacks candidate-specific audit coverage")
    if not isinstance(variants, list) or not variants:
        raise E26AGateBlocked("E26a config lacks candidate variants")
    expected_ids = tuple(str(candidate["id"]) for candidate in candidates)
    if set(rows) != set(expected_ids):
        raise E26AGateBlocked(
            "Numerical receipt candidate IDs differ from the locked candidate table"
        )
    coverage: dict[str, bool] = {}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        row = rows[candidate_id]
        if not isinstance(row, Mapping):
            raise E26AGateBlocked(f"Malformed candidate audit: {candidate_id}")
        expected_hash = sha256_canonical_json(candidate)
        if row.get("model_config_sha256") != expected_hash:
            raise E26AGateBlocked(f"Numerical audit config hash mismatch: {candidate_id}")
        variant_rows = row.get("variants")
        if not isinstance(variant_rows, Mapping) or set(variant_rows) != set(variants):
            raise E26AGateBlocked(f"Numerical audit variant coverage mismatch: {candidate_id}")
        variant_passes: list[bool] = []
        for variant in variants:
            variant_row = variant_rows[variant]
            if not isinstance(variant_row, Mapping):
                raise E26AGateBlocked(
                    f"Malformed variant numerical audit: {candidate_id}/{variant}"
                )
            partitions = variant_row.get("arbitrary_partitions")
            accumulation = variant_row.get("gradient_accumulation")
            partition_pass = (
                isinstance(partitions, Mapping)
                and set(partitions) == {"zero_state", "prefilled_state"}
                and all(
                    isinstance(state_row, Mapping)
                    and set(state_row) == {"fp32", "bf16"}
                    and all(
                        isinstance(state_row[precision], Mapping)
                        and state_row[precision].get("passed") is True
                        for precision in ("fp32", "bf16")
                    )
                    for state_row in partitions.values()
                )
            )
            accumulation_pass = (
                isinstance(accumulation, Mapping)
                and set(accumulation) == {"fp32", "bf16"}
                and all(
                    isinstance(accumulation[precision], list)
                    and bool(accumulation[precision])
                    and all(
                        isinstance(item, Mapping) and item.get("passed") is True
                        for item in accumulation[precision]
                    )
                    for precision in ("fp32", "bf16")
                )
            )
            variant_passes.append(
                bool(variant_row.get("passed") is True and partition_pass and accumulation_pass)
            )
        coverage[candidate_id] = bool(row.get("passed") is True and all(variant_passes))
    return coverage


def _protocol_input_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    value = protocol.get("execution_inputs")
    if not isinstance(value, dict):
        value = protocol.get("input_hashes")
    if not isinstance(value, dict):
        raise E26AGateBlocked(
            "Protocol lock must bind every Stage-2 execution input in execution_inputs"
        )
    hashes: dict[str, str] = {}
    for name, digest in value.items():
        normalized = name if str(name).endswith("_sha256") else f"{name}_sha256"
        if isinstance(digest, dict):
            digest = digest.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise E26AGateBlocked(f"Invalid protocol execution-input hash: {name}")
        hashes[normalized] = digest
    return hashes


def _require_hash_binding(protocol: dict[str, Any], observed: dict[str, str]) -> None:
    locked = _protocol_input_hashes(protocol)
    # Acyclic lock DAG:
    #   protocol -> upstream inputs
    #   numerical/restart audits -> protocol + upstream inputs
    # The protocol therefore cannot bind itself or the downstream audits.
    downstream_or_self = {
        "protocol_lock_sha256",
        "backend_manifest_sha256",
        "numerical_audit_sha256",
        "restart_audit_sha256",
        "resource_preflight_sha256",
    }
    bindable = {name: digest for name, digest in observed.items() if name not in downstream_or_self}
    missing = sorted(set(bindable) - set(locked))
    if missing:
        raise E26AGateBlocked(f"Protocol lock does not bind execution inputs: {missing}")
    mismatched = sorted(name for name, digest in bindable.items() if locked.get(name) != digest)
    if mismatched:
        raise E26AGateBlocked(f"Protocol execution-input hashes changed: {mismatched}")


def _resource_policy(
    config: dict[str, Any],
    data_lock: dict[str, Any],
) -> ResourcePolicy:
    throughput = config.get("throughput")
    resource = data_lock.get("resource_policy")
    if not isinstance(throughput, dict) or not isinstance(resource, dict):
        raise E26AGateBlocked("Config/data lock lacks throughput resource policy")
    deadline_fraction = float(throughput.get("deadline_fraction_max", -1.0))
    deadline_reference = float(resource.get("deadline_reference_hours", DEADLINE_REFERENCE_HOURS))
    max_wall = float(resource.get("max_main_wall_clock_hours", -1.0))
    safety = float(resource.get("safety_time_multiplier", -1.0))
    max_storage = float(resource.get("max_main_checkpoint_storage_gib", -1.0))
    budgets = tuple(int(value) for value in resource.get("main_token_budget_candidates", ()))
    config_budgets = tuple(
        int(value) for value in throughput.get("main_token_budget_candidates", ())
    )
    if budgets != config_budgets or not budgets:
        raise E26AGateBlocked("Data-lock token budgets differ from the prospective config")
    if (
        deadline_reference != DEADLINE_REFERENCE_HOURS
        or deadline_fraction != 0.70
        or max_wall != 168.0
        or safety != 1.25
        or max_storage != 100.0
    ):
        raise E26AGateBlocked("Stage-2 resource policy differs from the prospective lock")
    policy = ResourcePolicy(
        deadline_reference_hours=deadline_reference,
        deadline_fraction_max=deadline_fraction,
        max_main_wall_clock_hours=max_wall,
        safety_time_multiplier=safety,
        max_main_checkpoint_storage_gib=max_storage,
        token_budgets=budgets,
    )
    if policy.effective_wall_cap_hours != 168.0:
        raise E26AGateBlocked("Effective resource deadline is not exactly 168 hours")
    return policy


def _gpu_inventory(minimum_devices: int = 4) -> tuple[dict[str, Any], ...]:
    if not torch.cuda.is_available():
        raise E26AGateBlocked("E26a requires CUDA")
    if torch.cuda.device_count() < minimum_devices:
        raise E26AGateBlocked(
            f"E26a requires at least {minimum_devices} visible GPUs; "
            f"observed {torch.cuda.device_count()}"
        )
    records = []
    for index in range(minimum_devices):
        props = torch.cuda.get_device_properties(index)
        gpu_uuid = _normalize_gpu_uuid(str(getattr(props, "uuid", "")))
        if not gpu_uuid:
            raise E26AGateBlocked(f"CUDA logical device {index} does not expose a GPU UUID")
        records.append(
            {
                "logical_index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": [int(props.major), int(props.minor)],
                "uuid": gpu_uuid,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
    return tuple(records)


def _normalize_gpu_uuid(value: str) -> str:
    normalized = value.strip()
    if normalized and not normalized.startswith("GPU-"):
        normalized = f"GPU-{normalized}"
    return normalized


def _selected_resource_execution_device(
    payload: Mapping[str, Any],
    selection: CandidateSelection,
) -> dict[str, Any]:
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise E26AGateBlocked("Resource preflight lacks candidate device bindings")
    selected_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("candidate_id") == selection.candidate_id
    ]
    if len(selected_rows) != 1:
        raise E26AGateBlocked("Resource preflight selected-candidate device is ambiguous")
    execution_device = selected_rows[0].get("execution_device")
    if not isinstance(execution_device, Mapping):
        raise E26AGateBlocked("Selected resource candidate lacks an execution device")
    return dict(execution_device)


def bind_scientific_execution_device(
    *,
    requested_device: torch.device | str,
    resource_preflight: Mapping[str, Any],
    selection: CandidateSelection,
    gpu_inventory: Sequence[Mapping[str, Any]],
) -> ScientificExecutionDeviceBinding:
    """Bind an explicit CLI CUDA device to the selected preflight worker GPU."""

    resolved = torch.device(requested_device)
    if resolved.type != "cuda" or resolved.index is None or resolved.index < 0:
        raise E26AGateBlocked(
            "Scientific E26a requires an explicit CUDA index bound to the resource receipt"
        )
    inventory_by_logical = {
        int(row["logical_index"]): row
        for row in gpu_inventory
        if isinstance(row.get("logical_index"), int)
    }
    observed = inventory_by_logical.get(resolved.index)
    if observed is None:
        raise E26AGateBlocked(
            f"Requested CUDA logical device {resolved.index} is absent from admission inventory"
        )
    locked = _selected_resource_execution_device(resource_preflight, selection)
    physical_index = locked.get("physical_device_index")
    worker_visible_index = locked.get("worker_visible_cuda_index")
    locked_uuid = locked.get("gpu_uuid")
    if (
        isinstance(physical_index, bool)
        or not isinstance(physical_index, int)
        or worker_visible_index != 0
        or not isinstance(locked_uuid, str)
    ):
        raise E26AGateBlocked("Selected resource candidate has an invalid CUDA binding")
    normalized_locked_uuid = _normalize_gpu_uuid(locked_uuid)
    observed_uuid = _normalize_gpu_uuid(str(observed.get("uuid", "")))
    if not normalized_locked_uuid or observed_uuid != normalized_locked_uuid:
        raise E26AGateBlocked(
            "Requested CUDA device UUID differs from the selected resource-preflight worker"
        )
    return ScientificExecutionDeviceBinding(
        cli_device=f"cuda:{resolved.index}",
        logical_device_index=resolved.index,
        physical_device_index=physical_index,
        gpu_uuid=normalized_locked_uuid,
        resource_worker_visible_cuda_index=worker_visible_index,
        resource_worker_cuda_visible_devices=str(locked.get("cuda_visible_devices", "")),
    )


def _runtime_cuda_device(logical_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available() or logical_index >= torch.cuda.device_count():
        raise E26AGateBlocked("Locked scientific CUDA device is no longer available")
    properties = torch.cuda.get_device_properties(logical_index)
    return {
        "logical_index": logical_index,
        "gpu_uuid": _normalize_gpu_uuid(str(getattr(properties, "uuid", ""))),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def require_locked_execution_device(
    admission: E26AGateAdmission,
    requested_device: torch.device | str,
) -> torch.device:
    """Fail closed unless execution remains on the admitted physical GPU."""

    resolved = torch.device(requested_device)
    binding = admission.execution_device_binding
    if (
        resolved.type != "cuda"
        or resolved.index is None
        or f"cuda:{resolved.index}" != binding.cli_device
        or resolved.index != binding.logical_device_index
    ):
        raise E26AGateBlocked(
            "Requested scientific CUDA device differs from the admitted resource device"
        )
    locked = _selected_resource_execution_device(
        admission.resource_preflight,
        admission.locked_resource_selection,
    )
    expected_locked = {
        "physical_device_index": binding.physical_device_index,
        "gpu_uuid": binding.gpu_uuid,
        "worker_visible_cuda_index": binding.resource_worker_visible_cuda_index,
        "cuda_visible_devices": binding.resource_worker_cuda_visible_devices,
    }
    mismatched = [
        field
        for field, expected in expected_locked.items()
        if (
            _normalize_gpu_uuid(str(locked.get(field, "")))
            if field == "gpu_uuid"
            else locked.get(field)
        )
        != expected
    ]
    if mismatched:
        raise E26AGateBlocked(f"Admitted resource execution-device binding changed: {mismatched}")
    runtime = _runtime_cuda_device(binding.logical_device_index)
    if runtime.get("gpu_uuid") != binding.gpu_uuid:
        raise E26AGateBlocked(
            "Runtime CUDA UUID differs from the admitted resource-preflight worker"
        )
    return resolved


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise E26AGateBlocked(f"Git command failed: {' '.join(args)}") from error


def validate_resource_preflight_receipt(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    paths: E26AGateInputPaths,
    config: dict[str, Any],
    policy: ResourcePolicy,
    input_hashes: Mapping[str, str],
    backend_preflight: Mapping[str, Any],
    numerical_audit: Mapping[str, Any],
    restart_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], CandidateSelection]:
    """Revalidate the non-evidence resource lock and its fixed selection."""

    _require_embedded_canonical_hash(
        payload,
        field="receipt_sha256",
        label="resource_preflight",
    )
    required_values = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_RESOURCE_PREFLIGHT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "claim_ceiling": "RESOURCE_FEASIBILITY_ONLY",
        "main_test_opened": False,
        "scientific_e26a_started": False,
        "scientific_e26b_started": False,
        "scientific_main_started": False,
        "canonical_e26_artifact_created": False,
        "passed": True,
    }
    mismatched_values = [
        key for key, expected in required_values.items() if payload.get(key) != expected
    ]
    if mismatched_values:
        raise E26AGateBlocked(
            f"Resource preflight evidence/capability boundary failed: {mismatched_values}"
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise E26AGateBlocked("Resource preflight lacks a full source commit")
    try:
        _git(repo_root, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
        _git(repo_root, "merge-base", "--is-ancestor", source_commit, "HEAD")
    except E26AGateBlocked as error:
        raise E26AGateBlocked(
            "Resource preflight source commit is not an ancestor of HEAD"
        ) from error
    source_inventory = e26_execution_source_inventory(repo_root)
    if payload.get("source_inventory") != source_inventory:
        raise E26AGateBlocked("Resource preflight source inventory differs from execution source")
    raw_locked = payload.get("locked_hashes")
    if not isinstance(raw_locked, Mapping):
        raise E26AGateBlocked("Resource preflight lacks locked hashes")
    try:
        normalized_locked = validate_e26_audit_locked_hashes(raw_locked)
    except ValueError as error:
        raise E26AGateBlocked(f"Resource preflight locked hashes are invalid: {error}") from error
    expected_locked = {
        key: value for key, value in input_hashes.items() if key in E26_AUDIT_LOCKED_HASH_KEYS
    }
    expected_locked["source_tree_sha256"] = str(source_inventory["source_tree_sha256"])
    if normalized_locked != dict(sorted(expected_locked.items())):
        raise E26AGateBlocked("Resource preflight upstream locked hashes changed")

    upstream = payload.get("upstream_receipts")
    if not isinstance(upstream, Mapping):
        raise E26AGateBlocked("Resource preflight lacks upstream receipt bindings")
    expected_upstream = {
        "backend_manifest": {
            "sha256": sha256_file(paths.backend_manifest),
            "manifest_sha256": backend_preflight.get("manifest_sha256"),
        },
        "numerical_audit": {
            "sha256": sha256_file(paths.numerical_audit),
            "receipt_sha256": numerical_audit.get("receipt_sha256"),
        },
        "restart_audit": {
            "sha256": sha256_file(paths.restart_audit),
            "receipt_sha256": restart_audit.get("receipt_sha256"),
        },
    }
    if dict(upstream) != expected_upstream:
        raise E26AGateBlocked("Resource preflight upstream receipt hashes changed")

    recorded_hardware = payload.get("hardware_inventory")
    if not isinstance(recorded_hardware, list) or not recorded_hardware:
        raise E26AGateBlocked("Resource preflight lacks hardware inventory")
    physical_indices: list[str] = []
    for row in recorded_hardware:
        if not isinstance(row, Mapping):
            raise E26AGateBlocked("Resource preflight hardware row is malformed")
        physical_index = row.get("physical_device_index")
        if (
            isinstance(physical_index, bool)
            or not isinstance(physical_index, int)
            or physical_index < 0
        ):
            raise E26AGateBlocked("Resource preflight hardware index is invalid")
        physical_indices.append(str(physical_index))
    if len(set(physical_indices)) != len(physical_indices):
        raise E26AGateBlocked("Resource preflight repeats a physical GPU")
    try:
        current_hardware = cuda_hardware_inventory(physical_indices)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise E26AGateBlocked(f"Cannot re-inventory resource-preflight GPUs: {error}") from error
    if current_hardware != recorded_hardware:
        raise E26AGateBlocked("Resource preflight GPU inventory changed")
    hardware_by_index = {int(row["physical_device_index"]): row for row in current_hardware}

    candidates = config.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise E26AGateBlocked("E26a config lacks resource candidates")
    expected_ids = [str(candidate["id"]) for candidate in candidates]
    rows = payload.get("candidates")
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("candidate_id") for row in rows] != expected_ids
    ):
        raise E26AGateBlocked("Resource preflight candidate order differs from the config lock")
    measurements: list[CandidateMeasurement] = []
    for candidate, raw_row in zip(candidates, rows, strict=True):
        if not isinstance(raw_row, Mapping):
            raise E26AGateBlocked("Resource preflight candidate row is malformed")
        expected_config_sha = sha256_canonical_json(candidate)
        if (
            raw_row.get("candidate_config_sha256") != expected_config_sha
            or raw_row.get("model_config_sha256") != expected_config_sha
        ):
            raise E26AGateBlocked("Resource preflight candidate config hash changed")
        measurement = candidate_measurement_from_mapping(raw_row)
        expected_projections = list(project_candidate_resources(measurement, policy))
        if raw_row.get("resource_projections") != expected_projections:
            raise E26AGateBlocked("Resource preflight projections are not reproducible")
        for field in (
            "worker_spec_sha256",
            "worker_receipt_sha256",
            "worker_report_sha256",
        ):
            value = raw_row.get(field)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                raise E26AGateBlocked(f"Resource preflight candidate lacks valid {field}")
        execution_device = raw_row.get("execution_device")
        if not isinstance(execution_device, Mapping):
            raise E26AGateBlocked("Resource preflight candidate lacks execution device")
        physical_index = execution_device.get("physical_device_index")
        if (
            isinstance(physical_index, bool)
            or not isinstance(physical_index, int)
            or physical_index < 0
        ):
            raise E26AGateBlocked("Resource preflight worker physical device is invalid")
        hardware = hardware_by_index.get(physical_index, {})
        expected_device = {
            "physical_device_index": physical_index,
            "gpu_uuid": hardware.get("gpu_uuid"),
            "worker_visible_cuda_index": 0,
            "cuda_visible_devices": str(physical_index),
            "name": hardware.get("name"),
            "total_memory_bytes": hardware.get("total_memory_bytes"),
            "compute_capability": hardware.get("compute_capability"),
            "observation": "PYTORCH_VISIBLE_DEVICE_UUID_VERIFIED",
        }
        if dict(execution_device) != expected_device:
            raise E26AGateBlocked(
                "Resource preflight worker device is absent from hardware inventory"
            )
        target_accumulation = raw_row.get("target_gradient_accumulation")
        throughput_config = config.get("throughput")
        variants = config.get("variants")
        if (
            not isinstance(target_accumulation, Mapping)
            or not isinstance(throughput_config, Mapping)
            or not isinstance(variants, list)
            or not variants
        ):
            raise E26AGateBlocked(
                "Resource preflight lacks the target-layout gradient-accumulation audit"
            )
        accumulation_payload = dict(target_accumulation)
        accumulation_sha256 = accumulation_payload.pop("audit_sha256", None)
        if not isinstance(accumulation_sha256, str) or accumulation_sha256 != sha256_canonical_json(
            accumulation_payload
        ):
            raise E26AGateBlocked("Target-layout gradient-accumulation audit hash changed")
        context_length = int(measurement.context_length)
        target_global_batch_tokens = int(throughput_config["target_global_batch_tokens"])
        if target_global_batch_tokens % context_length:
            raise E26AGateBlocked("Locked global token batch is not divisible by context length")
        global_sequences = target_global_batch_tokens // context_length
        selected_microbatch = int(measurement.selected_microbatch_sequences)
        selected_and_smaller = sorted(
            {
                int(value)
                for value in throughput_config["microbatch_size_candidates"]
                if 0 < int(value) <= selected_microbatch and global_sequences % int(value) == 0
            },
            reverse=True,
        )
        expected_microbatches = [global_sequences]
        expected_microbatches.extend(
            value for value in selected_and_smaller if value != global_sequences
        )
        expected_layouts = [
            [microbatch] * (global_sequences // microbatch) for microbatch in expected_microbatches
        ]
        expected_bindings = {
            "candidate_id": measurement.candidate_id,
            "model_config_sha256": expected_config_sha,
            "context_length": context_length,
            "target_global_batch_tokens": target_global_batch_tokens,
            "global_batch_sequences": global_sequences,
            "selected_microbatch_sequences": selected_microbatch,
            "accumulation_steps": measurement.accumulation_steps,
            "audited_microbatch_sequences": expected_microbatches,
            "accumulation_layouts": expected_layouts,
            "passed": True,
        }
        if (
            len(expected_microbatches) < 2
            or not selected_and_smaller
            or selected_and_smaller[0] != selected_microbatch
            or any(
                target_accumulation.get(key) != expected
                for key, expected in expected_bindings.items()
            )
        ):
            raise E26AGateBlocked(
                "Target-layout gradient-accumulation audit differs from the locked layout"
            )
        variant_audits = target_accumulation.get("variants")
        if not isinstance(variant_audits, Mapping) or set(variant_audits) != set(variants):
            variant_coverage_valid = False
        else:
            variant_coverage_valid = True
            for raw_variant in variants:
                variant = str(raw_variant)
                variant_row = variant_audits.get(variant)
                if not isinstance(variant_row, Mapping):
                    variant_coverage_valid = False
                    break
                rows = variant_row.get("rows")
                diagnostics = variant_row.get("compiled_backend_diagnostics")
                if (
                    not isinstance(rows, list)
                    or not all(isinstance(row, Mapping) for row in rows)
                    or not isinstance(diagnostics, Mapping)
                    or variant_row.get("variant") != variant
                    or variant_row.get("precision") != "bf16_actual_training"
                    or variant_row.get("passed") is not True
                    or [row.get("microbatch_sizes") for row in rows] != expected_layouts
                    or any(row.get("passed") is not True for row in rows)
                    or diagnostics.get("fallback_count") != 0
                    or diagnostics.get("graph_break_count") != 0
                ):
                    variant_coverage_valid = False
                    break
        if not variant_coverage_valid:
            raise E26AGateBlocked("Target-layout gradient-accumulation variant coverage is invalid")
        measurements.append(measurement)
    expected_selection = select_candidate(
        config=config,
        measurements=measurements,
        policy=policy,
    )
    if payload.get("selection") != expected_selection.as_dict():
        raise E26AGateBlocked("Resource preflight selection differs from deterministic selection")
    return payload, expected_selection


def validate_scientific_gate_admission(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    execution_ack: str,
    paths: E26AGateInputPaths,
    expected_resource_preflight_sha256: str,
    execution_device: torch.device | str,
    require_gpu_inventory: bool = True,
) -> E26AGateAdmission:
    """Validate all immutable dependencies before a canonical run directory exists."""

    if execution_ack != E26A_EXECUTION_ACK:
        raise E26AGateBlocked("Explicit E26a scientific-gate authorization is absent")
    if not isinstance(expected_resource_preflight_sha256, str) or not SHA256_PATTERN.fullmatch(
        expected_resource_preflight_sha256
    ):
        raise E26AGateBlocked(
            "Explicit approved resource-preflight file SHA-256 is absent or invalid"
        )
    if sha256_file(paths.resource_preflight) != expected_resource_preflight_sha256:
        raise E26AGateBlocked(
            "Resource-preflight file differs from the explicitly approved SHA-256"
        )
    repo = Path(repo_root).expanduser().resolve(strict=True)
    config = _strict_yaml(paths.config)
    calibration_config = _strict_yaml(paths.calibration_config)
    safety = config.get("safety")
    if not isinstance(safety, dict):
        raise E26AGateBlocked("E26a config lacks safety settings")
    expected_worktree = Path(str(safety["expected_worktree"])).resolve(strict=True)
    if repo != expected_worktree:
        raise E26AGateBlocked(f"E26a must run from {expected_worktree}, got {repo}")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise E26AGateBlocked("E26a scientific gate requires a clean committed worktree")
    canonical = Path(str(safety["canonical_artifact_root"])).expanduser().resolve()
    observed_root = Path(artifact_root).expanduser().resolve()
    if observed_root != canonical:
        raise E26AGateBlocked(f"E26a scientific gate requires canonical artifact root {canonical}")

    readiness = validate_e26a_readiness(
        repo_root=repo,
        config_path=paths.config,
        protocol_lock_path=paths.protocol_lock,
        backend_manifest_path=paths.backend_manifest,
        tokenizer_manifest_path=paths.tokenizer_manifest,
        corpus_manifest_path=paths.corpus_manifest,
    )
    backend_source_commit = readiness.control.get("backend_source_commit")
    if not isinstance(backend_source_commit, str):
        raise E26AGateBlocked("Readiness lacks backend source commit")
    _git(repo, "merge-base", "--is-ancestor", backend_source_commit, "HEAD")
    protocol = read_json_object_strict(paths.protocol_lock)
    input_hashes = paths.hashes()
    input_hashes["source_tree_sha256"] = str(
        e26_execution_source_inventory(repo)["source_tree_sha256"]
    )
    _require_hash_binding(protocol, input_hashes)

    raw_backend_candidate_lock = read_json_object_strict(paths.backend_candidate_lock)
    candidates = config.get("model_candidates")
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, Mapping) for candidate in candidates
    ):
        raise E26AGateBlocked("E26a config lacks a valid candidate table")
    try:
        backend_candidate_lock = validate_backend_candidate_lock(
            raw_backend_candidate_lock,
            repo_root=repo,
            config_path=paths.config,
            candidates=candidates,
        )
    except (OSError, ValueError) as error:
        raise E26AGateBlocked(f"Backend candidate lock is invalid: {error}") from error
    backend_preflight = read_json_object_strict(paths.backend_manifest)
    data_lock = _strict_yaml(paths.data_lock)
    data_readiness = read_json_object_strict(paths.data_readiness)
    transaction_manifest = read_json_object_strict(paths.transaction_manifest)
    validation_population_lock = read_json_object_strict(paths.validation_population_lock)
    schedule_manifest = read_json_object_strict(paths.schedule_manifest)
    numerical_audit = read_json_object_strict(paths.numerical_audit)
    restart_audit = read_json_object_strict(paths.restart_audit)
    frozen_tree_receipt = read_json_object_strict(paths.frozen_tree_receipt)
    resource_preflight = read_json_object_strict(paths.resource_preflight)
    for label, receipt in (
        ("data_readiness", data_readiness),
        ("transaction_manifest", transaction_manifest),
        ("validation_population_lock", validation_population_lock),
        ("schedule_manifest", schedule_manifest),
        ("numerical_audit", numerical_audit),
        ("restart_audit", restart_audit),
        ("frozen_tree_receipt", frozen_tree_receipt),
        ("resource_preflight", resource_preflight),
    ):
        _require_receipt_pass(receipt, label)
    _require_embedded_canonical_hash(
        transaction_manifest,
        field="manifest_sha256",
        label="transaction_manifest",
    )
    _require_embedded_canonical_hash(
        validation_population_lock,
        field="manifest_sha256",
        label="validation_population_lock",
    )
    try:
        validation_episodes = validate_e26a_validation_population_lock(
            validation_population_lock,
            config=config,
        )
    except E26AValidationPopulationError as error:
        raise E26AGateBlocked(f"E26a validation population lock is invalid: {error}") from error
    _require_embedded_canonical_hash(
        schedule_manifest,
        field="manifest_sha256",
        label="schedule_manifest",
    )
    _require_embedded_canonical_hash(
        numerical_audit,
        field="receipt_sha256",
        label="numerical_audit",
    )
    _require_embedded_canonical_hash(
        restart_audit,
        field="receipt_sha256",
        label="restart_audit",
    )
    _require_embedded_canonical_hash(
        frozen_tree_receipt,
        field="receipt_sha256",
        label="frozen_tree_receipt",
    )
    _require_embedded_canonical_hash(
        resource_preflight,
        field="receipt_sha256",
        label="resource_preflight",
    )
    _require_audit_locked_hashes(
        numerical_audit,
        label="numerical_audit",
        repo_root=repo,
        input_hashes=input_hashes,
    )
    candidate_numerical_coverage(config, numerical_audit)
    _require_audit_locked_hashes(
        restart_audit,
        label="restart_audit",
        repo_root=repo,
        input_hashes=input_hashes,
    )
    try:
        restart_coverage = validate_restart_audit_coverage(
            restart_audit,
            expected_candidate_ids=[str(candidate["id"]) for candidate in candidates],
        )
    except ValueError as error:
        raise E26AGateBlocked(f"Restart audit candidate coverage is invalid: {error}") from error
    if not all(restart_coverage.values()):
        raise E26AGateBlocked("Restart audit did not pass every locked model candidate")
    restart_cases = restart_audit.get("resume_cases")
    assert isinstance(restart_cases, Mapping)
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        expected_config_hash = sha256_canonical_json(candidate)
        mismatched_cases = [
            case_id
            for case_id, row in restart_cases.items()
            if str(case_id).startswith(f"{candidate_id}__")
            and (
                not isinstance(row, Mapping)
                or row.get("model_config_sha256") != expected_config_hash
            )
        ]
        if mismatched_cases:
            raise E26AGateBlocked(f"Restart audit config hash mismatch: {mismatched_cases}")
    _require_backend_preflight_promotion(
        payload=backend_preflight,
        candidate_lock=backend_candidate_lock,
        repo_root=repo,
        paths=paths,
        numerical_audit=numerical_audit,
        restart_audit=restart_audit,
    )
    if data_readiness.get("scientific_main_input_eligible") is not True:
        raise E26AGateBlocked("Data readiness is not eligible as a scientific input")
    _revalidate_stage2_data_readiness(paths=paths, recorded=data_readiness)
    try:
        frozen_tree_receipt = validate_frozen_invariance_receipt(
            frozen_tree_receipt,
            data_lock=data_lock,
        )
    except FrozenInvarianceError as error:
        raise E26AGateBlocked(f"Frozen live/source/artifact invariance failed: {error}") from error
    policy = _resource_policy(config, data_lock)
    resource_preflight, locked_resource_selection = validate_resource_preflight_receipt(
        resource_preflight,
        repo_root=repo,
        paths=paths,
        config=config,
        policy=policy,
        input_hashes=input_hashes,
        backend_preflight=backend_preflight,
        numerical_audit=numerical_audit,
        restart_audit=restart_audit,
    )
    inventory = _gpu_inventory() if require_gpu_inventory else ()
    execution_device_binding = bind_scientific_execution_device(
        requested_device=execution_device,
        resource_preflight=resource_preflight,
        selection=locked_resource_selection,
        gpu_inventory=inventory,
    )
    return E26AGateAdmission(
        repo_root=repo,
        artifact_root=canonical,
        config=config,
        calibration_config=calibration_config,
        protocol=protocol,
        readiness=readiness,
        paths=paths,
        input_hashes=input_hashes,
        backend_candidate_lock=backend_candidate_lock,
        data_lock=data_lock,
        data_readiness=data_readiness,
        transaction_manifest=transaction_manifest,
        validation_population_lock=validation_population_lock,
        validation_episodes=validation_episodes,
        schedule_manifest=schedule_manifest,
        numerical_audit=numerical_audit,
        restart_audit=restart_audit,
        frozen_tree_receipt=frozen_tree_receipt,
        resource_preflight=resource_preflight,
        locked_resource_selection=locked_resource_selection,
        resource_policy=policy,
        gpu_inventory=inventory,
        execution_device_binding=execution_device_binding,
    )


def project_candidate_resources(
    measurement: CandidateMeasurement,
    policy: ResourcePolicy,
) -> tuple[dict[str, Any], ...]:
    tokens_per_second = measurement.conservative_tokens_per_second
    wave_count = math.ceil(policy.main_runs / policy.gpu_lanes)
    projections: list[dict[str, Any]] = []
    for token_budget in policy.token_budgets:
        single_hours = token_budget / tokens_per_second / 3600.0
        wall_hours = single_hours * wave_count
        adjusted_hours = wall_hours * policy.safety_time_multiplier
        checkpoint_count = math.ceil(token_budget / policy.save_every_tokens) + 1
        storage_gib = measurement.checkpoint_bytes * checkpoint_count * policy.main_runs / GIB
        deadline_fraction = adjusted_hours / policy.deadline_reference_hours
        eligible = (
            adjusted_hours <= policy.max_main_wall_clock_hours
            and deadline_fraction <= policy.deadline_fraction_max
            and storage_gib <= policy.max_main_checkpoint_storage_gib
        )
        projections.append(
            {
                "token_budget": token_budget,
                "single_run_hours": single_hours,
                "wave_count": wave_count,
                "wall_hours": wall_hours,
                "safety_adjusted_wall_hours": adjusted_hours,
                "deadline_reference_hours": policy.deadline_reference_hours,
                "deadline_fraction": deadline_fraction,
                "checkpoint_count_per_run": checkpoint_count,
                "checkpoint_storage_gib": storage_gib,
                "eligible": eligible,
            }
        )
    return tuple(projections)


def select_candidate(
    *,
    config: Mapping[str, Any],
    measurements: Sequence[CandidateMeasurement],
    policy: ResourcePolicy,
) -> CandidateSelection:
    """Select the first eligible candidate in locked config order, then max budget."""

    by_id = {row.candidate_id: row for row in measurements}
    candidates = config.get("model_candidates")
    if not isinstance(candidates, list):
        raise E26AGateBlocked("E26a config lacks model_candidates")
    selection_config = config.get("candidate_selection")
    if isinstance(selection_config, Mapping):
        parameter_min = int(selection_config.get("parameter_count_min", 35_000_000))
        parameter_max = int(selection_config.get("parameter_count_max", 50_000_000))
    else:
        parameter_min = 35_000_000
        parameter_max = 50_000_000
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            raise E26AGateBlocked("Malformed model candidate in locked config")
        candidate_id = str(candidate["id"])
        measurement = by_id.get(candidate_id)
        if measurement is None:
            raise E26AGateBlocked(f"Candidate table lacks {candidate_id}")
        if not parameter_min <= measurement.parameter_count <= parameter_max:
            continue
        if (
            not measurement.matching_passed
            or not measurement.numerical_passed
            or not measurement.token_mix_bounded_discrepancy_passed
        ):
            continue
        if measurement.graph_break_count != 0 or measurement.fallback_count != 0:
            continue
        projections = project_candidate_resources(measurement, policy)
        feasible = [row for row in projections if row["eligible"]]
        if not feasible:
            continue
        measured_global_batch_tokens = (
            measurement.context_length
            * measurement.selected_microbatch_sequences
            * measurement.accumulation_steps
        )
        throughput_config = config.get("throughput")
        locked_global_batch_tokens = (
            int(throughput_config["target_global_batch_tokens"])
            if isinstance(throughput_config, Mapping)
            and "target_global_batch_tokens" in throughput_config
            else measured_global_batch_tokens
        )
        if (
            measurement.context_length <= 0
            or measurement.selected_microbatch_sequences <= 0
            or measurement.accumulation_steps <= 0
            or measured_global_batch_tokens != locked_global_batch_tokens
        ):
            continue
        selected_budget = max(feasible, key=lambda row: int(row["token_budget"]))
        return CandidateSelection(
            candidate_id=candidate_id,
            token_budget=int(selected_budget["token_budget"]),
            candidate_config_index=index,
            parameter_count=measurement.parameter_count,
            context_length=measurement.context_length,
            target_global_batch_tokens=locked_global_batch_tokens,
            selected_microbatch_sequences=measurement.selected_microbatch_sequences,
            accumulation_steps=measurement.accumulation_steps,
            conservative_tokens_per_second=measurement.conservative_tokens_per_second,
            projected_single_run_hours=float(selected_budget["single_run_hours"]),
            projected_wave_count=int(selected_budget["wave_count"]),
            projected_wall_hours=float(selected_budget["wall_hours"]),
            safety_adjusted_wall_hours=float(selected_budget["safety_adjusted_wall_hours"]),
            projected_checkpoint_storage_gib=float(selected_budget["checkpoint_storage_gib"]),
            deadline_fraction=float(selected_budget["deadline_fraction"]),
            selection_rule=(
                "FIRST_PASSING_CONFIG_ORDER_AFTER_PARAMETER_MATCHING_NUMERICAL_"
                "AND_RESOURCE_GATES_THEN_LARGEST_ELIGIBLE_TOKEN_BUDGET"
            ),
            projections=projections,
        )
    raise E26AGateBlocked("No prospective candidate satisfies all resource and gate criteria")


def require_locked_resource_selection(
    admission: E26AGateAdmission,
    observed: CandidateSelection,
) -> None:
    """Prevent canonical remeasurement from changing the preflight choice."""

    locked = admission.locked_resource_selection
    locked_identity = (
        locked.candidate_id,
        locked.token_budget,
        locked.context_length,
        locked.target_global_batch_tokens,
        locked.selected_microbatch_sequences,
        locked.accumulation_steps,
    )
    observed_identity = (
        observed.candidate_id,
        observed.token_budget,
        observed.context_length,
        observed.target_global_batch_tokens,
        observed.selected_microbatch_sequences,
        observed.accumulation_steps,
    )
    if observed_identity != locked_identity:
        raise E26AGateBlocked(
            "Canonical resource remeasurement changed the locked selection: "
            f"preflight={locked_identity}, canonical={observed_identity}"
        )


def zero_main_test_access_ledger(
    *,
    validation_population_lock: Mapping[str, Any],
    validation_population_lock_sha256: str,
) -> dict[str, Any]:
    """Create the exact validation-only access contract for E26a.

    This is deliberately not a free-form declaration.  The only population
    materialized by the E26a admission path is the deterministic validation
    lock, and the ledger binds both that file and its episode records.  Final
    validation compares the complete payload with a fresh reconstruction, so
    adding a permitted split and recomputing a self-hash is rejected.
    """

    if not isinstance(validation_population_lock_sha256, str) or not SHA256_PATTERN.fullmatch(
        validation_population_lock_sha256
    ):
        raise ArtifactContractError("Validation-population file hash is not a lowercase SHA-256")
    records_sha256 = validation_population_lock.get("records_sha256")
    episode_count = validation_population_lock.get("episode_count")
    if (
        validation_population_lock.get("manifest_type") != "E26A_VALIDATION_POPULATION_LOCK"
        or validation_population_lock.get("split") != "validation"
        or not isinstance(records_sha256, str)
        or not SHA256_PATTERN.fullmatch(records_sha256)
        or isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
    ):
        raise ArtifactContractError(
            "Validation-population lock cannot establish a validation-only ledger"
        )
    payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26A_MAIN_TEST_ACCESS_LEDGER",
        "scientific_evidence": False,
        "main_test_opened": False,
        "main_test_access_count": 0,
        "heldout_domain_opened": False,
        "heldout_domain_access_count": 0,
        "permitted_materialized_splits": ["validation"],
        "forbidden_materialized_splits": ["main_test", "heldout_domain"],
        "access_instrumentation": "VALIDATION_POPULATION_LOCK_ONLY",
        "validation_population_lock_sha256": validation_population_lock_sha256,
        "validation_population_records_sha256": records_sha256,
        "validation_episode_count": episode_count,
    }
    payload["ledger_sha256"] = sha256_canonical_json(payload)
    return payload


@contextmanager
def exclusive_scientific_gate_lock(
    artifact_root: str | Path,
    *,
    experiment: str = "e26a_operator_data_gate",
) -> Iterator[Path]:
    """Hold a nonblocking OS lock for one canonical E26a gate execution."""

    root = Path(artifact_root).expanduser().resolve()
    experiment_root = root / experiment
    experiment_root.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_root / ".scientific_gate.lock"
    if lock_path.is_symlink():
        raise ArtifactContractError(f"Scientific-gate lock cannot be a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise E26AGateBlocked(
                "Another E26a scientific gate holds the execution lock"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def create_scientific_gate_run(admission: E26AGateAdmission) -> ArtifactRun:
    """Create a canonical evidence-bounded E26a run after admission and locking."""

    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=admission.artifact_root,
        canonical_artifact_root=admission.artifact_root,
        run_mode="MAIN",
        dry_run=False,
        source_root=admission.repo_root,
        scientific_evidence=False,
        evidence_tier=E26A_EVIDENCE_TIER,
        claim_ceiling=E26A_CLAIM_CEILING,
    )
    run.write("protocol_lock.json", admission.protocol)
    run.write(
        "input_hashes.json",
        {
            "schema_version": "catena-v8.1",
            "paths": admission.paths.as_dict(),
            "hashes": admission.input_hashes,
            "aggregate_sha256": sha256_canonical_json(admission.input_hashes),
        },
    )
    run.write("scientific_data_readiness.json", admission.data_readiness)
    run.write("backend_candidate_lock_input.json", admission.backend_candidate_lock)
    run.write(
        "validation_population_lock_input.json",
        admission.validation_population_lock,
    )
    run.write("numerical_audit_input.json", admission.numerical_audit)
    run.write("restart_audit_input.json", admission.restart_audit)
    run.write("resource_preflight_input.json", admission.resource_preflight)
    run.write("gpu_inventory.json", list(admission.gpu_inventory))
    run.write(
        "scientific_execution_device_binding.json",
        admission.execution_device_binding.as_dict(),
    )
    run.write(
        "main_test_access_ledger.json",
        zero_main_test_access_ledger(
            validation_population_lock=admission.validation_population_lock,
            validation_population_lock_sha256=admission.input_hashes[
                "validation_population_lock_sha256"
            ],
        ),
    )
    for name in (
        "training_metrics.jsonl",
        "evaluation_metrics.jsonl",
        "seed_effects.jsonl",
        "candidate_table.jsonl",
        "throughput_metrics.jsonl",
        "pilot_training_metrics.jsonl",
        "pilot_evaluation_metrics.jsonl",
    ):
        (run.run_dir / name).touch()
    return run


def assert_main_test_unopened(
    run: ArtifactRun,
    *,
    validation_population_lock: Mapping[str, Any],
    validation_population_lock_sha256: str,
) -> None:
    ledger = read_json_object_strict(run.run_dir / "main_test_access_ledger.json")
    expected = zero_main_test_access_ledger(
        validation_population_lock=validation_population_lock,
        validation_population_lock_sha256=validation_population_lock_sha256,
    )
    if ledger != expected:
        raise ArtifactContractError(
            "E26a main-test access ledger differs from the exact validation-only contract"
        )


def candidate_measurement_from_mapping(payload: Mapping[str, Any]) -> CandidateMeasurement:
    throughputs = payload.get("tokens_per_second_by_variant")
    if not isinstance(throughputs, dict):
        raise E26AGateBlocked("Candidate measurement lacks paired throughput")
    return CandidateMeasurement(
        candidate_id=str(payload["candidate_id"]),
        parameter_count=int(payload["parameter_count"]),
        matching_passed=bool(payload["matching_passed"]),
        numerical_passed=bool(payload["numerical_passed"]),
        tokens_per_second_by_variant={str(key): float(value) for key, value in throughputs.items()},
        checkpoint_bytes=int(payload["checkpoint_bytes"]),
        peak_allocated_bytes=int(payload["peak_allocated_bytes"]),
        peak_reserved_bytes=int(payload["peak_reserved_bytes"]),
        p50_step_seconds=float(payload["p50_step_seconds"]),
        p95_step_seconds=float(payload["p95_step_seconds"]),
        compile_seconds=float(payload["compile_seconds"]),
        graph_break_count=int(payload["graph_break_count"]),
        fallback_count=int(payload["fallback_count"]),
        context_length=int(payload.get("context_length", 0)),
        selected_microbatch_sequences=int(payload.get("selected_microbatch_sequences", 0)),
        accumulation_steps=int(payload.get("accumulation_steps", 0)),
        measured_optimizer_steps=int(payload.get("measured_optimizer_steps", 0)),
        descriptive_stability_steps=int(payload.get("descriptive_stability_steps", 0)),
        model_config_sha256=str(payload.get("model_config_sha256", "")),
        parameter_signature_sha256=str(payload.get("parameter_signature_sha256", "")),
        paired_initialization_digest=str(payload.get("paired_initialization_digest", "")),
        token_mix_bounded_discrepancy_passed=bool(
            payload.get("token_mix_bounded_discrepancy_passed", False)
        ),
    )


def load_candidate_measurements(path: str | Path) -> tuple[CandidateMeasurement, ...]:
    source = _resolve_file(path, "candidate_measurements")
    payload = read_json_object_strict(source)
    if payload.get("schema_version") != "catena-v8.1":
        raise E26AGateBlocked("Candidate measurement bundle has the wrong schema version")
    if payload.get("scientific_evidence") is not False:
        raise E26AGateBlocked("Candidate measurements must not claim scientific evidence")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise E26AGateBlocked("Candidate measurement bundle is empty")
    return tuple(candidate_measurement_from_mapping(row) for row in rows)


def _candidate_row(measurement: CandidateMeasurement) -> dict[str, Any]:
    payload = asdict(measurement)
    payload["conservative_tokens_per_second"] = measurement.conservative_tokens_per_second
    return payload


def finalize_scientific_gate_run(
    *,
    run: ArtifactRun,
    admission: E26AGateAdmission,
    measurements: Sequence[CandidateMeasurement],
    gates: Sequence[Mapping[str, Any]],
    model_manifest: Mapping[str, Any],
    backend_manifest: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    pilot_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize E26a only; this function has no downstream-launch capability."""

    assert_main_test_unopened(
        run,
        validation_population_lock=admission.validation_population_lock,
        validation_population_lock_sha256=admission.input_hashes[
            "validation_population_lock_sha256"
        ],
    )
    if pilot_summary.get("completed") is True:
        access = pilot_summary.get("evaluation_access")
        expected_episode_ids = sorted(
            episode.episode_id
            for episode in admission.validation_episodes
            if episode.operation in PRIMARY_OPERATIONS
        )
        if (
            not isinstance(access, Mapping)
            or access.get("instrumentation") != "PER_EVALUATION_ROW_SPLIT_AND_EPISODE_TRACE"
            or access.get("observed_splits") != ["validation"]
            or access.get("unique_episode_count") != len(expected_episode_ids)
            or access.get("episode_ids_sha256") != sha256_canonical_json(expected_episode_ids)
            or access.get("expected_episode_ids_sha256")
            != sha256_canonical_json(expected_episode_ids)
            or access.get("main_test_access_count") != 0
            or access.get("heldout_domain_access_count") != 0
        ):
            raise ArtifactContractError(
                "E26a evaluation access trace differs from the locked validation-only population"
            )
    normalized_gates: list[dict[str, Any]] = []
    for gate in gates:
        value = dict(gate)
        if not isinstance(value.get("name"), str) or value.get("passed") not in (
            True,
            False,
        ):
            raise ArtifactContractError("Every E26a gate requires name and boolean passed")
        normalized_gates.append(value)
    all_passed = bool(normalized_gates) and all(bool(gate["passed"]) for gate in normalized_gates)

    selection: CandidateSelection | None = None
    if measurements:
        try:
            selection = select_candidate(
                config=admission.config,
                measurements=measurements,
                policy=admission.resource_policy,
            )
            require_locked_resource_selection(admission, selection)
        except E26AGateBlocked as error:
            normalized_gates.append(
                {
                    "name": "throughput_deadline_storage_candidate_selection",
                    "passed": False,
                    "observed": str(error),
                    "criterion": "at least one prospectively ordered eligible candidate",
                }
            )
            all_passed = False
    run.append("candidate_table.jsonl", [_candidate_row(row) for row in measurements])
    resource_rows = []
    for measurement in measurements:
        for projection in project_candidate_resources(
            measurement,
            admission.resource_policy,
        ):
            resource_rows.append(
                {
                    "candidate_id": measurement.candidate_id,
                    **projection,
                }
            )
    run.write(
        "resource_projection.json",
        {
            "schema_version": "catena-v8.1",
            "scientific_evidence": False,
            "deadline_reference_hours": DEADLINE_REFERENCE_HOURS,
            "deadline_fraction_max": admission.resource_policy.deadline_fraction_max,
            "effective_wall_cap_hours": admission.resource_policy.effective_wall_cap_hours,
            "rows": resource_rows,
            "resource_projection_sha256": sha256_canonical_json(resource_rows),
        },
    )
    if selection is not None:
        run.write("candidate_selection_lock.json", selection.as_dict())
    else:
        run.write(
            "candidate_selection_lock.json",
            {
                "schema_version": "catena-v8.1",
                "selected": False,
                "reason": "ONE_OR_MORE_PREREGISTERED_E26A_GATES_FAILED",
            },
        )
    run.write("pilot_summary.json", dict(pilot_summary))
    run.write("model_manifest.json", dict(model_manifest))
    output_backend = dict(backend_manifest)
    output_backend.update(
        {
            "e26a_gate_capable": all_passed,
            "parity_verified": all_passed,
            "scientific_main_capable": all_passed,
            "scientific_evidence": False,
            "claim_ceiling": E26A_CLAIM_CEILING,
        }
    )
    output_backend.pop("manifest_sha256", None)
    output_backend["manifest_sha256"] = sha256_canonical_json(output_backend)
    run.write("backend_manifest.json", output_backend)
    run.write("data_manifest.json", dict(data_manifest))

    disposition = "GO_E26B" if all_passed else _failure_disposition(normalized_gates)
    report = {
        "schema_version": "catena-v8.1",
        "experiment": "e26a_operator_data_gate",
        "run_id": run.run_id,
        "run_mode": "MAIN",
        "status": "PASS" if all_passed else "FAIL",
        "scientific_evidence": False,
        "evidence_tier": E26A_EVIDENCE_TIER,
        "claim_ceiling": E26A_CLAIM_CEILING,
        "disposition": disposition,
        "allowed_claim": (
            "E26a establishes protocol identifiability and execution readiness only; "
            "it does not establish a Dual-vs-Tied LM effect."
            if all_passed
            else "No E26 LM claim is opened by this failed protocol gate."
        ),
        "forbidden_claims": [
            "autoregressive LM transfer",
            "Dual superiority",
            "official GDN2/KDA correspondence",
            "E26b or E26c completion",
            "agent or production superiority",
        ],
        "gates": normalized_gates,
        "artifacts": {"run_dir": str(run.run_dir)},
        "upstream_dependencies": [
            {
                "kind": "execution_inputs",
                "aggregate_sha256": sha256_canonical_json(admission.input_hashes),
            }
        ],
        "candidate_selection": selection.as_dict() if selection is not None else None,
        "scientific_execution_device_binding": {
            **admission.execution_device_binding.as_dict(),
            "artifact_sha256": sha256_file(
                run.run_dir / "scientific_execution_device_binding.json"
            ),
        },
        "primary_operations": list(PRIMARY_OPERATIONS),
        "stale_baseline": "IDENTICAL_PREFIX_WITHOUT_TRANSACTION_APPLICATION",
        "gate_saturation_definition": {
            "low_inclusive": SATURATION_LOW,
            "high_inclusive": SATURATION_HIGH,
        },
        "state_norm_definition": "MAX_NORMALIZED_FROBENIUS_ACROSS_RECURRENT_STATES",
        "main_test_opened": False,
        "e26b_started": False,
        "e26c_started": False,
    }
    lines = [
        "# E26a scientific protocol gate",
        "",
        f"- Disposition: `{disposition}`",
        "- Scientific evidence: `false`",
        f"- Claim ceiling: `{E26A_CLAIM_CEILING}`",
        "- Main test opened: `false`",
        "- E26b/E26c auto-launch: `disabled`",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    lines.extend(f"| `{gate['name']}` | {gate['passed']} |" for gate in normalized_gates)
    run.finalize(report, "\n".join(lines))
    return report


def _failure_disposition(gates: Sequence[Mapping[str, Any]]) -> str:
    failed = {str(gate["name"]).lower() for gate in gates if gate.get("passed") is False}
    if any(
        "numerical" in name or "parity" in name or "restart" in name or "backend" in name
        for name in failed
    ):
        return "NO_GO_NUMERICAL"
    if any(
        "data" in name
        or "leak" in name
        or "cursor" in name
        or "token_mix" in name
        or "80_20" in name
        for name in failed
    ):
        return "NO_GO_DATA"
    if any("matching" in name or "parameter" in name or "initial" in name for name in failed):
        return "NO_GO_MATCHING"
    if any("floor" in name or "headroom" in name or "ppl" in name for name in failed):
        return "NO_GO_FLOOR_HEADROOM"
    if any("throughput" in name or "deadline" in name or "storage" in name for name in failed):
        return "NO_GO_THROUGHPUT"
    return "BLOCKED_DEPENDENCY"
