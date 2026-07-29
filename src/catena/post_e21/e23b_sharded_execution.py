"""Conservative seed-sharded execution substrate for frozen E23b.

This module changes only the physical execution topology.  The registered
configuration, seeds, model, optimizer, precision, metrics, thresholds,
dependency decisions, and final statistical aggregation remain those of
``e23b_product_poset_confirmatory``.

Workers never create a scientific report or update ``latest.json``.  The
aggregator creates the canonical E23b artifact only after every registered seed,
row, checkpoint, and dependency hash has been validated.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import (
    environment_snapshot,
    file_sha256,
    utc_run_id,
    write_json,
    write_jsonl,
)
from catena.core.provenance_v61 import (
    read_json_object_strict,
    read_jsonl_strict,
    sha256_canonical_json,
    source_tree_fingerprint,
    write_json_strict,
)
from catena.data.controller_poset import (
    CANONICAL_CONTROLLERS,
    DEMAND_FAMILIES,
)
from catena.post_e21.contracts import (
    PostE21ContractError,
    ProtocolSnapshot,
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.product_poset_eval import (
    ensure_finite_rows,
    expected_grid_size,
    resolve_e18b_freeze,
    resolve_e22b_dependency,
    resolve_e23a_screen_dependency,
    summarize_seed_predictions,
    validate_theory_prediction_lock,
)
from catena.post_e21.product_poset_runner import (
    data_manifest_payload,
    generate_product_poset_rows,
    product_poset_runtime,
    results_summary_ko,
    validate_e23_config,
    write_cell_rows,
    write_theory_predictions,
    write_training_rows,
)
from catena.training.sequence_control_lattice import state_dict_sha256
from experiments.common import finalize_run, initialize_run

LOGICAL_EXPERIMENT_ID = "e23b_product_poset_confirmatory"
EXECUTION_EXPERIMENT_ID = "e23b_product_poset_confirmatory_sharded_execution"
STAGING_DIRECTORY_NAME = "_e23b_product_poset_confirmatory_shards"
DEFAULT_CONFIG = "configs/e23b_product_poset_confirmatory.yaml"
PARENT_LOCK_RELATIVE = Path("docs/E23B_PRODUCT_POSET_CONFIRMATORY_LOCK.json")
AMENDMENT_LOCK_RELATIVE = Path("docs/E23B_SHARDED_EXECUTION_AMENDMENT_LOCK.json")
PREPARED_MANIFEST_NAME = "prepared_execution.json"
SHARD_MANIFEST_NAME = "shard_manifest.json"
AGGREGATE_RECEIPT_NAME = "aggregate_receipt.json"
AGGREGATE_LOCK_NAME = ".aggregate.lock"
EQUIVALENCE_REPORT_NAME = "E23B_CPU_SERIAL_SHARD_EQUIVALENCE.json"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
EQUIVALENCE_CHECK_KEYS = (
    "registered_four_seed_subset_exact",
    "raw_row_count_exact",
    "canonical_scientific_raw_rows_exact",
    "canonical_scientific_training_rows_exact",
    "checkpoint_state_hashes_exact",
    "seed_statistics_exact",
    "cell_statistics_exact",
    "assessment_exact",
    "runtime_contract_exact",
)
EQUIVALENCE_COMPARISON_EXCLUSIONS = (
    "examples_per_second",
    "peak_memory_bytes",
    "checkpoint absolute path",
    "checkpoint container file SHA-256",
)
EQUIVALENCE_FIXED_FIELD_KEYS = (
    "boundary_mode",
    "locality_method",
    "locality_risk_scale",
    "seeds",
    "runtime_config_sha256",
    "serial_rows",
    "sharded_rows",
    "comparison_exclusions",
    "checkpoint_state_hash_comparison",
    "scientific_metric_comparison",
)
NO_CHANGE_FLAGS = (
    "precision_change",
    "metric_change",
    "threshold_change",
    "seed_change",
    "seed_offset_change",
    "model_change",
    "controller_set_change",
    "optimizer_change",
    "batch_size_change",
    "training_step_change",
    "data_or_namespace_change",
    "theory_boundary_change",
    "dependency_rule_change",
    "claim_wording_change",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PostE21ContractError(
            f"Git provenance command failed: git {' '.join(arguments)}: {detail}"
        )
    return result.stdout.strip()


def validate_source_lock_tag(
    *,
    repo_root: Path,
    tag: str,
    parent_protocol_sha256: str,
    amendment_sha256: str,
) -> dict[str, Any]:
    """Require a clean annotated tag that binds both E23b protocol locks."""

    if not TAG_PATTERN.fullmatch(tag):
        raise PostE21ContractError("E23b source-lock tag contains unsafe characters")
    root = repo_root.resolve(strict=True)
    head = _git(root, "rev-parse", "HEAD")
    tagged_commit = _git(root, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    tag_type = _git(root, "cat-file", "-t", f"refs/tags/{tag}")
    if tag_type != "tag":
        raise PostE21ContractError("E23b source lock must be an annotated tag")
    if tagged_commit != head:
        raise PostE21ContractError("E23b source-lock tag does not point to current HEAD")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise PostE21ContractError("E23b MAIN requires a clean source-locked worktree")
    message = _git(root, "for-each-ref", "--format=%(contents)", f"refs/tags/{tag}")
    required_lines = {
        f"E23B_BASE_PROTOCOL_LOCK_SHA256={parent_protocol_sha256}",
        f"E23B_SHARDED_EXECUTION_AMENDMENT_LOCK_SHA256={amendment_sha256}",
    }
    if not required_lines.issubset(set(message.splitlines())):
        raise PostE21ContractError(
            "E23b source-lock tag message lacks protocol/amendment SHA bindings"
        )
    return {
        "tag": tag,
        "tag_object_id": _git(root, "rev-parse", f"refs/tags/{tag}^{{tag}}"),
        "tag_message_sha256": _hash_text(message),
        "git_commit": head,
        "dirty_status": "clean",
    }


def _read_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = read_json_object_strict(path)
    payload = envelope.get("payload")
    digest = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        raise PostE21ContractError(f"Invalid sharded-execution envelope: {path}")
    if sha256_canonical_json(payload) != digest:
        raise PostE21ContractError(f"Sharded-execution envelope hash mismatch: {path}")
    return payload, digest


def _write_envelope(path: Path, payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    digest = str(sha256_canonical_json(normalized))
    write_json_strict(
        path,
        {
            "payload": normalized,
            "payload_sha256": digest,
        },
    )
    return digest


def _resolve_repo_child(repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise PostE21ContractError(f"Path escapes repository: {resolved}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise PostE21ContractError(f"Required repository file is unsafe: {resolved}")
    return resolved


def validate_execution_locks(
    *,
    repo_root: Path,
    config_path: str | Path,
) -> tuple[ProtocolSnapshot, ProtocolSnapshot]:
    """Validate both the frozen scientific protocol and topology amendment."""

    root = repo_root.resolve(strict=True)
    config = _resolve_repo_child(root, config_path)
    parent = validate_protocol_lock(
        lock_path=root / PARENT_LOCK_RELATIVE,
        config_path=config,
        experiment_id=LOGICAL_EXPERIMENT_ID,
        repo_root=root,
    )
    amendment = validate_protocol_lock(
        lock_path=root / AMENDMENT_LOCK_RELATIVE,
        config_path=config,
        experiment_id=EXECUTION_EXPERIMENT_ID,
        repo_root=root,
    )
    amendment_payload = amendment.payload
    no_change_flags = {name: amendment_payload.get(name) for name in NO_CHANGE_FLAGS}
    if (
        amendment_payload.get("scientific_protocol_unchanged") is not True
        or amendment_payload.get("parent_experiment_id") != LOGICAL_EXPERIMENT_ID
        or amendment_payload.get("parent_protocol_lock_sha256") != parent.sha256
        or amendment_payload.get("shard_axis") != "registered_training_seed"
        or amendment_payload.get("registered_main_shard_count") != 4
        or amendment_payload.get("aggregation_order") != "frozen_config_seed_order"
        or any(value is not False for value in no_change_flags.values())
        or amendment_payload.get("checkpoint_reuse_or_resume") is not False
        or amendment_payload.get("topology_optimization_authorized_on") != "2026-07-29"
        or amendment_payload.get("source_lock_required_before_main") is not True
        or amendment_payload.get("source_lock_must_be_annotated_tag_on_clean_head") is not True
        or amendment_payload.get("dependency_bound_cpu_equivalence_required_before_main")
        is not True
        or amendment_payload.get("cpu_equivalence_check_keys") != list(EQUIVALENCE_CHECK_KEYS)
        or amendment_payload.get("cpu_equivalence_fixed_fields")
        != list(EQUIVALENCE_FIXED_FIELD_KEYS)
        or amendment_payload.get("cpu_equivalence_comparison_exclusions")
        != list(EQUIVALENCE_COMPARISON_EXCLUSIONS)
        or amendment_payload.get("physical_gpu_identity_bound_before_workers") is not True
        or amendment_payload.get("physical_gpu_uuid_unique_and_homogeneous_required") is not True
        or amendment_payload.get("workspace_aggregate_lock_held_through_receipt_and_latest")
        is not True
        or amendment_payload.get("canonical_artifact_copies_execution_manifests") is not True
        or amendment_payload.get("registered_seed_partition_default")
        != [
            {"shard_id": "shard_00", "seeds": [2401, 2411]},
            {"shard_id": "shard_01", "seeds": [2423, 2437]},
            {"shard_id": "shard_02", "seeds": [2441, 2459]},
            {"shard_id": "shard_03", "seeds": [2473, 2477]},
        ]
    ):
        raise PostE21ContractError("E23b sharded-execution amendment contract changed")
    return parent, amendment


def balanced_seed_partitions(
    seeds: Sequence[int],
    shard_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic contiguous, non-empty balanced seed partitions."""

    normalized = tuple(int(seed) for seed in seeds)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("E23b sharding requires non-empty unique seeds")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    count = min(int(shard_count), len(normalized))
    quotient, remainder = divmod(len(normalized), count)
    partitions: list[tuple[int, ...]] = []
    offset = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        partitions.append(normalized[offset : offset + width])
        offset += width
    if tuple(seed for partition in partitions for seed in partition) != normalized:
        raise AssertionError("E23b seed partition does not preserve registered order")
    return tuple(partitions)


def _query_physical_gpu(index: int) -> dict[str, Any]:
    if index < 0:
        raise PostE21ContractError(f"Invalid physical GPU index: {index}")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PostE21ContractError(f"Cannot resolve physical GPU {index}: {detail}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PostE21ContractError(f"Expected one physical GPU record for index {index}")
    parts = [part.strip() for part in lines[0].split(",", maxsplit=3)]
    if len(parts) != 4 or int(parts[0]) != index:
        raise PostE21ContractError(f"Malformed physical GPU record for index {index}")
    physical_index, uuid, name, capability = parts
    if not uuid.startswith("GPU-") or not name or not re.fullmatch(r"\d+\.\d+", capability):
        raise PostE21ContractError(f"Incomplete physical GPU identity for index {index}")
    return {
        "device_type": "cuda",
        "physical_index": int(physical_index),
        "uuid": uuid,
        "name": name,
        "compute_capability": capability,
    }


def _dry_cpu_binding(shard_index: int) -> dict[str, Any]:
    return {
        "device_type": "cpu",
        "physical_index": None,
        "uuid": f"CPU_DRY_RUN_SHARD_{shard_index:02d}",
        "name": "CPU_DRY_RUN",
        "compute_capability": "N/A",
    }


def _validate_device_bindings(
    bindings: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    normalized = [dict(binding) for binding in bindings]
    if not normalized:
        raise PostE21ContractError("E23b execution plan has no device bindings")
    if dry_run:
        expected = [_dry_cpu_binding(index) for index in range(len(normalized))]
        if normalized != expected:
            raise PostE21ContractError("E23b dry-run CPU bindings are non-canonical")
        return normalized
    if len(normalized) != 4:
        raise PostE21ContractError("E23b MAIN requires four physical GPU bindings")
    if any(binding.get("device_type") != "cuda" for binding in normalized):
        raise PostE21ContractError("E23b MAIN device binding is not CUDA")
    indices = [binding.get("physical_index") for binding in normalized]
    uuids = [binding.get("uuid") for binding in normalized]
    if (
        any(not isinstance(index, int) for index in indices)
        or len(set(indices)) != len(indices)
        or any(not isinstance(uuid, str) or not uuid.startswith("GPU-") for uuid in uuids)
        or len(set(uuids)) != len(uuids)
    ):
        raise PostE21ContractError("E23b physical GPU indices/UUIDs are not unique")
    names = {str(binding.get("name")) for binding in normalized}
    capabilities = {str(binding.get("compute_capability")) for binding in normalized}
    if len(names) != 1 or len(capabilities) != 1:
        raise PostE21ContractError("E23b physical GPU workers are not homogeneous")
    for binding in normalized:
        observed = _query_physical_gpu(int(binding["physical_index"]))
        if observed != binding:
            raise PostE21ContractError("E23b physical GPU identity changed after prepare")
    return normalized


def _validate_worker_device_binding(
    *,
    binding: Mapping[str, Any],
    device: torch.device,
) -> None:
    normalized = dict(binding)
    if normalized["device_type"] == "cpu":
        if device.type != "cpu":
            raise PostE21ContractError("E23b dry shard must execute on CPU")
        return
    if device.type != "cuda" or device.index not in {None, 0}:
        raise PostE21ContractError("E23b shard worker must use its isolated logical cuda:0")
    physical_index = int(normalized["physical_index"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_index):
        raise PostE21ContractError("E23b worker CUDA_VISIBLE_DEVICES differs from its plan")
    if _query_physical_gpu(physical_index) != normalized:
        raise PostE21ContractError("E23b worker physical GPU identity changed")
    if torch.cuda.device_count() != 1:
        raise PostE21ContractError("E23b worker must see exactly one isolated CUDA device")
    observed_name = torch.cuda.get_device_name(0)
    observed_capability = ".".join(str(value) for value in torch.cuda.get_device_capability(0))
    if (
        observed_name != normalized["name"]
        or observed_capability != normalized["compute_capability"]
    ):
        raise PostE21ContractError("E23b worker torch CUDA identity differs from its plan")


def sharded_product_poset_runtime(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    shard_count: int,
) -> dict[str, Any]:
    runtime: dict[str, Any] = dict(product_poset_runtime(config, dry_run=dry_run))
    if dry_run:
        if shard_count <= 0 or shard_count > 4:
            raise PostE21ContractError("E23b dry-run supports one to four seed shards")
        runtime = dict(runtime)
        runtime["seeds"] = [int(value) for value in config["seeds"][: int(shard_count)]]
    return runtime


def _resolve_dependencies(
    *,
    e18_freeze: str | Path | None,
    e23a_screen: str | Path | None,
    e22b_run: str | Path | None,
    dry_run: bool,
) -> tuple[dict[str, Any], Any, Any, Any]:
    e18 = resolve_e18b_freeze(
        freeze_path=e18_freeze,
        dry_run=dry_run,
    )
    screen = resolve_e23a_screen_dependency(
        screen_run=e23a_screen,
        dry_run=dry_run,
        expected_e18_freeze_sha256=e18.freeze_sha256,
    )
    e22 = resolve_e22b_dependency(
        e22b_run=e22b_run,
        dry_run=dry_run,
    )
    statuses = (
        e18.execution_status,
        screen.execution_status,
        e22.execution_status,
    )
    overall = "PASS" if all(value == "PASS" for value in statuses) else "BLOCKED_DEPENDENCY"
    payload = {
        "e18": e18.as_dict(),
        "e23a_screen": screen.as_dict(),
        "e22": e22.as_dict(),
        "overall_execution_status": overall,
    }
    return payload, e18, screen, e22


def _dependency_paths(
    *,
    e18_freeze: str | Path | None,
    e23a_screen: str | Path | None,
    e22b_run: str | Path | None,
) -> dict[str, str | None]:
    def normalized(value: str | Path | None) -> str | None:
        return None if value is None else str(Path(value).resolve(strict=True))

    return {
        "e18_freeze": normalized(e18_freeze),
        "e23a_screen": normalized(e23a_screen),
        "e22b_run": normalized(e22b_run),
    }


def _expected_equivalence_fixed_fields(
    *,
    config: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive every non-provenance equivalence field from frozen inputs."""

    registered_seeds = [int(value) for value in config["seeds"][:4]]
    if len(registered_seeds) != 4:
        raise PostE21ContractError("E23b equivalence requires the registered first four seeds")
    e22 = dependency.get("e22")
    if not isinstance(e22, Mapping):
        raise PostE21ContractError("E23b equivalence dependency lacks the E22 decision")
    boundary_mode = e22.get("boundary_mode")
    locality_method = e22.get("locality_method")
    locality_risk_scale = e22.get("locality_risk_scale")
    if boundary_mode not in {"capacity_only", "safe_minimality"}:
        raise PostE21ContractError("E23b equivalence dependency lacks a valid boundary mode")
    if not isinstance(locality_method, dict):
        raise PostE21ContractError("E23b equivalence dependency lacks a locality method")
    if isinstance(locality_risk_scale, bool) or not isinstance(locality_risk_scale, (int, float)):
        raise PostE21ContractError("E23b equivalence dependency lacks a locality risk scale")

    serial_config = deepcopy(dict(config))
    serial_config["seeds"] = list(registered_seeds)
    serial_config["dry_run"] = {
        **deepcopy(dict(config["dry_run"])),
        "seed_count": len(registered_seeds),
    }
    runtime = product_poset_runtime(serial_config, dry_run=True)
    expected_rows = expected_grid_size(
        seeds=registered_seeds,
        intensities=[float(value) for value in runtime["intensities"]],
        updates=[int(value) for value in runtime["updates"]],
        gap_events=[int(value) for value in runtime["gap_events"]],
    )
    fixed_fields = {
        "boundary_mode": boundary_mode,
        "locality_method": dict(locality_method),
        "locality_risk_scale": float(locality_risk_scale),
        "seeds": registered_seeds,
        "runtime_config_sha256": sha256_canonical_json(runtime),
        "serial_rows": expected_rows,
        "sharded_rows": expected_rows,
        "comparison_exclusions": list(EQUIVALENCE_COMPARISON_EXCLUSIONS),
        "checkpoint_state_hash_comparison": "exact",
        "scientific_metric_comparison": "exact",
    }
    if tuple(fixed_fields) != EQUIVALENCE_FIXED_FIELD_KEYS:
        raise AssertionError("E23b equivalence fixed-field producer drifted")
    return fixed_fields


def validate_equivalence_report(
    *,
    path: str | Path,
    source: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    amendment_sha256: str,
    parent_protocol_sha256: str,
    config_sha256: str,
    config: Mapping[str, Any],
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the non-evidence CPU equivalence proof used to unlock MAIN."""

    report_path = Path(path).resolve(strict=True)
    if not report_path.is_file() or report_path.is_symlink():
        raise PostE21ContractError(f"Unsafe E23b equivalence report: {report_path}")
    report = read_json_object_strict(report_path)
    expected = {
        "schema_version": 1,
        "experiment_id": LOGICAL_EXPERIMENT_ID,
        "execution_experiment_id": EXECUTION_EXPERIMENT_ID,
        "status": "PASS",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "scientific_evidence": False,
        "claim_eligible": False,
        "source_fingerprint": dict(source),
        "source_lock": dict(source_lock),
        "amendment_lock_sha256": amendment_sha256,
        "base_protocol_lock_sha256": parent_protocol_sha256,
        "config_sha256": config_sha256,
        "dependency_sha256": sha256_canonical_json(dict(dependency)),
        "dependency": dict(dependency),
        **_expected_equivalence_fixed_fields(
            config=config,
            dependency=dependency,
        ),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise PostE21ContractError(f"E23b equivalence report binding mismatch: {key}")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(EQUIVALENCE_CHECK_KEYS):
        raise PostE21ContractError("E23b equivalence check-key set is not exact")
    if not all(checks[key] is True for key in EQUIVALENCE_CHECK_KEYS):
        raise PostE21ContractError("E23b CPU serial-vs-shard equivalence did not fully pass")
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "status": "PASS",
        "checks": dict(checks),
    }


def _expected_counts(runtime: Mapping[str, Any]) -> dict[str, int]:
    seeds = [int(value) for value in runtime["seeds"]]
    raw = expected_grid_size(
        seeds=seeds,
        intensities=[float(value) for value in runtime["intensities"]],
        updates=[int(value) for value in runtime["updates"]],
        gap_events=[int(value) for value in runtime["gap_events"]],
    )
    return {
        "raw_rows": raw,
        "training_rows": len(seeds) * len(CANONICAL_CONTROLLERS),
        "checkpoint_count": len(seeds) * len(CANONICAL_CONTROLLERS),
        "seed_rows": len(seeds),
        "cell_rows": len(seeds) * len(DEMAND_FAMILIES),
    }


def prepare_sharded_execution(
    *,
    repo_root: Path,
    config_path: str | Path,
    artifact_root: str | Path,
    e18_freeze: str | Path | None,
    e23a_screen: str | Path | None,
    e22b_run: str | Path | None,
    shard_count: int,
    dry_run: bool,
    source_lock_tag: str | None = None,
    equivalence_report: str | Path | None = None,
    gpu_indices: Sequence[int] | None = None,
) -> Path:
    """Create an immutable execution plan without starting training."""

    root = repo_root.resolve(strict=True)
    config_file = _resolve_repo_child(root, config_path)
    parent_snapshot, amendment_snapshot = validate_execution_locks(
        repo_root=root,
        config_path=config_file,
    )
    config = load_config(config_file)
    validate_e23_config(
        config,
        experiment_id=LOGICAL_EXPERIMENT_ID,
        expected_seed_count=8,
    )
    validate_theory_prediction_lock(
        snapshot=parent_snapshot,
        config=config,
    )
    dependency_payload, _, _, e22 = _resolve_dependencies(
        e18_freeze=e18_freeze,
        e23a_screen=e23a_screen,
        e22b_run=e22b_run,
        dry_run=dry_run,
    )
    if dependency_payload["overall_execution_status"] != "PASS":
        raise PostE21ContractError(
            "BLOCKED_DEPENDENCY: E23b sharded execution requires all frozen dependencies"
        )
    if e22.boundary_mode not in {"capacity_only", "safe_minimality"}:
        raise PostE21ContractError("PASS E22b dependency lacks a valid boundary mode")

    runtime = sharded_product_poset_runtime(
        config,
        dry_run=dry_run,
        shard_count=shard_count,
    )
    runtime_seeds = tuple(int(value) for value in runtime["seeds"])
    if not dry_run and shard_count != 4:
        raise PostE21ContractError("E23b sharded MAIN requires exactly four seed shards")
    partitions = balanced_seed_partitions(runtime_seeds, shard_count)
    if dry_run:
        bindings = _validate_device_bindings(
            [_dry_cpu_binding(index) for index in range(len(partitions))],
            dry_run=True,
        )
    else:
        if gpu_indices is None or len(gpu_indices) != 4:
            raise PostE21ContractError("E23b MAIN requires four explicit physical GPU indices")
        bindings = _validate_device_bindings(
            [_query_physical_gpu(int(index)) for index in gpu_indices],
            dry_run=False,
        )
    source = source_tree_fingerprint(root).as_dict()
    if dry_run:
        source_lock: dict[str, Any] = {
            "required": False,
            "validated": False,
            "reason": "DRY_RUN_NON_EVIDENCE",
        }
        equivalence: dict[str, Any] | None = None
    else:
        if source_lock_tag is None:
            raise PostE21ContractError("E23b sharded MAIN requires --source-lock-tag")
        if equivalence_report is None:
            raise PostE21ContractError("E23b sharded MAIN requires --equivalence-report")
        source_lock = validate_source_lock_tag(
            repo_root=root,
            tag=source_lock_tag,
            parent_protocol_sha256=parent_snapshot.sha256,
            amendment_sha256=amendment_snapshot.sha256,
        )
        equivalence = validate_equivalence_report(
            path=equivalence_report,
            source=source,
            source_lock=source_lock,
            amendment_sha256=amendment_snapshot.sha256,
            parent_protocol_sha256=parent_snapshot.sha256,
            config_sha256=parent_snapshot.config_sha256,
            config=config,
            dependency=dependency_payload,
        )
    artifact = Path(artifact_root).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    staging_root = artifact / STAGING_DIRECTORY_NAME
    staging_root.mkdir(parents=True, exist_ok=True)
    workspace = staging_root / str(utc_run_id())
    workspace.mkdir(parents=False, exist_ok=False)
    (workspace / "shards").mkdir()
    (workspace / "logs").mkdir()
    shutil.copyfile(parent_snapshot.path, workspace / "protocol_lock.json")
    shutil.copyfile(amendment_snapshot.path, workspace / "execution_amendment_lock.json")
    if equivalence is not None:
        equivalence_copy = workspace / EQUIVALENCE_REPORT_NAME
        shutil.copyfile(Path(str(equivalence["path"])), equivalence_copy)
        if file_sha256(equivalence_copy) != equivalence["sha256"]:
            raise PostE21ContractError("Copied E23b equivalence report hash mismatch")
        equivalence = {
            **equivalence,
            "path": str(equivalence_copy.resolve()),
        }

    paths = _dependency_paths(
        e18_freeze=e18_freeze,
        e23a_screen=e23a_screen,
        e22b_run=e22b_run,
    )
    payload = {
        "schema_version": 1,
        "execution_experiment_id": EXECUTION_EXPERIMENT_ID,
        "logical_experiment_id": LOGICAL_EXPERIMENT_ID,
        "status": "PREPARED",
        "run_mode": "DRY_RUN" if dry_run else "MAIN",
        "created_at_utc": _utc_now(),
        "workspace": str(workspace.resolve()),
        "repo_root": str(root),
        "artifact_root": str(artifact),
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "resolved_config_sha256": sha256_canonical_json(config),
        "parent_protocol_lock": {
            "path": str(parent_snapshot.path),
            "sha256": parent_snapshot.sha256,
        },
        "execution_amendment_lock": {
            "path": str(amendment_snapshot.path),
            "sha256": amendment_snapshot.sha256,
        },
        "source_fingerprint": source,
        "source_lock": source_lock,
        "equivalence_report": equivalence,
        "scientific_protocol_unchanged": True,
        "execution_topology_only": True,
        "no_change_flags": {name: amendment_snapshot.payload[name] for name in NO_CHANGE_FLAGS},
        "checkpoint_reuse_or_resume": amendment_snapshot.payload["checkpoint_reuse_or_resume"],
        "shard_axis": "registered_training_seed",
        "aggregation_order": "frozen_config_seed_order",
        "dependency_paths": paths,
        "dependency": dependency_payload,
        "boundary_mode": e22.boundary_mode,
        "locality_method": e22.locality_method,
        "locality_risk_scale": float(e22.locality_risk_scale),
        "runtime": runtime,
        "device_bindings": bindings,
        "shard_plan": [
            {
                "shard_id": f"shard_{index:02d}",
                "seeds": list(partition),
                "device_binding": bindings[index],
            }
            for index, partition in enumerate(partitions)
        ],
        "expected_counts": _expected_counts(runtime),
    }
    _write_envelope(workspace / PREPARED_MANIFEST_NAME, payload)
    return workspace


def _validate_prepared_workspace(
    workspace: str | Path,
) -> tuple[Path, dict[str, Any], str, dict[str, Any], ProtocolSnapshot, ProtocolSnapshot]:
    path = Path(workspace).resolve(strict=True)
    if not path.is_dir() or path.is_symlink():
        raise PostE21ContractError(f"Unsafe E23b shard workspace: {path}")
    payload, prepared_sha = _read_envelope(path / PREPARED_MANIFEST_NAME)
    if (
        payload.get("schema_version") != 1
        or payload.get("execution_experiment_id") != EXECUTION_EXPERIMENT_ID
        or payload.get("logical_experiment_id") != LOGICAL_EXPERIMENT_ID
        or payload.get("status") != "PREPARED"
        or payload.get("scientific_protocol_unchanged") is not True
        or payload.get("execution_topology_only") is not True
        or payload.get("workspace") != str(path)
    ):
        raise PostE21ContractError("Invalid E23b prepared-execution contract")
    repo_root = Path(str(payload["repo_root"])).resolve(strict=True)
    artifact_root = Path(str(payload["artifact_root"])).resolve(strict=True)
    try:
        path.relative_to(artifact_root)
    except ValueError as error:
        raise PostE21ContractError("E23b shard workspace escapes artifact root") from error
    config_path = _resolve_repo_child(repo_root, str(payload["config_path"]))
    parent, amendment = validate_execution_locks(
        repo_root=repo_root,
        config_path=config_path,
    )
    if (
        parent.sha256 != payload["parent_protocol_lock"]["sha256"]
        or amendment.sha256 != payload["execution_amendment_lock"]["sha256"]
        or file_sha256(path / "protocol_lock.json") != parent.sha256
        or file_sha256(path / "execution_amendment_lock.json") != amendment.sha256
        or file_sha256(config_path) != payload["config_sha256"]
        or source_tree_fingerprint(repo_root).as_dict() != payload["source_fingerprint"]
    ):
        raise PostE21ContractError("E23b source/config/lock changed after sharded prepare")
    config = load_config(config_path)
    if sha256_canonical_json(config) != payload["resolved_config_sha256"]:
        raise PostE21ContractError("E23b resolved config changed after sharded prepare")
    validate_e23_config(
        config,
        experiment_id=LOGICAL_EXPERIMENT_ID,
        expected_seed_count=8,
    )
    plan = payload.get("shard_plan")
    if not isinstance(plan, list) or not plan:
        raise PostE21ContractError("E23b prepared plan has no shards")
    dry_run = payload["run_mode"] == "DRY_RUN"
    runtime = sharded_product_poset_runtime(
        config,
        dry_run=dry_run,
        shard_count=len(plan),
    )
    if runtime != payload["runtime"] or _expected_counts(runtime) != payload["expected_counts"]:
        raise PostE21ContractError("E23b registered runtime differs from prepared plan")
    expected_no_change = {name: amendment.payload[name] for name in NO_CHANGE_FLAGS}
    if (
        payload.get("no_change_flags") != expected_no_change
        or any(value is not False for value in expected_no_change.values())
        or payload.get("checkpoint_reuse_or_resume") is not False
    ):
        raise PostE21ContractError("E23b prepared no-change invariants differ from amendment")
    raw_bindings = payload.get("device_bindings")
    if not isinstance(raw_bindings, list):
        raise PostE21ContractError("E23b prepared plan lacks device bindings")
    bindings = _validate_device_bindings(raw_bindings, dry_run=dry_run)
    if len(bindings) != len(plan):
        raise PostE21ContractError("E23b shard/device binding counts differ")
    planned_seeds: list[int] = []
    for index, shard in enumerate(plan):
        if (
            not isinstance(shard, dict)
            or shard.get("shard_id") != f"shard_{index:02d}"
            or not isinstance(shard.get("seeds"), list)
            or shard.get("device_binding") != bindings[index]
        ):
            raise PostE21ContractError("E23b shard plan is non-canonical")
        planned_seeds.extend(int(value) for value in shard["seeds"])
    if planned_seeds != [int(value) for value in runtime["seeds"]]:
        raise PostE21ContractError("E23b shard plan does not cover registered seeds exactly")

    dependency_paths = payload["dependency_paths"]
    dependency_payload, _, _, _ = _resolve_dependencies(
        e18_freeze=dependency_paths["e18_freeze"],
        e23a_screen=dependency_paths["e23a_screen"],
        e22b_run=dependency_paths["e22b_run"],
        dry_run=payload["run_mode"] == "DRY_RUN",
    )
    if dependency_payload != payload["dependency"]:
        raise PostE21ContractError("E23b dependency state changed after sharded prepare")
    if payload["run_mode"] == "MAIN":
        source_lock = payload.get("source_lock")
        equivalence = payload.get("equivalence_report")
        if not isinstance(source_lock, dict) or not isinstance(equivalence, dict):
            raise PostE21ContractError("E23b MAIN lacks source-lock/equivalence provenance")
        observed_source_lock = validate_source_lock_tag(
            repo_root=repo_root,
            tag=str(source_lock.get("tag", "")),
            parent_protocol_sha256=parent.sha256,
            amendment_sha256=amendment.sha256,
        )
        if observed_source_lock != source_lock:
            raise PostE21ContractError("E23b source-lock tag changed after prepare")
        observed_equivalence = validate_equivalence_report(
            path=str(equivalence.get("path", "")),
            source=payload["source_fingerprint"],
            source_lock=source_lock,
            amendment_sha256=amendment.sha256,
            parent_protocol_sha256=parent.sha256,
            config_sha256=parent.config_sha256,
            config=config,
            dependency=dependency_payload,
        )
        if observed_equivalence != equivalence:
            raise PostE21ContractError("E23b equivalence report changed after prepare")
    elif (
        payload.get("source_lock")
        != {
            "required": False,
            "validated": False,
            "reason": "DRY_RUN_NON_EVIDENCE",
        }
        or payload.get("equivalence_report") is not None
    ):
        raise PostE21ContractError("E23b DRY_RUN source/equivalence contract changed")
    return path, payload, prepared_sha, config, parent, amendment


def _add_dependency_provenance(
    rows: Sequence[dict[str, Any]],
    dependency: Mapping[str, Any],
) -> None:
    e18 = dependency["e18"]
    screen = dependency["e23a_screen"]
    e22 = dependency["e22"]
    for row in rows:
        row["e18_freeze_sha256"] = e18["freeze_sha256"]
        row["e23a_screen_report_sha256"] = screen["report_sha256"]
        row["e22_report_sha256"] = e22["report_sha256"]
        row["e22_protocol_lock_sha256"] = e22["protocol_lock_sha256"]
        row["safe_objective_implemented"] = e22["safe_objective_implemented"]


def _shard_record(
    payload: Mapping[str, Any],
    shard_id: str,
) -> dict[str, Any]:
    matches = [
        dict(record)
        for record in payload["shard_plan"]
        if isinstance(record, dict) and record.get("shard_id") == shard_id
    ]
    if len(matches) != 1:
        raise PostE21ContractError(f"Unknown or duplicate E23b shard id: {shard_id}")
    return matches[0]


def run_shard_worker(
    *,
    workspace: str | Path,
    shard_id: str,
    device: torch.device,
) -> Path:
    """Run one disjoint registered-seed shard and emit no scientific report."""

    (
        workspace_path,
        prepared,
        prepared_sha,
        config,
        _,
        amendment,
    ) = _validate_prepared_workspace(workspace)
    shard = _shard_record(prepared, shard_id)
    _validate_worker_device_binding(
        binding=shard["device_binding"],
        device=device,
    )
    shard_dir = workspace_path / "shards" / shard_id
    if shard_dir.exists():
        raise FileExistsError(f"Refusing to overwrite E23b shard: {shard_dir}")
    shard_dir.mkdir(parents=False, exist_ok=False)
    write_json_strict(shard_dir / "environment.json", environment_snapshot())

    execution_config = deepcopy(config)
    execution_config["seeds"] = [int(value) for value in shard["seeds"]]
    dry_run = prepared["run_mode"] == "DRY_RUN"
    if dry_run:
        execution_config["dry_run"]["seed_count"] = len(execution_config["seeds"])
    learned = generate_product_poset_rows(
        execution_config,
        boundary_mode=str(prepared["boundary_mode"]),
        locality_method_payload=prepared["locality_method"],
        locality_risk_scale=float(prepared["locality_risk_scale"]),
        device=device,
        run_dir=shard_dir,
        dry_run=dry_run,
    )
    if learned.runtime["seeds"] != execution_config["seeds"]:
        raise PostE21ContractError("E23b worker ran seeds outside its locked shard")
    _add_dependency_provenance(learned.rows, prepared["dependency"])
    _add_dependency_provenance(learned.training_rows, prepared["dependency"])
    raw_path = shard_dir / "product_poset_raw_metrics.jsonl"
    training_path = shard_dir / "product_poset_training_runs.jsonl"
    write_jsonl(raw_path, learned.rows)
    write_jsonl(training_path, learned.training_rows)
    checkpoint_map_path = shard_dir / "checkpoint_hashes.json"
    write_json(checkpoint_map_path, dict(sorted(learned.checkpoint_hashes.items())))
    expected = _expected_counts(learned.runtime)
    if (
        len(learned.rows) != expected["raw_rows"]
        or len(learned.training_rows) != expected["training_rows"]
        or len(learned.checkpoint_hashes) != expected["checkpoint_count"]
    ):
        raise PostE21ContractError("E23b worker output count does not match its seed shard")
    for key, digest in learned.checkpoint_hashes.items():
        checkpoint = shard_dir / "checkpoints" / f"{key}.pt"
        if not checkpoint.is_file() or checkpoint.is_symlink() or file_sha256(checkpoint) != digest:
            raise PostE21ContractError(f"E23b worker checkpoint integrity failed: {key}")

    manifest_payload = {
        "schema_version": 1,
        "execution_experiment_id": EXECUTION_EXPERIMENT_ID,
        "logical_experiment_id": LOGICAL_EXPERIMENT_ID,
        "status": "COMPLETE",
        "completed_at_utc": _utc_now(),
        "shard_id": shard_id,
        "seeds": list(execution_config["seeds"]),
        "device": str(device),
        "device_binding": dict(shard["device_binding"]),
        "workspace": str(workspace_path),
        "prepared_manifest_sha256": prepared_sha,
        "execution_amendment_lock_sha256": amendment.sha256,
        "source_fingerprint": prepared["source_fingerprint"],
        "config_sha256": prepared["config_sha256"],
        "resolved_config_sha256": prepared["resolved_config_sha256"],
        "boundary_mode": prepared["boundary_mode"],
        "locality_method": prepared["locality_method"],
        "locality_risk_scale": prepared["locality_risk_scale"],
        "dependency": prepared["dependency"],
        "runtime": learned.runtime,
        "counts": {
            "raw_rows": len(learned.rows),
            "training_rows": len(learned.training_rows),
            "checkpoint_count": len(learned.checkpoint_hashes),
        },
        "artifacts": {
            "raw": {
                "path": str(raw_path.resolve()),
                "sha256": file_sha256(raw_path),
            },
            "training": {
                "path": str(training_path.resolve()),
                "sha256": file_sha256(training_path),
            },
            "checkpoint_hashes": {
                "path": str(checkpoint_map_path.resolve()),
                "sha256": file_sha256(checkpoint_map_path),
            },
        },
        "checkpoint_hashes": dict(sorted(learned.checkpoint_hashes.items())),
    }
    _write_envelope(shard_dir / SHARD_MANIFEST_NAME, manifest_payload)
    return shard_dir


def _object_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl_strict(path)
    if any(not isinstance(row, dict) for row in rows):
        raise PostE21ContractError(f"E23b JSONL contains a non-object row: {path}")
    return [dict(row) for row in rows]


def _raw_sort_key(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        [int(value) for value in config["seeds"]].index(int(row["seed"])),
        [controller.controller_id for controller in CANONICAL_CONTROLLERS].index(
            str(row["controller_id"])
        ),
        list(DEMAND_FAMILIES).index(str(row["demand_family"])),
        [float(value) for value in config["intensities"]].index(float(row["intensity"])),
        [int(value) for value in config["evaluation"]["updates"]].index(int(row["updates"])),
        [int(value) for value in config["evaluation"]["gap_events"]].index(int(row["gap_events"])),
    )


def _training_sort_key(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, int]:
    return (
        [int(value) for value in config["seeds"]].index(int(row["seed"])),
        [controller.controller_id for controller in CANONICAL_CONTROLLERS].index(
            str(row["controller_id"])
        ),
    )


def _expected_raw_keys(
    runtime: Mapping[str, Any],
) -> set[tuple[int, str, str, float, int, int]]:
    return {
        (
            int(seed),
            controller.controller_id,
            demand,
            float(intensity),
            int(updates),
            int(gap),
        )
        for seed in runtime["seeds"]
        for controller in CANONICAL_CONTROLLERS
        for demand in DEMAND_FAMILIES
        for intensity in runtime["intensities"]
        for updates in runtime["updates"]
        for gap in runtime["gap_events"]
    }


def _validate_and_collect_shards(
    *,
    workspace: Path,
    prepared: Mapping[str, Any],
    prepared_sha: str,
    config: Mapping[str, Any],
    amendment: ProtocolSnapshot,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Path],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_sources: dict[str, Path] = {}
    shard_descriptors: list[dict[str, Any]] = []
    for shard in prepared["shard_plan"]:
        shard_id = str(shard["shard_id"])
        shard_dir = workspace / "shards" / shard_id
        manifest_path = shard_dir / SHARD_MANIFEST_NAME
        manifest, manifest_sha = _read_envelope(manifest_path)
        expected_runtime = dict(prepared["runtime"])
        expected_runtime["seeds"] = [int(value) for value in shard["seeds"]]
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("shard_id") != shard_id
            or manifest.get("seeds") != shard["seeds"]
            or manifest.get("workspace") != str(workspace)
            or manifest.get("prepared_manifest_sha256") != prepared_sha
            or manifest.get("execution_amendment_lock_sha256") != amendment.sha256
            or manifest.get("source_fingerprint") != prepared["source_fingerprint"]
            or manifest.get("config_sha256") != prepared["config_sha256"]
            or manifest.get("resolved_config_sha256") != prepared["resolved_config_sha256"]
            or manifest.get("boundary_mode") != prepared["boundary_mode"]
            or manifest.get("locality_method") != prepared["locality_method"]
            or manifest.get("locality_risk_scale") != prepared["locality_risk_scale"]
            or manifest.get("dependency") != prepared["dependency"]
            or manifest.get("runtime") != expected_runtime
            or manifest.get("device_binding") != shard["device_binding"]
        ):
            raise PostE21ContractError(f"E23b shard provenance mismatch: {shard_id}")
        raw_path = shard_dir / "product_poset_raw_metrics.jsonl"
        training_path = shard_dir / "product_poset_training_runs.jsonl"
        checkpoint_map_path = shard_dir / "checkpoint_hashes.json"
        artifacts = manifest["artifacts"]
        for artifact_name, artifact_path in (
            ("raw", raw_path),
            ("training", training_path),
            ("checkpoint_hashes", checkpoint_map_path),
        ):
            descriptor = artifacts[artifact_name]
            if descriptor["path"] != str(artifact_path.resolve()) or descriptor[
                "sha256"
            ] != file_sha256(artifact_path):
                raise PostE21ContractError(
                    f"E23b shard artifact hash mismatch: {shard_id}/{artifact_name}"
                )
        shard_rows = _object_rows(raw_path)
        shard_training = _object_rows(training_path)
        checkpoint_map = read_json_object_strict(checkpoint_map_path)
        expected_counts = _expected_counts(expected_runtime)
        if (
            len(shard_rows) != expected_counts["raw_rows"]
            or len(shard_training) != expected_counts["training_rows"]
            or len(checkpoint_map) != expected_counts["checkpoint_count"]
            or manifest["counts"]
            != {
                "raw_rows": len(shard_rows),
                "training_rows": len(shard_training),
                "checkpoint_count": len(checkpoint_map),
            }
            or manifest["checkpoint_hashes"] != checkpoint_map
        ):
            raise PostE21ContractError(f"E23b shard output count mismatch: {shard_id}")
        for key, digest_value in checkpoint_map.items():
            digest = str(digest_value)
            if key in checkpoint_hashes:
                raise PostE21ContractError(f"Duplicate E23b checkpoint key: {key}")
            source = shard_dir / "checkpoints" / f"{key}.pt"
            if (
                not source.is_file()
                or source.is_symlink()
                or len(digest) != 64
                or file_sha256(source) != digest
            ):
                raise PostE21ContractError(f"E23b checkpoint integrity failed: {key}")
            checkpoint_hashes[key] = digest
            checkpoint_sources[key] = source
        rows.extend(shard_rows)
        training_rows.extend(shard_training)
        shard_descriptors.append(
            {
                "shard_id": shard_id,
                "seeds": list(shard["seeds"]),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_payload_sha256": manifest_sha,
                "manifest_file_sha256": file_sha256(manifest_path),
                "device": manifest["device"],
                "device_binding": dict(manifest["device_binding"]),
                "counts": dict(manifest["counts"]),
            }
        )

    observed_bindings = _validate_device_bindings(
        [descriptor["device_binding"] for descriptor in shard_descriptors],
        dry_run=prepared["run_mode"] == "DRY_RUN",
    )
    if observed_bindings != prepared["device_bindings"]:
        raise PostE21ContractError("E23b shard GPU bindings differ from prepared plan")
    expected_counts = prepared["expected_counts"]
    if (
        len(rows) != expected_counts["raw_rows"]
        or len(training_rows) != expected_counts["training_rows"]
        or len(checkpoint_hashes) != expected_counts["checkpoint_count"]
    ):
        raise PostE21ContractError("Merged E23b shard counts are incomplete")
    ensure_finite_rows(rows)
    ensure_finite_rows(training_rows)
    raw_keys = [
        (
            int(row["seed"]),
            str(row["controller_id"]),
            str(row["demand_family"]),
            float(row["intensity"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        for row in rows
    ]
    if len(raw_keys) != len(set(raw_keys)) or set(raw_keys) != _expected_raw_keys(
        prepared["runtime"]
    ):
        raise PostE21ContractError("Merged E23b raw Cartesian grid is not exact")
    expected_training_keys = {
        (int(seed), controller.controller_id)
        for seed in prepared["runtime"]["seeds"]
        for controller in CANONICAL_CONTROLLERS
    }
    observed_training_keys = [
        (int(row["seed"]), str(row["controller_id"])) for row in training_rows
    ]
    if (
        len(observed_training_keys) != len(set(observed_training_keys))
        or set(observed_training_keys) != expected_training_keys
    ):
        raise PostE21ContractError("Merged E23b training grid is not exact")
    dependency = prepared["dependency"]
    expected_provenance = {
        "e18_freeze_sha256": dependency["e18"]["freeze_sha256"],
        "e23a_screen_report_sha256": dependency["e23a_screen"]["report_sha256"],
        "e22_report_sha256": dependency["e22"]["report_sha256"],
        "e22_protocol_lock_sha256": dependency["e22"]["protocol_lock_sha256"],
        "safe_objective_implemented": dependency["e22"]["safe_objective_implemented"],
    }
    for row in [*rows, *training_rows]:
        if any(row.get(key) != value for key, value in expected_provenance.items()):
            raise PostE21ContractError("Merged E23b row dependency provenance changed")
        checkpoint_key = f"{row['controller_id']}_seed{int(row['seed'])}"
        if (
            row.get("checkpoint_sha256") != checkpoint_hashes.get(checkpoint_key)
            or Path(str(row["checkpoint"])).resolve()
            != checkpoint_sources[checkpoint_key].resolve()
        ):
            raise PostE21ContractError("Merged E23b row checkpoint provenance changed")
    digest_groups: dict[tuple[int, str, float, int], set[str]] = {}
    for row in rows:
        digest_key = (
            int(row["seed"]),
            str(row["demand_family"]),
            float(row["intensity"]),
            int(row["updates"]),
        )
        digest_groups.setdefault(digest_key, set()).add(str(row["base_transaction_digest"]))
    if any(len(values) != 1 for values in digest_groups.values()):
        raise PostE21ContractError("Merged E23b paired-data digest changed across cells")
    rows.sort(key=lambda row: _raw_sort_key(row, config))
    training_rows.sort(key=lambda row: _training_sort_key(row, config))
    return rows, training_rows, checkpoint_hashes, checkpoint_sources, shard_descriptors


def _checkpoint_state_hashes(
    *,
    run_dir: Path,
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, expected_file_hash in checkpoint_hashes.items():
        path = run_dir / "checkpoints" / f"{key}.pt"
        if file_sha256(path) != expected_file_hash:
            raise PostE21ContractError(f"E23b equivalence checkpoint file changed: {key}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise PostE21ContractError(f"E23b checkpoint state is not a mapping: {key}")
        normalized_state: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise PostE21ContractError(f"E23b checkpoint state is malformed: {key}")
            normalized_state[name] = tensor
        hashes[key] = state_dict_sha256(normalized_state)
    return dict(sorted(hashes.items()))


def _normalized_equivalence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    training: bool,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row.pop("checkpoint", None)
        row.pop("checkpoint_sha256", None)
        if training:
            row.pop("examples_per_second", None)
            row.pop("peak_memory_bytes", None)
        normalized.append(row)
    return normalized


def run_cpu_serial_shard_equivalence(
    *,
    repo_root: Path,
    config_path: str | Path,
    output_root: Path,
    e18_freeze: str | Path,
    e23a_screen: str | Path,
    e22b_run: str | Path,
    source_lock_tag: str,
) -> Path:
    """Create a dependency-bound, non-evidence serial-vs-shard CPU proof."""

    root = repo_root.resolve(strict=True)
    config_file = _resolve_repo_child(root, config_path)
    parent, amendment = validate_execution_locks(
        repo_root=root,
        config_path=config_file,
    )
    config = load_config(config_file)
    validate_e23_config(
        config,
        experiment_id=LOGICAL_EXPERIMENT_ID,
        expected_seed_count=8,
    )
    validate_theory_prediction_lock(snapshot=parent, config=config)
    dependency, _, _, e22 = _resolve_dependencies(
        e18_freeze=e18_freeze,
        e23a_screen=e23a_screen,
        e22b_run=e22b_run,
        dry_run=False,
    )
    if dependency["overall_execution_status"] != "PASS":
        raise PostE21ContractError(
            "BLOCKED_DEPENDENCY: E23b equivalence requires actual completed dependencies"
        )
    if e22.boundary_mode not in {"capacity_only", "safe_minimality"}:
        raise PostE21ContractError("E23b equivalence dependency lacks a boundary mode")
    source = source_tree_fingerprint(root).as_dict()
    source_lock = validate_source_lock_tag(
        repo_root=root,
        tag=source_lock_tag,
        parent_protocol_sha256=parent.sha256,
        amendment_sha256=amendment.sha256,
    )
    target = output_root.resolve(strict=False)
    temp_root = Path("/tmp").resolve(strict=True)
    try:
        target.relative_to(temp_root)
    except ValueError as error:
        raise PostE21ContractError("E23b equivalence output must be below /tmp") from error
    if target == temp_root or target.exists():
        raise FileExistsError(f"E23b equivalence output must be a fresh /tmp child: {target}")
    target.mkdir(parents=True, exist_ok=False)

    equivalence_seeds = tuple(int(value) for value in config["seeds"][:4])
    serial_config = deepcopy(config)
    serial_config["seeds"] = list(equivalence_seeds)
    serial_config["dry_run"]["seed_count"] = len(equivalence_seeds)
    serial_dir = target / "serial"
    serial_dir.mkdir()
    serial = generate_product_poset_rows(
        serial_config,
        boundary_mode=e22.boundary_mode,
        locality_method_payload=e22.locality_method,
        locality_risk_scale=float(e22.locality_risk_scale),
        device=torch.device("cpu"),
        run_dir=serial_dir,
        dry_run=True,
    )
    serial.rows.sort(key=lambda row: _raw_sort_key(row, config))
    serial.training_rows.sort(key=lambda row: _training_sort_key(row, config))
    serial_states = _checkpoint_state_hashes(
        run_dir=serial_dir,
        checkpoint_hashes=serial.checkpoint_hashes,
    )

    sharded_rows: list[dict[str, Any]] = []
    sharded_training: list[dict[str, Any]] = []
    sharded_states: dict[str, str] = {}
    for index, seed in enumerate(equivalence_seeds):
        shard_config = deepcopy(config)
        shard_config["seeds"] = [int(seed)]
        shard_config["dry_run"]["seed_count"] = 1
        shard_dir = target / f"shard_{index:02d}"
        shard_dir.mkdir()
        shard = generate_product_poset_rows(
            shard_config,
            boundary_mode=e22.boundary_mode,
            locality_method_payload=e22.locality_method,
            locality_risk_scale=float(e22.locality_risk_scale),
            device=torch.device("cpu"),
            run_dir=shard_dir,
            dry_run=True,
        )
        sharded_rows.extend(shard.rows)
        sharded_training.extend(shard.training_rows)
        for key, value in _checkpoint_state_hashes(
            run_dir=shard_dir,
            checkpoint_hashes=shard.checkpoint_hashes,
        ).items():
            if key in sharded_states:
                raise PostE21ContractError(f"Duplicate E23b equivalence checkpoint: {key}")
            sharded_states[key] = value
    sharded_rows.sort(key=lambda row: _raw_sort_key(row, config))
    sharded_training.sort(key=lambda row: _training_sort_key(row, config))
    runtime = serial.runtime
    serial_seed_rows, serial_detail = summarize_seed_predictions(
        serial.rows,
        seeds=list(equivalence_seeds),
        intensities=[float(value) for value in runtime["intensities"]],
        updates=[int(value) for value in runtime["updates"]],
        gap_events=[int(value) for value in runtime["gap_events"]],
        affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
        target_margin=float(config["adequacy"]["target_margin"]),
        retention_margin=float(config["adequacy"]["retention_margin"]),
        locality_margin=float(config["adequacy"]["locality_margin"]),
        minimum_single_axis_exact_matches=int(
            config["adequacy"]["minimum_single_axis_exact_matches"]
        ),
        minimum_pairwise_exact_matches=int(config["adequacy"]["minimum_pairwise_exact_matches"]),
        incomparable_direction_margin=float(config["adequacy"]["incomparable_direction_margin"]),
        maximal_simpler_degradation_margin=float(
            config["adequacy"]["maximal_simpler_degradation_margin"]
        ),
        boundary_mode=e22.boundary_mode,
    )
    sharded_seed_rows, sharded_detail = summarize_seed_predictions(
        sharded_rows,
        seeds=list(equivalence_seeds),
        intensities=[float(value) for value in runtime["intensities"]],
        updates=[int(value) for value in runtime["updates"]],
        gap_events=[int(value) for value in runtime["gap_events"]],
        affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
        target_margin=float(config["adequacy"]["target_margin"]),
        retention_margin=float(config["adequacy"]["retention_margin"]),
        locality_margin=float(config["adequacy"]["locality_margin"]),
        minimum_single_axis_exact_matches=int(
            config["adequacy"]["minimum_single_axis_exact_matches"]
        ),
        minimum_pairwise_exact_matches=int(config["adequacy"]["minimum_pairwise_exact_matches"]),
        incomparable_direction_margin=float(config["adequacy"]["incomparable_direction_margin"]),
        maximal_simpler_degradation_margin=float(
            config["adequacy"]["maximal_simpler_degradation_margin"]
        ),
        boundary_mode=e22.boundary_mode,
    )
    expected_raw = expected_grid_size(
        seeds=equivalence_seeds,
        intensities=runtime["intensities"],
        updates=runtime["updates"],
        gap_events=runtime["gap_events"],
    )
    checks = {
        "registered_four_seed_subset_exact": equivalence_seeds
        == tuple(int(value) for value in config["seeds"][:4]),
        "raw_row_count_exact": len(serial.rows) == len(sharded_rows) == expected_raw,
        "canonical_scientific_raw_rows_exact": _normalized_equivalence_rows(
            serial.rows,
            training=False,
        )
        == _normalized_equivalence_rows(sharded_rows, training=False),
        "canonical_scientific_training_rows_exact": _normalized_equivalence_rows(
            serial.training_rows,
            training=True,
        )
        == _normalized_equivalence_rows(sharded_training, training=True),
        "checkpoint_state_hashes_exact": serial_states == dict(sorted(sharded_states.items())),
        "seed_statistics_exact": serial_seed_rows == sharded_seed_rows,
        "cell_statistics_exact": serial_detail["cells"] == sharded_detail["cells"],
        "assessment_exact": serial_detail["assessment"] == sharded_detail["assessment"],
        "runtime_contract_exact": all(
            {
                **runtime,
                "seeds": [int(seed)],
            }
            == {
                **product_poset_runtime(
                    {
                        **deepcopy(config),
                        "seeds": [int(seed)],
                        "dry_run": {
                            **deepcopy(config["dry_run"]),
                            "seed_count": 1,
                        },
                    },
                    dry_run=True,
                ),
            }
            for seed in equivalence_seeds
        ),
    }
    report = {
        "schema_version": 1,
        "experiment_id": LOGICAL_EXPERIMENT_ID,
        "execution_experiment_id": EXECUTION_EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "scientific_evidence": False,
        "claim_eligible": False,
        "created_at_utc": _utc_now(),
        "source_fingerprint": source,
        "source_lock": source_lock,
        "amendment_lock_sha256": amendment.sha256,
        "base_protocol_lock_sha256": parent.sha256,
        "config_sha256": parent.config_sha256,
        "dependency_sha256": sha256_canonical_json(dependency),
        "dependency": dependency,
        "boundary_mode": e22.boundary_mode,
        "locality_method": e22.locality_method,
        "locality_risk_scale": float(e22.locality_risk_scale),
        "seeds": list(equivalence_seeds),
        "runtime_config_sha256": sha256_canonical_json(runtime),
        "serial_rows": len(serial.rows),
        "sharded_rows": len(sharded_rows),
        "checks": checks,
        "comparison_exclusions": list(EQUIVALENCE_COMPARISON_EXCLUSIONS),
        "checkpoint_state_hash_comparison": "exact",
        "scientific_metric_comparison": "exact",
    }
    report_path = target / EQUIVALENCE_REPORT_NAME
    write_json_strict(report_path, report)
    if report["status"] != "PASS":
        raise PostE21ContractError(f"E23b CPU serial-vs-shard equivalence failed: {checks}")
    validate_equivalence_report(
        path=report_path,
        source=source,
        source_lock=source_lock,
        amendment_sha256=amendment.sha256,
        parent_protocol_sha256=parent.sha256,
        config_sha256=parent.config_sha256,
        config=config,
        dependency=dependency,
    )
    return report_path


def _claim_status(
    *,
    dry_run: bool,
    boundary_mode: str,
    assessment: Mapping[str, Any],
) -> tuple[str, bool]:
    if dry_run:
        return "DRY_RUN_ONLY", False
    if boundary_mode == "safe_minimality":
        supported = bool(assessment["safe_minimality_supported"])
        return (
            "SUPPORTED_SAFE_EPSILON_MINIMALITY_CONTROLLED" if supported else "NOT_SUPPORTED",
            supported,
        )
    if boundary_mode == "capacity_only":
        supported = bool(assessment["capacity_supported"])
        return (
            "SUPPORTED_CAPACITY_EPSILON_MINIMALITY_CONTROLLED" if supported else "NOT_SUPPORTED",
            supported,
        )
    raise PostE21ContractError("Unknown E23b boundary mode during aggregation")


def _aggregate_sharded_execution_locked(
    *,
    workspace: str | Path,
    artifact_root: str | Path,
    device_request: str = "cpu",
) -> Path:
    """Validate all shards, then create one canonical immutable E23b artifact."""

    if device_request != "cpu":
        raise ValueError("E23b aggregation is deterministic CPU-only bookkeeping")
    (
        workspace_path,
        prepared,
        prepared_sha,
        config,
        parent_snapshot,
        amendment_snapshot,
    ) = _validate_prepared_workspace(workspace)
    receipt_path = workspace_path / AGGREGATE_RECEIPT_NAME
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to aggregate E23b workspace twice: {receipt_path}")
    canonical_artifact_root = Path(str(prepared["artifact_root"])).resolve(strict=True)
    if Path(artifact_root).resolve(strict=True) != canonical_artifact_root:
        raise PostE21ContractError("E23b aggregate artifact root differs from prepared plan")
    (
        rows,
        training_rows,
        checkpoint_hashes,
        checkpoint_sources,
        shard_descriptors,
    ) = _validate_and_collect_shards(
        workspace=workspace_path,
        prepared=prepared,
        prepared_sha=prepared_sha,
        config=config,
        amendment=amendment_snapshot,
    )
    theory = validate_theory_prediction_lock(
        snapshot=parent_snapshot,
        config=config,
    )
    run_mode = str(prepared["run_mode"])
    dry_run = run_mode == "DRY_RUN"
    config_path = str(prepared["config_path"])
    config_loaded, run_dir, _ = initialize_run(
        experiment_id=LOGICAL_EXPERIMENT_ID,
        config_path=config_path,
        artifact_root=str(canonical_artifact_root),
        device_request=device_request,
        run_mode=run_mode,
    )
    if config_loaded != config:
        raise PostE21ContractError("Aggregator config differs from prepared config")
    copy_protocol_snapshot(snapshot=parent_snapshot, run_dir=run_dir)
    prepared_copy = run_dir / PREPARED_MANIFEST_NAME
    shutil.copyfile(workspace_path / PREPARED_MANIFEST_NAME, prepared_copy)
    if file_sha256(prepared_copy) != file_sha256(workspace_path / PREPARED_MANIFEST_NAME):
        raise PostE21ContractError("Copied E23b prepared manifest hash mismatch")
    amendment_copy = run_dir / "execution_amendment_lock.json"
    shutil.copyfile(amendment_snapshot.path, amendment_copy)
    if file_sha256(amendment_copy) != amendment_snapshot.sha256:
        raise PostE21ContractError("Copied E23b execution amendment hash mismatch")
    equivalence_copy: Path | None = None
    if prepared["equivalence_report"] is not None:
        equivalence_copy = run_dir / EQUIVALENCE_REPORT_NAME
        shutil.copyfile(
            Path(str(prepared["equivalence_report"]["path"])),
            equivalence_copy,
        )
        if file_sha256(equivalence_copy) != prepared["equivalence_report"]["sha256"]:
            raise PostE21ContractError("Copied E23b equivalence report hash mismatch")
    shard_manifest_dir = run_dir / "execution_shard_manifests"
    shard_manifest_dir.mkdir(parents=False, exist_ok=False)
    canonical_shard_descriptors: list[dict[str, Any]] = []
    for descriptor in shard_descriptors:
        source_manifest = Path(str(descriptor["manifest_path"]))
        destination_manifest = shard_manifest_dir / f"{descriptor['shard_id']}.json"
        shutil.copyfile(source_manifest, destination_manifest)
        if file_sha256(destination_manifest) != descriptor["manifest_file_sha256"]:
            raise PostE21ContractError(
                f"Copied E23b shard manifest changed: {descriptor['shard_id']}"
            )
        canonical_shard_descriptors.append(
            {
                **descriptor,
                "source_manifest_path": str(source_manifest.resolve()),
                "manifest_path": str(destination_manifest.resolve()),
            }
        )

    runtime = dict(prepared["runtime"])
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload=data_manifest_payload(
            config,
            phase="CONFIRMATORY",
            boundary_mode=str(prepared["boundary_mode"]),
            dependency=prepared["dependency"],
            locality_method=prepared["locality_method"],
            locality_risk_scale=float(prepared["locality_risk_scale"]),
            runtime=runtime,
        ),
    )
    theory_artifact = write_theory_predictions(
        run_dir=run_dir,
        config=config,
        locked_sha256=str(theory["sha256"]),
    )
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=False, exist_ok=False)
    final_checkpoint_paths: dict[str, Path] = {}
    for key in sorted(checkpoint_hashes):
        destination = checkpoints_dir / f"{key}.pt"
        shutil.copyfile(checkpoint_sources[key], destination)
        if file_sha256(destination) != checkpoint_hashes[key]:
            raise PostE21ContractError(f"Aggregated E23b checkpoint copy changed: {key}")
        final_checkpoint_paths[key] = destination
    for row in [*rows, *training_rows]:
        checkpoint_key = f"{row['controller_id']}_seed{int(row['seed'])}"
        row["checkpoint"] = str(final_checkpoint_paths[checkpoint_key].resolve())

    seed_rows, detail = summarize_seed_predictions(
        rows,
        seeds=[int(value) for value in runtime["seeds"]],
        intensities=[float(value) for value in runtime["intensities"]],
        updates=[int(value) for value in runtime["updates"]],
        gap_events=[int(value) for value in runtime["gap_events"]],
        affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
        target_margin=float(config["adequacy"]["target_margin"]),
        retention_margin=float(config["adequacy"]["retention_margin"]),
        locality_margin=float(config["adequacy"]["locality_margin"]),
        minimum_single_axis_exact_matches=int(
            config["adequacy"]["minimum_single_axis_exact_matches"]
        ),
        minimum_pairwise_exact_matches=int(config["adequacy"]["minimum_pairwise_exact_matches"]),
        incomparable_direction_margin=float(config["adequacy"]["incomparable_direction_margin"]),
        maximal_simpler_degradation_margin=float(
            config["adequacy"]["maximal_simpler_degradation_margin"]
        ),
        boundary_mode=str(prepared["boundary_mode"]),
    )
    if (
        len(seed_rows) != prepared["expected_counts"]["seed_rows"]
        or len(detail["cells"]) != prepared["expected_counts"]["cell_rows"]
    ):
        raise PostE21ContractError("Aggregated E23b statistical rows are incomplete")
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=rows,
        seed_rows=seed_rows,
        raw_filename="product_poset_raw_metrics.jsonl",
        seed_filename="product_poset_seed_metrics.jsonl",
    )
    cell_artifact = write_cell_rows(
        run_dir=run_dir,
        cell_rows=detail["cells"],
    )
    training_artifact = write_training_rows(
        run_dir=run_dir,
        rows=training_rows,
    )
    assessment = detail["assessment"]
    claim_status, supported = _claim_status(
        dry_run=dry_run,
        boundary_mode=str(prepared["boundary_mode"]),
        assessment=assessment,
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    dependency = prepared["dependency"]
    summary_path.write_text(
        results_summary_ko(
            phase="E23b Confirmatory",
            run_mode=run_mode,
            status=claim_status,
            boundary_mode=str(prepared["boundary_mode"]),
            assessment=assessment,
            dependency_reason=";".join(
                (
                    str(dependency["e18"]["reason"]),
                    str(dependency["e23a_screen"]["reason"]),
                    str(dependency["e22"]["reason"]),
                )
            ),
        ),
        encoding="utf-8",
    )
    claim_eligible = bool(not dry_run and supported)
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=parent_snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=claim_eligible,
    )
    topology_payload = {
        "mode": "REGISTERED_SEED_SHARDED_V1",
        "scientific_protocol_unchanged": True,
        "no_change_flags": dict(prepared["no_change_flags"]),
        "checkpoint_reuse_or_resume": prepared["checkpoint_reuse_or_resume"],
        "shard_axis": "registered_training_seed",
        "aggregation_order": "frozen_config_seed_order",
        "workspace": str(workspace_path),
        "prepared_manifest_sha256": prepared_sha,
        "execution_amendment_lock": {
            "source_path": str(amendment_snapshot.path),
            "sha256": amendment_snapshot.sha256,
            "run_snapshot_path": str(amendment_copy.resolve()),
        },
        "source_lock": dict(prepared["source_lock"]),
        "equivalence_report": (
            None
            if equivalence_copy is None
            else {
                **dict(prepared["equivalence_report"]),
                "path": str(equivalence_copy.resolve()),
            }
        ),
        "shards": canonical_shard_descriptors,
        "validation": {
            "registered_seed_cover_exact": True,
            "raw_cartesian_grid_exact": True,
            "training_grid_exact": True,
            "checkpoint_hashes_verified": True,
            "dependency_provenance_verified_per_row": True,
            "paired_data_digest_verified": True,
            "physical_gpu_bindings_verified": True,
            "physical_gpu_bindings_unique_and_homogeneous": True,
            "nonfinite_rows": 0,
            "duplicate_rows": 0,
        },
    }
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "experiment_id": LOGICAL_EXPERIMENT_ID,
        "run_mode": run_mode,
        "phase": "CONFIRMATORY",
        **metadata,
        "dependency": dependency,
        "e18_dependency": dependency["e18"],
        "e23a_screen_dependency": dependency["e23a_screen"],
        "e22_dependency": dependency["e22"],
        "boundary_mode": prepared["boundary_mode"],
        "boundary_selection": {
            "rule": "theory_boundary_only_v1",
            "result_independent": True,
            "selected_before_e23_outcomes": True,
            "e23a_outcomes_used": False,
            "e23a_screen_recorded_for_pipeline_provenance_only": True,
            "sets": theory["confirmatory_boundary_sets"],
        },
        "theory_prediction": {
            "locked_before_outcomes": True,
            "sha256": theory["sha256"],
            "poset_minimal_sets": theory["poset_minimal_sets"],
        },
        "execution_topology": topology_payload,
        "summary": assessment,
        "artifacts": {
            "prepared_execution": {
                "path": str(prepared_copy.resolve()),
                "sha256": file_sha256(prepared_copy),
                "payload_sha256": prepared_sha,
            },
            "shard_manifests": [
                {
                    "shard_id": descriptor["shard_id"],
                    "path": descriptor["manifest_path"],
                    "sha256": descriptor["manifest_file_sha256"],
                    "payload_sha256": descriptor["manifest_payload_sha256"],
                }
                for descriptor in canonical_shard_descriptors
            ],
            "data_manifest": {
                "path": str(data_manifest_path.resolve()),
                "sha256": file_sha256(data_manifest_path),
            },
            "theory_predictions": theory_artifact,
            "rows": row_artifacts,
            "training_runs": training_artifact,
            "poset_minimal_demands": cell_artifact,
            "results_summary_ko": {
                "path": str(summary_path.resolve()),
                "sha256": file_sha256(summary_path),
                "line_count": len(summary_path.read_text(encoding="utf-8").splitlines()),
            },
            "execution_amendment_lock": {
                "path": str(amendment_copy.resolve()),
                "sha256": file_sha256(amendment_copy),
            },
            "serial_shard_equivalence": (
                None
                if equivalence_copy is None
                else {
                    "path": str(equivalence_copy.resolve()),
                    "sha256": file_sha256(equivalence_copy),
                }
            ),
        },
        "claim_gate": {
            "status": claim_status,
            "supported": supported,
            "safe_locality_supported": bool(assessment["safe_minimality_supported"]),
            "capacity_supported": bool(assessment["capacity_supported"]),
            "allowed_claim": (
                "Safe absolute-adequacy minimal controller recovery in a controlled "
                "four-axis sequence poset."
                if prepared["boundary_mode"] == "safe_minimality"
                else "Capacity-only absolute-adequacy minimal controller recovery in a "
                "controlled four-axis sequence poset."
            ),
            "forbidden_claim": (
                "Locality when boundary_mode=capacity_only; semantic, natural-"
                "language, language-model, agent, official-backend, or runtime "
                "transfer in every mode."
            ),
        },
    }
    finalize_run(
        experiment_id=LOGICAL_EXPERIMENT_ID,
        artifact_root=str(canonical_artifact_root),
        run_dir=run_dir,
        report=report,
    )
    report_path = run_dir / "report.json"
    receipt_payload = {
        "schema_version": 1,
        "status": "AGGREGATED",
        "completed_at_utc": _utc_now(),
        "logical_experiment_id": LOGICAL_EXPERIMENT_ID,
        "run_mode": run_mode,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "prepared_manifest_sha256": prepared_sha,
        "execution_amendment_lock_sha256": amendment_snapshot.sha256,
    }
    _write_envelope(receipt_path, receipt_payload)
    return Path(run_dir)


def aggregate_sharded_execution(
    *,
    workspace: str | Path,
    artifact_root: str | Path,
    device_request: str = "cpu",
) -> Path:
    """Serialize aggregation per workspace through receipt and latest update."""

    raw_workspace = Path(workspace)
    if raw_workspace.is_symlink():
        raise PostE21ContractError(f"Unsafe symlinked E23b workspace: {raw_workspace}")
    workspace_path = raw_workspace.resolve(strict=True)
    if not (workspace_path / PREPARED_MANIFEST_NAME).is_file():
        raise PostE21ContractError(f"E23b prepared manifest is missing: {workspace_path}")
    lock_path = workspace_path / AGGREGATE_LOCK_NAME
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PostE21ContractError(
                f"E23b aggregate already active for workspace: {workspace_path}"
            ) from error
        try:
            return _aggregate_sharded_execution_locked(
                workspace=workspace_path,
                artifact_root=artifact_root,
                device_request=device_request,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def normalized_equivalence_payload(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Extract deterministic scientific fields for serial-vs-shard validation."""

    run = Path(run_dir).resolve(strict=True)
    raw = _object_rows(run / "product_poset_raw_metrics.jsonl")
    training = _object_rows(run / "product_poset_training_runs.jsonl")
    seed = _object_rows(run / "product_poset_seed_metrics.jsonl")
    cells = _object_rows(run / "poset_minimal_demands.jsonl")
    for row in raw:
        row.pop("checkpoint", None)
    for row in training:
        row.pop("checkpoint", None)
        row.pop("examples_per_second", None)
        row.pop("peak_memory_bytes", None)
    report = read_json_object_strict(run / "report.json")
    data_manifest = read_json_object_strict(run / "data_manifest.json")
    return {
        "raw": raw,
        "training": training,
        "seed": seed,
        "cells": cells,
        "summary": report["summary"],
        "claim_gate": report["claim_gate"],
        "boundary_mode": report["boundary_mode"],
        "data_sha256": data_manifest["data_sha256"],
        "checkpoint_hashes": report["checkpoint_hashes"],
    }


def require_main_acknowledgement() -> None:
    if os.environ.get("CATENA_POST_E21_MAIN_ACK") != "POST_E21_MAIN_AUTHORIZED":
        raise PermissionError("MAIN requires CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED")
