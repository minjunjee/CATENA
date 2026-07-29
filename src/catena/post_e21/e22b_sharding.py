"""Performance-only paired-seed sharding for the frozen E22b experiment.

This module changes execution topology only.  It delegates every model,
training, evaluation, metric, threshold, and claim decision to the frozen E22b
implementation.  A coordinator creates one canonical run, launches one worker
per registered seed shard, validates immutable worker manifests, restores the
serial row order, and only then finalizes the ordinary E22b report.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import environment_snapshot, file_sha256
from catena.core.provenance_v61 import (
    SourceTreeFingerprint,
    read_json_object_strict,
    read_jsonl_strict,
    sha256_canonical_json,
    source_tree_fingerprint,
    write_json_strict,
    write_jsonl_strict,
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
from catena.post_e21.locality_data import (
    LocalityMethod,
    method_by_id,
    parse_locality_methods,
)
from catena.post_e21.locality_eval import (
    assess_locality_confirmatory,
    build_active_cell_rows,
    compute_locality_seed_summaries,
    confirmatory_summary_ko,
    validate_paired_metric_grid,
)
from catena.post_e21.locality_protocol import (
    ParentThresholdContract,
    load_parent_threshold_contract,
    require_temp_dry_root,
    threshold_float,
    validate_parent_binding,
    validate_selection_run_dependency,
)
from catena.post_e21.locality_runner import (
    run_locality_method_grid,
    runtime_locality_config,
)
from catena.systems.device import resolve_device
from catena.training.structured_sequence_localization import (
    structured_state_dict_sha256,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e22b_active_path_locality"
DEFAULT_CONFIG = "configs/e22b_active_path_locality.yaml"
BASE_LOCK_RELATIVE = "docs/E22B_ACTIVE_PATH_LOCALITY_PROTOCOL_LOCK.json"
AMENDMENT_LOCK_RELATIVE = "docs/E22B_SEED_SHARD_EXECUTION_AMENDMENT_LOCK.json"
AMENDMENT_ID = "e22b_paired_seed_sharding_v1"
BASE_SOURCE_COMMIT = "51156242dfc429cb66d577c144b8d38a5ae38551"
SHARD_COUNT = 4
TAG_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
MAIN_ACK_ENV = "CATENA_POST_E21_MAIN_ACK"
MAIN_ACK_VALUE = "POST_E21_MAIN_AUTHORIZED"
CATENA_V6_PREFIX = Path("/home/minjun_dev/miniconda3/envs/catena-v6")
SCIENTIFIC_NO_CHANGE_FLAGS = {
    "batch_size_changed": False,
    "checkpoint_reuse_or_resume": False,
    "config_changed": False,
    "data_or_namespace_changed": False,
    "metric_or_claim_changed": False,
    "model_or_optimizer_changed": False,
    "precision_changed": False,
    "seed_or_seed_offset_changed": False,
    "steps_changed": False,
    "threshold_or_sesoi_changed": False,
}
EQUIVALENCE_CHECK_KEYS = (
    "raw_row_count_equal",
    "canonical_scientific_rows_exact",
    "checkpoint_state_hashes_exact",
    "identifier_codebook_hash_exact",
    "initialization_hashes_exact",
    "parameter_counts_exact",
)
EQUIVALENCE_COMPARISON_EXCLUSIONS = (
    "examples_per_second",
    "checkpoint container file SHA-256",
)
EQUIVALENCE_FIXED_FIELDS = (
    "scientific_evidence",
    "claim_eligible",
    "seeds",
    "selected_method_id",
    "baseline_method_id",
    "runtime_config_sha256",
    "serial_rows",
    "sharded_rows",
    "comparison_exclusions",
    "checkpoint_state_hash_comparison",
    "scientific_metric_comparison",
)


@dataclass(frozen=True, slots=True)
class E22BScientificContract:
    """Validated frozen inputs shared by serial and sharded E22b execution."""

    config: dict[str, Any]
    snapshot: ProtocolSnapshot
    parent: ParentThresholdContract
    dependency: dict[str, Any]
    methods: tuple[LocalityMethod, LocalityMethod]
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AmendmentSnapshot:
    """Validated performance-only execution amendment."""

    path: Path
    sha256: str
    payload: dict[str, Any]


@dataclass(slots=True)
class LauncherLockHandle:
    """Held non-blocking filesystem lock for one canonical E22b launcher."""

    descriptor: int
    path: Path
    record: dict[str, Any]

    def release(self) -> None:
        if self.descriptor < 0:
            return
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = -1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _require_regular_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise PostE21ContractError(f"{label} is missing or unsafe: {path}")
    return resolved


def _relative_contained(root: Path, path: Path, *, label: str) -> str:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise PostE21ContractError(f"{label} escapes its artifact root") from error


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
    base_protocol_sha256: str,
    amendment_sha256: str,
) -> dict[str, Any]:
    """Require an annotated tag on a clean current HEAD before scientific MAIN."""

    if not TAG_PATTERN.fullmatch(tag):
        raise PostE21ContractError("E22b source-lock tag contains unsafe characters")
    head = _git(repo_root, "rev-parse", "HEAD")
    tagged_commit = _git(repo_root, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    tag_type = _git(repo_root, "cat-file", "-t", f"refs/tags/{tag}")
    if tag_type != "tag":
        raise PostE21ContractError("E22b source lock must be an annotated tag")
    if tagged_commit != head:
        raise PostE21ContractError("E22b source-lock tag does not point to current HEAD")
    dirty = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise PostE21ContractError("E22b MAIN requires a clean source-locked worktree")
    message = _git(repo_root, "for-each-ref", "--format=%(contents)", f"refs/tags/{tag}")
    required_lines = {
        f"E22B_BASE_PROTOCOL_LOCK_SHA256={base_protocol_sha256}",
        f"E22B_SEED_SHARD_AMENDMENT_LOCK_SHA256={amendment_sha256}",
    }
    if not required_lines.issubset(set(message.splitlines())):
        raise PostE21ContractError(
            "E22b source-lock tag message lacks protocol/amendment SHA bindings"
        )
    return {
        "tag": tag,
        "tag_object_id": _git(repo_root, "rev-parse", f"refs/tags/{tag}^{{tag}}"),
        "tag_message_sha256": _hash_text(message),
        "git_commit": head,
        "dirty_status": "clean",
    }


def registered_equivalence_validation_contract() -> dict[str, Any]:
    """Return the exact execution-lock contract for CPU equivalence evidence."""

    return {
        "check_keys": list(EQUIVALENCE_CHECK_KEYS),
        "checks_exact_and_all_true": True,
        "comparison_exclusions": list(EQUIVALENCE_COMPARISON_EXCLUSIONS),
        "derived_fields": [
            "seeds",
            "selected_method_id",
            "baseline_method_id",
            "runtime_config_sha256",
            "serial_rows",
            "sharded_rows",
        ],
        "fixed_non_evidence_fields": {
            "checkpoint_state_hash_comparison": "exact",
            "claim_eligible": False,
            "scientific_evidence": False,
            "scientific_metric_comparison": "exact",
        },
        "producer_self_validation_required": True,
    }


def validate_amendment_payload(payload: Mapping[str, Any]) -> None:
    """Validate every authorization and declared scientific no-change field."""

    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("amendment_id") != AMENDMENT_ID
        or payload.get("performance_only") is not True
        or payload.get("scientific_protocol_changed") is not False
        or payload.get("base_source_commit") != BASE_SOURCE_COMMIT
        or payload.get("main_authorized") is not True
        or payload.get("source_lock_required_before_main") is not True
    ):
        raise PostE21ContractError("E22b execution amendment identity is invalid")
    authorization = payload.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("authorized_at_utc_date") != "2026-07-29"
        or authorization.get("scope") != "execution_topology_only"
        or authorization.get("scientific_change_authorized") is not False
        or authorization.get("user_authorized") is not True
    ):
        raise PostE21ContractError("E22b execution-topology authorization is invalid")
    invariants = payload.get("scientific_invariants")
    if invariants != SCIENTIFIC_NO_CHANGE_FLAGS:
        raise PostE21ContractError(
            "E22b execution amendment scientific no-change flags are incomplete or changed"
        )
    if (
        payload.get("registered_equivalence_validation")
        != registered_equivalence_validation_contract()
    ):
        raise PostE21ContractError("E22b registered equivalence-validation contract changed")


def validate_execution_amendment(*, repo_root: Path) -> AmendmentSnapshot:
    """Validate the additive topology lock without changing the E22b protocol."""

    path = _require_regular_file(
        repo_root / AMENDMENT_LOCK_RELATIVE,
        label="E22b shard amendment lock",
    )
    payload = read_json_object_strict(path)
    validate_amendment_payload(payload)
    base_lock = _require_regular_file(
        repo_root / BASE_LOCK_RELATIVE,
        label="E22b base protocol lock",
    )
    if payload.get("base_protocol_lock_sha256") != file_sha256(base_lock):
        raise PostE21ContractError("E22b amendment does not bind the frozen protocol lock")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise PostE21ContractError("E22b amendment lacks its immutable file map")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not _is_sha256(expected):
            raise PostE21ContractError("E22b amendment file map is malformed")
        candidate = _require_regular_file(
            repo_root / relative,
            label=f"E22b amendment source {relative}",
        )
        _relative_contained(repo_root, candidate, label=relative)
        if file_sha256(candidate) != expected:
            raise PostE21ContractError(f"E22b amendment source changed: {relative}")
    return AmendmentSnapshot(path=path, sha256=file_sha256(path), payload=payload)


def registered_seed_shards(seeds: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Partition paired training seeds round-robin without splitting a seed."""

    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) != 8 or len(set(normalized)) != 8:
        raise PostE21ContractError("E22b shard execution requires exactly 8 unique seeds")
    shards = tuple(
        tuple(normalized[index] for index in range(shard_index, len(normalized), SHARD_COUNT))
        for shard_index in range(SHARD_COUNT)
    )
    if any(len(shard) != 2 for shard in shards):
        raise PostE21ContractError("E22b shard partition must contain 2 paired seeds per shard")
    flattened = [seed for shard in shards for seed in shard]
    if sorted(flattened) != sorted(normalized):
        raise PostE21ContractError("E22b shard partition lost or duplicated seeds")
    return shards


def validate_registered_shard_plan(
    *,
    amendment: AmendmentSnapshot,
    seeds: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    shards = registered_seed_shards(seeds)
    registered = amendment.payload.get("registered_execution")
    if not isinstance(registered, dict):
        raise PostE21ContractError("E22b amendment lacks registered_execution")
    observed = registered.get("seed_shards")
    expected = [list(shard) for shard in shards]
    if (
        registered.get("shard_count") != SHARD_COUNT
        or registered.get("partition_rule") != "round_robin_registered_seed_order"
        or registered.get("worker_unit") != "complete_paired_seed"
        or registered.get("checkpoint_materialization") != "atomic_independent_byte_copy"
        or registered.get("canonical_aggregate_order")
        != [
            "seed",
            "method",
            "variant",
            "condition",
            "demand_family",
            "updates",
            "gap_events",
        ]
        or observed != expected
    ):
        raise PostE21ContractError("E22b registered seed-shard topology changed")
    main_gates = amendment.payload.get("registered_main_gates")
    if main_gates != {
        "acknowledgement": f"{MAIN_ACK_ENV}={MAIN_ACK_VALUE}",
        "canonical_equivalence_copy_required": True,
        "catena_v6_interpreter": str(CATENA_V6_PREFIX / "bin/python"),
        "catena_v6_prefix": str(CATENA_V6_PREFIX),
        "exclusive_launcher_lock_required": True,
        "four_idle_homogeneous_gpus_required": True,
    }:
        raise PostE21ContractError("E22b registered MAIN gates changed")
    if (
        amendment.payload.get("registered_equivalence_validation")
        != registered_equivalence_validation_contract()
    ):
        raise PostE21ContractError("E22b registered equivalence-validation contract changed")
    return shards


def load_scientific_contract(
    *,
    repo_root: Path,
    config_path: str | Path,
    selection_run: str | Path,
    dependency_run_mode: str,
) -> E22BScientificContract:
    """Re-run every frozen E22b protocol and E22a dependency check."""

    if dependency_run_mode not in {"MAIN", "DRY_RUN"}:
        raise ValueError("dependency_run_mode must be MAIN or DRY_RUN")
    config = load_config(config_path)
    if (
        config.get("experiment_id") != EXPERIMENT_ID
        or config.get("protocol", {}).get("phase") != "confirmatory"
        or "thresholds" in config
        or "claim_gate" in config
    ):
        raise PostE21ContractError("E22b config identity/threshold contract failed")
    snapshot = validate_protocol_lock(
        lock_path=repo_root / BASE_LOCK_RELATIVE,
        config_path=config_path,
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
    )
    parent = load_parent_threshold_contract(repo_root=repo_root)
    validate_parent_binding(snapshot=snapshot, parent=parent)

    selection_contract = config["selection_contract"]
    e22a_config_path = repo_root / str(selection_contract["config_path"])
    e22a_snapshot = validate_protocol_lock(
        lock_path=repo_root / str(selection_contract["protocol_lock_path"]),
        config_path=e22a_config_path,
        experiment_id=str(selection_contract["experiment_id"]),
        repo_root=repo_root,
    )
    if snapshot.payload.get("parent_e22a_static_lock_sha256") != e22a_snapshot.sha256:
        raise PostE21ContractError("E22b static lock does not bind E22a")
    dependency = validate_selection_run_dependency(
        selection_run=selection_run,
        parent=parent,
        expected_protocol_lock_sha256=e22a_snapshot.sha256,
        dry_run=dependency_run_mode == "DRY_RUN",
    )
    e22a_methods = parse_locality_methods(load_config(e22a_config_path)["methods"])
    selected_payload = dependency["selected_method"]
    baseline_payload = dependency["baseline_method"]
    selected = method_by_id(e22a_methods, str(selected_payload["method_id"]))
    baseline = method_by_id(e22a_methods, str(baseline_payload["method_id"]))
    if selected.as_dict() != selected_payload or baseline.as_dict() != baseline_payload:
        raise PostE21ContractError("E22a selected method differs from frozen method grid")
    seeds = tuple(int(value) for value in config["confirmatory_seeds"])
    development_seeds = {int(value) for value in load_config(e22a_config_path)["development_seeds"]}
    if len(seeds) != 8 or len(set(seeds)) != 8 or set(seeds) & development_seeds:
        raise PostE21ContractError("E22b requires eight unique fresh paired seeds")
    return E22BScientificContract(
        config=config,
        snapshot=snapshot,
        parent=parent,
        dependency=dependency,
        methods=(baseline, selected),
        seeds=seeds,
    )


def _method_payloads(methods: Sequence[LocalityMethod]) -> list[dict[str, Any]]:
    return [method.as_dict() for method in methods]


def build_e22b_data_payload(
    *,
    runtime: Mapping[str, Any],
    contract: E22BScientificContract,
    dry_run: bool,
) -> dict[str, Any]:
    """Return the exact data-manifest payload used by the serial E22b entrypoint."""

    return {
        "phase": "E22b",
        "dry_run": bool(dry_run),
        "confirmatory_seeds": list(contract.seeds),
        "methods": _method_payloads(contract.methods),
        "variants": list(runtime["model"]["variants"]),
        "conditions": list(runtime["conditions"]),
        "demand_families": list(runtime["demand_families"]),
        "training_grid": dict(runtime["training"]),
        "evaluation_grid": dict(runtime["evaluation"]),
        "namespaces": dict(runtime["namespaces"]),
        "parent_e21_lock_sha256": contract.parent.sha256,
        "selection_lock_sha256": contract.dependency["selection_lock_sha256"],
    }


def build_equivalence_validation_contract(
    *,
    confirmatory_seeds: Sequence[int],
    methods: Sequence[LocalityMethod],
    dry_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive every equivalence-only field from frozen scientific inputs."""

    normalized_seeds = tuple(int(seed) for seed in confirmatory_seeds)
    if len(normalized_seeds) != 8 or len(set(normalized_seeds)) != 8:
        raise PostE21ContractError(
            "E22b equivalence contract requires eight registered confirmatory seeds"
        )
    if len(methods) != 2:
        raise PostE21ContractError(
            "E22b equivalence contract requires baseline and selected methods"
        )
    baseline, selected = methods
    if (
        baseline.method_id != "mean_retention"
        or baseline.baseline is not True
        or selected.baseline is not False
        or selected.selection_eligible is not True
    ):
        raise PostE21ContractError(
            "E22b equivalence method identities differ from the selection contract"
        )
    equivalence_seeds = normalized_seeds[:4]
    expected_rows = (
        len(equivalence_seeds)
        * len(methods)
        * len(dry_runtime["model"]["variants"])
        * len(dry_runtime["conditions"])
        * len(dry_runtime["demand_families"])
        * len(dry_runtime["evaluation"]["updates"])
        * len(dry_runtime["evaluation"]["gap_events"])
    )
    if expected_rows <= 0:
        raise PostE21ContractError("E22b equivalence expected row count is invalid")
    return {
        "scientific_evidence": False,
        "claim_eligible": False,
        "seeds": list(equivalence_seeds),
        "selected_method_id": selected.method_id,
        "baseline_method_id": baseline.method_id,
        "runtime_config_sha256": sha256_canonical_json(dry_runtime),
        "serial_rows": expected_rows,
        "sharded_rows": expected_rows,
        "comparison_exclusions": list(EQUIVALENCE_COMPARISON_EXCLUSIONS),
        "checkpoint_state_hash_comparison": "exact",
        "scientific_metric_comparison": "exact",
    }


def _source_record(repo_root: Path) -> dict[str, int | str]:
    fingerprint = source_tree_fingerprint(repo_root)
    return {"sha256": fingerprint.sha256, "files": fingerprint.files}


def _descriptor(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path.resolve(strict=True)),
        "sha256": file_sha256(path),
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def _canonical_row_key(
    row: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    methods: Sequence[str],
    variants: Sequence[str],
    conditions: Sequence[str],
    demand_families: Sequence[str],
    updates: Sequence[int],
    gaps: Sequence[int],
) -> tuple[int, int, int, int, int, int, int]:
    dimensions: tuple[Sequence[Any], ...] = (
        seeds,
        methods,
        variants,
        conditions,
        demand_families,
        updates,
        gaps,
    )
    values: tuple[Any, ...] = (
        int(row["seed"]),
        str(row["method_id"]),
        str(row["variant"]),
        str(row["condition"]),
        str(row["demand_family"]),
        int(row["updates"]),
        int(row["gap_events"]),
    )
    try:
        return (
            dimensions[0].index(values[0]),
            dimensions[1].index(values[1]),
            dimensions[2].index(values[2]),
            dimensions[3].index(values[3]),
            dimensions[4].index(values[4]),
            dimensions[5].index(values[5]),
            dimensions[6].index(values[6]),
        )
    except ValueError as error:
        raise PostE21ContractError("E22b row contains an unregistered grid value") from error


def canonicalize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: Mapping[str, Any],
    methods: Sequence[LocalityMethod],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Restore the byte-order-equivalent serial nesting order."""

    method_ids = [method.method_id for method in methods]
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: _canonical_row_key(
            row,
            seeds=[int(seed) for seed in seeds],
            methods=method_ids,
            variants=[str(value) for value in runtime["model"]["variants"]],
            conditions=[str(value) for value in runtime["conditions"]],
            demand_families=[str(value) for value in runtime["demand_families"]],
            updates=[int(value) for value in runtime["evaluation"]["updates"]],
            gaps=[int(value) for value in runtime["evaluation"]["gap_events"]],
        ),
    )


def _checkpoint_state_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise PostE21ContractError(f"E22b checkpoint payload is malformed: {path}")
    digest = structured_state_dict_sha256(payload["model"])
    if not isinstance(digest, str):
        raise PostE21ContractError("E22b checkpoint state hash is not a string")
    return digest


def _checkpoint_paths(run_dir: Path, checkpoint_hashes: Mapping[str, str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key in checkpoint_hashes:
        path = run_dir / "checkpoints" / f"{key}.pt"
        if not path.is_file() or path.is_symlink():
            raise PostE21ContractError(f"E22b checkpoint is missing or unsafe: {path}")
        result[key] = path
    return result


def _write_worker_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite E22b shard manifest: {path}")
    write_json_strict(path, dict(payload))


def _read_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json_object_strict(_require_regular_file(plan_path, label="E22b execution plan"))
    if (
        plan.get("schema_version") != 1
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("amendment_id") != AMENDMENT_ID
        or plan.get("status") != "LOCKED_BEFORE_SHARD_EXECUTION"
    ):
        raise PostE21ContractError("E22b execution plan identity is invalid")
    detached_path = plan_path.with_suffix(".sha256.json")
    detached = read_json_object_strict(
        _require_regular_file(
            detached_path,
            label="E22b detached execution-plan hash",
        )
    )
    if (
        detached.get("schema_version") != 1
        or detached.get("experiment_id") != EXPERIMENT_ID
        or detached.get("run_id") != plan.get("run_id")
        or detached.get("sha256") != file_sha256(plan_path)
    ):
        raise PostE21ContractError("E22b detached execution-plan hash is invalid")
    return dict(plan)


def _validate_plan_bindings(
    *,
    repo_root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
) -> tuple[AmendmentSnapshot, SourceTreeFingerprint]:
    amendment = validate_execution_amendment(repo_root=repo_root)
    if plan.get("amendment_lock_sha256") != amendment.sha256:
        raise PostE21ContractError("E22b plan amendment binding changed")
    base_lock_sha = file_sha256(repo_root / BASE_LOCK_RELATIVE)
    if plan.get("base_protocol_lock_sha256") != base_lock_sha:
        raise PostE21ContractError("E22b plan base-protocol binding changed")
    source = source_tree_fingerprint(repo_root)
    if plan.get("source_fingerprint") != source.as_dict():
        raise PostE21ContractError("E22b source changed after the execution plan was locked")
    run_dir = Path(str(plan["run_dir"])).resolve(strict=True)
    if plan_path.resolve(strict=True) != run_dir / "execution_plan.json":
        raise PostE21ContractError("E22b execution plan is not inside its canonical run")
    protocol_copy = _require_regular_file(
        run_dir / "protocol_lock.json",
        label="E22b run protocol snapshot",
    )
    amendment_copy = _require_regular_file(
        run_dir / "execution_amendment_lock.json",
        label="E22b run amendment snapshot",
    )
    data_manifest_path = _require_regular_file(
        run_dir / "data_manifest.json",
        label="E22b run data manifest",
    )
    if file_sha256(protocol_copy) != str(plan["base_protocol_lock_sha256"]):
        raise PostE21ContractError("E22b run protocol snapshot changed")
    if file_sha256(amendment_copy) != str(plan["amendment_lock_sha256"]):
        raise PostE21ContractError("E22b run amendment snapshot changed")
    if file_sha256(data_manifest_path) != str(plan["data_manifest_sha256"]):
        raise PostE21ContractError("E22b run data manifest changed")
    data_manifest = read_json_object_strict(data_manifest_path)
    if data_manifest.get("data_sha256") != plan.get("data_sha256"):
        raise PostE21ContractError("E22b run data hash changed")
    run_manifest_path = _require_regular_file(
        run_dir / "run_manifest.json",
        label="E22b run manifest",
    )
    run_manifest = read_json_object_strict(run_manifest_path)
    if (
        run_manifest.get("experiment_id") != EXPERIMENT_ID
        or run_manifest.get("run_id") != run_dir.name
        or run_manifest.get("run_mode") != plan.get("run_mode")
        or run_manifest.get("config_file_sha256") != plan.get("config_sha256")
        or run_manifest.get("source_fingerprint") != source.as_dict()
    ):
        raise PostE21ContractError("E22b coordinator run manifest changed")
    topology = run_manifest.get("execution_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("mode") != "PAIRED_SEED_SHARDED"
        or topology.get("performance_only") is not True
        or topology.get("shard_count") != SHARD_COUNT
        or topology.get("devices") != plan.get("devices")
        or topology.get("execution_plan_sha256") != file_sha256(plan_path)
        or topology.get("execution_amendment_lock_sha256") != amendment.sha256
        or topology.get("main_preflight") != plan.get("main_preflight")
        or topology.get("equivalence_report") != plan.get("equivalence_report")
    ):
        raise PostE21ContractError("E22b coordinator topology manifest changed")
    if plan.get("run_mode") == "MAIN":
        source_lock = plan.get("source_lock")
        equivalence = plan.get("equivalence_report")
        if not isinstance(source_lock, dict) or not isinstance(equivalence, dict):
            raise PostE21ContractError("E22b MAIN plan lacks source/equivalence locks")
        validated_source_lock = validate_source_lock_tag(
            repo_root=repo_root,
            tag=str(source_lock.get("tag")),
            base_protocol_sha256=str(plan["base_protocol_lock_sha256"]),
            amendment_sha256=amendment.sha256,
        )
        if validated_source_lock != source_lock:
            raise PostE21ContractError("E22b MAIN source-lock record changed")
        validate_locked_main_preflight(
            preflight=plan.get("main_preflight"),
            devices=[str(value) for value in plan["devices"]],
            run_dir=run_dir,
        )
        equivalence_contract = build_equivalence_validation_contract(
            confirmatory_seeds=[int(value) for value in plan["confirmatory_seeds"]],
            methods=_methods_from_plan(repo_root=repo_root, plan=plan),
            dry_runtime=runtime_locality_config(
                load_config(repo_root / DEFAULT_CONFIG),
                dry_run=True,
            ),
        )
        validate_locked_equivalence_record(
            record=equivalence,
            run_dir=run_dir,
            source=source.as_dict(),
            amendment_sha256=amendment.sha256,
            base_protocol_sha256=str(plan["base_protocol_lock_sha256"]),
            config_sha256=str(plan["config_sha256"]),
            selection_lock_sha256=str(plan["selection_lock_sha256"]),
            equivalence_contract=equivalence_contract,
        )
    else:
        expected_non_evidence = {
            "required": False,
            "validated": False,
            "reason": "DRY_RUN_NON_EVIDENCE",
        }
        if (
            plan.get("run_mode") != "DRY_RUN"
            or plan.get("source_lock") != expected_non_evidence
            or plan.get("equivalence_report") is not None
            or plan.get("main_preflight") != expected_non_evidence
        ):
            raise PostE21ContractError("E22b dry-run preflight contract changed")
    return amendment, source


def _worker_seed_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    methods: Sequence[LocalityMethod],
    runtime: Mapping[str, Any],
) -> list[dict[str, Any]]:
    computed = compute_locality_seed_summaries(
        rows,
        seeds=[int(seed) for seed in seeds],
        method_ids=[method.method_id for method in methods],
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        stress_updates=int(runtime["evaluation"]["stress"]["updates"]),
        stress_gap_events=int(runtime["evaluation"]["stress"]["gap_events"]),
    )
    return [dict(row) for row in computed]


def run_shard_worker(
    *,
    repo_root: Path,
    plan_path: Path,
    shard_index: int,
    device_request: str,
) -> Path:
    """Execute one immutable complete-paired-seed shard."""

    plan = _read_plan(plan_path)
    amendment, source = _validate_plan_bindings(
        repo_root=repo_root,
        plan_path=plan_path,
        plan=plan,
    )
    shards_payload = plan.get("seed_shards")
    devices_payload = plan.get("devices")
    if not isinstance(shards_payload, list) or not isinstance(devices_payload, list):
        raise PostE21ContractError("E22b plan lacks shard/device lists")
    if shard_index < 0 or shard_index >= len(shards_payload):
        raise PostE21ContractError("E22b shard index is out of range")
    if str(devices_payload[shard_index]) != device_request:
        raise PostE21ContractError("E22b worker device differs from its locked plan")
    shard_seeds = tuple(int(seed) for seed in shards_payload[shard_index])
    runtime = plan.get("runtime_config")
    method_payloads = plan.get("methods")
    if not isinstance(runtime, dict) or not isinstance(method_payloads, list):
        raise PostE21ContractError("E22b plan lacks runtime/method payload")
    e22a_config = load_config(repo_root / "configs/e22a_locality_method_selection.yaml")
    registered_methods = parse_locality_methods(e22a_config["methods"])
    methods = tuple(
        method_by_id(registered_methods, str(payload["method_id"]))
        for payload in method_payloads
        if isinstance(payload, dict)
    )
    if len(methods) != 2 or _method_payloads(methods) != method_payloads:
        raise PostE21ContractError("E22b plan method payload changed")
    if sha256_canonical_json(runtime) != plan.get("runtime_config_sha256"):
        raise PostE21ContractError("E22b runtime config changed after plan lock")
    run_dir = Path(str(plan["run_dir"])).resolve(strict=True)
    shard_dir = run_dir / "shards" / f"shard-{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(device_request)
    started_at = utc_now()
    rows, checkpoint_hashes, runtime_metadata = run_locality_method_grid(
        runtime=runtime,
        methods=methods,
        seeds=shard_seeds,
        run_dir=shard_dir,
        device=device,
        parent_lock_sha256=str(plan["parent_e21_lock_sha256"]),
        protocol_lock_sha256=str(plan["base_protocol_lock_sha256"]),
        risk_scale=float(plan["risk_scale"]),
    )
    validate_paired_metric_grid(
        rows,
        seeds=shard_seeds,
        methods=[method.method_id for method in methods],
        variants=[str(value) for value in runtime["model"]["variants"]],
        conditions=[str(value) for value in runtime["conditions"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
    )
    rows = canonicalize_rows(
        rows,
        runtime=runtime,
        methods=methods,
        seeds=shard_seeds,
    )
    seed_rows = _worker_seed_rows(
        rows=rows,
        seeds=shard_seeds,
        methods=methods,
        runtime=runtime,
    )
    active_rows = build_active_cell_rows(rows)
    raw_path = shard_dir / "raw_metrics.jsonl"
    seed_path = shard_dir / "seed_metrics.jsonl"
    active_path = shard_dir / "active_cell_metrics.jsonl"
    environment_path = shard_dir / "environment.json"
    write_jsonl_strict(raw_path, rows)
    write_jsonl_strict(seed_path, seed_rows)
    write_jsonl_strict(active_path, active_rows)
    write_json_strict(environment_path, environment_snapshot())
    checkpoint_paths = _checkpoint_paths(shard_dir, checkpoint_hashes)
    state_hashes = {key: _checkpoint_state_hash(path) for key, path in checkpoint_paths.items()}
    manifest_path = shard_dir / "shard_manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "COMPLETE",
        "run_id": run_dir.name,
        "run_mode": str(plan["run_mode"]),
        "shard_index": shard_index,
        "shard_count": len(shards_payload),
        "paired_seeds": list(shard_seeds),
        "device_request": device_request,
        "resolved_device": str(device),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "base_protocol_lock_sha256": str(plan["base_protocol_lock_sha256"]),
        "amendment_lock_sha256": amendment.sha256,
        "selection_lock_sha256": str(plan["selection_lock_sha256"]),
        "config_sha256": str(plan["config_sha256"]),
        "data_sha256": str(plan["data_sha256"]),
        "execution_plan_sha256": file_sha256(plan_path),
        "source_fingerprint": source.as_dict(),
        "runtime_config_sha256": str(plan["runtime_config_sha256"]),
        "rows": {
            "raw": _descriptor(raw_path, rows=len(rows)),
            "seed": _descriptor(seed_path, rows=len(seed_rows)),
            "active_cell": _descriptor(active_path, rows=len(active_rows)),
        },
        "environment": _descriptor(environment_path),
        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
        "checkpoint_state_hashes": dict(sorted(state_hashes.items())),
        "checkpoint_paths": {
            key: _relative_contained(shard_dir, path, label=f"checkpoint {key}")
            for key, path in sorted(checkpoint_paths.items())
        },
        "runtime_metadata": runtime_metadata,
    }
    _write_worker_manifest(manifest_path, manifest)
    return manifest_path


def _expect_descriptor(
    *,
    descriptor: object,
    expected_path: Path,
    expected_rows: int | None,
    label: str,
) -> None:
    if not isinstance(descriptor, dict):
        raise PostE21ContractError(f"E22b {label} descriptor is missing")
    if Path(str(descriptor.get("path"))).resolve(strict=True) != expected_path.resolve(strict=True):
        raise PostE21ContractError(f"E22b {label} path differs from its shard manifest")
    if descriptor.get("sha256") != file_sha256(expected_path):
        raise PostE21ContractError(f"E22b {label} hash differs from its shard manifest")
    if expected_rows is not None and descriptor.get("rows") != expected_rows:
        raise PostE21ContractError(f"E22b {label} row count differs from its shard manifest")


def _validate_checkpoint_manifest(
    *,
    shard_dir: Path,
    manifest: Mapping[str, Any],
    shard_seeds: Sequence[int],
    method_ids: Sequence[str],
    variants: Sequence[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, Path]]:
    file_hashes = manifest.get("checkpoint_hashes")
    state_hashes = manifest.get("checkpoint_state_hashes")
    paths = manifest.get("checkpoint_paths")
    if not isinstance(file_hashes, dict) or not isinstance(state_hashes, dict):
        raise PostE21ContractError("E22b shard checkpoint hashes are missing")
    if not isinstance(paths, dict):
        raise PostE21ContractError("E22b shard checkpoint paths are missing")
    expected_keys = {
        f"seed{seed}_{method}_{variant}"
        for seed in shard_seeds
        for method in method_ids
        for variant in variants
    }
    if set(file_hashes) != expected_keys or set(state_hashes) != expected_keys:
        raise PostE21ContractError("E22b shard checkpoint key set is incomplete")
    if set(paths) != expected_keys:
        raise PostE21ContractError("E22b shard checkpoint path set is incomplete")
    resolved_paths: dict[str, Path] = {}
    normalized_file_hashes: dict[str, str] = {}
    normalized_state_hashes: dict[str, str] = {}
    for key in sorted(expected_keys):
        relative = paths[key]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise PostE21ContractError("E22b shard checkpoint path must be relative")
        checkpoint = _require_regular_file(
            shard_dir / relative,
            label=f"E22b shard checkpoint {key}",
        )
        _relative_contained(shard_dir, checkpoint, label=key)
        file_hash = str(file_hashes[key])
        state_hash = str(state_hashes[key])
        if not _is_sha256(file_hash) or file_sha256(checkpoint) != file_hash:
            raise PostE21ContractError(f"E22b shard checkpoint file hash changed: {key}")
        if not _is_sha256(state_hash) or _checkpoint_state_hash(checkpoint) != state_hash:
            raise PostE21ContractError(f"E22b shard checkpoint state hash changed: {key}")
        resolved_paths[key] = checkpoint
        normalized_file_hashes[key] = file_hash
        normalized_state_hashes[key] = state_hash
    return normalized_file_hashes, normalized_state_hashes, resolved_paths


def validate_and_load_shard(
    *,
    repo_root: Path,
    plan_path: Path,
    shard_index: int,
) -> dict[str, Any]:
    """Validate one worker's entire immutable artifact before aggregation."""

    plan = _read_plan(plan_path)
    amendment, source = _validate_plan_bindings(
        repo_root=repo_root,
        plan_path=plan_path,
        plan=plan,
    )
    run_dir = Path(str(plan["run_dir"])).resolve(strict=True)
    shard_dir = run_dir / "shards" / f"shard-{shard_index:03d}"
    manifest_path = _require_regular_file(
        shard_dir / "shard_manifest.json",
        label=f"E22b shard {shard_index} manifest",
    )
    manifest = read_json_object_strict(manifest_path)
    seed_shards = plan["seed_shards"]
    shard_seeds = tuple(int(seed) for seed in seed_shards[shard_index])
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "COMPLETE",
        "run_id": run_dir.name,
        "run_mode": str(plan["run_mode"]),
        "shard_index": shard_index,
        "shard_count": len(seed_shards),
        "paired_seeds": list(shard_seeds),
        "device_request": str(plan["devices"][shard_index]),
        "base_protocol_lock_sha256": str(plan["base_protocol_lock_sha256"]),
        "amendment_lock_sha256": amendment.sha256,
        "selection_lock_sha256": str(plan["selection_lock_sha256"]),
        "config_sha256": str(plan["config_sha256"]),
        "data_sha256": str(plan["data_sha256"]),
        "execution_plan_sha256": file_sha256(plan_path),
        "source_fingerprint": source.as_dict(),
        "runtime_config_sha256": str(plan["runtime_config_sha256"]),
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise PostE21ContractError(f"E22b shard {shard_index} identity mismatch: {key}")
    runtime = plan["runtime_config"]
    method_ids = [str(payload["method_id"]) for payload in plan["methods"]]
    variants = [str(value) for value in runtime["model"]["variants"]]
    raw_path = shard_dir / "raw_metrics.jsonl"
    seed_path = shard_dir / "seed_metrics.jsonl"
    active_path = shard_dir / "active_cell_metrics.jsonl"
    environment_path = shard_dir / "environment.json"
    raw_rows = read_jsonl_strict(raw_path)
    seed_rows = read_jsonl_strict(seed_path)
    active_rows = read_jsonl_strict(active_path)
    if not all(isinstance(row, dict) for row in [*raw_rows, *seed_rows, *active_rows]):
        raise PostE21ContractError("E22b shard JSONL contains a non-object row")
    expected_raw = (
        len(shard_seeds)
        * len(method_ids)
        * len(variants)
        * len(runtime["conditions"])
        * len(runtime["demand_families"])
        * len(runtime["evaluation"]["updates"])
        * len(runtime["evaluation"]["gap_events"])
    )
    expected_seed = len(shard_seeds) * len(method_ids)
    expected_active = len(build_active_cell_rows(raw_rows))
    descriptors = manifest.get("rows")
    if not isinstance(descriptors, dict):
        raise PostE21ContractError("E22b shard row descriptors are missing")
    _expect_descriptor(
        descriptor=descriptors.get("raw"),
        expected_path=raw_path,
        expected_rows=expected_raw,
        label="raw rows",
    )
    _expect_descriptor(
        descriptor=descriptors.get("seed"),
        expected_path=seed_path,
        expected_rows=expected_seed,
        label="seed rows",
    )
    _expect_descriptor(
        descriptor=descriptors.get("active_cell"),
        expected_path=active_path,
        expected_rows=expected_active,
        label="active-cell rows",
    )
    _expect_descriptor(
        descriptor=manifest.get("environment"),
        expected_path=environment_path,
        expected_rows=None,
        label="worker environment",
    )
    validate_paired_metric_grid(
        raw_rows,
        seeds=shard_seeds,
        methods=method_ids,
        variants=variants,
        conditions=[str(value) for value in runtime["conditions"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
    )
    checkpoint_hashes, state_hashes, checkpoint_paths = _validate_checkpoint_manifest(
        shard_dir=shard_dir,
        manifest=manifest,
        shard_seeds=shard_seeds,
        method_ids=method_ids,
        variants=variants,
    )
    metadata = manifest.get("runtime_metadata")
    if not isinstance(metadata, dict):
        raise PostE21ContractError("E22b shard runtime metadata is missing")
    initialization_hashes = metadata.get("initialization_hashes")
    parameter_counts = metadata.get("parameter_counts")
    identifier_hash = metadata.get("identifier_codebook_sha256")
    if not isinstance(initialization_hashes, dict) or not isinstance(parameter_counts, dict):
        raise PostE21ContractError("E22b shard runtime metadata is incomplete")
    methods_by_id = {str(payload["method_id"]): payload for payload in plan["methods"]}
    for row in raw_rows:
        seed = int(row["seed"])
        method_id = str(row["method_id"])
        variant = str(row["variant"])
        checkpoint_key = f"seed{seed}_{method_id}_{variant}"
        if str(row["checkpoint_sha256"]) != checkpoint_hashes[checkpoint_key]:
            raise PostE21ContractError("E22b shard row does not bind its checkpoint file")
        if str(row["initialization_sha256"]) != str(initialization_hashes[str(seed)]):
            raise PostE21ContractError("E22b shard row does not bind its paired initialization")
        if str(row["identifier_codebook_sha256"]) != str(identifier_hash):
            raise PostE21ContractError("E22b shard row codebook hash changed")
        parameter_key = f"{method_id}:{variant}"
        if int(row["parameter_count"]) != int(parameter_counts[parameter_key]):
            raise PostE21ContractError("E22b shard row parameter count changed")
        method_payload = methods_by_id[method_id]
        if str(row["objective"]) != str(method_payload["objective"]) or bool(
            row["selection_eligible"]
        ) != bool(method_payload["selection_eligible"]):
            raise PostE21ContractError("E22b shard row method metadata changed")
    expected_seed_rows = _worker_seed_rows(
        rows=raw_rows,
        seeds=shard_seeds,
        methods=_methods_from_plan(repo_root=repo_root, plan=plan),
        runtime=runtime,
    )
    if seed_rows != expected_seed_rows:
        raise PostE21ContractError("E22b shard seed summaries do not recompute exactly")
    if active_rows != build_active_cell_rows(raw_rows):
        raise PostE21ContractError("E22b shard active-cell rows do not recompute exactly")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "manifest": manifest,
        "raw_rows": raw_rows,
        "seed_rows": seed_rows,
        "active_rows": active_rows,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_state_hashes": state_hashes,
        "checkpoint_paths": checkpoint_paths,
    }


def _methods_from_plan(
    *,
    repo_root: Path,
    plan: Mapping[str, Any],
) -> tuple[LocalityMethod, LocalityMethod]:
    registered = parse_locality_methods(
        load_config(repo_root / "configs/e22a_locality_method_selection.yaml")["methods"]
    )
    payloads = plan["methods"]
    methods = tuple(method_by_id(registered, str(payload["method_id"])) for payload in payloads)
    if len(methods) != 2 or _method_payloads(methods) != payloads:
        raise PostE21ContractError("E22b execution plan method set changed")
    return methods[0], methods[1]


def aggregate_shards(
    *,
    repo_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    """Validate and merge all shards without consulting scientific outcomes."""

    plan = _read_plan(plan_path)
    _validate_plan_bindings(repo_root=repo_root, plan_path=plan_path, plan=plan)
    runtime = plan["runtime_config"]
    methods = _methods_from_plan(repo_root=repo_root, plan=plan)
    seeds = tuple(int(seed) for seed in plan["confirmatory_seeds"])
    shard_payloads = [
        validate_and_load_shard(
            repo_root=repo_root,
            plan_path=plan_path,
            shard_index=index,
        )
        for index in range(int(plan["shard_count"]))
    ]
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_state_hashes: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    initialization_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] | None = None
    identifier_hash: str | None = None
    for shard in shard_payloads:
        rows.extend(dict(row) for row in shard["raw_rows"])
        for key, value in shard["checkpoint_hashes"].items():
            if key in checkpoint_hashes:
                raise PostE21ContractError(f"E22b duplicate checkpoint across shards: {key}")
            checkpoint_hashes[key] = value
            checkpoint_state_hashes[key] = shard["checkpoint_state_hashes"][key]
            checkpoint_paths[key] = shard["checkpoint_paths"][key]
        metadata = shard["manifest"].get("runtime_metadata")
        if not isinstance(metadata, dict):
            raise PostE21ContractError("E22b shard runtime metadata is missing")
        current_identifier = str(metadata.get("identifier_codebook_sha256"))
        if identifier_hash is None:
            identifier_hash = current_identifier
        elif current_identifier != identifier_hash:
            raise PostE21ContractError("E22b codebook differs across seed shards")
        current_counts = metadata.get("parameter_counts")
        if not isinstance(current_counts, dict):
            raise PostE21ContractError("E22b shard parameter counts are missing")
        normalized_counts = {str(key): int(value) for key, value in current_counts.items()}
        if parameter_counts is None:
            parameter_counts = normalized_counts
        elif normalized_counts != parameter_counts:
            raise PostE21ContractError("E22b parameter surface differs across shards")
        current_initial = metadata.get("initialization_hashes")
        if not isinstance(current_initial, dict):
            raise PostE21ContractError("E22b shard initialization hashes are missing")
        for seed, value in current_initial.items():
            if seed in initialization_hashes:
                raise PostE21ContractError("E22b seed initialization appears in two shards")
            initialization_hashes[str(seed)] = str(value)
    rows = canonicalize_rows(rows, runtime=runtime, methods=methods, seeds=seeds)
    validate_paired_metric_grid(
        rows,
        seeds=seeds,
        methods=[method.method_id for method in methods],
        variants=[str(value) for value in runtime["model"]["variants"]],
        conditions=[str(value) for value in runtime["conditions"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
    )
    expected_checkpoint_count = len(seeds) * len(methods) * len(runtime["model"]["variants"])
    if len(checkpoint_hashes) != expected_checkpoint_count:
        raise PostE21ContractError("E22b merged checkpoint set is incomplete")
    if set(initialization_hashes) != {str(seed) for seed in seeds}:
        raise PostE21ContractError("E22b merged initialization set is incomplete")
    if identifier_hash is None or parameter_counts is None:
        raise PostE21ContractError("E22b merged runtime metadata is incomplete")
    return {
        "rows": rows,
        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
        "checkpoint_state_hashes": dict(sorted(checkpoint_state_hashes.items())),
        "checkpoint_paths": checkpoint_paths,
        "runtime_metadata": {
            "identifier_codebook_sha256": identifier_hash,
            "initialization_hashes": {
                str(seed): initialization_hashes[str(seed)] for seed in seeds
            },
            "parameter_counts": parameter_counts,
        },
        "shards": [
            {
                "shard_index": int(shard["manifest"]["shard_index"]),
                "paired_seeds": list(shard["manifest"]["paired_seeds"]),
                "device_request": str(shard["manifest"]["device_request"]),
                "resolved_device": str(shard["manifest"]["resolved_device"]),
                "manifest_path": str(shard["manifest_path"]),
                "manifest_sha256": str(shard["manifest_sha256"]),
            }
            for shard in shard_payloads
        ],
    }


def _atomic_copy_regular_file(*, source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite copied artifact: {target}")
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _materialize_canonical_checkpoints(
    *,
    run_dir: Path,
    checkpoint_paths: Mapping[str, Path],
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, str]:
    target_dir = run_dir / "checkpoints"
    target_dir.mkdir(parents=False, exist_ok=False)
    materialization: dict[str, str] = {}
    for key in sorted(checkpoint_paths):
        source = checkpoint_paths[key]
        target = target_dir / f"{key}.pt"
        _atomic_copy_regular_file(source=source, target=target)
        if os.stat(source).st_ino == os.stat(target).st_ino:
            raise PostE21ContractError(
                f"E22b canonical checkpoint unexpectedly shares an inode: {key}"
            )
        if file_sha256(target) != checkpoint_hashes[key]:
            raise PostE21ContractError(f"E22b canonical checkpoint hash mismatch: {key}")
        materialization[key] = "atomic_byte_copy"
    directory_descriptor = os.open(target_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return materialization


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = read_json_object_strict(path)
    if not isinstance(payload, dict):
        raise PostE21ContractError(f"Expected manifest object: {path}")
    return payload


def _update_run_manifest(path: Path, fields: Mapping[str, Any]) -> None:
    manifest = _read_manifest(path)
    for key in fields:
        if key in manifest:
            raise PostE21ContractError(f"Refusing to replace run-manifest field: {key}")
    manifest.update(dict(fields))
    write_json_strict(path, manifest)


def _write_execution_failure(
    *,
    run_dir: Path,
    stage: str,
    error: BaseException,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = run_dir / "execution_failure.json"
    if path.exists():
        return
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FAILED_OPERATIONAL_PRESERVED",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "recorded_at_utc": utc_now(),
        "latest_pointer_updated": False,
    }
    if extra:
        payload["details"] = dict(extra)
    write_json_strict(path, payload)


def _launch_workers(
    *,
    repo_root: Path,
    run_dir: Path,
    plan_path: Path,
    devices: Sequence[str],
) -> list[dict[str, Any]]:
    logs_dir = run_dir / "_worker_logs"
    logs_dir.mkdir(parents=False, exist_ok=False)
    processes: list[tuple[int, str, subprocess.Popen[bytes], Any, Path]] = []
    env = os.environ.copy()
    if all(device == "cpu" for device in devices):
        # Keep a non-evidence validation run from oversubscribing the host.
        # Scientific MAIN uses explicit CUDA devices and never enters this branch.
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
    source_paths = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        source_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    entrypoint = repo_root / "scripts/launch_e22b_seed_shards.py"
    results: list[dict[str, Any]] = []
    try:
        for index, device in enumerate(devices):
            log_path = logs_dir / f"shard-{index:03d}.log"
            log_handle = log_path.open("xb")
            command = [
                sys.executable,
                "-u",
                str(entrypoint),
                "_worker",
                "--plan",
                str(plan_path),
                "--shard-index",
                str(index),
                "--device",
                device,
            ]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log_handle.close()
                raise
            processes.append((index, device, process, log_handle, log_path))
        for index, device, process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            results.append(
                {
                    "shard_index": index,
                    "device": device,
                    "pid": process.pid,
                    "return_code": return_code,
                    "log_path": str(log_path.resolve()),
                    "log_sha256": file_sha256(log_path),
                }
            )
    except BaseException:
        for _, _, process, _, _ in processes:
            if process.poll() is None:
                with suppress(OSError):
                    process.terminate()
        for _, _, process, _, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        process.kill()
                    process.wait()
        raise
    finally:
        for _, _, _, log_handle, _ in processes:
            if not log_handle.closed:
                log_handle.close()
    failed = [result for result in results if int(result["return_code"]) != 0]
    if failed:
        raise PostE21ContractError(f"E22b shard worker failure: {failed}")
    return results


def _validate_devices(devices: Sequence[str], *, dry_run: bool) -> tuple[str, ...]:
    normalized = tuple(str(device).strip() for device in devices)
    if len(normalized) != SHARD_COUNT or any(not device for device in normalized):
        raise PostE21ContractError("E22b sharding requires exactly four device requests")
    if dry_run:
        if any(device != "cpu" for device in normalized):
            raise PostE21ContractError("E22b sharded dry-run requires four cpu workers")
        return normalized
    if len(set(normalized)) != SHARD_COUNT:
        raise PostE21ContractError("E22b MAIN requires four distinct GPU devices")
    if any(not re.fullmatch(r"cuda:\d+", device) for device in normalized):
        raise PostE21ContractError("E22b MAIN device map must contain explicit cuda:N devices")
    indices = [int(device.split(":", 1)[1]) for device in normalized]
    if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
        raise PostE21ContractError("E22b MAIN GPU map is unavailable")
    return normalized


def validate_main_authorization_and_runtime(
    *,
    environ: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
    prefix: str | Path | None = None,
    expected_prefix: str | Path = CATENA_V6_PREFIX,
) -> dict[str, Any]:
    """Require the explicit MAIN acknowledgement and exact catena-v6 runtime."""

    environment = os.environ if environ is None else environ
    if environment.get(MAIN_ACK_ENV) != MAIN_ACK_VALUE:
        raise PostE21ContractError(f"E22b MAIN requires {MAIN_ACK_ENV}={MAIN_ACK_VALUE}")
    expected = Path(expected_prefix).resolve(strict=True)
    observed_prefix = Path(sys.prefix if prefix is None else prefix).resolve(strict=True)
    observed_executable = Path(sys.executable if executable is None else executable).resolve(
        strict=True
    )
    expected_executable = (expected / "bin/python").resolve(strict=True)
    if observed_prefix != expected or observed_executable != expected_executable:
        raise PostE21ContractError("E22b MAIN requires the exact catena-v6 prefix and interpreter")
    return {
        "ack_environment": MAIN_ACK_ENV,
        "ack_value_sha256": _hash_text(MAIN_ACK_VALUE),
        "python_prefix": str(observed_prefix),
        "python_executable": str(observed_executable),
    }


def acquire_exclusive_launcher_lock(
    *,
    artifact_root: str | Path,
) -> LauncherLockHandle:
    """Acquire the non-blocking process lock held for the entire E22b launch."""

    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise PostE21ContractError("E22b artifact root is not a directory")
    lock_dir = root / "_launcher_locks"
    lock_dir.mkdir(parents=False, exist_ok=True)
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        raise PostE21ContractError("E22b launcher-lock directory is unsafe")
    lock_path = lock_dir / "e22b_paired_seed_sharding.lock"
    if lock_path.is_symlink():
        raise PostE21ContractError("E22b launcher lock cannot be a symlink")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PostE21ContractError("E22b launcher lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PostE21ContractError(
                "Another E22b sharded launcher holds the exclusive lock"
            ) from error
        record = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "amendment_id": AMENDMENT_ID,
            "pid": os.getpid(),
            "acquired_at_utc": utc_now(),
            "path": str(lock_path.resolve()),
            "exclusive_nonblocking": True,
        }
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, serialized.encode("utf-8"))
        os.fsync(descriptor)
        return LauncherLockHandle(
            descriptor=descriptor,
            path=lock_path,
            record=record,
        )
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def validate_gpu_inventory(
    *,
    devices: Sequence[str],
    gpu_rows: Sequence[Mapping[str, Any]],
    compute_apps: Sequence[Mapping[str, Any]],
    maximum_idle_memory_mib: int = 512,
) -> dict[str, Any]:
    """Require four selected GPUs to be present, idle, and hardware-homogeneous."""

    if len(devices) != SHARD_COUNT:
        raise PostE21ContractError("E22b MAIN GPU inventory requires exactly four devices")
    indices = [int(device.split(":", 1)[1]) for device in devices]
    if len(set(indices)) != SHARD_COUNT:
        raise PostE21ContractError("E22b MAIN GPU inventory contains duplicate devices")
    by_index = {int(row["index"]): dict(row) for row in gpu_rows}
    if len(by_index) != len(gpu_rows):
        raise PostE21ContractError("nvidia-smi returned duplicate GPU indices")
    if any(index not in by_index for index in indices):
        raise PostE21ContractError("E22b selected GPU is absent from nvidia-smi inventory")
    selected = [by_index[index] for index in indices]
    signatures = {
        (
            str(row["name"]),
            int(row["memory_total_mib"]),
            str(row["compute_capability"]),
        )
        for row in selected
    }
    if len(signatures) != 1:
        raise PostE21ContractError("E22b MAIN requires four homogeneous GPUs")
    if any(int(row["memory_used_mib"]) > maximum_idle_memory_mib for row in selected):
        raise PostE21ContractError("E22b MAIN selected GPU has non-idle memory use")
    selected_uuids = {str(row["uuid"]) for row in selected}
    if len(selected_uuids) != SHARD_COUNT:
        raise PostE21ContractError("E22b MAIN GPU inventory contains duplicate UUIDs")
    busy = [dict(app) for app in compute_apps if str(app.get("gpu_uuid")) in selected_uuids]
    if busy:
        raise PostE21ContractError("E22b MAIN selected GPU has an active compute process")
    return {
        "status": "PASS",
        "idle": True,
        "homogeneous": True,
        "maximum_idle_memory_mib": maximum_idle_memory_mib,
        "selected_gpus": selected,
        "active_compute_processes": [],
    }


def validate_locked_main_preflight(
    *,
    preflight: object,
    devices: Sequence[str],
    run_dir: Path,
) -> dict[str, Any]:
    """Validate the immutable MAIN preflight record without re-querying live GPUs."""

    if not isinstance(preflight, dict):
        raise PostE21ContractError("E22b MAIN plan lacks its preflight record")
    if preflight.get("required") is not True or preflight.get("validated") is not True:
        raise PostE21ContractError("E22b MAIN preflight was not validated")
    authorization = preflight.get("authorization_runtime")
    expected_prefix = CATENA_V6_PREFIX.resolve(strict=True)
    expected_executable = (expected_prefix / "bin/python").resolve(strict=True)
    if authorization != {
        "ack_environment": MAIN_ACK_ENV,
        "ack_value_sha256": _hash_text(MAIN_ACK_VALUE),
        "python_prefix": str(expected_prefix),
        "python_executable": str(expected_executable),
    }:
        raise PostE21ContractError("E22b MAIN authorization/runtime record changed")
    lock_record = preflight.get("exclusive_launcher_lock")
    artifact_root = run_dir.parents[1].resolve(strict=True)
    expected_lock_path = (artifact_root / "_launcher_locks/e22b_paired_seed_sharding.lock").resolve(
        strict=True
    )
    if (
        not isinstance(lock_record, dict)
        or lock_record.get("schema_version") != 1
        or lock_record.get("experiment_id") != EXPERIMENT_ID
        or lock_record.get("amendment_id") != AMENDMENT_ID
        or not isinstance(lock_record.get("pid"), int)
        or int(lock_record["pid"]) <= 0
        or not isinstance(lock_record.get("acquired_at_utc"), str)
        or Path(str(lock_record.get("path"))).resolve(strict=True) != expected_lock_path
        or lock_record.get("exclusive_nonblocking") is not True
    ):
        raise PostE21ContractError("E22b MAIN exclusive launcher-lock record changed")
    gpu_inventory = preflight.get("gpu_inventory")
    if not isinstance(gpu_inventory, dict):
        raise PostE21ContractError("E22b MAIN GPU inventory record is missing")
    selected = gpu_inventory.get("selected_gpus")
    compute_apps = gpu_inventory.get("active_compute_processes")
    maximum_idle_memory_mib = gpu_inventory.get("maximum_idle_memory_mib")
    if (
        not isinstance(selected, list)
        or not isinstance(compute_apps, list)
        or not isinstance(maximum_idle_memory_mib, int)
    ):
        raise PostE21ContractError("E22b MAIN GPU inventory record is malformed")
    reconstructed = validate_gpu_inventory(
        devices=devices,
        gpu_rows=selected,
        compute_apps=compute_apps,
        maximum_idle_memory_mib=maximum_idle_memory_mib,
    )
    if reconstructed != gpu_inventory:
        raise PostE21ContractError("E22b MAIN GPU identity/provenance record changed")
    return dict(preflight)


def _nvidia_smi_rows(*, query: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PostE21ContractError(f"nvidia-smi {query} query failed: {detail}")
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "No running processes found" not in line
    ]
    return [[part.strip() for part in line.split(",")] for line in lines]


def collect_and_validate_main_gpu_state(
    *,
    devices: Sequence[str],
) -> dict[str, Any]:
    gpu_values = _nvidia_smi_rows(query="gpu=index,uuid,name,memory.total,memory.used,compute_cap")
    if any(len(row) != 6 for row in gpu_values):
        raise PostE21ContractError("nvidia-smi GPU inventory shape changed")
    gpu_rows = [
        {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_total_mib": int(row[3]),
            "memory_used_mib": int(row[4]),
            "compute_capability": row[5],
        }
        for row in gpu_values
    ]
    app_values = _nvidia_smi_rows(query="compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    if any(len(row) != 4 for row in app_values):
        raise PostE21ContractError("nvidia-smi compute-process inventory shape changed")
    compute_apps = [
        {
            "gpu_uuid": row[0],
            "pid": int(row[1]),
            "process_name": row[2],
            "used_memory_mib": int(row[3]),
        }
        for row in app_values
    ]
    return validate_gpu_inventory(
        devices=devices,
        gpu_rows=gpu_rows,
        compute_apps=compute_apps,
    )


def validate_equivalence_report(
    *,
    path: Path,
    source: Mapping[str, Any],
    amendment_sha256: str,
    base_protocol_sha256: str,
    config_sha256: str,
    selection_lock_sha256: str,
    equivalence_contract: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = _require_regular_file(path, label="E22b CPU equivalence report")
    report = read_json_object_strict(report_path)
    expected = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "PASS",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "source_fingerprint": dict(source),
        "amendment_lock_sha256": amendment_sha256,
        "base_protocol_lock_sha256": base_protocol_sha256,
        "config_sha256": config_sha256,
        "selection_lock_sha256": selection_lock_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise PostE21ContractError(f"E22b equivalence report binding mismatch: {key}")
    if set(equivalence_contract) != set(EQUIVALENCE_FIXED_FIELDS):
        raise PostE21ContractError("E22b derived equivalence field contract is malformed")
    for key in EQUIVALENCE_FIXED_FIELDS:
        if key not in report or report[key] != equivalence_contract[key]:
            raise PostE21ContractError(f"E22b equivalence report fixed-field mismatch: {key}")
    checks = report.get("checks")
    expected_checks = {key: True for key in EQUIVALENCE_CHECK_KEYS}
    if checks != expected_checks:
        raise PostE21ContractError("E22b CPU equivalence requires the exact six all-True checks")
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "status": "PASS",
        "checks": dict(checks),
    }


def copy_equivalence_report_into_run(
    *,
    validated_source: Mapping[str, Any],
    run_dir: Path,
    source: Mapping[str, Any],
    amendment_sha256: str,
    base_protocol_sha256: str,
    config_sha256: str,
    selection_lock_sha256: str,
    equivalence_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the external proof so the canonical run is provenance-self-contained."""

    source_path = _require_regular_file(
        Path(str(validated_source["path"])),
        label="E22b validated external equivalence report",
    )
    if file_sha256(source_path) != validated_source.get("sha256"):
        raise PostE21ContractError("E22b external equivalence report changed before copy")
    target = run_dir / "cpu_serial_shard_equivalence.json"
    _atomic_copy_regular_file(source=source_path, target=target)
    canonical = validate_equivalence_report(
        path=target,
        source=source,
        amendment_sha256=amendment_sha256,
        base_protocol_sha256=base_protocol_sha256,
        config_sha256=config_sha256,
        selection_lock_sha256=selection_lock_sha256,
        equivalence_contract=equivalence_contract,
    )
    if canonical["sha256"] != validated_source.get("sha256"):
        raise PostE21ContractError("E22b copied equivalence report differs from source")
    return {
        **canonical,
        "copied_into_canonical_run": True,
        "external_source_sha256": str(validated_source["sha256"]),
    }


def validate_locked_equivalence_record(
    *,
    record: Mapping[str, Any],
    run_dir: Path,
    source: Mapping[str, Any],
    amendment_sha256: str,
    base_protocol_sha256: str,
    config_sha256: str,
    selection_lock_sha256: str,
    equivalence_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical equivalence proof and every locked descriptor field."""

    canonical_path = run_dir / "cpu_serial_shard_equivalence.json"
    try:
        recorded_path = Path(str(record["path"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise PostE21ContractError(
            "E22b canonical equivalence path is missing or unsafe"
        ) from error
    if recorded_path != canonical_path.resolve(strict=True):
        raise PostE21ContractError("E22b equivalence proof is not inside its canonical run")
    validated = validate_equivalence_report(
        path=canonical_path,
        source=source,
        amendment_sha256=amendment_sha256,
        base_protocol_sha256=base_protocol_sha256,
        config_sha256=config_sha256,
        selection_lock_sha256=selection_lock_sha256,
        equivalence_contract=equivalence_contract,
    )
    for key, value in validated.items():
        if record.get(key) != value:
            raise PostE21ContractError(f"E22b canonical equivalence record changed: {key}")
    if (
        record.get("copied_into_canonical_run") is not True
        or record.get("external_source_sha256") != validated["sha256"]
        or set(record)
        != {
            "path",
            "sha256",
            "status",
            "checks",
            "copied_into_canonical_run",
            "external_source_sha256",
        }
    ):
        raise PostE21ContractError("E22b canonical equivalence provenance changed")
    return dict(record)


def _build_plan(
    *,
    run_dir: Path,
    runtime: dict[str, Any],
    contract: E22BScientificContract,
    amendment: AmendmentSnapshot,
    source: Mapping[str, Any],
    devices: Sequence[str],
    run_mode: str,
    data_sha256: str,
    data_manifest_sha256: str,
    source_lock: Mapping[str, Any],
    equivalence: Mapping[str, Any] | None,
    main_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    shards = validate_registered_shard_plan(amendment=amendment, seeds=contract.seeds)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "LOCKED_BEFORE_SHARD_EXECUTION",
        "created_at_utc": utc_now(),
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_mode": run_mode,
        "base_protocol_lock_sha256": contract.snapshot.sha256,
        "amendment_lock_sha256": amendment.sha256,
        "config_sha256": contract.snapshot.config_sha256,
        "parent_e21_lock_sha256": contract.parent.sha256,
        "selection_lock_sha256": contract.dependency["selection_lock_sha256"],
        "confirmatory_seeds": list(contract.seeds),
        "shard_count": SHARD_COUNT,
        "seed_shards": [list(shard) for shard in shards],
        "devices": list(devices),
        "methods": _method_payloads(contract.methods),
        "runtime_config": runtime,
        "runtime_config_sha256": sha256_canonical_json(runtime),
        "risk_scale": threshold_float(
            contract.parent.thresholds,
            "maximum_nontarget_degradation",
        ),
        "data_sha256": data_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "source_fingerprint": dict(source),
        "source_lock": dict(source_lock),
        "equivalence_report": None if equivalence is None else dict(equivalence),
        "main_preflight": dict(main_preflight),
        "scientific_invariants": dict(amendment.payload["scientific_invariants"]),
    }
    return plan


def _finalize_canonical_run(
    *,
    repo_root: Path,
    artifact_root: str | Path,
    run_dir: Path,
    config: Mapping[str, Any],
    contract: E22BScientificContract,
    amendment: AmendmentSnapshot,
    runtime: Mapping[str, Any],
    data_manifest_path: Path,
    data_sha256: str,
    plan_path: Path,
    worker_results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protocol_copy: Path,
    amendment_copy: Path,
) -> dict[str, Any]:
    plan = _read_plan(plan_path)
    _validate_plan_bindings(repo_root=repo_root, plan_path=plan_path, plan=plan)
    rows = aggregate["rows"]
    methods = contract.methods
    method_ids = [method.method_id for method in methods]
    seed_rows = compute_locality_seed_summaries(
        rows,
        seeds=contract.seeds,
        method_ids=method_ids,
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        stress_updates=int(runtime["evaluation"]["stress"]["updates"]),
        stress_gap_events=int(runtime["evaluation"]["stress"]["gap_events"]),
    )
    assessment = assess_locality_confirmatory(
        seed_rows,
        selected_method_id=methods[1].method_id,
        baseline_method_id=methods[0].method_id,
        required_seeds=contract.seeds,
        thresholds=contract.parent.thresholds,
        dry_run=str(config["__run_mode__"]) == "DRY_RUN",
    )
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=rows,
        seed_rows=seed_rows,
    )
    active_rows = build_active_cell_rows(rows)
    active_path = run_dir / "active_cell_metrics.jsonl"
    write_jsonl_strict(active_path, active_rows)
    selection_path = run_dir / "selection_provenance.json"
    write_json_strict(selection_path, contract.dependency)
    summary_path = run_dir / str(config["results_summary"]["filename"])
    summary_path.write_text(
        confirmatory_summary_ko(
            assessment=assessment,
            dry_run=str(config["__run_mode__"]) == "DRY_RUN",
        ),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > int(
        config["results_summary"]["maximum_lines"]
    ):
        raise PostE21ContractError("E22b sharded results summary exceeds one page")
    checkpoint_hashes = aggregate["checkpoint_hashes"]
    materialization = _materialize_canonical_checkpoints(
        run_dir=run_dir,
        checkpoint_paths=aggregate["checkpoint_paths"],
        checkpoint_hashes=checkpoint_hashes,
    )
    common = report_contract_metadata(
        run_dir=run_dir,
        snapshot=contract.snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=bool(
            assessment["status"] == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
            and assessment["supported"]
        ),
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PASS",
        "status": "PASS",
        "run_mode": str(config["__run_mode__"]),
        "run_scope": "E22B_ACTIVE_PATH_LOCALITY_CONFIRMATORY",
        **common,
        "parent_e21": {
            "lock_path": str(contract.parent.path),
            "lock_sha256": contract.parent.sha256,
            "inherited_thresholds": contract.parent.thresholds,
        },
        "phase_dependency": contract.dependency,
        "runtime_metadata": aggregate["runtime_metadata"],
        "summary": assessment,
        "claim_gate": {
            "status": assessment["status"],
            "supported": bool(assessment["supported"]),
            "allowed_claim": (
                "Selected locality objective versus paired mean retention in "
                "controlled structured-event sequences, if every gate passes."
            ),
            "forbidden_claim": (
                "E21 retrospective repair, H5, natural-language, novel-ID, "
                "LM, agent/planning, official backend, or runtime transfer."
            ),
        },
        "execution_topology": {
            "mode": "PAIRED_SEED_SHARDED",
            "performance_only": True,
            "scientific_protocol_changed": False,
            "shard_count": SHARD_COUNT,
            "devices": list(plan["devices"]),
            "seed_shards": [list(shard) for shard in registered_seed_shards(contract.seeds)],
            "canonical_serial_row_order_restored": True,
            "main_preflight": dict(plan["main_preflight"]),
            "equivalence_report": (
                None if plan["equivalence_report"] is None else dict(plan["equivalence_report"])
            ),
            "worker_results": [dict(result) for result in worker_results],
            "shard_manifests": list(aggregate["shards"]),
            "checkpoint_state_hashes": aggregate["checkpoint_state_hashes"],
            "canonical_checkpoint_materialization": materialization,
            "canonical_checkpoint_copy_mode": "atomic_independent_byte_copy",
        },
        "artifacts": {
            "protocol_lock": _descriptor(protocol_copy),
            "execution_amendment_lock": _descriptor(amendment_copy),
            "execution_plan": _descriptor(plan_path),
            "execution_plan_hash": _descriptor(plan_path.with_suffix(".sha256.json")),
            "cpu_serial_shard_equivalence": (
                None if plan["equivalence_report"] is None else dict(plan["equivalence_report"])
            ),
            "data_manifest": _descriptor(data_manifest_path),
            "rows": row_artifacts,
            "active_cell_metrics": _descriptor(active_path, rows=len(active_rows)),
            "selection_provenance": _descriptor(selection_path),
            "results_summary_ko": {
                **_descriptor(summary_path),
                "line_count": len(summary_path.read_text(encoding="utf-8").splitlines()),
            },
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=str(artifact_root),
        run_dir=run_dir,
        report=report,
    )
    return report


def coordinate_sharded_run(
    *,
    repo_root: Path,
    config_path: str | Path,
    artifact_root: str | Path,
    selection_run: str | Path,
    devices: Sequence[str],
    dry_run: bool,
    source_lock_tag: str | None,
    equivalence_report: str | Path | None,
) -> Path:
    """Create, execute, validate, and finalize one canonical E22b sharded run."""

    if dry_run:
        require_temp_dry_root(artifact_root)
    normalized_devices = _validate_devices(devices, dry_run=dry_run)
    contract = load_scientific_contract(
        repo_root=repo_root,
        config_path=config_path,
        selection_run=selection_run,
        dependency_run_mode="DRY_RUN" if dry_run else "MAIN",
    )
    amendment = validate_execution_amendment(repo_root=repo_root)
    validate_registered_shard_plan(amendment=amendment, seeds=contract.seeds)
    source = _source_record(repo_root)
    equivalence_validation_contract = build_equivalence_validation_contract(
        confirmatory_seeds=contract.seeds,
        methods=contract.methods,
        dry_runtime=runtime_locality_config(contract.config, dry_run=True),
    )
    launcher_lock: LauncherLockHandle | None = None
    run_dir: Path | None = None
    stage = "MAIN_PREFLIGHT" if not dry_run else "DRY_RUN_INITIALIZATION"
    if dry_run:
        non_evidence_record: dict[str, Any] = {
            "required": False,
            "validated": False,
            "reason": "DRY_RUN_NON_EVIDENCE",
        }
        source_lock = dict(non_evidence_record)
        main_preflight = dict(non_evidence_record)
        external_equivalence: dict[str, Any] | None = None
    else:
        authorization_runtime = validate_main_authorization_and_runtime()
        if source_lock_tag is None:
            raise PostE21ContractError("E22b sharded MAIN requires --source-lock-tag")
        if equivalence_report is None:
            raise PostE21ContractError("E22b sharded MAIN requires --equivalence-report")
        source_lock = validate_source_lock_tag(
            repo_root=repo_root,
            tag=source_lock_tag,
            base_protocol_sha256=contract.snapshot.sha256,
            amendment_sha256=amendment.sha256,
        )
        external_equivalence = validate_equivalence_report(
            path=Path(equivalence_report),
            source=source,
            amendment_sha256=amendment.sha256,
            base_protocol_sha256=contract.snapshot.sha256,
            config_sha256=contract.snapshot.config_sha256,
            selection_lock_sha256=contract.dependency["selection_lock_sha256"],
            equivalence_contract=equivalence_validation_contract,
        )
        launcher_lock = acquire_exclusive_launcher_lock(artifact_root=artifact_root)
        try:
            gpu_inventory = collect_and_validate_main_gpu_state(devices=normalized_devices)
        except BaseException:
            launcher_lock.release()
            launcher_lock = None
            raise
        main_preflight = {
            "required": True,
            "validated": True,
            "authorization_runtime": authorization_runtime,
            "exclusive_launcher_lock": dict(launcher_lock.record),
            "gpu_inventory": gpu_inventory,
        }
    try:
        stage = "CANONICAL_RUN_INITIALIZATION"
        initialized, initialized_run_dir, coordinator_device = initialize_run(
            experiment_id=EXPERIMENT_ID,
            config_path=str(config_path),
            artifact_root=str(artifact_root),
            device_request="cpu",
            run_mode="DRY_RUN" if dry_run else "MAIN",
        )
        run_dir = Path(initialized_run_dir)
        if initialized != contract.config or str(coordinator_device) != "cpu":
            raise PostE21ContractError("E22b coordinator initialization changed")
        protocol_copy = copy_protocol_snapshot(snapshot=contract.snapshot, run_dir=run_dir)
        amendment_copy = run_dir / "execution_amendment_lock.json"
        _atomic_copy_regular_file(source=amendment.path, target=amendment_copy)
        if file_sha256(amendment_copy) != amendment.sha256:
            raise PostE21ContractError("E22b amendment copy hash mismatch")
        equivalence: dict[str, Any] | None
        if dry_run:
            equivalence = None
        else:
            if external_equivalence is None:
                raise PostE21ContractError("E22b MAIN equivalence preflight disappeared")
            equivalence = copy_equivalence_report_into_run(
                validated_source=external_equivalence,
                run_dir=run_dir,
                source=source,
                amendment_sha256=amendment.sha256,
                base_protocol_sha256=contract.snapshot.sha256,
                config_sha256=contract.snapshot.config_sha256,
                selection_lock_sha256=contract.dependency["selection_lock_sha256"],
                equivalence_contract=equivalence_validation_contract,
            )
        runtime = runtime_locality_config(contract.config, dry_run=dry_run)
        data_manifest_path, data_sha256 = write_data_manifest(
            run_dir=run_dir,
            payload=build_e22b_data_payload(
                runtime=runtime,
                contract=contract,
                dry_run=dry_run,
            ),
        )
        plan = _build_plan(
            run_dir=run_dir,
            runtime=runtime,
            contract=contract,
            amendment=amendment,
            source=source,
            devices=normalized_devices,
            run_mode="DRY_RUN" if dry_run else "MAIN",
            data_sha256=data_sha256,
            data_manifest_sha256=file_sha256(data_manifest_path),
            source_lock=source_lock,
            equivalence=equivalence,
            main_preflight=main_preflight,
        )
        plan_path = run_dir / "execution_plan.json"
        write_json_strict(plan_path, plan)
        detached_plan_sha = file_sha256(plan_path)
        write_json_strict(
            plan_path.with_suffix(".sha256.json"),
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "run_id": run_dir.name,
                "sha256": detached_plan_sha,
            },
        )
        _update_run_manifest(
            run_dir / "run_manifest.json",
            {
                "execution_topology": {
                    "mode": "PAIRED_SEED_SHARDED",
                    "performance_only": True,
                    "shard_count": SHARD_COUNT,
                    "devices": list(normalized_devices),
                    "execution_plan_sha256": detached_plan_sha,
                    "execution_amendment_lock_sha256": amendment.sha256,
                    "main_preflight": dict(main_preflight),
                    "equivalence_report": (None if equivalence is None else dict(equivalence)),
                }
            },
        )
        stage = "SHARD_EXECUTION_OR_AGGREGATION"
        worker_results = _launch_workers(
            repo_root=repo_root,
            run_dir=run_dir,
            plan_path=plan_path,
            devices=normalized_devices,
        )
        aggregate = aggregate_shards(repo_root=repo_root, plan_path=plan_path)
        report_config = dict(contract.config)
        report_config["__run_mode__"] = "DRY_RUN" if dry_run else "MAIN"
        stage = "CANONICAL_FINALIZATION"
        _finalize_canonical_run(
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_dir=run_dir,
            config=report_config,
            contract=contract,
            amendment=amendment,
            runtime=runtime,
            data_manifest_path=data_manifest_path,
            data_sha256=data_sha256,
            plan_path=plan_path,
            worker_results=worker_results,
            aggregate=aggregate,
            protocol_copy=protocol_copy,
            amendment_copy=amendment_copy,
        )
    except BaseException as error:
        if run_dir is not None:
            _write_execution_failure(
                run_dir=run_dir,
                stage=stage,
                error=error,
            )
        raise
    finally:
        if launcher_lock is not None:
            launcher_lock.release()
    if run_dir is None:
        raise AssertionError("E22b coordinator completed without a canonical run")
    return run_dir


def _normalize_equivalence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.pop("examples_per_second", None)
    normalized.pop("checkpoint_sha256", None)
    return normalized


def compare_serial_and_sharded_outputs(
    *,
    serial_rows: Sequence[Mapping[str, Any]],
    sharded_rows: Sequence[Mapping[str, Any]],
    serial_checkpoint_states: Mapping[str, str],
    sharded_checkpoint_states: Mapping[str, str],
    serial_metadata: Mapping[str, Any],
    sharded_metadata: Mapping[str, Any],
) -> dict[str, bool]:
    """Compare every scientific value while excluding throughput/file-container hashes."""

    normalized_serial = [_normalize_equivalence_row(row) for row in serial_rows]
    normalized_sharded = [_normalize_equivalence_row(row) for row in sharded_rows]
    return {
        "raw_row_count_equal": len(serial_rows) == len(sharded_rows),
        "canonical_scientific_rows_exact": normalized_serial == normalized_sharded,
        "checkpoint_state_hashes_exact": (
            dict(serial_checkpoint_states) == dict(sharded_checkpoint_states)
        ),
        "identifier_codebook_hash_exact": (
            serial_metadata.get("identifier_codebook_sha256")
            == sharded_metadata.get("identifier_codebook_sha256")
        ),
        "initialization_hashes_exact": (
            serial_metadata.get("initialization_hashes")
            == sharded_metadata.get("initialization_hashes")
        ),
        "parameter_counts_exact": (
            serial_metadata.get("parameter_counts") == sharded_metadata.get("parameter_counts")
        ),
    }


def _state_hashes_for_run(
    *,
    run_dir: Path,
    checkpoint_hashes: Mapping[str, str],
) -> dict[str, str]:
    return {
        key: _checkpoint_state_hash(path)
        for key, path in _checkpoint_paths(run_dir, checkpoint_hashes).items()
    }


def run_cpu_serial_shard_equivalence(
    *,
    repo_root: Path,
    config_path: str | Path,
    selection_run: str | Path,
    output_root: Path,
    selection_is_dry_run: bool,
) -> Path:
    """Run a non-evidence CPU proof that seed decomposition preserves E22b outputs."""

    require_temp_dry_root(output_root)
    if output_root.exists():
        raise FileExistsError(f"Equivalence output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    contract = load_scientific_contract(
        repo_root=repo_root,
        config_path=config_path,
        selection_run=selection_run,
        dependency_run_mode="DRY_RUN" if selection_is_dry_run else "MAIN",
    )
    amendment = validate_execution_amendment(repo_root=repo_root)
    source = _source_record(repo_root)
    runtime = runtime_locality_config(contract.config, dry_run=True)
    equivalence_contract = build_equivalence_validation_contract(
        confirmatory_seeds=contract.seeds,
        methods=contract.methods,
        dry_runtime=runtime,
    )
    equivalence_seeds = tuple(int(seed) for seed in equivalence_contract["seeds"])
    serial_dir = output_root / "serial"
    serial_dir.mkdir()
    serial_rows, serial_hashes, serial_metadata = run_locality_method_grid(
        runtime=runtime,
        methods=contract.methods,
        seeds=equivalence_seeds,
        run_dir=serial_dir,
        device=torch.device("cpu"),
        parent_lock_sha256=contract.parent.sha256,
        protocol_lock_sha256=contract.snapshot.sha256,
        risk_scale=threshold_float(
            contract.parent.thresholds,
            "maximum_nontarget_degradation",
        ),
    )
    serial_rows = canonicalize_rows(
        serial_rows,
        runtime=runtime,
        methods=contract.methods,
        seeds=equivalence_seeds,
    )
    serial_states = _state_hashes_for_run(
        run_dir=serial_dir,
        checkpoint_hashes=serial_hashes,
    )

    shard_rows: list[dict[str, Any]] = []
    shard_states: dict[str, str] = {}
    shard_initializations: dict[str, str] = {}
    shard_parameter_counts: dict[str, int] | None = None
    shard_identifier: str | None = None
    for index, seed in enumerate(equivalence_seeds):
        shard_dir = output_root / f"shard-{index:03d}"
        shard_dir.mkdir()
        rows, hashes, metadata = run_locality_method_grid(
            runtime=runtime,
            methods=contract.methods,
            seeds=[seed],
            run_dir=shard_dir,
            device=torch.device("cpu"),
            parent_lock_sha256=contract.parent.sha256,
            protocol_lock_sha256=contract.snapshot.sha256,
            risk_scale=threshold_float(
                contract.parent.thresholds,
                "maximum_nontarget_degradation",
            ),
        )
        shard_rows.extend(rows)
        shard_states.update(_state_hashes_for_run(run_dir=shard_dir, checkpoint_hashes=hashes))
        shard_initializations.update(metadata["initialization_hashes"])
        current_counts = {
            str(key): int(value) for key, value in metadata["parameter_counts"].items()
        }
        if shard_parameter_counts is None:
            shard_parameter_counts = current_counts
        elif current_counts != shard_parameter_counts:
            raise PostE21ContractError("Equivalence shard parameter counts differ")
        current_identifier = str(metadata["identifier_codebook_sha256"])
        if shard_identifier is None:
            shard_identifier = current_identifier
        elif current_identifier != shard_identifier:
            raise PostE21ContractError("Equivalence shard codebook hashes differ")
    sharded_rows = canonicalize_rows(
        shard_rows,
        runtime=runtime,
        methods=contract.methods,
        seeds=equivalence_seeds,
    )
    sharded_metadata = {
        "identifier_codebook_sha256": shard_identifier,
        "initialization_hashes": shard_initializations,
        "parameter_counts": shard_parameter_counts,
    }
    checks = compare_serial_and_sharded_outputs(
        serial_rows=serial_rows,
        sharded_rows=sharded_rows,
        serial_checkpoint_states=serial_states,
        sharded_checkpoint_states=shard_states,
        serial_metadata=serial_metadata,
        sharded_metadata=sharded_metadata,
    )
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_mode": "CPU_SERIAL_VS_SHARD_EQUIVALENCE",
        "created_at_utc": utc_now(),
        "source_fingerprint": source,
        "amendment_lock_sha256": amendment.sha256,
        "base_protocol_lock_sha256": contract.snapshot.sha256,
        "config_sha256": contract.snapshot.config_sha256,
        "selection_lock_sha256": contract.dependency["selection_lock_sha256"],
        **equivalence_contract,
        "checks": checks,
    }
    report_path = output_root / "E22B_CPU_SERIAL_SHARD_EQUIVALENCE.json"
    write_json_strict(report_path, report)
    validate_equivalence_report(
        path=report_path,
        source=source,
        amendment_sha256=amendment.sha256,
        base_protocol_sha256=contract.snapshot.sha256,
        config_sha256=contract.snapshot.config_sha256,
        selection_lock_sha256=contract.dependency["selection_lock_sha256"],
        equivalence_contract=equivalence_contract,
    )
    return report_path
