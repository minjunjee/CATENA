from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

from .audit_contract import e26_execution_source_inventory
from .backend_lock import validate_backend_candidate_lock
from .e26a_population_lock import write_e26a_validation_population_lock
from .frozen_invariance import (
    FrozenInvarianceError,
    validate_frozen_invariance_receipt,
)


class Stage2ProtocolLockError(RuntimeError):
    """Raised when an immutable E26a Stage-2 lock cannot be constructed."""


@dataclass(frozen=True, slots=True)
class Stage2ProtocolInputs:
    config: Path
    calibration_config: Path
    backend_candidate_lock: Path
    tokenizer_manifest: Path
    corpus_manifest: Path
    data_lock: Path
    data_readiness: Path
    transaction_manifest: Path
    schedule_manifest: Path
    frozen_tree_receipt: Path

    def resolved(self) -> Stage2ProtocolInputs:
        values: dict[str, Path] = {}
        for field in fields(self):
            name = field.name
            raw_path = getattr(self, name)
            path = Path(raw_path).expanduser().resolve(strict=True)
            if not path.is_file() or path.is_symlink():
                raise Stage2ProtocolLockError(f"{name} must be a regular non-symlink file: {path}")
            values[name] = path
        return Stage2ProtocolInputs(**values)

    def paths(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}

    def hashes(self) -> dict[str, str]:
        return {
            f"{field.name}_sha256": sha256_file(getattr(self, field.name)) for field in fields(self)
        }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise Stage2ProtocolLockError(f"Git command failed: git {' '.join(args)}") from error


def _canonical_lock_utc(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise Stage2ProtocolLockError("--lock-utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage2ProtocolLockError("--lock-utc must include a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2ProtocolLockError(f"{label} must be a YAML mapping: {path}")
    return payload


def _registered_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "matching",
        "backend_gates",
        "data",
        "gate_population",
        "floor_gate",
        "throughput",
        "candidate_selection",
        "claim_gates",
        "mechanism_gates",
        "safety",
    )
    return {key: config[key] for key in keys if key in config}


def build_stage2_protocol_lock(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    lock_utc: str,
    inputs: Stage2ProtocolInputs,
) -> dict[str, Any]:
    """Write the acyclic source inventory and prospective E26a protocol lock.

    The output directory must be outside the repository. This makes the source
    inventory independent of the files it creates. The protocol binds only
    upstream execution inputs; its own hash and the numerical/restart receipts
    are bound later by those downstream audit receipts.
    """

    repo = Path(repo_root).expanduser().resolve(strict=True)
    if not repo.is_dir():
        raise Stage2ProtocolLockError(f"Repository root is not a directory: {repo}")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise Stage2ProtocolLockError("Stage-2 protocol lock requires a clean committed worktree")
    head = _git(repo, "rev-parse", "HEAD")
    destination = Path(output_dir).expanduser().resolve()
    try:
        destination.relative_to(repo)
    except ValueError:
        pass
    else:
        raise Stage2ProtocolLockError("Stage-2 lock output must be outside the source repository")
    if destination.exists():
        raise Stage2ProtocolLockError(
            f"Refusing to overwrite an existing Stage-2 lock directory: {destination}"
        )

    resolved_inputs = inputs.resolved()
    config = _yaml_mapping(resolved_inputs.config, "config")
    if config.get("schema_version") != "catena-v8.1":
        raise Stage2ProtocolLockError("E26a config schema_version must be catena-v8.1")
    if config.get("experiment") != "e26a_operator_data_gate":
        raise Stage2ProtocolLockError("Stage-2 lock only supports e26a_operator_data_gate")
    candidates = config.get("model_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, Mapping) for candidate in candidates)
    ):
        raise Stage2ProtocolLockError("E26a config lacks a valid candidate table")
    try:
        validate_backend_candidate_lock(
            read_json_object_strict(resolved_inputs.backend_candidate_lock),
            repo_root=repo,
            config_path=resolved_inputs.config,
            candidates=candidates,
        )
    except (OSError, ValueError) as error:
        raise Stage2ProtocolLockError(f"Backend candidate lock is invalid: {error}") from error
    data_lock = _yaml_mapping(resolved_inputs.data_lock, "data lock")
    try:
        validate_frozen_invariance_receipt(
            read_json_object_strict(resolved_inputs.frozen_tree_receipt),
            data_lock=data_lock,
        )
    except (FrozenInvarianceError, OSError, ValueError) as error:
        raise Stage2ProtocolLockError(
            f"Frozen live/source/artifact invariance receipt is invalid: {error}"
        ) from error
    inventory = e26_execution_source_inventory(repo)
    destination.mkdir(parents=True, exist_ok=False)
    validation_population_path = destination / "e26a_validation_population_lock.json"
    write_e26a_validation_population_lock(validation_population_path, config)
    execution_inputs = {
        "source_tree_sha256": str(inventory["source_tree_sha256"]),
        **resolved_inputs.hashes(),
        "validation_population_lock_sha256": sha256_file(validation_population_path),
    }
    canonical_utc = _canonical_lock_utc(lock_utc)
    protocol: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26A_STAGE2_PROSPECTIVE_PROTOCOL_LOCK",
        "experiment": config["experiment"],
        "stage": config["stage"],
        "locked": True,
        "lock_utc": canonical_utc,
        "git_head": head,
        "git_status": "",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_PROTOCOL_GATE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "source_hash": inventory["source_tree_sha256"],
        "config_hash": sha256_canonical_json(config),
        "primary_question": (
            "Can the matched autoregressive E26 design be identified and executed "
            "without opening the frozen main test?"
        ),
        "primary_estimand": (
            "PROTOCOL_FLOOR_HEADROOM_NUMERICAL_PARITY_AND_RESOURCE_FEASIBILITY_ONLY"
        ),
        "inference_unit": "one_paired_calibration_seed",
        "registered_dispositions": config.get("registered_dispositions", []),
        "thresholds": _registered_thresholds(config),
        "full_config_snapshot": config,
        "dependencies": config.get(
            "dependencies",
            [config.get("dependency", {})],
        ),
        "execution_inputs": dict(sorted(execution_inputs.items())),
        "execution_input_paths": {
            **resolved_inputs.paths(),
            "validation_population_lock": ("BUNDLE_RELATIVE:e26a_validation_population_lock.json"),
        },
        "hash_dag": {
            "protocol_binds": "UPSTREAM_EXECUTION_INPUTS_ONLY",
            "downstream_nodes_bind_protocol": [
                "numerical_audit",
                "restart_audit",
                "backend_preflight_manifest",
                "resource_preflight",
            ],
            "omitted_from_protocol_by_design": [
                "protocol_lock_sha256",
                "backend_manifest_sha256",
                "numerical_audit_sha256",
                "restart_audit_sha256",
                "resource_preflight_sha256",
            ],
        },
        "main_test_opened": False,
        "main_test_access_count": 0,
        "e26b_started": False,
        "e26c_started": False,
    }
    protocol["payload_sha256"] = sha256_canonical_json(protocol)

    inventory_path = destination / "e26_execution_source_inventory.json"
    protocol_path = destination / "e26a_protocol_lock.json"
    write_json_strict(inventory_path, inventory)
    write_json_strict(protocol_path, protocol)
    receipt: dict[str, Any] = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26A_STAGE2_LOCK_BUNDLE_RECEIPT",
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "passed": True,
        "lock_utc": canonical_utc,
        "git_head": head,
        "git_status": "",
        "source_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
            "source_tree_sha256": inventory["source_tree_sha256"],
            "files": inventory["files"],
        },
        "protocol_lock": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "payload_sha256": protocol["payload_sha256"],
        },
        "upstream_inputs": {
            name: {
                "path": resolved_inputs.paths()[name.removesuffix("_sha256")],
                "sha256": digest,
            }
            for name, digest in sorted(resolved_inputs.hashes().items())
        },
        "validation_population_lock": {
            "path": str(validation_population_path),
            "sha256": sha256_file(validation_population_path),
        },
        "acyclic_hash_dag": True,
        "main_test_opened": False,
        "main_test_access_count": 0,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    receipt_path = destination / "e26a_stage2_lock_bundle_receipt.json"
    write_json_strict(receipt_path, receipt)
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
    }
