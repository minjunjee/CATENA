"""Composite final-data lock for the non-evidence E26 Stage-3C preflight."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from copy import deepcopy
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
from .data_readiness_v3 import validate_zero_tolerance_data_bundle


class Stage3CDataLockError(RuntimeError):
    """Raised when final repaired data cannot be bound to the inherited V1 contract."""


_SCHEMA_VERSION = "catena-e26-data-lock-v3-final-preflight"
_REPAIR_POLICY = "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS"
_ZERO_FLAGS = "ZERO_PROTECTED_TRAIN_FLAGS"


def _git(worktree: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=worktree,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise Stage3CDataLockError(
            f"Git command failed in Stage-3C worktree: git {' '.join(arguments)}"
        ) from error


def _git_is_ancestor(worktree: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=worktree,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise Stage3CDataLockError(
            f"Cannot verify Stage-3C source ancestry: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise Stage3CDataLockError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise Stage3CDataLockError(f"{label} must be a regular non-symlink file: {resolved}")
    return resolved


def _yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage3CDataLockError(f"{label} must contain a YAML mapping")
    return payload


def _bound(path: str | Path, label: str = "input") -> dict[str, str]:
    resolved = _regular_file(path, label)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _binding(mapping: Mapping[str, Any], key: str, label: str) -> dict[str, str]:
    record = mapping.get(key)
    if not isinstance(record, Mapping):
        raise Stage3CDataLockError(f"{label} binding is missing")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise Stage3CDataLockError(f"{label} binding is incomplete")
    observed = _bound(path, label)
    if observed["sha256"] != digest:
        raise Stage3CDataLockError(f"{label} binding SHA changed")
    return observed


def _require_parent_binding(
    *,
    parent_path: Path,
    repair_protocol: Mapping[str, Any],
) -> dict[str, str]:
    original_inputs = repair_protocol.get("original_inputs")
    if not isinstance(original_inputs, Mapping):
        raise Stage3CDataLockError("Repair protocol lacks original_inputs")
    frozen_parent = _binding(original_inputs, "data_lock", "frozen V1 data lock")
    supplied_parent = _bound(parent_path, "supplied V1 data lock")
    if supplied_parent != frozen_parent:
        raise Stage3CDataLockError(
            "Stage-3C parent must be the exact V1 data lock bound by the repair protocol"
        )
    return frozen_parent


def _execution_snapshot(
    worktree: Path,
    recorded: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise Stage3CDataLockError("Stage-3C execution worktree must be clean")
    head = _git(worktree, "rev-parse", "HEAD")
    inventory = e26_execution_source_inventory(worktree)
    current = {
        "worktree": str(worktree),
        "mode": "NON_EVIDENCE_FINAL_PREFLIGHT_ONLY",
        "git_head": head,
        "git_status": "",
        "source_tree_sha256": inventory["source_tree_sha256"],
        "source_file_count": inventory["files"],
        "non_evidence_preflight_authorized": True,
        "scientific_e26a_authorized": False,
        "e26b_or_later_authorized": False,
    }
    if recorded is None:
        return current
    snapshot = dict(recorded)
    required_exact = (
        "worktree",
        "mode",
        "git_status",
        "source_tree_sha256",
        "source_file_count",
        "non_evidence_preflight_authorized",
        "scientific_e26a_authorized",
        "e26b_or_later_authorized",
    )
    if any(snapshot.get(field) != current[field] for field in required_exact):
        raise Stage3CDataLockError("Stage-3C execution source snapshot changed")
    recorded_head = snapshot.get("git_head")
    if not isinstance(recorded_head, str) or not _git_is_ancestor(worktree, recorded_head, head):
        raise Stage3CDataLockError(
            "Recorded Stage-3C commit is not an ancestor of the current source"
        )
    if set(snapshot) != set(current):
        raise Stage3CDataLockError("Stage-3C execution snapshot fields changed")
    return snapshot


def build_stage3c_data_lock(
    *,
    parent_data_lock_path: str | Path,
    repair_protocol_path: str | Path,
    repair_receipt_path: str | Path,
    repair_source_receipt_path: str | Path,
    readiness_path: str | Path,
    expected_readiness_sha256: str,
    stage3c_worktree: str | Path,
    _recorded_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inherit the exact V1 contract and add only final-data/preflight bindings."""

    parent_path = _regular_file(parent_data_lock_path, "parent data lock")
    parent = _yaml_mapping(parent_path, "parent data lock")
    if parent.get("schema_version") != "catena-e26-data-lock-v1":
        raise Stage3CDataLockError("Stage-3C parent must be the frozen E26 V1 data lock")
    for field in ("repository", "source", "content_partition", "tokenizer", "memmaps"):
        if not isinstance(parent.get(field), dict):
            raise Stage3CDataLockError(f"Parent data lock lacks {field}")
    for field in ("transaction", "tooling", "resource_policy", "stop_policy"):
        if not isinstance(parent.get(field), dict):
            raise Stage3CDataLockError(f"Parent data lock lacks {field}")

    protocol_path = _regular_file(repair_protocol_path, "repair protocol")
    repair_protocol = _yaml_mapping(protocol_path, "repair protocol")
    if repair_protocol.get("schema_version") != "catena-e26-data-lock-v2-zero-tolerance":
        raise Stage3CDataLockError("Stage-3C requires the zero-tolerance repair protocol")
    frozen_parent_binding = _require_parent_binding(
        parent_path=parent_path,
        repair_protocol=repair_protocol,
    )

    readiness_file = _regular_file(readiness_path, "scientific readiness-v3")
    if sha256_file(readiness_file) != expected_readiness_sha256:
        raise Stage3CDataLockError("Final readiness file SHA differs from user-approved input")
    observed_readiness = read_json_object_strict(readiness_file)
    recomputed = validate_zero_tolerance_data_bundle(
        data_lock_path=protocol_path,
        repair_receipt_path=repair_receipt_path,
        source_receipt_path=repair_source_receipt_path,
    ).as_dict()
    if observed_readiness != recomputed:
        raise Stage3CDataLockError("Final readiness differs from independent reconstruction")
    if (
        observed_readiness.get("scientific_main_input_eligible") is not True
        or observed_readiness.get("repair_disposition") != _ZERO_FLAGS
        or observed_readiness.get("near_duplicate_flagged_pair_count") != 0
        or observed_readiness.get("human_labels_used") is not False
        or observed_readiness.get("main_test_opened") is not False
        or observed_readiness.get("gpu_preflight_started") is not False
        or observed_readiness.get("scientific_e26a_started") is not False
    ):
        raise Stage3CDataLockError("Final repaired-data readiness does not open Stage-3C")

    repair_file = _regular_file(repair_receipt_path, "zero-tolerance repair receipt")
    repair = read_json_object_strict(repair_file)
    if (
        repair.get("disposition") != _ZERO_FLAGS
        or repair.get("policy") != _REPAIR_POLICY
        or repair.get("human_labels_used") is not False
        or repair.get("gpu_preflight_started") is not False
        or repair.get("scientific_e26a_started") is not False
        or repair.get("scientific_main_started") is not False
    ):
        raise Stage3CDataLockError("Zero-tolerance repair receipt is not eligible")
    repair_internal_sha = repair.get("repair_receipt_sha256")
    if not isinstance(repair_internal_sha, str):
        raise Stage3CDataLockError("Repair receipt lacks its canonical SHA")

    protocol_binding = _bound(protocol_path, "repair protocol")
    readiness_protocol = observed_readiness.get("protocol_lock")
    if not isinstance(readiness_protocol, dict) or readiness_protocol != protocol_binding:
        raise Stage3CDataLockError("Readiness binds a different repair protocol")
    repair_binding = _bound(repair_file, "zero-tolerance repair receipt")
    if observed_readiness.get("repair_receipt") != repair_binding:
        raise Stage3CDataLockError("Readiness binds a different repair receipt")

    worktree = Path(stage3c_worktree).expanduser()
    if worktree.is_symlink():
        raise Stage3CDataLockError("Stage-3C worktree must not be a symlink")
    worktree = worktree.resolve(strict=True)
    if not worktree.is_dir():
        raise Stage3CDataLockError("Stage-3C worktree is unavailable")

    payload = deepcopy(parent)
    payload["schema_version"] = _SCHEMA_VERSION
    payload["status"] = "PROSPECTIVE_POST_REPAIR_PRE_E26A_LOCK"
    payload["scientific_evidence"] = False
    payload["evidence_tier"] = "SCIENTIFIC_INPUT_PROVENANCE"
    payload["claim_ceiling"] = "PROTOCOL_IDENTIFIABILITY_ONLY"
    payload["main_test_opened"] = False
    payload["main_test_access_count"] = 0
    payload["data_root"] = str(repair_file.parent)
    near_duplicate = payload["content_partition"]["near_duplicate"]
    near_duplicate["original_flagged_pair_policy"] = near_duplicate["flagged_pair_policy"]
    near_duplicate["prospective_repair_policy"] = _REPAIR_POLICY
    near_duplicate["final_protected_train_flag_count"] = 0
    payload["stage3c_execution"] = _execution_snapshot(
        worktree,
        recorded=_recorded_execution,
    )
    payload["final_repaired_data"] = {
        "policy": _REPAIR_POLICY,
        "human_labels_used": False,
        "parent_data_lock": frozen_parent_binding,
        "parent_payload_sha256": sha256_canonical_json(parent),
        "repair_protocol": protocol_binding,
        "repair_receipt": repair_binding,
        "repair_receipt_internal_sha256": repair_internal_sha,
        "repair_source_receipt": _bound(repair_source_receipt_path, "repair source receipt"),
        "scientific_data_readiness_v3": _bound(readiness_file, "scientific readiness-v3"),
        "readiness_internal_sha256": observed_readiness["readiness_sha256"],
        "general_memmap_receipt": dict(observed_readiness["general_memmap_receipt"]),
        "train_corpus_manifest": {
            "path": observed_readiness["general_corpora"]["general_train"]["corpus_manifest_path"],
            "sha256": observed_readiness["general_corpora"]["general_train"][
                "corpus_manifest_sha256"
            ],
        },
        "schedule_manifest": dict(observed_readiness["schedule_manifest"]),
        "transaction_manifest": dict(observed_readiness["transaction_manifest"]),
        "final_protected_train_flag_count": 0,
        "main_test_opened": False,
    }
    stop_policy = payload["stop_policy"]
    stop_policy["execute_stage3c_non_evidence_preflight"] = True
    stop_policy["execute_scientific_e26a"] = False
    stop_policy["execute_e26b"] = False
    stop_policy["execute_e26c_or_later"] = False
    payload["lock_sha256"] = sha256_canonical_json(payload)
    return payload


def _embedded_build_arguments(payload: Mapping[str, Any]) -> dict[str, Any]:
    final = payload.get("final_repaired_data")
    execution = payload.get("stage3c_execution")
    if not isinstance(final, Mapping) or not isinstance(execution, Mapping):
        raise Stage3CDataLockError("Stage-3C lock lacks repaired-data/execution bindings")
    parent = _binding(final, "parent_data_lock", "parent V1 data lock")
    protocol = _binding(final, "repair_protocol", "repair protocol")
    repair = _binding(final, "repair_receipt", "repair receipt")
    source = _binding(final, "repair_source_receipt", "repair source receipt")
    readiness = _binding(
        final,
        "scientific_data_readiness_v3",
        "scientific readiness-v3",
    )
    worktree = execution.get("worktree")
    if not isinstance(worktree, str):
        raise Stage3CDataLockError("Stage-3C execution worktree binding is missing")
    return {
        "parent_data_lock_path": parent["path"],
        "repair_protocol_path": protocol["path"],
        "repair_receipt_path": repair["path"],
        "repair_source_receipt_path": source["path"],
        "readiness_path": readiness["path"],
        "expected_readiness_sha256": readiness["sha256"],
        "stage3c_worktree": worktree,
        "_recorded_execution": execution,
    }


def validate_stage3c_data_lock(
    payload: Mapping[str, Any],
    *,
    parent_data_lock_path: str | Path | None = None,
) -> None:
    """Reconstruct the lock from its bound files and require exact equality."""

    observed = dict(payload)
    claimed = observed.pop("lock_sha256", None)
    if claimed != sha256_canonical_json(observed):
        raise Stage3CDataLockError("Stage-3C data lock canonical hash changed")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise Stage3CDataLockError("Stage-3C data lock schema changed")
    arguments = _embedded_build_arguments(payload)
    if parent_data_lock_path is not None:
        supplied = _bound(parent_data_lock_path, "supplied validation parent")
        if supplied["path"] != arguments["parent_data_lock_path"]:
            raise Stage3CDataLockError("Validator was supplied a different V1 parent")
    expected = build_stage3c_data_lock(**arguments)
    if dict(payload) != expected:
        raise Stage3CDataLockError(
            "Stage-3C data lock differs from exact inherited/repaired-data reconstruction"
        )


def write_stage3c_data_lock(path: str | Path, **kwargs: Any) -> Path:
    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Stage-3C data lock: {destination}")
    payload = build_stage3c_data_lock(**kwargs)
    write_json_strict(destination, payload)
    written = read_json_object_strict(destination.resolve(strict=True))
    validate_stage3c_data_lock(written)
    return destination.resolve(strict=True)
