"""Aggregate fail-closed Stage-2 scientific-input readiness receipt."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

from .construction_source import (
    CONSTRUCTION_SOURCE_FILES,
    CRITICAL_TOOL_VERSIONS,
    REQUIRED_ARTIFACT_BINDINGS,
)
from .general_corpus import load_scientific_corpus_manifest
from .tokenizer import load_scientific_tokenizer_manifest


class Stage2DataReadinessError(RuntimeError):
    """Raised when any source/tokenizer/split/replay input is not locked."""


@dataclass(frozen=True, slots=True)
class Stage2DataReadiness:
    payload: dict[str, Any]

    @property
    def readiness_sha256(self) -> str:
        return str(self.payload["readiness_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _load(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise Stage2DataReadinessError(f"{label} is not a regular file: {resolved}")
    return resolved, read_json_object_strict(resolved), sha256_file(resolved)


def _load_yaml(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise Stage2DataReadinessError(f"{label} is not a regular file: {resolved}")
    parsed = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise Stage2DataReadinessError(f"{label} must contain a YAML object")
    return resolved, parsed, sha256_file(resolved)


def _require(payload: dict[str, Any], field: str, expected: Any, label: str) -> None:
    if payload.get(field) != expected:
        raise Stage2DataReadinessError(
            f"{label}.{field} expected {expected!r}, got {payload.get(field)!r}"
        )


def _validate_construction_source(
    path: str | Path,
    *,
    artifact_paths: dict[str, Path],
) -> tuple[Path, dict[str, Any], str]:
    receipt_path, receipt, receipt_hash = _load(path, "construction_source")
    _require(
        receipt,
        "schema_version",
        "catena-e26-construction-source-v1",
        "construction_source",
    )
    _require(
        receipt,
        "construction_contract_complete",
        True,
        "construction_source",
    )
    _require(receipt, "git_clean", True, "construction_source")
    _require(
        receipt,
        "scientific_main_input_eligible",
        True,
        "construction_source",
    )
    claimed_hash = receipt.get("receipt_sha256")
    without_hash = dict(receipt)
    without_hash.pop("receipt_sha256", None)
    if claimed_hash != sha256_canonical_json(without_hash):
        raise Stage2DataReadinessError("construction source receipt hash mismatch")

    repo_root = Path(str(receipt.get("repo_root", ""))).resolve(strict=True)
    construction_head = str(receipt.get("git_head", ""))
    current_head = subprocess.run(
        ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_status = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestry = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            construction_head,
            current_head,
        ),
        check=False,
    )
    if ancestry.returncode != 0 or current_status:
        raise Stage2DataReadinessError(
            "construction commit is not an ancestor or current tree is dirty"
        )

    source_rows = receipt.get("builder_files")
    if not isinstance(source_rows, list):
        raise Stage2DataReadinessError("construction builder_files is not a list")
    observed_paths = tuple(str(row.get("path")) for row in source_rows if isinstance(row, dict))
    if observed_paths != CONSTRUCTION_SOURCE_FILES:
        raise Stage2DataReadinessError("construction builder file set/order mismatch")
    for row in source_rows:
        if not isinstance(row, dict):
            raise Stage2DataReadinessError("invalid construction builder record")
        source_path = (repo_root / str(row["path"])).resolve(strict=True)
        if source_path.stat().st_size != row.get("bytes") or sha256_file(source_path) != row.get(
            "sha256"
        ):
            raise Stage2DataReadinessError(f"construction builder changed: {row.get('path')}")
    if receipt.get("builder_source_sha256") != sha256_canonical_json(source_rows):
        raise Stage2DataReadinessError("construction builder aggregate hash mismatch")

    tool_environment = receipt.get("tool_environment")
    if not isinstance(tool_environment, dict):
        raise Stage2DataReadinessError("construction tool environment is missing")
    if tool_environment.get("critical_versions") != CRITICAL_TOOL_VERSIONS:
        raise Stage2DataReadinessError("construction critical tool versions mismatch")
    freeze = tool_environment.get("pip_freeze")
    if not isinstance(freeze, list) or tool_environment.get(
        "pip_freeze_sha256"
    ) != sha256_canonical_json(freeze):
        raise Stage2DataReadinessError("construction pip-freeze lock mismatch")

    binding_rows = receipt.get("artifact_bindings")
    if not isinstance(binding_rows, list):
        raise Stage2DataReadinessError("construction artifact bindings are missing")
    labels = tuple(str(row.get("label")) for row in binding_rows if isinstance(row, dict))
    if labels != REQUIRED_ARTIFACT_BINDINGS:
        raise Stage2DataReadinessError("construction artifact binding set/order mismatch")
    for row in binding_rows:
        if not isinstance(row, dict):
            raise Stage2DataReadinessError("invalid construction artifact record")
        label = str(row["label"])
        expected_path = artifact_paths[label].resolve(strict=True)
        recorded_path = Path(str(row["path"])).resolve(strict=True)
        if expected_path != recorded_path:
            raise Stage2DataReadinessError(f"construction artifact path mismatch for {label}")
        if recorded_path.stat().st_size != row.get("bytes") or sha256_file(
            recorded_path
        ) != row.get("sha256"):
            raise Stage2DataReadinessError(f"construction-bound artifact changed: {label}")
    if receipt.get("artifact_binding_sha256") != sha256_canonical_json(binding_rows):
        raise Stage2DataReadinessError("construction artifact aggregate hash mismatch")
    return receipt_path, receipt, receipt_hash


def validate_stage2_data_bundle(
    *,
    data_lock_path: str | Path,
    construction_receipt_path: str | Path,
    source_inventory_path: str | Path,
    source_metadata_path: str | Path,
    download_receipt_path: str | Path,
    tokenizer_manifest_path: str | Path,
    tokenizer_replay_path: str | Path,
    dedup_receipt_path: str | Path,
    near_duplicate_audit_path: str | Path,
    memmap_receipt_path: str | Path,
    transaction_manifest_path: str | Path,
    schedule_manifest_path: str | Path,
) -> Stage2DataReadiness:
    data_lock, lock_payload, lock_hash = _load_yaml(data_lock_path, "data_lock")
    _require(lock_payload, "schema_version", "catena-e26-data-lock-v1", "data_lock")
    source_path, source, source_hash = _load(source_inventory_path, "source_inventory")
    _require(source, "revision", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "source")
    expected_source_hash = source.get("inventory_sha256")
    if not isinstance(expected_source_hash, str):
        raise Stage2DataReadinessError("source inventory lacks inventory_sha256")
    source_without_hash = dict(source)
    source_without_hash.pop("inventory_sha256")
    if expected_source_hash != sha256_canonical_json(source_without_hash):
        raise Stage2DataReadinessError("source inventory canonical hash mismatch")
    metadata_path, metadata, metadata_hash = _load(source_metadata_path, "source_metadata")
    _require(metadata, "revision", source["revision"], "source_metadata")
    _require(metadata, "expected_license", "odc-by", "source_metadata")
    metadata_files = metadata.get("files")
    if not isinstance(metadata_files, list) or len(metadata_files) != 2:
        raise Stage2DataReadinessError("source metadata snapshot is incomplete")
    for row in metadata_files:
        if not isinstance(row, dict):
            raise Stage2DataReadinessError("invalid source metadata file record")
        recorded = (metadata_path.parent / str(row.get("path", ""))).resolve(strict=True)
        if recorded.stat().st_size != row.get("bytes") or sha256_file(recorded) != row.get(
            "sha256"
        ):
            raise Stage2DataReadinessError("source metadata snapshot bytes changed")

    download_path, download, download_hash = _load(download_receipt_path, "download_receipt")
    _require(download, "revision", source["revision"], "download")
    _require(download, "all_verified", True, "download")
    if download.get("inventory_sha256") != expected_source_hash:
        raise Stage2DataReadinessError("download receipt belongs to a different inventory")
    _require(
        download,
        "selection_policy",
        "INITIAL_PLUS_PREFIX_OF_MISSING_8_GRID_V1",
        "download",
    )
    _require(download, "expansion_additions", 1, "download")
    _require(download, "added_indices", [2], "download")
    _require(download, "selected_indices", [0, 2, 4, 9, 13], "download")
    amendment = download.get("capacity_amendment")
    if (
        not isinstance(amendment, dict)
        or amendment.get("trigger") != "REGISTERED_MINIMUM_TOKEN_CAPACITY_ONLY"
        or amendment.get("outcome_based_selection") is not False
        or amendment.get("prior_validation_tokens") != 4_971_104
        or amendment.get("required_validation_tokens") != 5_000_000
        or amendment.get("prior_build_disposition") != "FAILED_CAPACITY_IMMUTABLE"
    ):
        raise Stage2DataReadinessError("download capacity-expansion provenance mismatch")

    tokenizer = load_scientific_tokenizer_manifest(tokenizer_manifest_path)
    replay_path, replay, replay_hash = _load(tokenizer_replay_path, "tokenizer_replay")
    _require(replay, "artifact_hash_sets_identical", True, "tokenizer_replay")
    if replay.get("tokenizer_manifest_sha256") != tokenizer.manifest_hash:
        raise Stage2DataReadinessError("tokenizer replay receipt hash mismatch")
    tokenizer_checksum = replay.get("tokenizer_sha256_file")
    if not isinstance(tokenizer_checksum, dict):
        raise Stage2DataReadinessError("tokenizer replay lacks TOKENIZER_SHA256.txt binding")
    checksum_path = Path(str(tokenizer_checksum.get("path", ""))).resolve(strict=True)
    if (
        sha256_file(checksum_path) != tokenizer_checksum.get("sha256")
        or checksum_path.read_text(encoding="utf-8")
        != f"{tokenizer.model_sha256}  tokenizer.json\n"
    ):
        raise Stage2DataReadinessError("TOKENIZER_SHA256.txt binding changed")

    dedup_path, dedup, dedup_hash = _load(dedup_receipt_path, "dedup_receipt")
    if int(dedup.get("unique_documents", 0)) < 1:
        raise Stage2DataReadinessError("dedup receipt has no unique documents")
    for field in ("manifest_path", "sqlite_path"):
        recorded = Path(str(dedup.get(field, ""))).resolve(strict=True)
        expected = dedup.get(field.replace("_path", "_sha256"))
        if sha256_file(recorded) != expected:
            raise Stage2DataReadinessError(f"dedup {field} changed after lock")

    near_path, near, near_hash = _load(near_duplicate_audit_path, "near_duplicate_audit")
    _require(near, "pass", True, "near_duplicate_audit")
    _require(near, "flagged_pair_count", 0, "near_duplicate_audit")
    _require(
        near,
        "candidate_generation",
        "ASYMMETRIC_PROTECTED_INDEX_STREAMED_TRAIN_LSH_32X4",
        "near_duplicate_audit",
    )
    _require(near, "permutations", 128, "near_duplicate_audit")
    _require(near, "field_prime", 4_294_967_291, "near_duplicate_audit")
    _require(
        near,
        "field_reduction",
        "SHINGLE_HASH_MOD_FIELD_BEFORE_AFFINE_PERMUTATION",
        "near_duplicate_audit",
    )
    _require(near, "bands", 32, "near_duplicate_audit")
    _require(near, "rows_per_band", 4, "near_duplicate_audit")
    history = near.get("implementation_history")
    if (
        not isinstance(history, list)
        or len(history) != 1
        or not isinstance(history[0], dict)
        or history[0].get("status") != "INVALID_IMPLEMENTATION_ATTEMPT_INTERRUPTED"
        or history[0].get("scientific_input_eligible") is not False
        or not history[0].get("command")
    ):
        raise Stage2DataReadinessError(
            "near-duplicate implementation amendment history is incomplete"
        )

    memmap_path, memmaps, memmap_hash = _load(memmap_receipt_path, "memmap_receipt")
    split_rows = memmaps.get("splits")
    if not isinstance(split_rows, list) or len(split_rows) != 3:
        raise Stage2DataReadinessError("memmap receipt must contain three general splits")
    required = {
        "general_train": 400_000_000,
        "general_validation": 5_000_000,
        "general_test": 5_000_000,
    }
    corpus_records: dict[str, Any] = {}
    for row in split_rows:
        if not isinstance(row, dict):
            raise Stage2DataReadinessError("invalid memmap split receipt")
        split = row.get("split")
        if split not in required:
            raise Stage2DataReadinessError(f"unexpected memmap split {split!r}")
        manifest_path = Path(str(row.get("manifest_path", ""))).resolve(strict=True)
        if sha256_file(manifest_path) != row.get("manifest_sha256"):
            raise Stage2DataReadinessError(f"{split} manifest hash mismatch")
        manifest = load_scientific_corpus_manifest(
            manifest_path,
            tokenizer_manifest=tokenizer,
            verify_token_ids=True,
        )
        raw_manifest = read_json_object_strict(manifest_path)
        token_file = raw_manifest.get("token_file")
        if (
            not isinstance(token_file, dict)
            or token_file.get("byte_order") != "little"
            or manifest.dtype != "<u2"
            or manifest.token_count < required[str(split)]
        ):
            raise Stage2DataReadinessError(f"{split} violates uint16/capacity lock")
        corpus_records[str(split)] = manifest.as_dict()
    if set(corpus_records) != set(required):
        raise Stage2DataReadinessError("general split set is incomplete")

    transaction_path, transaction, transaction_hash = _load(
        transaction_manifest_path, "transaction_manifest"
    )
    _require(transaction, "generator_version", "v8.1", "transaction")
    _require(transaction, "replay_identical", True, "transaction")
    _require(transaction, "main_test_opened", False, "transaction")
    _require(
        transaction,
        "visible_operation_gate_address_future_query_leakage",
        0,
        "transaction",
    )

    schedule_path, schedule, schedule_hash = _load(schedule_manifest_path, "schedule_manifest")
    _require(schedule, "paired_variants_identical", True, "schedule")
    _require(schedule, "resume_identical", True, "schedule")
    _require(schedule, "main_test_opened", False, "schedule")
    _require(
        schedule,
        "algorithm",
        "token_balanced_complete_example_80_20_v2",
        "schedule",
    )
    _require(schedule, "target_general_fraction", 0.8, "schedule")
    _require(schedule, "target_transaction_fraction", 0.2, "schedule")
    _require(schedule, "actual_loss_bearing_mix_valid", True, "schedule")
    _require(
        schedule,
        "mix_validation",
        "ABS_4T_MINUS_G_LE_4_TIMES_SEQUENCE_LENGTH",
        "schedule",
    )
    for probe_name in ("first_probe", "post_resume_probe"):
        probe = schedule.get(probe_name)
        if not isinstance(probe, dict):
            raise Stage2DataReadinessError(f"schedule {probe_name} is missing")
        snapshot = probe.get("cursor_snapshot")
        if not isinstance(snapshot, dict):
            raise Stage2DataReadinessError(f"schedule {probe_name} cursor snapshot is missing")
        general_tokens = snapshot.get("general_unpadded_tokens")
        transaction_tokens = snapshot.get("transaction_unpadded_tokens")
        sequence_length = snapshot.get("sequence_length")
        if (
            not isinstance(general_tokens, int)
            or not isinstance(transaction_tokens, int)
            or not isinstance(sequence_length, int)
            or general_tokens <= 0
            or transaction_tokens <= 0
            or abs(4 * transaction_tokens - general_tokens) > 4 * sequence_length
        ):
            raise Stage2DataReadinessError(
                f"schedule {probe_name} violates actual token-mix balance"
            )

    construction_path, construction, construction_hash = _validate_construction_source(
        construction_receipt_path,
        artifact_paths={
            "data_lock": data_lock,
            "source_inventory": source_path,
            "source_metadata": metadata_path,
            "download_receipt": download_path,
            "dedup_receipt": dedup_path,
            "tokenizer_manifest": Path(tokenizer_manifest_path).expanduser().resolve(strict=True),
            "tokenizer_replay": replay_path,
            "near_duplicate_audit": near_path,
            "memmap_receipt": memmap_path,
            "transaction_manifest": transaction_path,
            "schedule_manifest": schedule_path,
        },
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-scientific-data-readiness-v2",
        "manifest_type": "E26_SCIENTIFIC_DATA_READINESS_V2",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "scientific_main_input_eligible": True,
        "main_test_opened": False,
        "data_lock": {"path": str(data_lock), "sha256": lock_hash},
        "construction_source": {
            "path": str(construction_path),
            "sha256": construction_hash,
            "git_head": construction["git_head"],
            "builder_source_sha256": construction["builder_source_sha256"],
            "artifact_binding_sha256": construction["artifact_binding_sha256"],
            "pip_freeze_sha256": construction["tool_environment"]["pip_freeze_sha256"],
        },
        "source_inventory": {"path": str(source_path), "sha256": source_hash},
        "source_metadata": {"path": str(metadata_path), "sha256": metadata_hash},
        "download_receipt": {"path": str(download_path), "sha256": download_hash},
        "tokenizer": tokenizer.as_dict(),
        "tokenizer_manifest_sha256": tokenizer.manifest_hash,
        "tokenizer_replay": {"path": str(replay_path), "sha256": replay_hash},
        "dedup_receipt": {"path": str(dedup_path), "sha256": dedup_hash},
        "near_duplicate_audit": {"path": str(near_path), "sha256": near_hash},
        "general_corpora": corpus_records,
        "general_memmap_receipt": {"path": str(memmap_path), "sha256": memmap_hash},
        "transaction_manifest": {
            "path": str(transaction_path),
            "sha256": transaction_hash,
        },
        "schedule_manifest": {"path": str(schedule_path), "sha256": schedule_hash},
        "paired_cursor_probe": schedule["first_probe"],
        "all_input_hashes_verified": True,
        "split_collision_count": 0,
        "exact_duplicate_cross_split_count": 0,
        "near_duplicate_flagged_pair_count": 0,
    }
    payload["readiness_sha256"] = sha256_canonical_json(payload)
    return Stage2DataReadiness(payload)
