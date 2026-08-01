"""Fail-closed readiness validation for E26 zero-tolerance Data Repair R1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

from .data_lock import ContentSplit, SQLiteDocumentIndex
from .frozen_invariance import verify_frozen_artifacts
from .general_corpus import load_scientific_corpus_manifest
from .tokenizer import load_scientific_tokenizer_manifest
from .zero_tolerance_repair import (
    ZERO_FLAGS,
    RepairProvenanceError,
    _expected_detector_contract,
    _validate_detector_payload,
    load_repair_protocol,
    validate_original_bindings,
    validate_repair_source_receipt,
)


class DataReadinessV3Error(RuntimeError):
    """Raised when repaired scientific inputs violate the R1 protocol."""


@dataclass(frozen=True, slots=True)
class ScientificDataReadinessV3:
    payload: dict[str, Any]

    @property
    def readiness_sha256(self) -> str:
        return str(self.payload["readiness_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _read_bound(record: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any], str]:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise DataReadinessV3Error(f"{label} binding is incomplete")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or sha256_file(path) != expected:
        raise DataReadinessV3Error(f"{label} hash binding failed")
    return path, read_json_object_strict(path), expected


def _validate_internal_hash(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> None:
    without_hash = dict(payload)
    claimed = without_hash.pop(field, None)
    if claimed != sha256_canonical_json(without_hash):
        raise DataReadinessV3Error(f"{label} internal hash mismatch")


def _split_rows(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = receipt.get("splits")
    if not isinstance(rows, list):
        raise DataReadinessV3Error("General memmap receipt lacks split rows")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("split"), str):
            raise DataReadinessV3Error("Malformed general memmap split row")
        result[str(row["split"])] = row
    required = {
        ContentSplit.GENERAL_TRAIN.value,
        ContentSplit.GENERAL_VALIDATION.value,
        ContentSplit.GENERAL_TEST.value,
    }
    if set(result) != required:
        raise DataReadinessV3Error("Repaired general memmap split set changed")
    return result


def _selected_hashes(document_manifest: Path) -> tuple[str, ...]:
    import json

    rows: list[str] = []
    with document_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("content_sha256"), str):
                raise DataReadinessV3Error(f"Malformed selected-document row at line {line_number}")
            rows.append(str(row["content_sha256"]))
    return tuple(rows)


def validate_zero_tolerance_data_bundle(
    *,
    data_lock_path: str | Path,
    repair_receipt_path: str | Path,
    source_receipt_path: str | Path,
) -> ScientificDataReadinessV3:
    protocol_path, protocol, protocol_sha = load_repair_protocol(data_lock_path)
    original = validate_original_bindings(protocol)

    repair_path = Path(repair_receipt_path).expanduser().resolve(strict=True)
    repair = read_json_object_strict(repair_path)
    if repair.get("schema_version") != "catena-e26-zero-tolerance-repair-v1":
        raise DataReadinessV3Error("Repair receipt schema changed")
    if repair.get("disposition") != ZERO_FLAGS:
        raise DataReadinessV3Error("Only zero-flag repair may produce readiness-v3")
    if repair.get("protocol_sha256") != protocol_sha:
        raise DataReadinessV3Error("Repair receipt belongs to another protocol")
    if repair.get("scientific_evidence") is not False:
        raise DataReadinessV3Error("Data repair receipt cannot be scientific evidence")
    if repair.get("policy") != "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS":
        raise DataReadinessV3Error("Repair receipt policy changed")
    if repair.get("human_labels_used") is not False:
        raise DataReadinessV3Error("Human/model labels entered the zero-tolerance repair")
    original_stage2 = repair.get("original_stage2")
    if (
        not isinstance(original_stage2, Mapping)
        or original_stage2.get("disposition") != "BLOCKED_DATA_SOURCE"
        or original_stage2.get("policy") != "FAIL_PENDING_MANUAL_AUDIT"
        or original_stage2.get("preserved") is not True
    ):
        raise DataReadinessV3Error("Original Stage-2 boundary was not preserved")
    if repair.get("gpu_preflight_started") is not False:
        raise DataReadinessV3Error("GPU preflight started before repaired-data readiness")
    if repair.get("scientific_e26a_started") is not False:
        raise DataReadinessV3Error("Scientific E26a started before repaired-data readiness")
    if repair.get("scientific_main_started") is not False:
        raise DataReadinessV3Error("Scientific MAIN started before repaired-data readiness")
    _validate_internal_hash(repair, "repair_receipt_sha256", "repair receipt")

    source_path = Path(source_receipt_path).expanduser().resolve(strict=True)
    source_receipt = read_json_object_strict(source_path)
    try:
        validate_repair_source_receipt(source_receipt)
    except RepairProvenanceError as error:
        raise DataReadinessV3Error(str(error)) from error
    source_binding = repair.get("repair_source")
    if (
        not isinstance(source_binding, Mapping)
        or Path(str(source_binding.get("path", ""))).resolve(strict=True) != source_path
        or source_binding.get("sha256") != sha256_file(source_path)
        or source_binding.get("git_head") != source_receipt.get("git_head")
    ):
        raise DataReadinessV3Error("Repair receipt binds a different construction source")

    artifact_lock = protocol["repository"]["frozen_artifact_lock"]
    lock_path = Path(str(artifact_lock["path"])).resolve(strict=True)
    if sha256_file(lock_path) != artifact_lock["sha256"]:
        raise DataReadinessV3Error("Frozen E00-E25 lock changed")
    completed = read_json_object_strict(lock_path)
    if (
        completed.get("file_count") != artifact_lock["file_count"]
        or completed.get("aggregate_sha256") != artifact_lock["aggregate_sha256"]
    ):
        raise DataReadinessV3Error("Frozen E00-E25 lock payload changed")
    base = completed.get("base_manifest")
    if not isinstance(base, Mapping):
        raise DataReadinessV3Error("Frozen E00-E25 lock lacks base manifest")
    frozen = verify_frozen_artifacts(
        baseline_manifest=lock_path,
        expected_file_count=int(base["file_count"]),
        expected_aggregate_sha256=str(base["aggregate_sha256"]),
    )
    if frozen.get("passed") is not True:
        raise DataReadinessV3Error("Frozen E00-E25 evidence changed")

    exclusion_path, exclusion, exclusion_hash = _read_bound(
        repair["exclusion_manifest"], "exclusion_manifest"
    )
    _validate_internal_hash(exclusion, "exclusion_manifest_sha256", "exclusion manifest")
    if exclusion.get("policy") != "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS":
        raise DataReadinessV3Error("Exclusion policy changed")
    if exclusion.get("human_labels_used") is not False:
        raise DataReadinessV3Error("Human/model semantic labels entered zero-tolerance repair")
    expected = protocol.get("outcome_independent_diagnostic_expectations")
    if not isinstance(expected, Mapping):
        raise DataReadinessV3Error("Protocol lacks diagnostic expectations")
    if exclusion.get("unique_train_exclusion_count") != expected["exclusion_count"]:
        raise DataReadinessV3Error("Initial exclusion count changed")
    if exclusion.get("exclusion_sorted_json_sha256") != expected["exclusion_sorted_json_sha256"]:
        raise DataReadinessV3Error("Initial exclusion identity changed")
    if (
        exclusion.get("excluded_normalized_utf8_bytes")
        != expected["excluded_normalized_utf8_bytes"]
    ):
        raise DataReadinessV3Error("Excluded raw-byte count changed")

    final = repair.get("final")
    if not isinstance(final, Mapping):
        raise DataReadinessV3Error("Repair receipt lacks final artifact bindings")
    dedup_path, dedup, dedup_hash = _read_bound(final["dedup_receipt"], "dedup_receipt")
    audit_path, audit, audit_hash = _read_bound(final["near_duplicate_audit"], "audit")
    memmap_path, memmaps, memmap_hash = _read_bound(
        final["general_memmap_receipt"], "general_memmap_receipt"
    )
    schedule_path, schedule, schedule_hash = _read_bound(
        final["paired_schedule"], "paired_schedule"
    )
    index_path = Path(str(dedup.get("sqlite_path", ""))).resolve(strict=True)
    manifest_path = Path(str(dedup.get("manifest_path", ""))).resolve(strict=True)
    if sha256_file(index_path) != dedup.get("sqlite_sha256") or sha256_file(
        manifest_path
    ) != dedup.get("manifest_sha256"):
        raise DataReadinessV3Error("Filtered exact-dedup artifacts changed")
    if dedup.get("exclusion_count") != repair.get("final_exclusion_count"):
        raise DataReadinessV3Error("Filtered index exclusion count changed")

    if audit.get("pass") is not True or audit.get("flagged_pair_count") != 0:
        raise DataReadinessV3Error("Final protected-train near-duplicate audit is not zero")
    if audit.get("flagged_pair_policy") != "FAIL_PENDING_MANUAL_AUDIT":
        raise DataReadinessV3Error("Registered detector output policy was relabelled")
    try:
        _validate_detector_payload(audit, _expected_detector_contract(protocol))
    except RepairProvenanceError as error:
        raise DataReadinessV3Error(str(error)) from error

    initial_exclusions = {
        str(row["content_sha256"])
        for row in exclusion.get("exclusions", ())
        if isinstance(row, Mapping)
    }
    final_exclusions_raw = repair.get("final_exclusions")
    if (
        not isinstance(final_exclusions_raw, list)
        or any(not isinstance(value, str) for value in final_exclusions_raw)
        or final_exclusions_raw != sorted(set(final_exclusions_raw))
    ):
        raise DataReadinessV3Error("Final monotonic exclusion set is malformed")
    final_exclusions = set(final_exclusions_raw)
    if not initial_exclusions.issubset(final_exclusions):
        raise DataReadinessV3Error("Final exclusions are not a monotonic superset")
    if len(final_exclusions) != repair.get("final_exclusion_count") or sha256_canonical_json(
        final_exclusions_raw
    ) != repair.get("final_exclusion_sha256"):
        raise DataReadinessV3Error("Final exclusion identity changed")
    if dedup.get("exclusion_sha256") != repair.get("final_exclusion_sha256"):
        raise DataReadinessV3Error("Filtered exact-dedup receipt binds another exclusion set")
    round_rows = repair.get("rounds")
    if not isinstance(round_rows, list) or not round_rows:
        raise DataReadinessV3Error("Repair receipt lacks monotonic round provenance")
    reconstructed = set(initial_exclusions)
    for expected_iteration, row in enumerate(round_rows, start=1):
        if not isinstance(row, Mapping) or row.get("iteration") != expected_iteration:
            raise DataReadinessV3Error("Repair round order changed")
        _, ledger, _ = _read_bound(row["round_ledger"], "round_ledger")
        _validate_internal_hash(ledger, "round_sha256", "round ledger")
        if ledger.get("input_exclusion_count") != len(reconstructed) or ledger.get(
            "input_exclusion_sha256"
        ) != sha256_canonical_json(sorted(reconstructed)):
            raise DataReadinessV3Error("Repair round input exclusions changed")
        additions = ledger.get("automatic_train_side_additions")
        if not isinstance(additions, list) or any(
            not isinstance(value, str) for value in additions
        ):
            raise DataReadinessV3Error("Repair round additions are malformed")
        reconstructed.update(additions)
    if reconstructed != final_exclusions:
        raise DataReadinessV3Error("Final exclusions do not replay from round ledgers")
    last_round = round_rows[-1]
    if last_round.get("dedup_receipt") != final.get("dedup_receipt") or last_round.get(
        "audit"
    ) != final.get("near_duplicate_audit"):
        raise DataReadinessV3Error("Final artifacts do not match the terminal repair round")
    with SQLiteDocumentIndex(index_path, create=False) as index:
        placeholders = ",".join("?" for _ in final_exclusions)
        count = index.connection.execute(
            f"SELECT COUNT(*) FROM documents WHERE content_sha256 IN ({placeholders})",
            tuple(sorted(final_exclusions)),
        ).fetchone()[0]
        if int(count) != 0:
            raise DataReadinessV3Error("Excluded train documents remain in filtered index")

    tokenizer = load_scientific_tokenizer_manifest(original["tokenizer_manifest"])
    new_rows = _split_rows(memmaps)
    old_rows = _split_rows(read_json_object_strict(original["general_memmap_receipt"]))
    corpus_records: dict[str, Any] = {}
    for split, row in new_rows.items():
        manifest_file = Path(str(row.get("manifest_path", ""))).resolve(strict=True)
        if sha256_file(manifest_file) != row.get("manifest_sha256"):
            raise DataReadinessV3Error(f"{split} manifest hash changed")
        manifest = load_scientific_corpus_manifest(
            manifest_file,
            tokenizer_manifest=tokenizer,
            verify_token_ids=True,
        )
        corpus_records[split] = manifest.as_dict()
        if split in {
            ContentSplit.GENERAL_VALIDATION.value,
            ContentSplit.GENERAL_TEST.value,
        }:
            old = old_rows[split]
            old_manifest_file = Path(str(old.get("manifest_path", ""))).resolve(strict=True)
            if sha256_file(old_manifest_file) != old.get("manifest_sha256"):
                raise DataReadinessV3Error(f"Frozen V1 protected manifest changed: {split}")
            load_scientific_corpus_manifest(
                old_manifest_file,
                tokenizer_manifest=tokenizer,
                verify_token_ids=True,
            )
            if (
                row.get("token_sha256") != old.get("token_sha256")
                or row.get("document_selection_sha256") != old.get("document_selection_sha256")
                or row.get("document_count") != old.get("document_count")
            ):
                raise DataReadinessV3Error(f"Protected memmap changed: {split}")

    train = new_rows[ContentSplit.GENERAL_TRAIN.value]
    if final_exclusions == initial_exclusions and (
        train.get("token_count") != expected["repaired_train_token_count"]
        or train.get("document_count") != expected["repaired_train_document_count"]
        or train.get("token_sha256") != expected["repaired_train_token_sha256"]
        or train.get("document_selection_sha256") != expected["repaired_train_selection_sha256"]
    ):
        raise DataReadinessV3Error("Repaired train prefix differs from prospective diagnostic")
    train_manifest_payload = read_json_object_strict(str(train["manifest_path"]))
    document_record = train_manifest_payload.get("document_manifest")
    if not isinstance(document_record, Mapping):
        raise DataReadinessV3Error("Train corpus manifest lacks document binding")
    train_document_path = Path(str(train["manifest_path"])).parent / str(document_record["path"])
    selected = _selected_hashes(train_document_path.resolve(strict=True))
    if len(selected) != len(set(selected)) or final_exclusions.intersection(selected):
        raise DataReadinessV3Error("Repaired train selection contains excluded/duplicate hashes")
    if (
        final_exclusions == initial_exclusions
        and sha256_file(train_document_path) != expected["repaired_train_documents_sha256"]
    ):
        raise DataReadinessV3Error("Repaired train document manifest differs from lock")

    if schedule.get("train_corpus_manifest_sha256") != train.get("manifest_sha256"):
        raise DataReadinessV3Error("Paired schedule is not bound to repaired train corpus")
    for field in (
        "paired_variants_identical",
        "resume_identical",
        "actual_loss_bearing_mix_valid",
    ):
        if schedule.get(field) is not True:
            raise DataReadinessV3Error(f"Paired schedule failed {field}")
    if schedule.get("main_test_opened") is not False:
        raise DataReadinessV3Error("Main test was opened during data repair")

    payload: dict[str, Any] = {
        "schema_version": "catena-e26-scientific-data-readiness-v3",
        "manifest_type": "E26_SCIENTIFIC_DATA_READINESS_V3",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "scientific_main_input_eligible": True,
        "main_test_opened": False,
        "repair_disposition": ZERO_FLAGS,
        "original_stage2_disposition": "BLOCKED_DATA_SOURCE",
        "original_stage2_disposition_preserved": True,
        "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
        "human_labels_used": False,
        "protocol_lock": {"path": str(protocol_path), "sha256": protocol_sha},
        "repair_receipt": {"path": str(repair_path), "sha256": sha256_file(repair_path)},
        "repair_source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "git_head": source_receipt["git_head"],
            "builder_source_sha256": source_receipt["builder_source_sha256"],
        },
        "frozen_e00_e25": frozen,
        "exclusion_manifest": {"path": str(exclusion_path), "sha256": exclusion_hash},
        "dedup_receipt": {"path": str(dedup_path), "sha256": dedup_hash},
        "near_duplicate_audit": {"path": str(audit_path), "sha256": audit_hash},
        "near_duplicate_flagged_pair_count": 0,
        "general_memmap_receipt": {"path": str(memmap_path), "sha256": memmap_hash},
        "general_corpora": corpus_records,
        "transaction_manifest": {
            "path": str(original["transaction_manifest"]),
            "sha256": sha256_file(original["transaction_manifest"]),
        },
        "schedule_manifest": {"path": str(schedule_path), "sha256": schedule_hash},
        "all_input_hashes_verified": True,
        "split_collision_count": 0,
        "exact_duplicate_cross_split_count": 0,
        "gpu_preflight_started": False,
        "scientific_e26a_started": False,
    }
    payload["readiness_sha256"] = sha256_canonical_json(payload)
    return ScientificDataReadinessV3(payload)
