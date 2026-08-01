#!/usr/bin/env python3
"""Build E26 Data Repair R1 without starting any GPU or scientific run."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.data_lock import ContentSplit, LockedDocument, SQLiteDocumentIndex
from catena.lm.data_readiness_v3 import validate_zero_tolerance_data_bundle
from catena.lm.frozen_invariance import verify_frozen_artifacts
from catena.lm.memmap_builder import MemmapInputDocument, build_general_memmap
from catena.lm.parquet_documents import iter_parquet_documents
from catena.lm.schedule_manifest import write_schedule_manifest
from catena.lm.tokenizer import ExternalScientificTokenizer
from catena.lm.zero_tolerance_repair import (
    BLOCKED_CAPACITY,
    BLOCKED_PROVENANCE,
    REQUIRED_STAGE2_BASE_COMMIT,
    ZERO_FLAGS,
    RepairCapacityError,
    RepairProvenanceError,
    additional_train_exclusions,
    build_filtered_document_index,
    build_initial_exclusion_manifest,
    build_repair_source_receipt,
    derived_data_root,
    load_initial_exclusions,
    load_repair_protocol,
    run_locked_near_duplicate_audit,
    validate_filtered_population,
    validate_original_bindings,
    verified_shard_paths,
    write_repair_round_ledger,
    write_terminal_status,
    write_zero_tolerance_audit,
)


def _documents(rows: Iterator[tuple[LockedDocument, str]]) -> Iterator[MemmapInputDocument]:
    for record, text in rows:
        yield MemmapInputDocument(
            content_sha256=record.content_sha256,
            text=text,
            source_location=f"{record.shard_path}:{record.row_group}:{record.row_index}",
        )


def _old_split_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = read_json_object_strict(path)
    rows = payload.get("splits")
    if not isinstance(rows, list):
        raise RepairProvenanceError("Original memmap receipt lacks split rows")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("split"), str):
            raise RepairProvenanceError("Original memmap receipt contains a malformed row")
        result[str(row["split"])] = row
    return result


def _document_manifest_path(corpus_manifest_path: str | Path) -> Path:
    manifest_path = Path(corpus_manifest_path).resolve(strict=True)
    manifest = read_json_object_strict(manifest_path)
    record = manifest.get("document_manifest")
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise RepairProvenanceError("Corpus manifest lacks selected-document binding")
    path = (manifest_path.parent / str(record["path"])).resolve(strict=True)
    if sha256_file(path) != record.get("sha256"):
        raise RepairProvenanceError("Selected-document manifest bytes changed")
    return path


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RepairProvenanceError("Selected-document manifest row is not an object")
            rows.append(value)
    return rows


def _selection_proof(
    *,
    old_documents: Path,
    new_documents: Path,
    exclusions: Sequence[str],
    minimum_tokens: int,
) -> dict[str, Any]:
    old_rows = _jsonl_rows(old_documents)
    new_rows = _jsonl_rows(new_documents)
    exclusion_set = set(exclusions)
    retained = [row for row in old_rows if row["content_sha256"] not in exclusion_set]
    identity_fields = ("content_sha256", "source_location", "token_count")
    retained_identity = [tuple(row[field] for field in identity_fields) for row in retained]
    new_prefix_identity = [
        tuple(row[field] for field in identity_fields) for row in new_rows[: len(retained)]
    ]
    if new_prefix_identity != retained_identity:
        raise RepairProvenanceError("V1 retained train prefix changed order or token content")
    additions = new_rows[len(retained) :]
    old_boundary = str(old_rows[-1]["content_sha256"])
    if any(str(row["content_sha256"]) <= old_boundary for row in additions):
        raise RepairProvenanceError("Backfill is not after the V1 content-hash boundary")
    if any(row["content_sha256"] in exclusion_set for row in new_rows):
        raise RepairProvenanceError("Excluded document remains in repaired train selection")
    hashes = [str(row["content_sha256"]) for row in new_rows]
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise RepairProvenanceError("Repaired train selection is not a unique SHA prefix")
    final_tokens = int(new_rows[-1]["token_end"])
    penultimate_tokens = int(new_rows[-2]["token_end"]) if len(new_rows) > 1 else 0
    if final_tokens < minimum_tokens or penultimate_tokens >= minimum_tokens:
        raise RepairProvenanceError("Whole-document stopping boundary changed")
    payload: dict[str, Any] = {
        "algorithm": "RETAIN_V1_PREFIX_THEN_NEXT_ELIGIBLE_CONTENT_SHA256",
        "minimum_tokens": minimum_tokens,
        "old_document_count": len(old_rows),
        "old_boundary_sha256": old_boundary,
        "removed_selected_document_count": len(old_rows) - len(retained),
        "retained_document_count": len(retained),
        "retained_token_count": sum(int(row["token_count"]) for row in retained),
        "backfill_document_count": len(additions),
        "backfill_first_sha256": additions[0]["content_sha256"] if additions else None,
        "backfill_last_sha256": additions[-1]["content_sha256"] if additions else None,
        "backfill_rows_sha256": sha256_canonical_json(additions),
        "final_document_count": len(new_rows),
        "final_token_count": final_tokens,
        "penultimate_token_count": penultimate_tokens,
        "whole_document_overshoot_tokens": final_tokens - minimum_tokens,
        "content_sha_order_preserved": True,
        "excluded_hashes_absent": True,
    }
    payload["selection_proof_sha256"] = sha256_canonical_json(payload)
    return payload


def _build_memmaps(
    *,
    root: Path,
    document_index: Path,
    tokenizer_manifest: Path,
    source_revision: str,
    protocol: Mapping[str, Any],
    old_receipt_path: Path,
    exclusions: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    general_root = root / "general"
    if general_root.exists():
        raise FileExistsError(general_root)
    general_root.mkdir(parents=True)
    tokenizer = ExternalScientificTokenizer.from_manifest(tokenizer_manifest)
    capacity = protocol["capacity"]
    specifications = (
        (ContentSplit.GENERAL_TRAIN, int(capacity["general_train_tokens_min"])),
        (ContentSplit.GENERAL_VALIDATION, int(capacity["general_validation_tokens_min"])),
        (ContentSplit.GENERAL_TEST, int(capacity["general_test_tokens_min"])),
    )
    receipts: list[dict[str, Any]] = []
    for split, minimum_tokens in specifications:
        with SQLiteDocumentIndex(document_index, create=False) as index:
            try:
                receipt = build_general_memmap(
                    _documents(index.texts(split)),
                    split=split.value,
                    minimum_tokens=minimum_tokens,
                    output_root=general_root / split.value,
                    tokenizer_manifest_path=tokenizer_manifest,
                    runtime_tokenizer=tokenizer,
                    source_revisions=[f"HuggingFaceFW/fineweb-edu@{source_revision}"],
                )
            except Exception as error:
                if "below required" in str(error):
                    raise RepairCapacityError(str(error)) from error
                raise
        receipts.append(receipt)
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-general-memmaps-v2-zero-tolerance",
        "scientific_evidence": False,
        "dtype": "<u2",
        "shared_across_variants": True,
        "shared_across_runs": True,
        "splits": receipts,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    receipt_path = general_root / "general_memmaps_receipt.json"
    write_json_strict(receipt_path, payload)

    old_rows = _old_split_rows(old_receipt_path)
    new_rows = {str(row["split"]): row for row in receipts}
    for protected_split_name in (
        ContentSplit.GENERAL_VALIDATION.value,
        ContentSplit.GENERAL_TEST.value,
    ):
        old = old_rows[protected_split_name]
        new = new_rows[protected_split_name]
        if (
            old.get("token_sha256") != new.get("token_sha256")
            or old.get("document_selection_sha256") != new.get("document_selection_sha256")
            or old.get("document_count") != new.get("document_count")
        ):
            raise RepairProvenanceError(f"Protected memmap changed: {protected_split_name}")

    expected = protocol["outcome_independent_diagnostic_expectations"]
    train = new_rows[ContentSplit.GENERAL_TRAIN.value]
    initial_exclusion_identity = (
        len(exclusions) == expected["exclusion_count"]
        and sha256_canonical_json(list(exclusions)) == expected["exclusion_sorted_json_sha256"]
    )
    if initial_exclusion_identity:
        locked_fields = {
            "token_count": "repaired_train_token_count",
            "document_count": "repaired_train_document_count",
            "token_sha256": "repaired_train_token_sha256",
            "document_selection_sha256": "repaired_train_selection_sha256",
        }
        for observed_field, expected_field in locked_fields.items():
            if train.get(observed_field) != expected[expected_field]:
                raise RepairProvenanceError(
                    f"Repaired train {observed_field} differs from outcome-independent diagnostic"
                )
    new_train_documents = _document_manifest_path(str(train["manifest_path"]))
    if (
        initial_exclusion_identity
        and sha256_file(new_train_documents) != expected["repaired_train_documents_sha256"]
    ):
        raise RepairProvenanceError("Repaired train document JSONL differs from diagnostic lock")
    old_train_documents = _document_manifest_path(str(old_rows["general_train"]["manifest_path"]))
    proof = _selection_proof(
        old_documents=old_train_documents,
        new_documents=new_train_documents,
        exclusions=exclusions,
        minimum_tokens=int(capacity["general_train_tokens_min"]),
    )
    return {
        "path": str(receipt_path.resolve()),
        "sha256": sha256_file(receipt_path),
        "payload": payload,
    }, proof


def _preflight_frozen_evidence(protocol: Mapping[str, Any]) -> dict[str, Any]:
    lock = protocol["repository"]["frozen_artifact_lock"]
    path = Path(str(lock["path"])).resolve(strict=True)
    if sha256_file(path) != lock["sha256"]:
        raise RepairProvenanceError("Frozen E00-E25 composite lock bytes changed")
    payload = read_json_object_strict(path)
    if payload.get("file_count") != lock["file_count"]:
        raise RepairProvenanceError("Frozen E00-E25 file count changed")
    if payload.get("aggregate_sha256") != lock["aggregate_sha256"]:
        raise RepairProvenanceError("Frozen E00-E25 aggregate changed")
    base = payload["base_manifest"]
    result = verify_frozen_artifacts(
        baseline_manifest=path,
        expected_file_count=int(base["file_count"]),
        expected_aggregate_sha256=str(base["aggregate_sha256"]),
    )
    if result.get("passed") is not True:
        raise RepairProvenanceError("Frozen E00-E25 artifact verification failed")
    return result


def _write_external_report(root: Path, repair: Mapping[str, Any]) -> Path:
    stats = repair["exclusion_statistics"]
    train = repair["train_selection_proof"]
    final = repair["final"]
    pair_summary = f"{stats['flagged_pair_count']} / {stats['unique_train_exclusion_count']}"
    v1_exposure = (
        f"{stats['v1_selected_exclusion_count']} / {stats['v1_selected_exclusion_tokens']:,}"
    )
    repaired_train = f"{train['final_document_count']:,} / {train['final_token_count']:,}"
    text = f"""# E26 zero-tolerance data repair 결과

## 판정

```text
execution_status: COMPLETED
disposition: {repair["disposition"]}
scientific_data_readiness_v3: PASS
gpu_preflight_started: false
scientific_e26a_started: false
```

Stage-2 원본 `BLOCKED_DATA_SOURCE`와 `FAIL_PENDING_MANUAL_AUDIT`는 그대로
보존했다. 별도 prospective R1에서 frozen detector가 flag한 541개
`general_train` 문서를 semantic 판단 없이 모두 제외했다.

| 항목 | 결과 |
|---|---:|
| Flagged pair / unique train exclusion | {pair_summary} |
| 제외 raw UTF-8 bytes | {stats["excluded_normalized_utf8_bytes"]:,} |
| 제외 locked content tokens | {stats["excluded_locked_content_tokens"]:,} |
| V1 train-token fraction | {stats["v1_selected_exclusion_fraction"]:.8%} |
| Exclusion graph connected components | {stats["connected_component_count"]} |
| V1 train에 실제 노출된 제외 문서/token | {v1_exposure} |
| Deterministic backfill 문서 | {train["backfill_document_count"]} |
| Repaired train 문서/token | {repaired_train} |
| Final protected–train flags | 0 |

## Artifact

- Root: `{root}`
- Protocol SHA: `{repair["protocol_sha256"]}`
- Repair receipt SHA: `{repair["repair_receipt_sha256"]}`
- Final audit: `{final["near_duplicate_audit"]["path"]}`
- Readiness-v3: `{repair["readiness_path"]}`

## Claim 경계

이 결과는 E26 scientific input의 contamination gate만 연다. LM 성능,
Dual–Tied 차이, numerical/resource readiness 또는 E26a GO를 의미하지 않는다.
GPU preflight와 scientific E26a는 실행하지 않았다.
"""
    path = root / "E26_ZERO_TOLERANCE_DATA_REPAIR_REPORT_KO.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_blocked_report(
    root: Path,
    *,
    disposition: str,
    error: BaseException,
    implementation_defect: bool,
) -> Path:
    """Record a fail-closed repair outcome without altering the frozen Stage-2 result."""

    text = f"""# E26 zero-tolerance data repair 결과

## 판정

```text
execution_status: BLOCKED
disposition: {disposition}
scientific_data_readiness_v3: NOT_CREATED
gpu_preflight_started: false
scientific_e26a_started: false
implementation_defect: {str(implementation_defect).lower()}
```

Stage-2 원본 `BLOCKED_DATA_SOURCE`와 `FAIL_PENDING_MANUAL_AUDIT`는 변경하지
않았다. Prospective zero-tolerance R1은 fail-closed했으며 부분 namespace를
진단 가능하도록 보존했다.

```text
error_type: {type(error).__name__}
error: {error}
```

이 결과는 repaired-data readiness, GPU preflight 또는 E26a 실행 권한을 열지
않는다.
"""
    path = root / "E26_ZERO_TOLERANCE_DATA_REPAIR_REPORT_KO.md"
    path.write_text(text, encoding="utf-8")
    return path


def _record_blocked_terminal(
    root: Path | None,
    *,
    disposition: str,
    error: BaseException,
    implementation_defect: bool,
) -> None:
    if root is None or not root.is_dir() or (root / "terminal_status.json").exists():
        return
    report = _write_blocked_report(
        root,
        disposition=disposition,
        error=error,
        implementation_defect=implementation_defect,
    )
    write_terminal_status(
        root / "terminal_status.json",
        disposition=disposition,
        detail={
            "error_type": type(error).__name__,
            "error": str(error),
            "implementation_defect": implementation_defect,
            "report": str(report),
            "report_sha256": sha256_file(report),
        },
    )


def run(config_path: str | Path, repo_root: str | Path) -> Path:
    protocol_path, protocol, protocol_sha = load_repair_protocol(config_path)
    bindings = validate_original_bindings(protocol)
    root = derived_data_root(protocol, protocol_sha)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite repair namespace: {root}")
    root.mkdir(parents=True)
    protocol_lock: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-protocol-lock-v1",
        "scientific_evidence": False,
        "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "output_root": str(root),
        "original_stage2_disposition": "BLOCKED_DATA_SOURCE",
        "original_stage2_policy": "FAIL_PENDING_MANUAL_AUDIT",
        "original_disposition_preserved": True,
        "gpu_preflight_allowed": False,
        "scientific_e26a_allowed": False,
    }
    protocol_lock["protocol_lock_sha256"] = sha256_canonical_json(protocol_lock)
    write_json_strict(root / "protocol_lock.json", protocol_lock)

    frozen = _preflight_frozen_evidence(protocol)
    source_receipt = build_repair_source_receipt(repo_root)
    if source_receipt["git_branch"] != protocol["repository"]["branch"]:
        raise RepairProvenanceError("Repair executed from the wrong Git branch")
    if protocol["repository"]["required_stage2_base_commit"] != REQUIRED_STAGE2_BASE_COMMIT:
        raise RepairProvenanceError("Protocol Stage-2 base commit changed")
    source_receipt_path = root / "repair_source_receipt.json"
    write_json_strict(source_receipt_path, source_receipt)

    initial, graph, flags = load_initial_exclusions(bindings["corrected_audit"], protocol=protocol)
    old_rows = _old_split_rows(bindings["general_memmap_receipt"])
    old_train_documents = _document_manifest_path(
        str(old_rows[ContentSplit.GENERAL_TRAIN.value]["manifest_path"])
    )
    exclusion_path = root / "initial_exclusion_manifest.json"
    exclusion = build_initial_exclusion_manifest(
        exclusions=initial,
        flags=flags,
        graph=graph,
        original_index_path=bindings["document_index"],
        original_train_documents_path=old_train_documents,
        tokenizer_manifest_path=bindings["tokenizer_manifest"],
        output_path=exclusion_path,
    )
    expected = protocol["outcome_independent_diagnostic_expectations"]
    checks = {
        "unique_train_exclusion_count": "exclusion_count",
        "exclusion_sorted_json_sha256": "exclusion_sorted_json_sha256",
        "excluded_normalized_utf8_bytes": "excluded_normalized_utf8_bytes",
        "excluded_locked_content_tokens": "excluded_locked_content_tokens",
    }
    for observed_field, expected_field in checks.items():
        if exclusion.get(observed_field) != expected[expected_field]:
            raise RepairProvenanceError(f"Initial exclusion diagnostic changed: {observed_field}")
    exposure = exclusion["v1_train_exposure"]
    if (
        exposure["selected_exclusion_count"] != expected["v1_selected_exclusion_count"]
        or exposure["selected_exclusion_token_count"] != expected["v1_selected_exclusion_tokens"]
        or exposure["selected_rows_sha256"] != expected["v1_selected_exclusion_rows_sha256"]
    ):
        raise RepairProvenanceError("V1 train exposure diagnostic changed")
    if exclusion["raw_boundary_digest_sha256"] != expected["excluded_raw_boundary_sha256"]:
        raise RepairProvenanceError("Excluded raw-byte identity changed")

    original_audit = read_json_object_strict(bindings["corrected_audit"])
    history = original_audit.get("implementation_history")
    if not isinstance(history, list):
        raise RepairProvenanceError("Original audit implementation history is missing")
    exclusions = tuple(initial)
    rounds: list[dict[str, Any]] = []
    final_index: Path | None = None
    final_dedup: Path | None = None
    final_audit_path: Path | None = None
    max_iterations = int(protocol["iteration"]["max_iterations"])
    shard_paths = verified_shard_paths(bindings["download_receipt"])
    for iteration in range(1, max_iterations + 1):
        round_root = root / "iterations" / f"iteration_{iteration:03d}"
        index_root = round_root / "document_index"
        audit_root = round_root / "near_duplicate"
        index_root.mkdir(parents=True)
        audit_root.mkdir(parents=True)
        receipt = build_filtered_document_index(
            iter_parquet_documents(shard_paths),
            exclusions=exclusions,
            sqlite_path=index_root / "documents.sqlite3",
            manifest_path=index_root / "documents.jsonl",
        )
        dedup_path = index_root / "dedup_receipt.json"
        write_json_strict(dedup_path, receipt.as_dict())
        population = validate_filtered_population(
            original_index_path=bindings["document_index"],
            filtered_index_path=receipt.sqlite_path,
            exclusions=exclusions,
        )
        detector_flags = run_locked_near_duplicate_audit(receipt.sqlite_path)
        audit_path = audit_root / "near_duplicate_audit.json"
        audit = write_zero_tolerance_audit(
            audit_path,
            detector_flags,
            original_implementation_history=[dict(row) for row in history],
        )
        ledger_path = round_root / "round_ledger.json"
        ledger = write_repair_round_ledger(
            ledger_path,
            iteration=iteration,
            exclusion_manifest_sha256=exclusion["exclusion_manifest_sha256"],
            exclusions=exclusions,
            audit_path=audit_path,
            audit=audit,
            population_validation=population,
        )
        rounds.append(
            {
                "iteration": iteration,
                "dedup_receipt": {"path": str(dedup_path), "sha256": sha256_file(dedup_path)},
                "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
                "round_ledger": {
                    "path": str(ledger_path),
                    "sha256": sha256_file(ledger_path),
                    "round_sha256": ledger["round_sha256"],
                },
            }
        )
        if not detector_flags:
            final_index = Path(receipt.sqlite_path)
            final_dedup = dedup_path
            final_audit_path = audit_path
            break
        additions = additional_train_exclusions(detector_flags)
        next_exclusions = tuple(sorted(set(exclusions).union(additions)))
        if len(next_exclusions) == len(exclusions):
            raise RepairProvenanceError("Near-duplicate repair made no monotonic progress")
        exclusions = next_exclusions
    if final_index is None or final_dedup is None or final_audit_path is None:
        raise RepairProvenanceError("Maximum monotonic exclusion iterations exhausted")

    memmap, selection_proof = _build_memmaps(
        root=root,
        document_index=final_index,
        tokenizer_manifest=bindings["tokenizer_manifest"],
        source_revision=str(protocol["source"]["revision"]),
        protocol=protocol,
        old_receipt_path=bindings["general_memmap_receipt"],
        exclusions=exclusions,
    )
    if (
        len(exclusions) == expected["exclusion_count"]
        and sha256_canonical_json(list(exclusions)) == expected["exclusion_sorted_json_sha256"]
        and selection_proof["backfill_document_count"] != expected["backfill_document_count"]
    ):
        raise RepairProvenanceError("Backfill document count differs from prospective diagnostic")

    schedule_root = root / "schedule"
    schedule_root.mkdir()
    train_row = next(
        row
        for row in memmap["payload"]["splits"]
        if row["split"] == ContentSplit.GENERAL_TRAIN.value
    )
    schedule_path = write_schedule_manifest(
        schedule_root / "paired_schedule_manifest.json",
        train_corpus_manifest=str(train_row["manifest_path"]),
        tokenizer_manifest=bindings["tokenizer_manifest"],
        seed=260_026,
        sequence_length=4_096,
        probe_tokens=1_000_000,
    )

    repair: dict[str, Any] = {
        "schema_version": "catena-e26-zero-tolerance-repair-v1",
        "manifest_type": "E26_ZERO_TOLERANCE_DATA_REPAIR_R1",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "disposition": ZERO_FLAGS,
        "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
        "human_labels_used": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "repair_source": {
            "path": str(source_receipt_path),
            "sha256": sha256_file(source_receipt_path),
            "git_head": source_receipt["git_head"],
        },
        "original_stage2": {
            "disposition": "BLOCKED_DATA_SOURCE",
            "policy": "FAIL_PENDING_MANUAL_AUDIT",
            "audit_path": str(bindings["corrected_audit"]),
            "audit_sha256": sha256_file(bindings["corrected_audit"]),
            "preserved": True,
        },
        "frozen_e00_e25": frozen,
        "exclusion_manifest": {
            "path": str(exclusion_path),
            "sha256": sha256_file(exclusion_path),
        },
        "final_exclusion_count": len(exclusions),
        "final_exclusions": list(exclusions),
        "final_exclusion_sha256": sha256_canonical_json(list(exclusions)),
        "rounds": rounds,
        "final": {
            "dedup_receipt": {"path": str(final_dedup), "sha256": sha256_file(final_dedup)},
            "near_duplicate_audit": {
                "path": str(final_audit_path),
                "sha256": sha256_file(final_audit_path),
            },
            "general_memmap_receipt": {
                "path": memmap["path"],
                "sha256": memmap["sha256"],
            },
            "paired_schedule": {
                "path": str(schedule_path),
                "sha256": sha256_file(schedule_path),
            },
        },
        "exclusion_statistics": {
            "flagged_pair_count": exclusion["flagged_pair_count"],
            "unique_train_exclusion_count": exclusion["unique_train_exclusion_count"],
            "excluded_normalized_utf8_bytes": exclusion["excluded_normalized_utf8_bytes"],
            "excluded_locked_content_tokens": exclusion["excluded_locked_content_tokens"],
            "v1_selected_exclusion_count": exposure["selected_exclusion_count"],
            "v1_selected_exclusion_tokens": exposure["selected_exclusion_token_count"],
            "v1_selected_exclusion_fraction": exposure["selected_exclusion_fraction"],
            "connected_component_count": exclusion["graph"]["component_count"],
        },
        "train_selection_proof": selection_proof,
        "gpu_preflight_started": False,
        "scientific_e26a_started": False,
        "scientific_main_started": False,
    }
    repair["repair_receipt_sha256"] = sha256_canonical_json(repair)
    repair_path = root / "zero_tolerance_repair_receipt.json"
    write_json_strict(repair_path, repair)

    readiness = validate_zero_tolerance_data_bundle(
        data_lock_path=protocol_path,
        repair_receipt_path=repair_path,
        source_receipt_path=source_receipt_path,
    )
    readiness_path = root / "scientific_data_readiness_v3.json"
    write_json_strict(readiness_path, readiness.as_dict())
    repair_with_report = dict(repair)
    repair_with_report["readiness_path"] = str(readiness_path)
    report_path = _write_external_report(root, repair_with_report)
    write_terminal_status(
        root / "terminal_status.json",
        disposition=ZERO_FLAGS,
        detail={
            "repair_receipt": str(repair_path),
            "repair_receipt_sha256": sha256_file(repair_path),
            "readiness_v3": str(readiness_path),
            "readiness_v3_sha256": sha256_file(readiness_path),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    root: Path | None = None
    try:
        _, protocol, protocol_sha = load_repair_protocol(args.config)
        root = derived_data_root(protocol, protocol_sha)
        output = run(args.config, args.repo_root)
    except RepairCapacityError as error:
        _record_blocked_terminal(
            root,
            disposition=BLOCKED_CAPACITY,
            error=error,
            implementation_defect=False,
        )
        print(f"E26 zero-tolerance repair: {BLOCKED_CAPACITY}: {error}", file=sys.stderr)
        return 2
    except RepairProvenanceError as error:
        _record_blocked_terminal(
            root,
            disposition=BLOCKED_PROVENANCE,
            error=error,
            implementation_defect=False,
        )
        print(f"E26 zero-tolerance repair: {BLOCKED_PROVENANCE}: {error}", file=sys.stderr)
        return 3
    except Exception as error:
        # An implementation failure cannot create readiness.  Record it under
        # the preregistered fail-closed provenance disposition and retain its
        # distinct implementation-defect marker for amendment handling.
        if not isinstance(error, FileExistsError):
            _record_blocked_terminal(
                root,
                disposition=BLOCKED_PROVENANCE,
                error=error,
                implementation_defect=True,
            )
        traceback.print_exc()
        return 4
    print(f"E26 zero-tolerance repair: {ZERO_FLAGS} ({output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
