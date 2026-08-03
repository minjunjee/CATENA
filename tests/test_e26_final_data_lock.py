from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm.data_lock import ContentSplit, SourceDocument, SQLiteDocumentIndex, lock_document
from catena.lm.e26_final_data_lock import (
    BoundFileExpectation,
    E26FinalDataLockError,
    E26FinalDataSourceExpectation,
    build_e26_final_data_source_receipt,
    build_e26_final_heldout_domain_receipt,
    count_e26_final_general_train_capacity,
    iter_e26_final_general_train,
    validate_e26_final_data_source_receipt,
    validate_e26_final_heldout_domain_receipt,
    write_e26_final_capacity_receipt,
    write_e26_final_data_source_receipt,
    write_e26_final_heldout_domain_receipt,
)
from catena.lm.e26_final_provenance import TOKENIZER_FILES
from catena.lm.e26_final_tokenizer import (
    TOKENIZER_BACKEND,
    TOKENIZER_RECEIPT_SCHEMA,
)


def _general_train_documents(count: int) -> list[tuple[Any, bytes]]:
    rows: list[tuple[Any, bytes]] = []
    candidate = 0
    while len(rows) < count:
        document = SourceDocument(
            text=f"deterministic general training document {candidate}",
            shard_path="sample/10BT/000_00000.parquet",
            row_group=0,
            row_index=candidate,
            source_id=f"id-{candidate}",
            source_url=f"https://example.invalid/{candidate}",
        )
        record, raw = lock_document(document)
        if record.split == ContentSplit.GENERAL_TRAIN.value:
            rows.append((record, raw))
        candidate += 1
    return rows


def _with_internal_sha(payload: dict[str, Any], field: str) -> dict[str, Any]:
    output = dict(payload)
    output[field] = sha256_canonical_json(output)
    return output


def _source_fixture(
    tmp_path: Path,
) -> tuple[E26FinalDataSourceExpectation, str, int]:
    rows = _general_train_documents(4)
    sqlite_path = tmp_path / "documents.sqlite3"
    with SQLiteDocumentIndex(sqlite_path, create=True) as index:
        for record, raw in rows:
            assert index.add(record, raw)
        index.commit()
    sqlite_sha = sha256_file(sqlite_path)
    split_counts = {ContentSplit.GENERAL_TRAIN.value: len(rows)}
    split_bytes = {
        ContentSplit.GENERAL_TRAIN.value: sum(record.normalized_utf8_bytes for record, _ in rows)
    }
    selection_sha = "a" * 64
    dedup_path = tmp_path / "dedup_receipt.json"
    write_json_strict(
        dedup_path,
        {
            "document_selection_sha256": selection_sha,
            "documents_seen": len(rows),
            "exact_duplicates": 0,
            "manifest_path": str(tmp_path / "documents.jsonl"),
            "manifest_sha256": selection_sha,
            "split_counts": split_counts,
            "sqlite_path": str(sqlite_path),
            "sqlite_sha256": sqlite_sha,
            "unique_documents": len(rows),
        },
    )
    revision = "f" * 40
    inventory_path = tmp_path / "fineweb_inventory.json"
    inventory = _with_internal_sha(
        {
            "dataset_id": "fixture/fineweb",
            "subset": "sample",
            "revision": revision,
            "all_shards": [],
        },
        "inventory_sha256",
    )
    write_json_strict(inventory_path, inventory)
    download_path = tmp_path / "fineweb_download_receipt.json"
    download = _with_internal_sha(
        {
            "dataset_id": "fixture/fineweb",
            "subset": "sample",
            "revision": revision,
            "selected_indices": [0],
            "all_verified": True,
            "shards": [{"verified": True}],
        },
        "receipt_sha256",
    )
    write_json_strict(download_path, download)

    excluded_record, _excluded_raw = rows[1]
    exclusion_path = tmp_path / "initial_exclusion_manifest.json"
    exclusion = _with_internal_sha(
        {
            "schema_version": "catena-e26-zero-tolerance-exclusions-v1",
            "manifest_type": "E26_ZERO_TOLERANCE_EXCLUSION_MANIFEST",
            "scientific_evidence": False,
            "policy": "EXCLUDE_ALL_FLAGGED_TRAIN_DOCUMENTS",
            "outcome_independent": True,
            "human_labels_used": False,
            "flagged_pair_count": 1,
            "unique_train_exclusion_count": 1,
            "exclusions": [
                {
                    "content_sha256": excluded_record.content_sha256,
                    "normalized_utf8_bytes": excluded_record.normalized_utf8_bytes,
                    "locked_content_tokens": 7,
                    "split": ContentSplit.GENERAL_TRAIN.value,
                }
            ],
            "exclusion_sorted_json_sha256": sha256_canonical_json([excluded_record.content_sha256]),
            "excluded_normalized_utf8_bytes": excluded_record.normalized_utf8_bytes,
            "excluded_locked_content_tokens": 7,
            "excluded_locked_tokens_with_one_separator_per_document": 8,
        },
        "exclusion_manifest_sha256",
    )
    write_json_strict(exclusion_path, exclusion)
    expectation = E26FinalDataSourceExpectation(
        sqlite=BoundFileExpectation(str(sqlite_path), sqlite_sha),
        dedup_receipt=BoundFileExpectation(str(dedup_path), sha256_file(dedup_path)),
        source_inventory=BoundFileExpectation(str(inventory_path), sha256_file(inventory_path)),
        download_receipt=BoundFileExpectation(str(download_path), sha256_file(download_path)),
        exclusion_manifest=BoundFileExpectation(str(exclusion_path), sha256_file(exclusion_path)),
        source_revision=revision,
        dataset_id="fixture/fineweb",
        subset="sample",
        document_selection_sha256=selection_sha,
        expected_unique_documents=len(rows),
        expected_documents_seen=len(rows),
        expected_exact_duplicates=0,
        expected_exclusion_count=1,
        expected_excluded_utf8_bytes=excluded_record.normalized_utf8_bytes,
        expected_selected_shard_indices=(0,),
        expected_split_counts=tuple(split_counts.items()),
        expected_split_utf8_bytes=tuple(split_bytes.items()),
    )
    return expectation, excluded_record.content_sha256, len(rows) - 1


def _synthetic_tokenizer_receipt(root: Path) -> dict[str, Any]:
    root.mkdir()
    payload: dict[str, Any] = {
        "schema_version": TOKENIZER_RECEIPT_SCHEMA,
        "manifest_type": "E26_FINAL_32K_TOKENIZER_LOCK",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "TOKENIZER_IDENTITY_AND_BOUNDARY_POLICY_ONLY",
        "repo_id": "TinyLlama/TinyLlama_v1.1",
        "revision": "ff3c701f2424c7625fdefb9dd470f45ef18b02d6",
        "bundle_root": str(root),
        "runtime_backend": TOKENIZER_BACKEND,
        "runtime_primary_file": "tokenizer.model",
        "vocab_size": 32_000,
        "special_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 2},
        "pad_policy": "SET_PAD_TO_EOS_ID_2_AT_RUNTIME",
        "document_separator_policy": "EOS_AFTER_EACH_DOCUMENT",
        "document_separator_id": 2,
        "files": {row.filename: {} for row in TOKENIZER_FILES},
        "hard_checks": {"fixture": True},
        "scientific_main_started": False,
        "passed": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


class _WordTokenizer:
    @staticmethod
    def encode(
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        assert add_bos is False
        assert add_eos is False
        return [100 + index for index, _word in enumerate(text.split())]

    @staticmethod
    def encode_batch(
        texts: list[str],
        *,
        num_threads: int,
    ) -> list[list[int]]:
        assert num_threads > 0
        return [[100 + index for index, _word in enumerate(text.split())] for text in texts]


def test_source_lock_iteration_and_32k_capacity_are_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    expectation, excluded_sha, eligible_count = _source_fixture(tmp_path)
    receipt = build_e26_final_data_source_receipt(expectation=expectation)
    sqlite_path = Path(expectation.sqlite.path)
    sqlite_sha_before = sha256_file(sqlite_path)

    first = list(iter_e26_final_general_train(receipt, expectation=expectation))
    second = list(iter_e26_final_general_train(receipt, expectation=expectation))
    assert first == second
    assert len(first) == eligible_count
    assert [row.content_sha256 for row in first] == sorted(row.content_sha256 for row in first)
    assert excluded_sha not in {row.content_sha256 for row in first}
    assert sha256_file(sqlite_path) == sqlite_sha_before

    tokenizer_receipt = _synthetic_tokenizer_receipt(tmp_path / "tokenizer-lock-root")
    expected_content_tokens = sum(len(row.text.split()) for row in first)
    capacity = count_e26_final_general_train_capacity(
        source_receipt=receipt,
        tokenizer_receipt=tokenizer_receipt,
        tokenizer=_WordTokenizer(),
        minimum_required_tokens=expected_content_tokens + eligible_count,
        source_expectation=expectation,
        verify_bound_inputs=False,
    )
    assert capacity["disposition"] == "PASS"
    assert capacity["content_token_count"] == expected_content_tokens
    assert capacity["separator_token_count"] == eligible_count
    assert capacity["total_token_count"] == expected_content_tokens + eligible_count
    assert capacity["document_separator_id"] == 2


def test_source_and_capacity_receipts_refuse_overwrite_and_tampering(tmp_path: Path) -> None:
    expectation, _excluded_sha, _eligible_count = _source_fixture(tmp_path)
    receipt = build_e26_final_data_source_receipt(expectation=expectation)
    source_output = tmp_path / "source-lock.json"
    write_e26_final_data_source_receipt(source_output, receipt, expectation=expectation)
    assert read_json_object_strict(source_output) == receipt
    with pytest.raises(FileExistsError):
        write_e26_final_data_source_receipt(source_output, receipt, expectation=expectation)

    tampered = dict(receipt)
    tampered["train_selection_order"] = "ROWID"
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = sha256_canonical_json(tampered)
    with pytest.raises(E26FinalDataLockError, match="identity changed"):
        validate_e26_final_data_source_receipt(
            tampered, expectation=expectation, verify_bound_inputs=False
        )

    tokenizer_receipt = _synthetic_tokenizer_receipt(tmp_path / "tokenizer-lock-root")
    capacity = count_e26_final_general_train_capacity(
        source_receipt=receipt,
        tokenizer_receipt=tokenizer_receipt,
        tokenizer=_WordTokenizer(),
        minimum_required_tokens=10**9,
        source_expectation=expectation,
        verify_bound_inputs=False,
    )
    assert capacity["disposition"] == "BLOCKED_SOURCE_CAPACITY"
    capacity_output = tmp_path / "capacity.json"
    write_e26_final_capacity_receipt(capacity_output, capacity)
    with pytest.raises(FileExistsError):
        write_e26_final_capacity_receipt(capacity_output, capacity)


def test_source_lock_fails_closed_when_a_bound_input_changes(tmp_path: Path) -> None:
    expectation, _excluded_sha, _eligible_count = _source_fixture(tmp_path)
    receipt = build_e26_final_data_source_receipt(expectation=expectation)
    inventory = Path(expectation.source_inventory.path)
    inventory.write_bytes(inventory.read_bytes() + b"\n")
    with pytest.raises(E26FinalDataLockError, match="SHA-256 changed"):
        validate_e26_final_data_source_receipt(receipt, expectation=expectation)


def test_heldout_domain_is_outcome_independent_lexicographic_compliance(
    tmp_path: Path,
) -> None:
    receipt = build_e26_final_heldout_domain_receipt()
    assert receipt["candidate_domains"] == ["compliance", "inventory", "logistics"]
    assert receipt["selection_algorithm"] == "LEXICOGRAPHIC_FIRST_V1"
    assert receipt["selected_domain"] == "compliance"
    assert receipt["model_outcomes_accessed"] is False
    assert receipt["evaluation_outcomes_accessed"] is False
    assert receipt["main_test_opened"] is False
    assert validate_e26_final_heldout_domain_receipt(receipt) == receipt

    output = tmp_path / "heldout-domain-lock.json"
    write_e26_final_heldout_domain_receipt(output, receipt)
    with pytest.raises(FileExistsError):
        write_e26_final_heldout_domain_receipt(output, receipt)

    tampered = dict(receipt)
    tampered["selected_domain"] = "inventory"
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = sha256_canonical_json(tampered)
    with pytest.raises(E26FinalDataLockError, match="selected_domain"):
        validate_e26_final_heldout_domain_receipt(tampered)
