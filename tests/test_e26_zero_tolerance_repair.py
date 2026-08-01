from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from catena.core.provenance_v61 import read_json_object_strict
from catena.lm.data_lock import (
    ContentSplit,
    NearDuplicateFlag,
    SourceDocument,
    audit_near_duplicates_asymmetric,
    bucket_for_sha256,
    build_document_index,
    content_digest,
    split_for_bucket,
    write_near_duplicate_audit,
)
from catena.lm.data_readiness_v3 import DataReadinessV3Error, _validate_schedule_probe
from catena.lm.zero_tolerance_repair import (
    BLOCKED_CAPACITY,
    BLOCKED_PROVENANCE,
    TERMINAL_DISPOSITIONS,
    ZERO_FLAGS,
    additional_train_exclusions,
    build_filtered_document_index,
    load_initial_exclusions,
    validate_filtered_population,
)


def _text_for_split(split: ContentSplit, prefix: str) -> str:
    for index in range(200_000):
        text = f"{prefix} {index}"
        digest = content_digest(text)[0]
        if split_for_bucket(bucket_for_sha256(digest)) is split:
            return text
    raise AssertionError(f"could not construct fixture text for {split}")


def _detector_protocol(audit: dict[str, object]) -> dict[str, object]:
    return {
        "detector": {
            key: audit[key]
            for key in (
                "algorithm",
                "normalization",
                "shingle_width_words",
                "permutations",
                "field_prime",
                "shingle_hash",
                "field_reduction",
                "bands",
                "rows_per_band",
                "seed",
                "estimated_jaccard_flag_threshold",
                "candidate_generation",
            )
        },
        "original_inputs": {
            "corrected_audit": {
                "flagged_pair_count": 3,
                "internal_audit_sha256": audit["audit_sha256"],
                "strata_counts": {
                    "general_test__general_train": 1,
                    "general_validation__general_train": 1,
                    "tokenizer_only__general_train": 1,
                },
            }
        },
    }


def test_frozen_audit_derives_only_train_side_exclusions(tmp_path: Path) -> None:
    flags = (
        NearDuplicateFlag(
            "1" * 64,
            ContentSplit.GENERAL_TEST.value,
            "a" * 64,
            ContentSplit.GENERAL_TRAIN.value,
            0.9,
        ),
        NearDuplicateFlag(
            "2" * 64,
            ContentSplit.GENERAL_VALIDATION.value,
            "b" * 64,
            ContentSplit.GENERAL_TRAIN.value,
            0.9,
        ),
        NearDuplicateFlag(
            "3" * 64,
            ContentSplit.TOKENIZER_ONLY.value,
            "c" * 64,
            ContentSplit.GENERAL_TRAIN.value,
            0.9,
        ),
    )
    path = write_near_duplicate_audit(tmp_path / "audit.json", flags)
    audit = read_json_object_strict(path)
    exclusions, graph, observed = load_initial_exclusions(path, protocol=_detector_protocol(audit))
    assert exclusions == ("a" * 64, "b" * 64, "c" * 64)
    assert len(observed) == 3
    assert graph["component_count"] == 3
    assert graph["cycle_rank"] == 0


def test_filtered_exact_dedup_is_original_minus_train_exclusion(tmp_path: Path) -> None:
    train_a = _text_for_split(ContentSplit.GENERAL_TRAIN, "train-a")
    train_b = _text_for_split(ContentSplit.GENERAL_TRAIN, "train-b")
    validation = _text_for_split(ContentSplit.GENERAL_VALIDATION, "validation")
    train_a_sha = content_digest(train_a)[0]
    documents = [
        SourceDocument(train_a, "000.parquet", 0, 0),
        SourceDocument(train_a, "000.parquet", 0, 1),
        SourceDocument(train_b, "000.parquet", 0, 2),
        SourceDocument(validation, "000.parquet", 0, 3),
    ]
    original = build_document_index(
        documents,
        sqlite_path=tmp_path / "original.sqlite3",
        manifest_path=tmp_path / "original.jsonl",
    )
    filtered = build_filtered_document_index(
        documents,
        exclusions=(train_a_sha,),
        sqlite_path=tmp_path / "filtered.sqlite3",
        manifest_path=tmp_path / "filtered.jsonl",
    )
    assert original.unique_documents == 3
    assert filtered.unique_documents == 2
    assert filtered.excluded_source_occurrences == 2
    proof = validate_filtered_population(
        original_index_path=original.sqlite_path,
        filtered_index_path=filtered.sqlite_path,
        exclusions=(train_a_sha,),
    )
    assert proof["exact_subset_relation"] is True
    assert proof["removed_train_document_count"] == 1
    assert proof["protected"][ContentSplit.GENERAL_VALIDATION.value]["byte_identical_records"]


def test_document_removal_cannot_create_new_locked_lsh_flags() -> None:
    base = "one two three four five six seven eight nine ten"
    protected = [("1" * 64, ContentSplit.GENERAL_TEST.value, base)]
    train = [
        ("a" * 64, ContentSplit.GENERAL_TRAIN.value, base + " eleven"),
        ("b" * 64, ContentSplit.GENERAL_TRAIN.value, "unrelated words only here"),
    ]
    flags = audit_near_duplicates_asymmetric(protected, train)
    assert [item.right_sha256 for item in flags] == ["a" * 64]
    additions = additional_train_exclusions(flags)
    assert additions == ("a" * 64,)
    retained = [row for row in train if row[0] not in set(additions)]
    assert audit_near_duplicates_asymmetric(protected, retained) == ()


def test_protocol_keeps_original_block_and_only_registered_terminals() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/e26_data_lock_v2_zero_tolerance.yaml").read_text(encoding="utf-8")
    )
    assert config["repair"]["original_stage2_disposition"] == "BLOCKED_DATA_SOURCE"
    assert config["repair"]["original_policy"] == "FAIL_PENDING_MANUAL_AUDIT"
    assert config["repair"]["human_adjudication_required"] is False
    assert config["source"]["train_selection_order"] == "CONTENT_SHA256_ASCENDING"
    assert config["source"]["additional_shard_download_allowed"] is False
    assert (
        config["repository"]["required_stage2_base_commit"]
        == "55975897b441891312e977ce3734c6b9d2e3c36e"
    )
    assert {ZERO_FLAGS, BLOCKED_CAPACITY, BLOCKED_PROVENANCE} == TERMINAL_DISPOSITIONS
    assert set(config["repair"]["terminal_dispositions"]) == TERMINAL_DISPOSITIONS


def test_schedule_probe_enforces_locked_80_20_tolerance() -> None:
    schedule = {
        "algorithm": "token_balanced_complete_example_80_20_v2",
        "sequence_length": 4_096,
        "target_general_fraction": 0.8,
        "target_transaction_fraction": 0.2,
        "probe": {
            "cursor_snapshot": {
                "cursor_algorithm": "token_balanced_complete_example_80_20_v2",
                "general_unpadded_tokens": 800_000,
                "transaction_unpadded_tokens": 200_000,
                "sequence_length": 4_096,
                "target_general_fraction": 0.8,
                "target_transaction_fraction": 0.2,
            }
        },
    }
    _validate_schedule_probe(schedule, "probe")
    schedule["probe"]["cursor_snapshot"]["transaction_unpadded_tokens"] = 150_000
    with pytest.raises(DataReadinessV3Error, match="80:20"):
        _validate_schedule_probe(schedule, "probe")


def test_new_readiness_and_repair_schemas_are_valid_json() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "e26_scientific_data_readiness_v3.schema.json",
        "e26_zero_tolerance_repair_receipt.schema.json",
    ):
        payload = json.loads((root / "schemas/v8_1" / name).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
