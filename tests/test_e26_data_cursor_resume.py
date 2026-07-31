from __future__ import annotations

import copy

import numpy as np
import pytest

from catena.lm.general_corpus import (
    PairedTokenCursor,
    ScientificCorpusContractError,
    TokenMemmap,
    write_synthetic_token_memmap,
)
from catena.lm.paired_stream import (
    PackedTransactionCursor,
    PairedStreamContractError,
    PairedTrainingCursor,
    PairedTransactionCursor,
    StreamSource,
    TokenBalancedPairedTrainingCursor,
    replay_digest,
)
from catena.lm.tokenizer import ByteTokenizer


def _mixed_cursor(tmp_path):
    manifest = write_synthetic_token_memmap(
        tmp_path / "general",
        vocab_size=259,
        token_count=8_192,
        seed=260,
    )
    corpus = TokenMemmap(manifest)
    tokenizer = ByteTokenizer()
    tokenizer_hash = tokenizer.manifest().manifest_hash
    general = corpus.paired_cursor(seed=261, sequence_length=32)
    transaction = PairedTransactionCursor(
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        seed=262,
        sequence_length=32,
        pad_token_id=tokenizer.pad_id,
    )
    return (
        corpus,
        tokenizer,
        tokenizer_hash,
        PairedTrainingCursor(general, transaction),
    )


def test_general_cursor_snapshot_is_validated_and_hashes_token_bytes(tmp_path) -> None:
    manifest = write_synthetic_token_memmap(tmp_path / "corpus", token_count=512)
    corpus = TokenMemmap(manifest)
    cursor = corpus.paired_cursor(seed=11, sequence_length=24)
    rows, receipt = cursor.take(3)
    restored = PairedTokenCursor.from_snapshot(corpus, cursor.snapshot())
    expected, expected_receipt = cursor.take(2)
    observed, observed_receipt = restored.take(2)

    assert len(receipt.token_bytes_sha256) == 64
    assert receipt.starts
    assert observed_receipt == expected_receipt
    assert all(np.array_equal(left, right) for left, right in zip(observed, expected, strict=True))
    assert all(row.shape == (24,) for row in rows)

    corrupted = copy.deepcopy(cursor.snapshot())
    corrupted["sequence_index"] += 1
    with pytest.raises(ScientificCorpusContractError, match="SHA-256"):
        PairedTokenCursor.from_snapshot(corpus, corrupted)


def test_paired_mixed_cursor_replays_source_metadata_and_bytes(tmp_path) -> None:
    corpus, tokenizer, tokenizer_hash, left = _mixed_cursor(tmp_path)
    _, _, _, right = _mixed_cursor(tmp_path)
    left_rows, left_receipt = left.take(7)
    right_rows, right_receipt = right.take(7)

    assert left_receipt == right_receipt
    assert [row.source_type for row in left_rows] == [
        StreamSource.GENERAL.value,
        StreamSource.GENERAL.value,
        StreamSource.GENERAL.value,
        StreamSource.GENERAL.value,
        StreamSource.TRANSACTION.value,
        StreamSource.GENERAL.value,
        StreamSource.GENERAL.value,
    ]
    assert all(
        np.array_equal(left_row.token_ids, right_row.token_ids)
        and left_row.audit_record() == right_row.audit_record()
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    )

    snapshot = left.snapshot()
    resumed = PairedTrainingCursor.from_snapshot(
        corpus,
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        snapshot=snapshot,
    )
    left_next, left_next_receipt = left.take(11)
    resumed_next, resumed_receipt = resumed.take(11)
    assert resumed_receipt == left_next_receipt
    assert all(
        np.array_equal(left_row.token_ids, right_row.token_ids)
        for left_row, right_row in zip(left_next, resumed_next, strict=True)
    )

    left_digest = replay_digest(left, minimum_tokens=1_024)
    resumed_after = PairedTrainingCursor.from_snapshot(
        corpus,
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        snapshot=resumed.snapshot(),
    )
    resumed_digest = replay_digest(resumed_after, minimum_tokens=1_024)
    assert left_digest == resumed_digest


def test_mixed_cursor_rejects_tampered_child_progress(tmp_path) -> None:
    corpus, tokenizer, tokenizer_hash, cursor = _mixed_cursor(tmp_path)
    cursor.take(5)
    snapshot = copy.deepcopy(cursor.snapshot())
    child = dict(snapshot["general_cursor"])
    child_payload = dict(child)
    child_payload.pop("snapshot_sha256")
    child_payload["sequence_index"] += 1
    child_payload["tokens_emitted"] = (
        child_payload["sequence_index"] * child_payload["sequence_length"]
    )
    from catena.core.provenance_v61 import sha256_canonical_json

    child_payload["snapshot_sha256"] = sha256_canonical_json(child_payload)
    snapshot["general_cursor"] = child_payload
    outer = dict(snapshot)
    outer.pop("snapshot_sha256")
    outer["snapshot_sha256"] = sha256_canonical_json(outer)
    with pytest.raises(PairedStreamContractError, match="child indices"):
        PairedTrainingCursor.from_snapshot(
            corpus,
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=outer,
        )


def test_complete_example_token_balanced_cursor_tracks_actual_tokens_and_resume(
    tmp_path,
) -> None:
    manifest = write_synthetic_token_memmap(
        tmp_path / "balanced-general",
        vocab_size=259,
        token_count=1_048_576,
        seed=270,
    )
    corpus = TokenMemmap(manifest)
    tokenizer = ByteTokenizer()
    tokenizer_hash = tokenizer.manifest().manifest_hash

    def new_cursor() -> TokenBalancedPairedTrainingCursor:
        return TokenBalancedPairedTrainingCursor(
            corpus.paired_cursor(seed=271, sequence_length=4_096),
            PackedTransactionCursor(
                tokenizer,
                tokenizer_hash=tokenizer_hash,
                seed=272,
                sequence_length=4_096,
                pad_token_id=tokenizer.pad_id,
            ),
        )

    left = new_cursor()
    right = new_cursor()
    left_rows, left_receipt = left.take(80)
    right_rows, right_receipt = right.take(80)
    assert left_receipt == right_receipt
    assert left_receipt.loss_bearing_tokens == (
        left_receipt.general_unpadded_tokens + left_receipt.transaction_unpadded_tokens
    )
    assert (
        abs(4 * left.transaction_unpadded_tokens - left.general_unpadded_tokens)
        <= 4 * left.sequence_length
    )
    assert any(
        row.source_type == StreamSource.TRANSACTION.value
        and row.packed_examples >= 1
        and len(row.component_source_ids) == row.packed_examples
        and row.unpadded_tokens + row.padding_tokens == 4_096
        for row in left_rows
    )
    assert all(
        np.array_equal(left_row.token_ids, right_row.token_ids)
        and left_row.audit_record() == right_row.audit_record()
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    )

    restored = TokenBalancedPairedTrainingCursor.from_snapshot(
        corpus,
        tokenizer,
        tokenizer_hash=tokenizer_hash,
        snapshot=left.snapshot(),
    )
    expected, expected_receipt = left.take(31)
    observed, observed_receipt = restored.take(31)
    assert observed_receipt == expected_receipt
    assert all(
        np.array_equal(left_row.token_ids, right_row.token_ids)
        for left_row, right_row in zip(expected, observed, strict=True)
    )
