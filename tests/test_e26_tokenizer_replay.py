import hashlib
from pathlib import Path

import pytest

from catena.lm.tokenizer_builder import (
    SPECIAL_TOKENS,
    build_replayed_tokenizer,
    select_tokenizer_training_chunks,
)


def test_training_byte_limit_uses_content_hash_order_and_valid_utf8_prefix() -> None:
    chunks = select_tokenizer_training_chunks(
        [
            ("f" * 64, "later"),
            ("0" * 64, "ééé"),
            ("1" * 64, "tail"),
        ],
        byte_limit=7,
    )
    assert [item.content_sha256 for item in chunks] == ["0" * 64, "1" * 64]
    assert chunks[0].utf8_bytes_used == 6
    assert chunks[1].utf8_bytes_used == 1
    assert chunks[1].text == "t"
    assert chunks[1].partial_final_document


def test_tokenizer_replay_twice_has_fixed_16k_ids(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers", reason="run in the pinned E26 data-tool environment")
    # Repeating a large set of opaque strings gives every candidate merge the
    # registered minimum frequency while retaining enough distinct merges to
    # exercise the exact 16K vocabulary contract.
    words = [
        hashlib.sha256(f"opaque-token-{index}".encode()).hexdigest()
        for index in range(25_000)
    ]
    corpus = " ".join(words)
    receipt = build_replayed_tokenizer(
        [
            ("0" * 64, corpus),
            ("1" * 64, corpus),
        ],
        output_root=tmp_path / "tokenizer",
        source_revisions=["fixture@0123456789abcdef"],
        byte_limit=len(corpus.encode()) * 2,
    )
    assert receipt["artifact_hash_sets_identical"] is True
    assert receipt["stress_audit"]["pass"] is True
    assert [token_id for _, token_id in SPECIAL_TOKENS.values()] == [0, 1, 2, 3, 4]
