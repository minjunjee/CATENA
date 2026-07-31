from pathlib import Path

from catena.lm.data_lock import (
    _MINHASH_COEFFICIENTS,
    _MINHASH_PRIME,
    ContentSplit,
    SourceDocument,
    SQLiteDocumentIndex,
    _minhash_signature_from_hashes,
    audit_near_duplicates_asymmetric,
    build_document_index,
    content_digest,
    normalize_document,
)


def test_nfc_lf_identity_and_exact_dedup_are_split_safe(tmp_path: Path) -> None:
    first = "cafe\u0301\r\nline"
    second = "café\nline"
    assert normalize_document(first) == second
    assert content_digest(first)[0] == content_digest(second)[0]
    documents = [
        SourceDocument(first, "000.parquet", 0, 0, "a", "https://a"),
        SourceDocument(second, "000.parquet", 0, 1, "b", "https://b"),
        SourceDocument("independent text", "001.parquet", 0, 0, "c", "https://c"),
    ]
    receipt = build_document_index(
        documents,
        sqlite_path=tmp_path / "documents.sqlite3",
        manifest_path=tmp_path / "documents.jsonl",
    )
    assert receipt.documents_seen == 3
    assert receipt.unique_documents == 2
    assert receipt.exact_duplicates == 1
    assert sum(receipt.split_counts.values()) == 2


def test_sqlite_spool_emits_each_split_in_content_hash_order(tmp_path: Path) -> None:
    receipt = build_document_index(
        [
            SourceDocument("zeta", "000.parquet", 0, 0),
            SourceDocument("alpha", "000.parquet", 0, 1),
            SourceDocument("middle", "000.parquet", 0, 2),
        ],
        sqlite_path=tmp_path / "documents.sqlite3",
        manifest_path=tmp_path / "documents.jsonl",
    )
    with SQLiteDocumentIndex(receipt.sqlite_path, create=False) as index:
        for split in ContentSplit:
            rows = list(index.texts(split))
            hashes = [record.content_sha256 for record, _ in rows]
            assert hashes == sorted(hashes)


def test_locked_minhash_flags_cross_split_near_duplicate() -> None:
    base = "one two three four five six seven eight nine ten"
    flags = audit_near_duplicates_asymmetric(
        [("a" * 64, ContentSplit.TOKENIZER_ONLY.value, base)],
        [("b" * 64, ContentSplit.GENERAL_TRAIN.value, base + " eleven")],
    )
    assert len(flags) == 1
    assert flags[0].estimated_jaccard >= 0.80


def test_vector_minhash_matches_exact_bigint_at_uint64_boundary() -> None:
    hashes = (0, _MINHASH_PRIME - 1, _MINHASH_PRIME, 2**64 - 2, 2**64 - 1)
    observed = _minhash_signature_from_hashes(hashes)
    expected = tuple(
        min(
            (first * (value % _MINHASH_PRIME) + second) % _MINHASH_PRIME
            for value in hashes
        )
        for first, second in _MINHASH_COEFFICIENTS
    )
    assert observed == expected
