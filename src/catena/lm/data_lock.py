"""Deterministic E26 document partition, deduplication, and leakage contracts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

PARTITION_MODULUS = 10_000
NEAR_DUP_SHINGLE_WORDS = 5
NEAR_DUP_PERMUTATIONS = 128
NEAR_DUP_BANDS = 32
NEAR_DUP_ROWS_PER_BAND = 4
NEAR_DUP_SEED = 260_026
NEAR_DUP_THRESHOLD = 0.80
_MINHASH_PRIME = 4_294_967_291
_MINHASH_VECTOR_CHUNK = 4_096


class DataLockError(RuntimeError):
    """Raised when deterministic source-data construction cannot be certified."""


class ContentSplit(StrEnum):
    TOKENIZER_ONLY = "tokenizer_only"
    GENERAL_VALIDATION = "general_validation"
    GENERAL_TEST = "general_test"
    GENERAL_TRAIN = "general_train"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    text: str
    shard_path: str
    row_group: int
    row_index: int
    source_id: str = ""
    source_url: str = ""

    @property
    def location_key(self) -> tuple[str, int, int]:
        return (self.shard_path, self.row_group, self.row_index)


@dataclass(frozen=True, slots=True)
class LockedDocument:
    content_sha256: str
    content_sha512: str
    split: str
    bucket: int
    normalized_utf8_bytes: int
    shard_path: str
    row_group: int
    row_index: int
    source_id_sha256: str
    source_url_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DedupReceipt:
    documents_seen: int
    unique_documents: int
    exact_duplicates: int
    split_counts: dict[str, int]
    manifest_path: str
    manifest_sha256: str
    document_selection_sha256: str
    sqlite_path: str
    sqlite_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NearDuplicateFlag:
    left_sha256: str
    left_split: str
    right_sha256: str
    right_split: str
    estimated_jaccard: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_document(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("FineWeb text must be a string")
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def content_digest(text: str) -> tuple[str, str, bytes]:
    normalized = normalize_document(text).encode("utf-8", errors="strict")
    return (
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha512(normalized).hexdigest(),
        normalized,
    )


def bucket_for_sha256(content_sha256: str) -> int:
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
        raise ValueError("content_sha256 must be 64 lowercase hex characters")
    return int(content_sha256[:16], 16) % PARTITION_MODULUS


def split_for_bucket(bucket: int) -> ContentSplit:
    if not 0 <= bucket < PARTITION_MODULUS:
        raise ValueError("bucket outside [0,10000)")
    if bucket <= 99:
        return ContentSplit.TOKENIZER_ONLY
    if bucket <= 119:
        return ContentSplit.GENERAL_VALIDATION
    if bucket <= 139:
        return ContentSplit.GENERAL_TEST
    return ContentSplit.GENERAL_TRAIN


def lock_document(document: SourceDocument) -> tuple[LockedDocument, bytes]:
    sha256, sha512, normalized = content_digest(document.text)
    bucket = bucket_for_sha256(sha256)
    return (
        LockedDocument(
            content_sha256=sha256,
            content_sha512=sha512,
            split=split_for_bucket(bucket).value,
            bucket=bucket,
            normalized_utf8_bytes=len(normalized),
            shard_path=document.shard_path,
            row_group=document.row_group,
            row_index=document.row_index,
            source_id_sha256=hashlib.sha256(document.source_id.encode()).hexdigest(),
            source_url_sha256=hashlib.sha256(document.source_url.encode()).hexdigest(),
        ),
        normalized,
    )


class SQLiteDocumentIndex:
    """Disk-backed exact-dedup index with a canonical first-location representative."""

    def __init__(self, path: str | Path, *, create: bool) -> None:
        self.path = Path(path)
        if create and (self.path.exists() or self.path.is_symlink()):
            raise FileExistsError(f"Refusing to overwrite document index: {self.path}")
        if not create and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        if create:
            self.connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE documents (
                    content_sha256 TEXT PRIMARY KEY,
                    content_sha512 TEXT NOT NULL,
                    split TEXT NOT NULL,
                    bucket INTEGER NOT NULL,
                    normalized_utf8_bytes INTEGER NOT NULL,
                    shard_path TEXT NOT NULL,
                    row_group INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    source_id_sha256 TEXT NOT NULL,
                    source_url_sha256 TEXT NOT NULL,
                    normalized_utf8 BLOB NOT NULL
                );
                CREATE INDEX documents_split_sha ON documents(split, content_sha256);
                """
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteDocumentIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add(self, record: LockedDocument, normalized_utf8: bytes) -> bool:
        existing = self.connection.execute(
            "SELECT content_sha512, normalized_utf8_bytes, normalized_utf8 FROM documents "
            "WHERE content_sha256 = ?",
            (record.content_sha256,),
        ).fetchone()
        if existing is not None:
            if existing != (
                record.content_sha512,
                record.normalized_utf8_bytes,
                normalized_utf8,
            ):
                raise DataLockError(
                    "SHA-256 collision detected between unequal normalized documents"
                )
            return False
        self.connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.content_sha256,
                record.content_sha512,
                record.split,
                record.bucket,
                record.normalized_utf8_bytes,
                record.shard_path,
                record.row_group,
                record.row_index,
                record.source_id_sha256,
                record.source_url_sha256,
                normalized_utf8,
            ),
        )
        return True

    def commit(self) -> None:
        self.connection.commit()

    def records(self, *, split: ContentSplit | None = None) -> Iterator[LockedDocument]:
        query = (
            "SELECT content_sha256, content_sha512, split, bucket, normalized_utf8_bytes, "
            "shard_path, row_group, row_index, source_id_sha256, source_url_sha256 "
            "FROM documents"
        )
        parameters: tuple[str, ...] = ()
        if split is not None:
            query += " WHERE split = ?"
            parameters = (split.value,)
        query += " ORDER BY content_sha256"
        for row in self.connection.execute(query, parameters):
            yield LockedDocument(*row)

    def representative_locations(
        self,
        split: ContentSplit,
    ) -> dict[tuple[str, int, int], str]:
        return {
            (str(shard), int(row_group), int(row_index)): str(content_sha)
            for content_sha, shard, row_group, row_index in self.connection.execute(
                "SELECT content_sha256, shard_path, row_group, row_index "
                "FROM documents WHERE split = ?",
                (split.value,),
            )
        }

    def texts(
        self,
        split: ContentSplit,
    ) -> Iterator[tuple[LockedDocument, str]]:
        query = (
            "SELECT content_sha256, content_sha512, split, bucket, normalized_utf8_bytes, "
            "shard_path, row_group, row_index, source_id_sha256, source_url_sha256, "
            "normalized_utf8 FROM documents WHERE split = ? ORDER BY content_sha256"
        )
        for row in self.connection.execute(query, (split.value,)):
            record = LockedDocument(*row[:10])
            raw = row[10]
            if not isinstance(raw, bytes):
                raw = bytes(raw)
            text = raw.decode("utf-8", errors="strict")
            if len(raw) != record.normalized_utf8_bytes:
                raise DataLockError("Stored normalized text byte count changed")
            yield record, text


def build_document_index(
    documents: Iterable[SourceDocument],
    *,
    sqlite_path: str | Path,
    manifest_path: str | Path,
) -> DedupReceipt:
    destination = Path(manifest_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite document manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    duplicates = 0
    previous_location: tuple[str, int, int] | None = None
    with SQLiteDocumentIndex(sqlite_path, create=True) as index:
        for document in documents:
            if previous_location is not None and document.location_key < previous_location:
                raise DataLockError("Source documents are not in canonical shard/row order")
            previous_location = document.location_key
            seen += 1
            record, normalized = lock_document(document)
            if not index.add(record, normalized):
                duplicates += 1
        index.commit()
        split_counts: dict[str, int] = defaultdict(int)
        digest = hashlib.sha256()
        unique_documents = 0
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            for record in index.records():
                unique_documents += 1
                split_counts[record.split] += 1
                line = json.dumps(
                    record.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write(line + "\n")
                digest.update(line.encode() + b"\n")
            handle.flush()
    return DedupReceipt(
        documents_seen=seen,
        unique_documents=unique_documents,
        exact_duplicates=duplicates,
        split_counts=dict(sorted(split_counts.items())),
        manifest_path=str(destination.resolve()),
        manifest_sha256=sha256_file(destination),
        document_selection_sha256=digest.hexdigest(),
        sqlite_path=str(Path(sqlite_path).resolve()),
        sqlite_sha256=sha256_file(sqlite_path),
    )


def _near_duplicate_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_document(text).lower().split())


def _shingle_hashes(text: str) -> tuple[int, ...]:
    tokens = _near_duplicate_tokens(text)
    shingles: tuple[str, ...]
    if len(tokens) < NEAR_DUP_SHINGLE_WORDS:
        shingles = (" ".join(tokens),)
    else:
        shingles = tuple(
            " ".join(tokens[index : index + NEAR_DUP_SHINGLE_WORDS])
            for index in range(len(tokens) - NEAR_DUP_SHINGLE_WORDS + 1)
        )
    return tuple(
        int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8], "big")
        % _MINHASH_PRIME
        for shingle in set(shingles)
    )


def _minhash_coefficients() -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for index in range(NEAR_DUP_PERMUTATIONS):
        raw = hashlib.sha256(f"{NEAR_DUP_SEED}:{index}".encode()).digest()
        first = 1 + int.from_bytes(raw[:8], "big") % (_MINHASH_PRIME - 1)
        second = int.from_bytes(raw[8:16], "big") % _MINHASH_PRIME
        result.append((first, second))
    return tuple(result)


_MINHASH_COEFFICIENTS = _minhash_coefficients()


def minhash_signature(text: str) -> tuple[int, ...]:
    return _minhash_signature_from_hashes(_shingle_hashes(text))


def _minhash_signature_from_hashes(hashes: Sequence[int]) -> tuple[int, ...]:
    if not hashes:
        raise ValueError("MinHash input must contain at least one shingle hash")
    coefficients = np.asarray(_MINHASH_COEFFICIENTS, dtype=np.uint64)
    first = coefficients[:, 0:1]
    second = coefficients[:, 1:2]
    minima = np.full(NEAR_DUP_PERMUTATIONS, _MINHASH_PRIME, dtype=np.uint64)
    for start in range(0, len(hashes), _MINHASH_VECTOR_CHUNK):
        values = np.asarray(
            hashes[start : start + _MINHASH_VECTOR_CHUNK],
            dtype=np.uint64,
        )
        values = (values % np.uint64(_MINHASH_PRIME))[None, :]
        # The registered 32-bit prime guarantees that a*x+b fits uint64
        # exactly, avoiding platform-dependent overflow before the modulus.
        transformed = (first * values + second) % _MINHASH_PRIME
        minima = np.minimum(minima, transformed.min(axis=1))
    return tuple(int(value) for value in minima)


def minhash_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != NEAR_DUP_PERMUTATIONS or len(right) != NEAR_DUP_PERMUTATIONS:
        raise ValueError(f"MinHash signatures must have {NEAR_DUP_PERMUTATIONS} entries")
    return sum(a == b for a, b in zip(left, right, strict=True)) / NEAR_DUP_PERMUTATIONS


def audit_near_duplicates(
    records: Iterable[tuple[str, str, str]],
) -> tuple[NearDuplicateFlag, ...]:
    """Flag cross-split near duplicates using deterministic 32x4-band LSH.

    ``records`` contains ``(content_sha256, split, normalized_text)``.  The
    routine is exact over its deterministic LSH candidate set and deliberately
    returns flags rather than silently removing documents.
    """

    prepared = [
        (content_sha, split, minhash_signature(text))
        for content_sha, split, text in records
    ]
    bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for item_index, (_, split, signature) in enumerate(prepared):
        for band in range(NEAR_DUP_BANDS):
            lower = band * NEAR_DUP_ROWS_PER_BAND
            upper = lower + NEAR_DUP_ROWS_PER_BAND
            key = (band, signature[lower:upper])
            for other in bands[key]:
                if prepared[other][1] != split:
                    candidates.add((other, item_index))
            bands[key].append(item_index)
    flags: list[NearDuplicateFlag] = []
    for left_index, right_index in sorted(candidates):
        left_sha, left_split, left_signature = prepared[left_index]
        right_sha, right_split, right_signature = prepared[right_index]
        similarity = minhash_similarity(left_signature, right_signature)
        if similarity >= NEAR_DUP_THRESHOLD:
            flags.append(
                NearDuplicateFlag(
                    left_sha,
                    left_split,
                    right_sha,
                    right_split,
                    similarity,
                )
            )
    return tuple(flags)


def audit_near_duplicates_asymmetric(
    protected_records: Iterable[tuple[str, str, str]],
    train_records: Iterable[tuple[str, str, str]],
) -> tuple[NearDuplicateFlag, ...]:
    """Memory-bounded cross-split audit with tokenizer/evaluation splits indexed.

    Protected records are the tokenizer, validation, and test partitions.  They
    are small under the locked hash buckets, so their 32 LSH bands are retained.
    General-train documents are streamed once and are never compared with one
    another because that is not a cross-split leakage question.
    """

    protected = [
        (content_sha, split, minhash_signature(text))
        for content_sha, split, text in protected_records
    ]
    bands: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()
    for index, (_, split, signature) in enumerate(protected):
        for band in range(NEAR_DUP_BANDS):
            lower = band * NEAR_DUP_ROWS_PER_BAND
            upper = lower + NEAR_DUP_ROWS_PER_BAND
            key = (band, signature[lower:upper])
            for other in bands[key]:
                if protected[other][1] != split:
                    candidate_pairs.add((other, index))
            bands[key].append(index)

    flags: list[NearDuplicateFlag] = []
    for left, right in sorted(candidate_pairs):
        left_sha, left_split, left_signature = protected[left]
        right_sha, right_split, right_signature = protected[right]
        similarity = minhash_similarity(left_signature, right_signature)
        if similarity >= NEAR_DUP_THRESHOLD:
            flags.append(
                NearDuplicateFlag(
                    left_sha,
                    left_split,
                    right_sha,
                    right_split,
                    similarity,
                )
            )
    for train_sha, train_split, train_text in train_records:
        signature = minhash_signature(train_text)
        candidates: set[int] = set()
        for band in range(NEAR_DUP_BANDS):
            lower = band * NEAR_DUP_ROWS_PER_BAND
            upper = lower + NEAR_DUP_ROWS_PER_BAND
            key = (band, signature[lower:upper])
            candidates.update(bands.get(key, ()))
        if not candidates:
            continue
        for index in sorted(candidates):
            other_sha, other_split, other_signature = protected[index]
            similarity = minhash_similarity(other_signature, signature)
            if similarity >= NEAR_DUP_THRESHOLD:
                flags.append(
                    NearDuplicateFlag(
                        other_sha,
                        other_split,
                        train_sha,
                        train_split,
                        similarity,
                    )
                )
    flags.sort(
        key=lambda item: (
            item.left_sha256,
            item.right_sha256,
            item.left_split,
            item.right_split,
        )
    )
    return tuple(flags)


def write_near_duplicate_audit(
    path: str | Path,
    flags: Sequence[NearDuplicateFlag],
    *,
    implementation_history: Sequence[dict[str, Any]] = (),
) -> Path:
    payload = {
        "schema_version": "catena-e26-near-duplicate-audit-v1",
        "scientific_evidence": False,
        "algorithm": "SHA256_PERMUTED_MINHASH_V1",
        "normalization": "LOWERCASE_NFC_WHITESPACE_TOKENS",
        "shingle_width_words": NEAR_DUP_SHINGLE_WORDS,
        "permutations": NEAR_DUP_PERMUTATIONS,
        "field_prime": _MINHASH_PRIME,
        "shingle_hash": "SHA256_PREFIX_64_BIG_ENDIAN",
        "field_reduction": "SHINGLE_HASH_MOD_FIELD_BEFORE_AFFINE_PERMUTATION",
        "bands": NEAR_DUP_BANDS,
        "rows_per_band": NEAR_DUP_ROWS_PER_BAND,
        "candidate_generation": "ASYMMETRIC_PROTECTED_INDEX_STREAMED_TRAIN_LSH_32X4",
        "seed": NEAR_DUP_SEED,
        "estimated_jaccard_flag_threshold": NEAR_DUP_THRESHOLD,
        "flagged_pairs": [item.as_dict() for item in flags],
        "flagged_pair_count": len(flags),
        "pass": not flags,
        "flagged_pair_policy": "FAIL_PENDING_MANUAL_AUDIT",
        "implementation_history": list(implementation_history),
    }
    payload["audit_sha256"] = sha256_canonical_json(payload)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite near-duplicate audit: {destination}")
    write_json_strict(destination, payload)
    return destination
