from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from catena.core.provenance_v61 import (
    SHA256_PATTERN,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
)

from .tokenizer import (
    SCIENTIFIC_VOCAB_SIZE,
    TokenizerManifest,
    load_scientific_tokenizer_manifest,
)


class ScientificCorpusContractError(ValueError):
    """Raised when corpus bytes/provenance are not admissible for E26 MAIN."""


@dataclass(frozen=True, slots=True)
class TokenMemmapManifest:
    token_path: str
    dtype: str
    token_count: int
    token_file_sha256: str
    document_manifest_path: str
    document_manifest_sha256: str
    corpus_revision: str
    schema_version: str = "catena-v8.1"
    corpus_id: str = "non-evidence-reference"
    corpus_manifest_path: str | None = None
    corpus_manifest_sha256: str | None = None
    tokenizer_manifest_sha256: str | None = None
    tokenizer_model_sha256: str | None = None
    tokenizer_vocab_size: int | None = None
    document_selection_sha256: str | None = None
    document_count: int | None = None
    source_revisions: tuple[str, ...] = ()
    token_id_min: int | None = None
    token_id_max: int | None = None
    evidence_tier: str = "NON_EVIDENCE_VALIDATION"
    scientific_main_eligible: bool = False
    synthetic: bool = True
    reference_only: bool = True
    selection_frozen: bool = False

    @property
    def manifest_hash(self) -> str:
        if self.corpus_manifest_sha256 is not None:
            return self.corpus_manifest_sha256
        return sha256_canonical_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_revisions"] = list(self.source_revisions)
        return payload

    def assert_scientific_main(self) -> None:
        if not self.scientific_main_eligible:
            raise ScientificCorpusContractError(
                "General corpus manifest is not eligible for scientific MAIN"
            )
        if self.synthetic or self.reference_only:
            raise ScientificCorpusContractError(
                "Synthetic/reference corpus cannot be used for scientific MAIN"
            )
        if not self.selection_frozen:
            raise ScientificCorpusContractError("Document selection is not frozen")
        required_hashes = (
            self.token_file_sha256,
            self.document_manifest_sha256,
            self.document_selection_sha256,
            self.tokenizer_manifest_sha256,
            self.tokenizer_model_sha256,
            self.corpus_manifest_sha256,
        )
        if any(value is None or not SHA256_PATTERN.fullmatch(value) for value in required_hashes):
            raise ScientificCorpusContractError(
                "Scientific corpus lacks complete SHA-256 provenance"
            )
        if self.tokenizer_vocab_size != SCIENTIFIC_VOCAB_SIZE:
            raise ScientificCorpusContractError(
                f"Scientific corpus must be encoded with the locked {SCIENTIFIC_VOCAB_SIZE} "
                "token vocabulary"
            )
        if (
            self.token_id_min is None
            or self.token_id_max is None
            or self.token_id_min < 0
            or self.token_id_max >= SCIENTIFIC_VOCAB_SIZE
        ):
            raise ScientificCorpusContractError("Corpus contains an out-of-vocabulary token ID")


@dataclass(frozen=True, slots=True)
class TokenCursorReceipt:
    manifest_hash: str
    seed: int
    sequence_length: int
    start_sequence_index: int
    end_sequence_index: int
    sequences: int
    tokens: int
    starts: tuple[int, ...]
    starts_sha256: str
    token_bytes_sha256: str
    data_order_sha256: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["starts"] = list(self.starts)
        return payload


class PairedTokenCursor:
    """Variant-independent counter cursor over one immutable token memmap.

    Start positions are derived from a counter-based SHA-256 mapping.  Worker
    scheduling and model variant never enter the mapping, so a paired tied/dual
    run can recreate the exact token exposure from ``snapshot()`` alone.
    """

    def __init__(
        self,
        corpus: TokenMemmap,
        *,
        seed: int,
        sequence_length: int,
        start_sequence_index: int = 0,
    ) -> None:
        if sequence_length <= 1:
            raise ValueError("sequence_length must exceed one")
        if isinstance(start_sequence_index, bool) or start_sequence_index < 0:
            raise ValueError("start_sequence_index must be non-negative")
        valid_starts = len(corpus) - sequence_length + 1
        if valid_starts <= 0:
            raise ValueError("Token memmap is shorter than one sequence")
        self.corpus = corpus
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.sequence_index = int(start_sequence_index)
        self._valid_starts = valid_starts

    def _start_for(self, sequence_index: int) -> int:
        payload = (
            f"catena-e26-token-cursor-v1\0{self.corpus.manifest.manifest_hash}\0"
            f"{self.seed}\0{self.sequence_length}\0{sequence_index}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % self._valid_starts

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": "catena-v8.1",
            "cursor_algorithm": "sha256_counter_v1",
            "manifest_hash": self.corpus.manifest.manifest_hash,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "sequence_index": self.sequence_index,
            "tokens_emitted": self.sequence_index * self.sequence_length,
        }
        payload["snapshot_sha256"] = sha256_canonical_json(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        corpus: TokenMemmap,
        snapshot: Mapping[str, Any],
    ) -> PairedTokenCursor:
        expected_fields = {
            "schema_version",
            "cursor_algorithm",
            "manifest_hash",
            "seed",
            "sequence_length",
            "sequence_index",
            "tokens_emitted",
            "snapshot_sha256",
        }
        if set(snapshot) != expected_fields:
            raise ScientificCorpusContractError(
                "Token cursor snapshot fields do not match the locked schema"
            )
        payload = {key: snapshot[key] for key in expected_fields - {"snapshot_sha256"}}
        observed_hash = snapshot.get("snapshot_sha256")
        if not isinstance(observed_hash, str) or observed_hash != sha256_canonical_json(payload):
            raise ScientificCorpusContractError("Token cursor snapshot SHA-256 mismatch")
        if snapshot["schema_version"] != "catena-v8.1":
            raise ScientificCorpusContractError("Unsupported token cursor snapshot schema")
        if snapshot["cursor_algorithm"] != "sha256_counter_v1":
            raise ScientificCorpusContractError("Unsupported token cursor algorithm")
        if snapshot["manifest_hash"] != corpus.manifest.manifest_hash:
            raise ScientificCorpusContractError("Token cursor corpus manifest changed")
        seed = snapshot["seed"]
        sequence_length = snapshot["sequence_length"]
        sequence_index = snapshot["sequence_index"]
        tokens_emitted = snapshot["tokens_emitted"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or isinstance(sequence_index, bool)
            or not isinstance(sequence_index, int)
            or isinstance(tokens_emitted, bool)
            or not isinstance(tokens_emitted, int)
        ):
            raise ScientificCorpusContractError("Token cursor counters must be integers")
        if tokens_emitted != sequence_index * sequence_length:
            raise ScientificCorpusContractError(
                "Token cursor exposure count disagrees with its sequence index"
            )
        return cls(
            corpus,
            seed=seed,
            sequence_length=sequence_length,
            start_sequence_index=sequence_index,
        )

    def fork(self) -> PairedTokenCursor:
        return self.from_snapshot(self.corpus, self.snapshot())

    def take(self, count: int) -> tuple[list[NDArray[np.int64]], TokenCursorReceipt]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        start_index = self.sequence_index
        starts = [self._start_for(start_index + offset) for offset in range(count)]
        sequences = [
            np.asarray(
                self.corpus._tokens[start : start + self.sequence_length],
                dtype=np.int64,
            )
            for start in starts
        ]
        self.sequence_index += count
        starts_sha256 = sha256_canonical_json(starts)
        token_digest = hashlib.sha256()
        locked_dtype = np.dtype(self.corpus.manifest.dtype)
        for sequence in sequences:
            encoded = np.asarray(sequence, dtype=locked_dtype)
            token_digest.update(len(encoded).to_bytes(8, "big"))
            token_digest.update(encoded.tobytes(order="C"))
        token_bytes_sha256 = token_digest.hexdigest()
        receipt_payload: dict[str, Any] = {
            "manifest_hash": self.corpus.manifest.manifest_hash,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "start_sequence_index": start_index,
            "end_sequence_index": self.sequence_index,
            "sequences": count,
            "tokens": count * self.sequence_length,
            "starts": starts,
            "starts_sha256": starts_sha256,
            "token_bytes_sha256": token_bytes_sha256,
        }
        return sequences, TokenCursorReceipt(
            manifest_hash=self.corpus.manifest.manifest_hash,
            seed=self.seed,
            sequence_length=self.sequence_length,
            start_sequence_index=start_index,
            end_sequence_index=self.sequence_index,
            sequences=count,
            tokens=count * self.sequence_length,
            starts=tuple(starts),
            starts_sha256=starts_sha256,
            token_bytes_sha256=token_bytes_sha256,
            data_order_sha256=sha256_canonical_json(receipt_payload),
        )


class TokenMemmap:
    def __init__(
        self,
        manifest: TokenMemmapManifest,
        *,
        scientific_main: bool = False,
        verify_file_hash: bool | None = None,
    ) -> None:
        if scientific_main:
            manifest.assert_scientific_main()
        self.manifest = manifest
        self.path = Path(manifest.token_path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        dtype = np.dtype(manifest.dtype)
        self._tokens = np.memmap(self.path, mode="r", dtype=dtype)
        if len(self._tokens) != manifest.token_count:
            raise ValueError(
                f"Token count mismatch: manifest={manifest.token_count}, file={len(self._tokens)}"
            )
        should_hash = scientific_main if verify_file_hash is None else verify_file_hash
        if should_hash and sha256_file(self.path) != manifest.token_file_sha256:
            raise ScientificCorpusContractError("Token file changed after manifest validation")

    def __len__(self) -> int:
        return int(len(self._tokens))

    @classmethod
    def validate_files(
        cls,
        *,
        token_path: str | Path,
        dtype: str,
        document_manifest_path: str | Path,
        corpus_revision: str,
    ) -> TokenMemmapManifest:
        """Validate loose files for non-evidence smoke only.

        A manifest produced here is deliberately ineligible for MAIN.  MAIN
        must use :func:`load_scientific_corpus_manifest`.
        """

        token_file = Path(token_path)
        doc_file = Path(document_manifest_path)
        if not token_file.is_file():
            raise FileNotFoundError(token_file)
        if not doc_file.is_file():
            raise FileNotFoundError(doc_file)
        itemsize = np.dtype(dtype).itemsize
        if token_file.stat().st_size % itemsize:
            raise ValueError("Token file size is not divisible by dtype itemsize")
        count = token_file.stat().st_size // itemsize
        return TokenMemmapManifest(
            token_path=str(token_file.resolve()),
            dtype=np.dtype(dtype).name,
            token_count=count,
            token_file_sha256=sha256_file(token_file),
            document_manifest_path=str(doc_file.resolve()),
            document_manifest_sha256=sha256_file(doc_file),
            corpus_revision=corpus_revision,
        )

    @classmethod
    def from_scientific_manifest(
        cls,
        corpus_manifest_path: str | Path,
        *,
        tokenizer_manifest: TokenizerManifest,
        verify_token_ids: bool = True,
    ) -> TokenMemmap:
        manifest = load_scientific_corpus_manifest(
            corpus_manifest_path,
            tokenizer_manifest=tokenizer_manifest,
            verify_token_ids=verify_token_ids,
        )
        return cls(manifest, scientific_main=True)

    def paired_cursor(
        self,
        *,
        seed: int,
        sequence_length: int,
        start_sequence_index: int = 0,
    ) -> PairedTokenCursor:
        return PairedTokenCursor(
            self,
            seed=seed,
            sequence_length=sequence_length,
            start_sequence_index=start_sequence_index,
        )

    def deterministic_sequences(
        self,
        *,
        seed: int,
        sequence_length: int,
        count: int,
        start_cursor: int = 0,
    ) -> Iterator[NDArray[np.int64]]:
        cursor = self.paired_cursor(
            seed=seed,
            sequence_length=sequence_length,
            start_sequence_index=start_cursor,
        )
        sequences, _ = cursor.take(count)
        yield from sequences


def _require_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if value is not expected:
        raise ScientificCorpusContractError(f"{field} must be {expected!r}, got {value!r}")


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScientificCorpusContractError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ScientificCorpusContractError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _resolve_recorded_file(manifest_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ScientificCorpusContractError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ScientificCorpusContractError(f"{field} is missing: {candidate}") from error
    if not resolved.is_file():
        raise ScientificCorpusContractError(f"{field} is not a regular file: {resolved}")
    return resolved


def _scan_token_ids(
    path: Path,
    *,
    dtype: np.dtype[Any],
    vocab_size: int,
    chunk_tokens: int = 8_388_608,
) -> tuple[int, int]:
    tokens = np.memmap(path, mode="r", dtype=dtype)
    if len(tokens) == 0:
        raise ScientificCorpusContractError("Token memmap is empty")
    minimum = vocab_size
    maximum = -1
    for start in range(0, len(tokens), chunk_tokens):
        chunk = np.asarray(tokens[start : start + chunk_tokens])
        minimum = min(minimum, int(chunk.min()))
        maximum = max(maximum, int(chunk.max()))
        if minimum < 0 or maximum >= vocab_size:
            raise ScientificCorpusContractError(
                f"Token IDs outside [0,{vocab_size}): observed [{minimum},{maximum}]"
            )
    return minimum, maximum


def load_scientific_corpus_manifest(
    corpus_manifest_path: str | Path,
    *,
    tokenizer_manifest: TokenizerManifest,
    verify_token_ids: bool = True,
) -> TokenMemmapManifest:
    """Validate an external fixed document selection and token memmap.

    The token bytes must already exist.  This routine does not download,
    tokenize, shuffle, or select documents.
    """

    tokenizer_manifest.assert_scientific_main()
    path = Path(corpus_manifest_path).expanduser().resolve(strict=True)
    payload = read_json_object_strict(path)
    if payload.get("schema_version") != "catena-v8.1":
        raise ScientificCorpusContractError("Corpus manifest must use schema_version=catena-v8.1")
    if payload.get("manifest_type") != "E26_SCIENTIFIC_GENERAL_CORPUS":
        raise ScientificCorpusContractError("Corpus manifest has the wrong manifest_type")
    if payload.get("evidence_tier") != "SCIENTIFIC_INPUT":
        raise ScientificCorpusContractError("Corpus evidence_tier must be SCIENTIFIC_INPUT")
    _require_bool(payload, "scientific_main_eligible", True)
    _require_bool(payload, "synthetic", False)
    _require_bool(payload, "reference_only", False)
    _require_bool(payload, "selection_frozen", True)

    corpus_id = _require_string(payload, "corpus_id")
    corpus_revision = _require_string(payload, "corpus_revision")
    revision_upper = corpus_revision.upper()
    if "SYNTHETIC" in revision_upper or "REFERENCE_ONLY" in revision_upper:
        raise ScientificCorpusContractError(
            "Synthetic/reference corpus revision cannot be admitted to MAIN"
        )
    source_revisions = payload.get("source_revisions")
    if (
        not isinstance(source_revisions, list)
        or not source_revisions
        or any(not isinstance(item, str) or not item for item in source_revisions)
    ):
        raise ScientificCorpusContractError("Corpus requires pinned source_revisions")

    recorded_tokenizer_hash = _require_sha256(
        payload.get("tokenizer_manifest_sha256"),
        "tokenizer_manifest_sha256",
    )
    if recorded_tokenizer_hash != tokenizer_manifest.manifest_hash:
        raise ScientificCorpusContractError("Corpus was not encoded by the selected tokenizer")
    recorded_model_hash = _require_sha256(
        payload.get("tokenizer_model_sha256"),
        "tokenizer_model_sha256",
    )
    if recorded_model_hash != tokenizer_manifest.model_sha256:
        raise ScientificCorpusContractError("Corpus tokenizer model hash mismatch")
    vocab_size = payload.get("tokenizer_vocab_size")
    if vocab_size != tokenizer_manifest.vocab_size or vocab_size != SCIENTIFIC_VOCAB_SIZE:
        raise ScientificCorpusContractError("Corpus tokenizer vocabulary size mismatch")

    documents = payload.get("document_manifest")
    if not isinstance(documents, dict):
        raise ScientificCorpusContractError("document_manifest must be an object")
    document_path = _resolve_recorded_file(
        path,
        documents.get("path"),
        "document_manifest.path",
    )
    expected_document_hash = _require_sha256(
        documents.get("sha256"),
        "document_manifest.sha256",
    )
    if sha256_file(document_path) != expected_document_hash:
        raise ScientificCorpusContractError("Document manifest SHA-256 mismatch")
    document_count = documents.get("document_count")
    if (
        isinstance(document_count, bool)
        or not isinstance(document_count, int)
        or document_count < 1
    ):
        raise ScientificCorpusContractError("document_manifest.document_count must be >= 1")
    selection_hash = _require_sha256(
        documents.get("document_selection_sha256"),
        "document_manifest.document_selection_sha256",
    )

    token_file = payload.get("token_file")
    if not isinstance(token_file, dict):
        raise ScientificCorpusContractError("token_file must be an object")
    token_path = _resolve_recorded_file(path, token_file.get("path"), "token_file.path")
    dtype_name = _require_string(token_file, "dtype")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as error:
        raise ScientificCorpusContractError(f"Unsupported token dtype: {dtype_name}") from error
    if dtype.str != "<u2":
        raise ScientificCorpusContractError(
            "Scientific token dtype must be explicit little-endian uint16"
        )
    expected_bytes = token_file.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ScientificCorpusContractError("token_file.bytes must be an integer")
    if token_path.stat().st_size != expected_bytes:
        raise ScientificCorpusContractError("Token file byte count mismatch")
    if expected_bytes % dtype.itemsize:
        raise ScientificCorpusContractError("Token file size is not divisible by dtype itemsize")
    token_count = token_file.get("token_count")
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 1
        or token_count != expected_bytes // dtype.itemsize
    ):
        raise ScientificCorpusContractError("Token count does not match file size and dtype")
    expected_token_hash = _require_sha256(token_file.get("sha256"), "token_file.sha256")
    if sha256_file(token_path) != expected_token_hash:
        raise ScientificCorpusContractError("Token file SHA-256 mismatch")

    recorded_min = token_file.get("token_id_min")
    recorded_max = token_file.get("token_id_max")
    if (
        isinstance(recorded_min, bool)
        or not isinstance(recorded_min, int)
        or isinstance(recorded_max, bool)
        or not isinstance(recorded_max, int)
    ):
        raise ScientificCorpusContractError("token_file requires integer token_id_min/max")
    if verify_token_ids:
        observed_min, observed_max = _scan_token_ids(
            token_path,
            dtype=dtype,
            vocab_size=tokenizer_manifest.vocab_size,
        )
        if (recorded_min, recorded_max) != (observed_min, observed_max):
            raise ScientificCorpusContractError(
                "Recorded token ID range does not match the token file"
            )
    elif recorded_min < 0 or recorded_max >= tokenizer_manifest.vocab_size:
        raise ScientificCorpusContractError("Recorded token ID range is out of vocabulary")

    manifest = TokenMemmapManifest(
        token_path=str(token_path),
        dtype=dtype.str,
        token_count=token_count,
        token_file_sha256=expected_token_hash,
        document_manifest_path=str(document_path),
        document_manifest_sha256=expected_document_hash,
        corpus_revision=corpus_revision,
        corpus_id=corpus_id,
        corpus_manifest_path=str(path),
        corpus_manifest_sha256=sha256_file(path),
        tokenizer_manifest_sha256=recorded_tokenizer_hash,
        tokenizer_model_sha256=recorded_model_hash,
        tokenizer_vocab_size=vocab_size,
        document_selection_sha256=selection_hash,
        document_count=document_count,
        source_revisions=tuple(source_revisions),
        token_id_min=recorded_min,
        token_id_max=recorded_max,
        evidence_tier="SCIENTIFIC_INPUT",
        scientific_main_eligible=True,
        synthetic=False,
        reference_only=False,
        selection_frozen=True,
    )
    manifest.assert_scientific_main()
    return manifest


@dataclass(frozen=True, slots=True)
class ScientificDataReadiness:
    tokenizer: TokenizerManifest
    corpus: TokenMemmapManifest
    cursor_probe: TokenCursorReceipt
    readiness_sha256: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_SCIENTIFIC_DATA_READINESS",
            "scientific_evidence": False,
            "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
            "scientific_main_input_eligible": True,
            "tokenizer": self.tokenizer.as_dict(),
            "general_corpus": self.corpus.as_dict(),
            "paired_cursor_probe": self.cursor_probe.as_dict(),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload_without_hash()
        payload["readiness_sha256"] = self.readiness_sha256
        return payload


def validate_scientific_data_bundle(
    *,
    tokenizer_manifest_path: str | Path,
    corpus_manifest_path: str | Path,
    cursor_seed: int = 26_000,
    sequence_length: int = 4_096,
    cursor_probe_sequences: int = 4,
) -> ScientificDataReadiness:
    """Validate tokenizer/corpus provenance and a deterministic paired cursor."""

    tokenizer = load_scientific_tokenizer_manifest(tokenizer_manifest_path)
    corpus_manifest = load_scientific_corpus_manifest(
        corpus_manifest_path,
        tokenizer_manifest=tokenizer,
        verify_token_ids=True,
    )
    corpus = TokenMemmap(corpus_manifest, scientific_main=True)
    first = corpus.paired_cursor(seed=cursor_seed, sequence_length=sequence_length)
    second = first.fork()
    first_sequences, first_receipt = first.take(cursor_probe_sequences)
    second_sequences, second_receipt = second.take(cursor_probe_sequences)
    if first_receipt != second_receipt or any(
        not np.array_equal(left, right)
        for left, right in zip(first_sequences, second_sequences, strict=True)
    ):
        raise ScientificCorpusContractError("Paired token cursor is not deterministic")
    readiness_payload = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_SCIENTIFIC_DATA_READINESS",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "scientific_main_input_eligible": True,
        "tokenizer": tokenizer.as_dict(),
        "general_corpus": corpus_manifest.as_dict(),
        "paired_cursor_probe": first_receipt.as_dict(),
    }
    return ScientificDataReadiness(
        tokenizer=tokenizer,
        corpus=corpus_manifest,
        cursor_probe=first_receipt,
        readiness_sha256=sha256_canonical_json(readiness_payload),
    )


def write_synthetic_token_memmap(
    root: str | Path,
    *,
    vocab_size: int = 259,
    token_count: int = 20_000,
    seed: int = 7,
) -> TokenMemmapManifest:
    """Create a deterministic non-evidence corpus for packet smoke tests."""

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    tokens = rng.integers(0, vocab_size, size=token_count, dtype=dtype)
    token_path = destination / "synthetic_tokens.bin"
    tokens.tofile(token_path)
    document_manifest_path = destination / "synthetic_documents.json"
    document_manifest_path.write_text(
        '{"scientific_evidence":false,"documents":["synthetic-reference-only"]}\n',
        encoding="utf-8",
    )
    return TokenMemmap.validate_files(
        token_path=token_path,
        dtype=np.dtype(dtype).name,
        document_manifest_path=document_manifest_path,
        corpus_revision="SYNTHETIC_NON_EVIDENCE",
    )
