"""Deterministic replay-checked 16K ByteLevel BPE construction for E26."""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

from .data_lock import normalize_document

TOKENIZER_TRAINING_BYTE_LIMIT = 100_000_000
SPECIAL_TOKENS = {
    "pad": ("<pad>", 0),
    "bos": ("<bos>", 1),
    "eos": ("<eos>", 2),
    "doc": ("<doc>", 3),
    "unk": ("<unk>", 4),
}


class TokenizerBuildError(RuntimeError):
    """Raised when independent tokenizer construction is not byte reproducible."""


@dataclass(frozen=True, slots=True)
class TrainingChunk:
    content_sha256: str
    text: str
    utf8_bytes_used: int
    partial_final_document: bool


def _valid_utf8_prefix(data: bytes, limit: int) -> bytes:
    candidate = data[:limit]
    while candidate:
        try:
            candidate.decode("utf-8", errors="strict")
            return candidate
        except UnicodeDecodeError as error:
            candidate = candidate[: error.start]
    return b""


def select_tokenizer_training_chunks(
    documents: list[tuple[str, str]],
    *,
    byte_limit: int = TOKENIZER_TRAINING_BYTE_LIMIT,
) -> tuple[TrainingChunk, ...]:
    """Select content-hash-ordered whole documents plus one UTF-8-valid prefix."""

    if byte_limit < 1:
        raise ValueError("byte_limit must be positive")
    chunks: list[TrainingChunk] = []
    consumed = 0
    for content_sha256, raw_text in sorted(documents):
        normalized = normalize_document(raw_text)
        encoded = normalized.encode("utf-8")
        remaining = byte_limit - consumed
        if remaining <= 0:
            break
        if len(encoded) <= remaining:
            chunks.append(TrainingChunk(content_sha256, normalized, len(encoded), False))
            consumed += len(encoded)
            continue
        prefix = _valid_utf8_prefix(encoded, remaining)
        if prefix:
            text = prefix.decode("utf-8", errors="strict")
            chunks.append(TrainingChunk(content_sha256, text, len(prefix), True))
            consumed += len(prefix)
        break
    if not chunks:
        raise TokenizerBuildError("Tokenizer-only partition yielded no training text")
    return tuple(chunks)


def _load_tokenizers() -> Any:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    try:
        return importlib.import_module("tokenizers")
    except ModuleNotFoundError as error:
        raise TokenizerBuildError(
            "Tokenizer construction requires pinned tokenizers==0.23.1"
        ) from error


def _build_once(chunks: tuple[TrainingChunk, ...], destination: Path) -> dict[str, str]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite tokenizer replay directory: {destination}")
    destination.mkdir(parents=True)
    package = _load_tokenizers()
    models = importlib.import_module("tokenizers.models")
    normalizers = importlib.import_module("tokenizers.normalizers")
    pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
    decoders = importlib.import_module("tokenizers.decoders")
    trainers = importlib.import_module("tokenizers.trainers")

    tokenizer = package.Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=16_384,
        min_frequency=2,
        show_progress=False,
        special_tokens=[value[0] for value in SPECIAL_TOKENS.values()],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator((chunk.text for chunk in chunks), trainer=trainer)
    if tokenizer.get_vocab_size(with_added_tokens=True) != 16_384:
        raise TokenizerBuildError("Tokenizer trainer did not produce exactly 16,384 tokens")
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    for name, (token, expected_id) in SPECIAL_TOKENS.items():
        if vocabulary.get(token) != expected_id:
            raise TokenizerBuildError(
                f"Special token {name} expected id {expected_id}, got {vocabulary.get(token)}"
            )
    tokenizer_path = destination / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    model_files = tokenizer.model.save(str(destination))
    config = {
        "schema_version": "catena-e26-tokenizer-config-v1",
        "implementation": "huggingface_tokenizers",
        "model": "BPE",
        "vocab_size": 16_384,
        "normalizer": "NFC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "decoder": "ByteLevel",
        "min_frequency": 2,
        "threads": 1,
        "special_tokens": {
            name: {"token": token, "id": token_id}
            for name, (token, token_id) in SPECIAL_TOKENS.items()
        },
    }
    config_path = destination / "tokenizer_config.json"
    write_json_strict(config_path, config)
    paths = [tokenizer_path, config_path, *(Path(item) for item in model_files)]
    return {path.name: sha256_file(path) for path in sorted(paths)}


def _stress_tokenizer(tokenizer_path: Path) -> dict[str, Any]:
    package = _load_tokenizers()
    tokenizer = package.Tokenizer.from_file(str(tokenizer_path))
    cases = (
        "opaque-id-qz-00ff",
        '{"tool":"configure_api","route":"/v2/search"}',
        "https://example.test/a?x=1&y=two",
        "line one\n\tline two  \r\nline three",
        "NFC café — Ελληνικά — 日本語",
    )
    failures: list[str] = []
    maximum = -1
    for text in cases:
        normalized = normalize_document(text)
        encoding = tokenizer.encode(normalized, add_special_tokens=False)
        maximum = max(maximum, max(encoding.ids, default=-1))
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if decoded != normalized:
            failures.append(text)
    return {
        "cases": len(cases),
        "round_trip_failures": len(failures),
        "maximum_token_id": maximum,
        "pass": not failures and maximum < 16_384,
    }


def build_replayed_tokenizer(
    documents: list[tuple[str, str]],
    *,
    output_root: str | Path,
    source_revisions: list[str],
    byte_limit: int = TOKENIZER_TRAINING_BYTE_LIMIT,
) -> dict[str, Any]:
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite tokenizer output: {root}")
    root.mkdir(parents=True)
    chunks = select_tokenizer_training_chunks(documents, byte_limit=byte_limit)
    first_hashes = _build_once(chunks, root / "replay_a")
    second_hashes = _build_once(chunks, root / "replay_b")
    if first_hashes != second_hashes:
        raise TokenizerBuildError("Independent tokenizer artifact hashes differ")

    canonical = root / "canonical"
    canonical.mkdir()
    for name in sorted(first_hashes):
        shutil.copy2(root / "replay_a" / name, canonical / name)
    selection_records = [
        {
            "content_sha256": chunk.content_sha256,
            "utf8_bytes_used": chunk.utf8_bytes_used,
            "partial_final_document": chunk.partial_final_document,
        }
        for chunk in chunks
    ]
    training_manifest = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_TOKENIZER_TRAINING_DOCUMENTS",
        "selection_frozen": True,
        "synthetic": False,
        "reference_only": False,
        "document_count": len(chunks),
        "source_revisions": source_revisions,
        "training_byte_limit": byte_limit,
        "actual_training_bytes": sum(chunk.utf8_bytes_used for chunk in chunks),
        "training_byte_shortfall": byte_limit - sum(chunk.utf8_bytes_used for chunk in chunks),
        "final_document_policy": "MAXIMAL_UTF8_VALID_PREFIX_NOT_EXCEEDING_LIMIT",
        "document_selection_sha256": sha256_canonical_json(selection_records),
        "documents": selection_records,
    }
    training_path = canonical / "tokenizer_training_manifest.json"
    write_json_strict(training_path, training_manifest)
    tokenizer_path = canonical / "tokenizer.json"
    tokenizer_manifest = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_SCIENTIFIC_TOKENIZER",
        "evidence_tier": "SCIENTIFIC_INPUT",
        "scientific_main_eligible": True,
        "synthetic": False,
        "reference_only": False,
        "shared_across_variants": True,
        "trained_once": True,
        "tokenizer_id": "catena-e26-fineweb-edu-bytelevel-bpe-16k-v1",
        "tokenizer_type": "EXTERNAL_SCIENTIFIC_TOKENIZER",
        "tokenizer_family": "BPE",
        "vocab_size": 16_384,
        "model": {
            "path": tokenizer_path.name,
            "sha256": sha256_file(tokenizer_path),
            "bytes": tokenizer_path.stat().st_size,
        },
        "training_manifest": {
            "path": training_path.name,
            "sha256": sha256_file(training_path),
        },
        "special_tokens": {
            name: token_id for name, (_, token_id) in SPECIAL_TOKENS.items()
        },
    }
    tokenizer_manifest_path = canonical / "tokenizer_manifest.json"
    write_json_strict(tokenizer_manifest_path, tokenizer_manifest)
    stress = _stress_tokenizer(tokenizer_path)
    if not stress["pass"]:
        raise TokenizerBuildError("Tokenizer round-trip/OOV stress audit failed")
    receipt = {
        "schema_version": "catena-e26-tokenizer-replay-v1",
        "scientific_evidence": False,
        "replay_count": 2,
        "artifact_hashes": first_hashes,
        "artifact_hash_sets_identical": True,
        "tokenizer_manifest_path": str(tokenizer_manifest_path.resolve()),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
        "training_manifest_sha256": sha256_file(training_path),
        "stress_audit": stress,
    }
    receipt["replay_receipt_sha256"] = sha256_canonical_json(receipt)
    receipt_path = root / "tokenizer_replay_receipt.json"
    write_json_strict(receipt_path, receipt)
    return receipt
