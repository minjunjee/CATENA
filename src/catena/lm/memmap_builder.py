"""Streaming little-endian uint16 memmap construction for frozen E26 inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

from .data_lock import normalize_document
from .tokenizer import SCIENTIFIC_VOCAB_SIZE, load_scientific_tokenizer_manifest


class RuntimeTokenizer(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Any: ...


class MemmapBuildError(RuntimeError):
    """Raised when a scientific token file violates its prospective contract."""


@dataclass(frozen=True, slots=True)
class MemmapInputDocument:
    content_sha256: str
    text: str
    source_location: str


def _encoding_ids(encoding: Any) -> list[int]:
    values = getattr(encoding, "ids", encoding)
    if not isinstance(values, (list, tuple)):
        raise MemmapBuildError("Tokenizer encode result does not expose a token id sequence")
    result = [int(value) for value in values]
    if any(value < 0 or value >= SCIENTIFIC_VOCAB_SIZE for value in result):
        raise MemmapBuildError("Tokenizer emitted an id outside [0,16384)")
    return result


def build_general_memmap(
    documents: Iterable[MemmapInputDocument],
    *,
    split: str,
    minimum_tokens: int,
    output_root: str | Path,
    tokenizer_manifest_path: str | Path,
    runtime_tokenizer: RuntimeTokenizer,
    source_revisions: list[str],
) -> dict[str, Any]:
    """Write whole documents in caller-supplied canonical order until capacity is met."""

    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be positive")
    tokenizer_manifest = load_scientific_tokenizer_manifest(tokenizer_manifest_path)
    doc_separator = tokenizer_manifest.special_tokens.get("doc")
    if doc_separator != 3:
        raise MemmapBuildError("Scientific tokenizer must use <doc> id 3")
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite memmap split directory: {root}")
    root.mkdir(parents=True)
    token_path = root / f"{split}.tokens.uint16le"
    index_path = root / f"{split}.documents.jsonl"
    digest = hashlib.sha256()
    minimum = SCIENTIFIC_VOCAB_SIZE
    maximum = -1
    token_count = 0
    selected: list[dict[str, Any]] = []
    with token_path.open("xb") as token_handle, index_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as index_handle:
        for document_index, document in enumerate(documents):
            ids = _encoding_ids(
                runtime_tokenizer.encode(
                    normalize_document(document.text),
                    add_bos=False,
                    add_eos=False,
                )
            )
            if document_index:
                ids.insert(0, doc_separator)
            if not ids:
                continue
            array = np.asarray(ids, dtype=np.dtype("<u2"))
            raw = array.tobytes(order="C")
            token_handle.write(raw)
            digest.update(raw)
            start = token_count
            token_count += len(ids)
            minimum = min(minimum, min(ids))
            maximum = max(maximum, max(ids))
            row = {
                "content_sha256": document.content_sha256,
                "source_location": document.source_location,
                "token_start": start,
                "token_end": token_count,
                "token_count": len(ids),
            }
            selected.append(row)
            index_handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
            if token_count >= minimum_tokens:
                break
        token_handle.flush()
        os.fsync(token_handle.fileno())
        index_handle.flush()
        os.fsync(index_handle.fileno())
    if token_count < minimum_tokens:
        raise MemmapBuildError(
            f"{split} yielded {token_count} tokens, below required {minimum_tokens}"
        )
    token_sha = digest.hexdigest()
    if token_sha != sha256_file(token_path):
        raise MemmapBuildError("Streaming token digest differs from final file digest")
    selection_sha = sha256_canonical_json(selected)
    manifest = {
        "schema_version": "catena-v8.1",
        "manifest_type": "E26_SCIENTIFIC_GENERAL_CORPUS",
        "evidence_tier": "SCIENTIFIC_INPUT",
        "scientific_main_eligible": True,
        "synthetic": False,
        "reference_only": False,
        "selection_frozen": True,
        "corpus_id": f"fineweb-edu-sample-10BT-{split}-v1",
        "corpus_revision": source_revisions[0],
        "source_revisions": source_revisions,
        "split": split,
        "tokenizer_manifest_sha256": tokenizer_manifest.manifest_hash,
        "tokenizer_model_sha256": tokenizer_manifest.model_sha256,
        "tokenizer_vocab_size": tokenizer_manifest.vocab_size,
        "document_manifest": {
            "path": index_path.name,
            "sha256": sha256_file(index_path),
            "document_count": len(selected),
            "document_selection_sha256": selection_sha,
        },
        "token_file": {
            "path": token_path.name,
            "sha256": token_sha,
            "bytes": token_path.stat().st_size,
            "dtype": "<u2",
            "byte_order": "little",
            "token_count": token_count,
            "token_id_min": minimum,
            "token_id_max": maximum,
            "minimum_required_tokens": minimum_tokens,
            "whole_document_overshoot_tokens": token_count - minimum_tokens,
        },
    }
    manifest_path = root / f"{split}.corpus_manifest.json"
    write_json_strict(manifest_path, manifest)
    return {
        "split": split,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "token_path": str(token_path.resolve()),
        "token_sha256": token_sha,
        "token_count": token_count,
        "document_count": len(selected),
        "document_selection_sha256": selection_sha,
    }
