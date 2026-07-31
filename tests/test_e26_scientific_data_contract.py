from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict
from catena.lm.general_corpus import (
    ScientificCorpusContractError,
    TokenMemmap,
    validate_scientific_data_bundle,
    write_synthetic_token_memmap,
)
from catena.lm.tokenizer import (
    ExternalScientificTokenizer,
    ScientificTokenizerContractError,
    TokenizerManifest,
    load_scientific_tokenizer_manifest,
)


def _materialized_inputs(root: Path) -> tuple[Path, Path]:
    model_path = root / "tokenizer.model"
    model_path.write_bytes(b"external-bpe-model-for-contract-test")
    tokenizer_training = root / "tokenizer_training_documents.json"
    write_json_strict(
        tokenizer_training,
        {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_TOKENIZER_TRAINING_DOCUMENTS",
            "selection_frozen": True,
            "synthetic": False,
            "reference_only": False,
            "document_count": 3,
            "source_revisions": ["fixture-corpus@0123456"],
            "document_selection_sha256": sha256_canonical_json(["doc-a", "doc-b", "doc-c"]),
        },
    )
    tokenizer_manifest = root / "tokenizer_manifest.json"
    write_json_strict(
        tokenizer_manifest,
        {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_SCIENTIFIC_TOKENIZER",
            "evidence_tier": "SCIENTIFIC_INPUT",
            "scientific_main_eligible": True,
            "synthetic": False,
            "reference_only": False,
            "shared_across_variants": True,
            "trained_once": True,
            "tokenizer_id": "fixture-bpe-16k",
            "tokenizer_type": "EXTERNAL_SCIENTIFIC_TOKENIZER",
            "tokenizer_family": "BPE",
            "vocab_size": 16_384,
            "model": {
                "path": model_path.name,
                "sha256": sha256_file(model_path),
                "bytes": model_path.stat().st_size,
            },
            "training_manifest": {
                "path": tokenizer_training.name,
                "sha256": sha256_file(tokenizer_training),
            },
            "special_tokens": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        },
    )

    document_manifest = root / "general_documents.jsonl"
    document_manifest.write_text(
        "\n".join(json.dumps({"id": value}) for value in ("g-a", "g-b", "g-c")) + "\n",
        encoding="utf-8",
    )
    token_path = root / "general_tokens.uint16"
    tokens = (np.arange(256, dtype=np.uint16) * 61) % 16_384
    tokens.tofile(token_path)
    corpus_manifest = root / "corpus_manifest.json"
    write_json_strict(
        corpus_manifest,
        {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_SCIENTIFIC_GENERAL_CORPUS",
            "evidence_tier": "SCIENTIFIC_INPUT",
            "scientific_main_eligible": True,
            "synthetic": False,
            "reference_only": False,
            "selection_frozen": True,
            "corpus_id": "fixture-general-corpus",
            "corpus_revision": "fixture@0123456",
            "source_revisions": ["fixture-source@0123456"],
            "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest),
            "tokenizer_model_sha256": sha256_file(model_path),
            "tokenizer_vocab_size": 16_384,
            "document_manifest": {
                "path": document_manifest.name,
                "sha256": sha256_file(document_manifest),
                "document_count": 3,
                "document_selection_sha256": sha256_canonical_json(["g-a", "g-b", "g-c"]),
            },
            "token_file": {
                "path": token_path.name,
                "sha256": sha256_file(token_path),
                "bytes": token_path.stat().st_size,
                "dtype": "uint16",
                "token_count": int(tokens.size),
                "token_id_min": int(tokens.min()),
                "token_id_max": int(tokens.max()),
            },
        },
    )
    return tokenizer_manifest, corpus_manifest


def test_scientific_bundle_pins_bytes_and_paired_cursor(tmp_path: Path) -> None:
    tokenizer_path, corpus_path = _materialized_inputs(tmp_path)
    readiness = validate_scientific_data_bundle(
        tokenizer_manifest_path=tokenizer_path,
        corpus_manifest_path=corpus_path,
        sequence_length=16,
        cursor_probe_sequences=3,
    )
    assert readiness.tokenizer.vocab_size == 16_384
    assert readiness.tokenizer.scientific_main_eligible
    assert readiness.corpus.scientific_main_eligible
    assert readiness.cursor_probe.tokens == 48
    assert len(readiness.readiness_sha256) == 64
    assert readiness.readiness_sha256 == sha256_canonical_json(readiness.payload_without_hash())
    receipt_path = tmp_path / "readiness.json"
    write_json_strict(receipt_path, readiness.as_dict())
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["scientific_main_input_eligible"]

    corpus = TokenMemmap(readiness.corpus, scientific_main=True)
    tied = corpus.paired_cursor(seed=26011, sequence_length=24)
    dual = corpus.paired_cursor(seed=26011, sequence_length=24)
    tied_rows, tied_receipt = tied.take(4)
    dual_rows, dual_receipt = dual.take(4)
    assert tied_receipt == dual_receipt
    assert all(
        np.array_equal(left, right) for left, right in zip(tied_rows, dual_rows, strict=True)
    )
    resumed = tied.fork()
    tied_next, tied_next_receipt = tied.take(2)
    resumed_next, resumed_receipt = resumed.take(2)
    assert tied_next_receipt == resumed_receipt
    assert all(
        np.array_equal(left, right) for left, right in zip(tied_next, resumed_next, strict=True)
    )


def test_tokenizer_main_rejects_non_16k_and_reference_flags(tmp_path: Path) -> None:
    tokenizer_path, _ = _materialized_inputs(tmp_path)
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    payload["vocab_size"] = 259
    write_json_strict(tokenizer_path, payload)
    with pytest.raises(ScientificTokenizerContractError, match="vocab_size"):
        load_scientific_tokenizer_manifest(tokenizer_path)

    payload["vocab_size"] = 16_384
    payload["reference_only"] = True
    write_json_strict(tokenizer_path, payload)
    with pytest.raises(ScientificTokenizerContractError, match="reference_only"):
        load_scientific_tokenizer_manifest(tokenizer_path)


def test_scientific_corpus_detects_token_or_manifest_substitution(tmp_path: Path) -> None:
    tokenizer_path, corpus_path = _materialized_inputs(tmp_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    token_path = tmp_path / corpus_payload["token_file"]["path"]
    with token_path.open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(ScientificCorpusContractError, match="byte count"):
        validate_scientific_data_bundle(
            tokenizer_manifest_path=tokenizer_path,
            corpus_manifest_path=corpus_path,
            sequence_length=8,
        )


def test_synthetic_smoke_manifest_cannot_be_promoted_to_main(tmp_path: Path) -> None:
    manifest = write_synthetic_token_memmap(tmp_path / "synthetic", token_count=64)
    with pytest.raises(ScientificCorpusContractError, match="not eligible"):
        TokenMemmap(manifest, scientific_main=True)


class _FakeEncoding:
    ids = [7, 8]


class _FakeRuntimeTokenizer:
    def get_vocab_size(self, *, with_added_tokens: bool) -> int:
        assert with_added_tokens
        return 16_384

    def encode(self, text: str, *, add_special_tokens: bool) -> _FakeEncoding:
        assert text
        assert not add_special_tokens
        return _FakeEncoding()

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return ",".join(str(value) for value in token_ids)


def test_external_scientific_tokenizer_runtime_adapter() -> None:
    manifest = TokenizerManifest(
        tokenizer_id="test-16k",
        tokenizer_type="EXTERNAL_SCIENTIFIC_TOKENIZER",
        tokenizer_family="BPE",
        vocab_size=16_384,
        model_path="/unused/tokenizer.json",
        model_sha256="0" * 64,
        training_manifest_path="/unused/training.json",
        training_manifest_sha256="1" * 64,
        training_document_selection_sha256="2" * 64,
        manifest_path="/unused/manifest.json",
        manifest_sha256="3" * 64,
        special_tokens={"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        evidence_tier="SCIENTIFIC_INPUT",
        scientific_main_eligible=True,
        synthetic=False,
        reference_only=False,
    )
    tokenizer = ExternalScientificTokenizer(manifest, _FakeRuntimeTokenizer())
    assert tokenizer.encode("x", add_bos=True, add_eos=True) == [1, 7, 8, 2]
    assert tokenizer.decode([7, 8]) == "7,8"
