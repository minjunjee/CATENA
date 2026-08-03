from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import catena.lm.e26_final_tokenizer as final_tokenizer
from catena.core.provenance_v61 import read_json_object_strict, sha256_canonical_json
from catena.lm.e26_final_provenance import HfFileExpectation
from catena.lm.e26_final_tokenizer import (
    E26FinalSentencePieceTokenizer,
    E26FinalTokenizerError,
    E26FinalTokenizerExpectation,
    build_e26_final_tokenizer_receipt,
    validate_e26_final_tokenizer_receipt,
    write_e26_final_tokenizer_receipt,
)


def _write_payloads(root: Path) -> tuple[E26FinalTokenizerExpectation, dict[str, bytes]]:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"piece_{index}": index for index in range(3, 32_000)})
    payloads = {
        "tokenizer.json": json.dumps(
            {"model": {"type": "BPE", "vocab": vocab}}, sort_keys=True
        ).encode(),
        "tokenizer.model": b"locked-sentencepiece-fixture",
        "tokenizer_config.json": json.dumps(
            {
                "add_bos_token": True,
                "add_eos_token": False,
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
                "pad_token": None,
            },
            sort_keys=True,
        ).encode(),
        "special_tokens_map.json": json.dumps(
            {
                "bos_token": {"content": "<s>"},
                "eos_token": {"content": "</s>"},
                "unk_token": {"content": "<unk>"},
            },
            sort_keys=True,
        ).encode(),
        "config.json": json.dumps({"vocab_size": 32_000}, sort_keys=True).encode(),
        "generation_config.json": json.dumps({"pad_token_id": 0}, sort_keys=True).encode(),
    }
    root.mkdir()
    (root / ".cache").mkdir()
    expectations = []
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
        expectations.append(
            HfFileExpectation(
                filename=name,
                size=len(payload),
                blob_id=hashlib.sha1(name.encode()).hexdigest(),  # noqa: S324 - Git fixture
                sha256=hashlib.sha256(payload).hexdigest(),
                lfs=name == "tokenizer.model",
            )
        )
    return E26FinalTokenizerExpectation(files=tuple(expectations)), payloads


def test_exact_32k_tokenizer_lock_binds_eos_pad_and_document_separator(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    expectation, _payloads = _write_payloads(root)
    receipt = build_e26_final_tokenizer_receipt(root, expectation=expectation)

    assert receipt["passed"] is True
    assert receipt["vocab_size"] == 32_000
    assert receipt["special_token_ids"] == {"unk": 0, "bos": 1, "eos": 2, "pad": 2}
    assert receipt["document_separator_id"] == 2
    assert receipt["document_separator_policy"] == "EOS_AFTER_EACH_DOCUMENT"
    assert all(receipt["hard_checks"].values())
    assert validate_e26_final_tokenizer_receipt(receipt, expectation=expectation) == receipt

    output = tmp_path / "tokenizer-lock.json"
    write_e26_final_tokenizer_receipt(output, receipt, expectation=expectation)
    assert read_json_object_strict(output) == receipt
    with pytest.raises(FileExistsError):
        write_e26_final_tokenizer_receipt(output, receipt, expectation=expectation)


def test_tokenizer_lock_rejects_changed_bytes_population_and_rehashed_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    expectation, _payloads = _write_payloads(root)
    receipt = build_e26_final_tokenizer_receipt(root, expectation=expectation)

    changed = bytearray((root / "tokenizer.model").read_bytes())
    changed[-1] ^= 1
    (root / "tokenizer.model").write_bytes(changed)
    with pytest.raises(E26FinalTokenizerError, match="admission checks failed"):
        build_e26_final_tokenizer_receipt(root, expectation=expectation)
    (root / "tokenizer.model").write_bytes(b"locked-sentencepiece-fixture")

    (root / "unexpected.txt").write_text("ambiguous runtime input", encoding="utf-8")
    with pytest.raises(E26FinalTokenizerError, match="file population differs"):
        build_e26_final_tokenizer_receipt(root, expectation=expectation)
    (root / "unexpected.txt").unlink()

    tampered = dict(receipt)
    tampered["document_separator_id"] = 0
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = sha256_canonical_json(tampered)
    with pytest.raises(E26FinalTokenizerError, match="document_separator_id"):
        validate_e26_final_tokenizer_receipt(tampered, expectation=expectation)


def test_sentencepiece_runtime_uses_only_locked_model_and_explicit_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tokenizer"
    expectation, _payloads = _write_payloads(root)
    receipt = build_e26_final_tokenizer_receipt(root, expectation=expectation)

    class _Processor:
        def __init__(self, *, model_file: str) -> None:
            assert model_file == str(root / "tokenizer.model")

        @staticmethod
        def vocab_size() -> int:
            return 32_000

        @staticmethod
        def unk_id() -> int:
            return 0

        @staticmethod
        def bos_id() -> int:
            return 1

        @staticmethod
        def eos_id() -> int:
            return 2

        @staticmethod
        def encode(
            text: str | list[str],
            *,
            out_type: type[int],
            num_threads: int | None = None,
        ) -> list[int] | list[list[int]]:
            assert out_type is int
            if isinstance(text, list):
                assert num_threads == 3
                return [[10, 11] if value else [] for value in text]
            return [10, 11] if text else []

    original_import = final_tokenizer.importlib.import_module

    def _import(name: str) -> Any:
        if name == "sentencepiece":
            return SimpleNamespace(SentencePieceProcessor=_Processor)
        return original_import(name)

    monkeypatch.setattr(final_tokenizer.importlib, "import_module", _import)
    runtime = E26FinalSentencePieceTokenizer(receipt, expectation=expectation)
    assert runtime.pad_token_id == runtime.eos_token_id == runtime.document_separator_id == 2
    assert runtime.encode("text", add_bos=True, add_eos=True) == [1, 10, 11, 2]
    assert runtime.encode_batch(["text", ""], num_threads=3) == [[10, 11], []]
