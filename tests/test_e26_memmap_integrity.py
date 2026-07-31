from pathlib import Path

import numpy as np

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict
from catena.lm.general_corpus import load_scientific_corpus_manifest
from catena.lm.memmap_builder import MemmapInputDocument, build_general_memmap
from catena.lm.tokenizer import load_scientific_tokenizer_manifest


class _Tokenizer:
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        assert not add_bos and not add_eos
        return [5 + ord(character) % 100 for character in text]


def _tokenizer_manifest(root: Path) -> Path:
    model = root / "tokenizer.json"
    model.write_text('{"fixture":true}\n', encoding="utf-8")
    training = root / "training.json"
    write_json_strict(
        training,
        {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_TOKENIZER_TRAINING_DOCUMENTS",
            "selection_frozen": True,
            "synthetic": False,
            "reference_only": False,
            "document_count": 1,
            "source_revisions": ["fixture@revision"],
            "document_selection_sha256": sha256_canonical_json(["fixture"]),
        },
    )
    manifest = root / "tokenizer_manifest.json"
    write_json_strict(
        manifest,
        {
            "schema_version": "catena-v8.1",
            "manifest_type": "E26_SCIENTIFIC_TOKENIZER",
            "evidence_tier": "SCIENTIFIC_INPUT",
            "scientific_main_eligible": True,
            "synthetic": False,
            "reference_only": False,
            "shared_across_variants": True,
            "trained_once": True,
            "tokenizer_id": "fixture-16k",
            "tokenizer_type": "EXTERNAL_SCIENTIFIC_TOKENIZER",
            "tokenizer_family": "BPE",
            "vocab_size": 16_384,
            "model": {
                "path": model.name,
                "sha256": sha256_file(model),
                "bytes": model.stat().st_size,
            },
            "training_manifest": {
                "path": training.name,
                "sha256": sha256_file(training),
            },
            "special_tokens": {"pad": 0, "bos": 1, "eos": 2, "doc": 3, "unk": 4},
        },
    )
    return manifest


def test_memmap_is_content_ordered_little_endian_and_whole_document(tmp_path: Path) -> None:
    tokenizer_path = _tokenizer_manifest(tmp_path)
    receipt = build_general_memmap(
        [
            MemmapInputDocument("0" * 64, "abc", "s:0:0"),
            MemmapInputDocument("1" * 64, "defgh", "s:0:1"),
        ],
        split="general_validation",
        minimum_tokens=6,
        output_root=tmp_path / "validation",
        tokenizer_manifest_path=tokenizer_path,
        runtime_tokenizer=_Tokenizer(),
        source_revisions=["fixture@revision"],
    )
    manifest = load_scientific_corpus_manifest(
        receipt["manifest_path"],
        tokenizer_manifest=load_scientific_tokenizer_manifest(tokenizer_path),
    )
    assert manifest.dtype == "<u2"
    assert manifest.token_count == 9
    values = np.fromfile(manifest.token_path, dtype="<u2")
    assert int(values[3]) == 3
    assert int(values.max()) < 16_384
