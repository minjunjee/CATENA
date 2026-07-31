#!/usr/bin/env python3
"""Build locked general train/validation/test uint16-LE token files."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    write_json_strict,
)
from catena.lm.data_lock import ContentSplit, LockedDocument, SQLiteDocumentIndex
from catena.lm.memmap_builder import MemmapInputDocument, build_general_memmap
from catena.lm.tokenizer import ExternalScientificTokenizer


def _source_revision(download_receipt: str) -> str:
    payload = read_json_object_strict(download_receipt)
    revision = payload.get("revision")
    shards = payload.get("shards")
    if not isinstance(revision, str) or not isinstance(shards, list) or not shards:
        raise ValueError("Malformed FineWeb download receipt")
    for row in shards:
        if not isinstance(row, dict) or row.get("verified") is not True:
            raise ValueError("FineWeb download receipt contains an unverified shard")
    return revision


def _documents(
    rows: Iterator[tuple[LockedDocument, str]],
) -> Iterator[MemmapInputDocument]:
    for record, text in rows:
        yield MemmapInputDocument(
            content_sha256=str(record.content_sha256),
            text=text,
            source_location=(
                f"{record.shard_path}:{record.row_group}:{record.row_index}"
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-receipt", required=True)
    parser.add_argument("--document-index", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-tokens", type=int, default=400_000_000)
    parser.add_argument("--validation-tokens", type=int, default=5_000_000)
    parser.add_argument("--test-tokens", type=int, default=5_000_000)
    args = parser.parse_args()
    root = Path(args.output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite memmap root: {root}")
    root.mkdir(parents=True)
    revision = _source_revision(args.download_receipt)
    tokenizer = ExternalScientificTokenizer.from_manifest(args.tokenizer_manifest)
    specifications = (
        (ContentSplit.GENERAL_TRAIN, args.train_tokens),
        (ContentSplit.GENERAL_VALIDATION, args.validation_tokens),
        (ContentSplit.GENERAL_TEST, args.test_tokens),
    )
    receipts: list[dict[str, object]] = []
    for split, required_tokens in specifications:
        with SQLiteDocumentIndex(args.document_index, create=False) as index:
            receipt = build_general_memmap(
                _documents(index.texts(split)),
                split=split.value,
                minimum_tokens=required_tokens,
                output_root=root / split.value,
                tokenizer_manifest_path=args.tokenizer_manifest,
                runtime_tokenizer=tokenizer,
                source_revisions=[f"HuggingFaceFW/fineweb-edu@{revision}"],
            )
        receipts.append(receipt)
    payload = {
        "schema_version": "catena-e26-general-memmaps-v1",
        "scientific_evidence": False,
        "dtype": "<u2",
        "shared_across_variants": True,
        "shared_across_runs": True,
        "splits": receipts,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    write_json_strict(root / "general_memmaps_receipt.json", payload)
    print(f"E26 general memmaps: PASS ({payload['receipt_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
