#!/usr/bin/env python3
"""Replay-build the locked E26 16K tokenizer from tokenizer-only documents."""

from __future__ import annotations

import argparse

from catena.core.provenance_v61 import read_json_object_strict
from catena.lm.data_lock import ContentSplit, SQLiteDocumentIndex
from catena.lm.tokenizer_builder import build_replayed_tokenizer


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-receipt", required=True)
    parser.add_argument("--document-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-byte-limit", type=int, default=100_000_000)
    args = parser.parse_args()
    revision = _source_revision(args.download_receipt)
    with SQLiteDocumentIndex(args.document_index, create=False) as index:
        documents = [
            (record.content_sha256, text)
            for record, text in index.texts(ContentSplit.TOKENIZER_ONLY)
        ]
    receipt = build_replayed_tokenizer(
        documents,
        output_root=args.output_root,
        source_revisions=[f"HuggingFaceFW/fineweb-edu@{revision}"],
        byte_limit=args.training_byte_limit,
    )
    print(f"E26 tokenizer replay: PASS ({receipt['replay_receipt_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
