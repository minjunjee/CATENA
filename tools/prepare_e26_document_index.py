#!/usr/bin/env python3
"""Build the Stage-2 NFC/content-hash split and exact-dedup index."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import read_json_object_strict, sha256_file, write_json_strict
from catena.lm.data_lock import build_document_index
from catena.lm.parquet_documents import iter_parquet_documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-receipt", required=True)
    parser.add_argument("--sqlite-output", required=True)
    parser.add_argument("--document-manifest-output", required=True)
    parser.add_argument("--dedup-receipt-output", required=True)
    args = parser.parse_args()
    source = read_json_object_strict(args.download_receipt)
    shards = source.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Download receipt has no verified shards")
    paths: list[Path] = []
    for row in shards:
        if not isinstance(row, dict) or row.get("verified") is not True:
            raise ValueError("Download receipt contains an unverified shard")
        path = Path(str(row["local_path"])).resolve(strict=True)
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get(
            "local_sha256"
        ):
            raise ValueError(f"Pinned shard changed after download: {path}")
        paths.append(path)
    receipt = build_document_index(
        iter_parquet_documents(paths),
        sqlite_path=args.sqlite_output,
        manifest_path=args.document_manifest_output,
    )
    output = Path(args.dedup_receipt_output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite dedup receipt: {output}")
    write_json_strict(output, receipt.as_dict())
    print(
        f"E26 content lock: {receipt.unique_documents} unique / "
        f"{receipt.exact_duplicates} exact duplicates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
