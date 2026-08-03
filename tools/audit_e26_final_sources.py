#!/usr/bin/env python3
"""Create the E26 Final external provenance receipt without model download."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.lm.e26_final_provenance import (
    CHECKPOINT_REPO_ID,
    CHECKPOINT_REVISION,
    OFFICIAL_SOURCE,
    TOKENIZER_FILES,
    TOKENIZER_REPO_ID,
    TOKENIZER_REVISION,
    audit_checkpoint_metadata,
    audit_official_source,
    audit_tokenizer_metadata,
    build_final_provenance_receipt,
    fetch_json_metadata,
    fetch_small_bytes,
    git_ls_remote,
    hf_model_api_url,
    hf_resolve_url,
    write_final_provenance_receipt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact official source, checkpoint metadata, and TinyLlama tokenizer "
            "metadata/small files; never download the checkpoint"
        )
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/external/gdn2_official"),
    )
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout_seconds <= 0.0:
        raise ValueError("--timeout-seconds must be positive")
    remote_refs = git_ls_remote(OFFICIAL_SOURCE.remote_url)
    official = audit_official_source(args.official_repo, remote_refs=remote_refs)
    checkpoint_metadata = fetch_json_metadata(
        hf_model_api_url(CHECKPOINT_REPO_ID, CHECKPOINT_REVISION),
        timeout_seconds=args.timeout_seconds,
    )
    checkpoint, checkpoint_warnings = audit_checkpoint_metadata(
        checkpoint_metadata,
        local_checkpoint=args.checkpoint_file,
    )
    tokenizer_metadata = fetch_json_metadata(
        hf_model_api_url(TOKENIZER_REPO_ID, TOKENIZER_REVISION),
        timeout_seconds=args.timeout_seconds,
    )
    tokenizer_files = {
        expected.filename: fetch_small_bytes(
            hf_resolve_url(TOKENIZER_REPO_ID, TOKENIZER_REVISION, expected.filename),
            timeout_seconds=args.timeout_seconds,
        )
        for expected in TOKENIZER_FILES
    }
    tokenizer, tokenizer_warnings = audit_tokenizer_metadata(
        tokenizer_metadata,
        files=tokenizer_files,
    )
    receipt = build_final_provenance_receipt(
        official_source=official,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        warnings=[*checkpoint_warnings, *tokenizer_warnings],
    )
    write_final_provenance_receipt(args.output, receipt)
    print(f"E26 Final external provenance: {'PASS' if receipt['passed'] else 'BLOCKED'}")
    print(f"receipt: {args.output.expanduser().resolve()}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print(f"checkpoint_bytes_ready: {receipt['checkpoint_bytes_ready']}")
    print(f"warning_count: {receipt['warning_count']}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
