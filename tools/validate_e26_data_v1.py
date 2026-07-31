#!/usr/bin/env python3
"""Validate all Stage-2 scientific data dependencies and write one receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import read_json_object_strict, write_json_strict
from catena.lm.data_readiness_v2 import validate_stage2_data_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-lock", required=True)
    parser.add_argument("--construction-receipt", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--download-receipt", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--tokenizer-replay", required=True)
    parser.add_argument("--dedup-receipt", required=True)
    parser.add_argument("--near-duplicate-audit", required=True)
    parser.add_argument("--memmap-receipt", required=True)
    parser.add_argument("--transaction-manifest", required=True)
    parser.add_argument("--schedule-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--expected-readiness")
    args = parser.parse_args()
    if args.check_only == (args.output is not None):
        parser.error("choose exactly one of --output or --check-only")
    readiness = validate_stage2_data_bundle(
        data_lock_path=args.data_lock,
        construction_receipt_path=args.construction_receipt,
        source_inventory_path=args.source_inventory,
        source_metadata_path=args.source_metadata,
        download_receipt_path=args.download_receipt,
        tokenizer_manifest_path=args.tokenizer_manifest,
        tokenizer_replay_path=args.tokenizer_replay,
        dedup_receipt_path=args.dedup_receipt,
        near_duplicate_audit_path=args.near_duplicate_audit,
        memmap_receipt_path=args.memmap_receipt,
        transaction_manifest_path=args.transaction_manifest,
        schedule_manifest_path=args.schedule_manifest,
    )
    if args.expected_readiness is not None:
        observed = read_json_object_strict(args.expected_readiness)
        if observed != readiness.as_dict():
            raise ValueError("Existing readiness receipt differs from fresh validation")
    if args.output is not None:
        output = Path(args.output)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite Stage-2 readiness receipt: {output}"
            )
        write_json_strict(output, readiness.as_dict())
    print(f"E26 Stage-2 data readiness: PASS ({readiness.readiness_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
