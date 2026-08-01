#!/usr/bin/env python3
"""Revalidate E26 zero-tolerance repaired inputs and readiness-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import read_json_object_strict, write_json_strict
from catena.lm.data_readiness_v3 import validate_zero_tolerance_data_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-lock", required=True)
    parser.add_argument("--repair-receipt", required=True)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-readiness")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only == (args.output is not None):
        parser.error("choose exactly one of --output or --check-only")
    readiness = validate_zero_tolerance_data_bundle(
        data_lock_path=args.data_lock,
        repair_receipt_path=args.repair_receipt,
        source_receipt_path=args.source_receipt,
    )
    if args.expected_readiness is not None:
        observed = read_json_object_strict(args.expected_readiness)
        if observed != readiness.as_dict():
            raise ValueError("Existing readiness-v3 differs from fresh validation")
    if args.output is not None:
        output = Path(args.output)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite readiness-v3: {output}")
        write_json_strict(output, readiness.as_dict())
    print(f"E26 zero-tolerance data readiness: PASS ({readiness.readiness_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
