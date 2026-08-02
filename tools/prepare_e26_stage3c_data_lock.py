#!/usr/bin/env python3
"""Write the inherited V1 + zero-tolerance repaired-data Stage-3C lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.lm.stage3c_data_lock import write_stage3c_data_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-data-lock", required=True)
    parser.add_argument("--repair-protocol", required=True)
    parser.add_argument("--repair-receipt", required=True)
    parser.add_argument("--repair-source-receipt", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--expected-readiness-sha256", required=True)
    parser.add_argument("--stage3c-worktree", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = write_stage3c_data_lock(
        Path(args.output),
        parent_data_lock_path=args.parent_data_lock,
        repair_protocol_path=args.repair_protocol,
        repair_receipt_path=args.repair_receipt,
        repair_source_receipt_path=args.repair_source_receipt,
        readiness_path=args.readiness,
        expected_readiness_sha256=args.expected_readiness_sha256,
        stage3c_worktree=args.stage3c_worktree,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
