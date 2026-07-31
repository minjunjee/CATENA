#!/usr/bin/env python3
"""Freeze the v8.1 development transaction generator replay receipt."""

from __future__ import annotations

import argparse

from catena.lm.transaction_data import (
    default_stage2_transaction_spec,
    write_transaction_replay_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = write_transaction_replay_manifest(
        args.output,
        default_stage2_transaction_spec(),
    )
    print(f"E26 transaction replay: PASS ({output.resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
