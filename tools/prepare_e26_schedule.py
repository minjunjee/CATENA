#!/usr/bin/env python3
"""Create the paired 4:1 general/transaction schedule replay manifest."""

from __future__ import annotations

import argparse

from catena.lm.schedule_manifest import write_schedule_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-corpus-manifest", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=260_026)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--probe-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    output = write_schedule_manifest(
        args.output,
        train_corpus_manifest=args.train_corpus_manifest,
        tokenizer_manifest=args.tokenizer_manifest,
        seed=args.seed,
        sequence_length=args.sequence_length,
        probe_tokens=args.probe_tokens,
    )
    print(f"E26 paired schedule replay: PASS ({output.resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
