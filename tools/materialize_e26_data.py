#!/usr/bin/env python3
"""Validate already-materialized E26 scientific tokenizer/corpus inputs.

Despite the historical filename, this tool deliberately performs no network
access, document selection, tokenizer training, or tokenization.  It converts
two externally prepared, hash-pinned manifests into one readiness receipt only
after every byte/hash/range/cursor contract passes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import write_json_strict
from catena.lm.general_corpus import validate_scientific_data_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate external E26 16K tokenizer and fixed token memmap provenance"
    )
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cursor-seed", type=int, default=26_000)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--cursor-probe-sequences", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite readiness receipt: {output}")
    readiness = validate_scientific_data_bundle(
        tokenizer_manifest_path=args.tokenizer_manifest,
        corpus_manifest_path=args.corpus_manifest,
        cursor_seed=args.cursor_seed,
        sequence_length=args.sequence_length,
        cursor_probe_sequences=args.cursor_probe_sequences,
    )
    write_json_strict(output, readiness.as_dict())
    print(f"E26 scientific data input readiness: PASS ({readiness.readiness_sha256})")
    print(f"receipt: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
