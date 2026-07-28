"""Reference-only interface mock for e23_product_poset_sequence.

Do not copy blindly. Integrate into the live repository using the same runner,
manifest, artifact, and config conventions as E18--E21.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Contract mock only: Codex must wire this entry point to the existing "
        "E18--E21 runner and artifact infrastructure."
    )


if __name__ == "__main__":
    raise SystemExit(main())
