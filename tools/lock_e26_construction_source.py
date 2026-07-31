#!/usr/bin/env python3
"""Bind built Stage-2 data artifacts to one exact committed builder tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.lm.construction_source import (
    REQUIRED_ARTIFACT_BINDINGS,
    write_construction_source_receipt,
)


def _binding(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("artifact binding must be LABEL=PATH")
    return label, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_binding,
        metavar="LABEL=PATH",
    )
    args = parser.parse_args()
    bindings = dict(args.artifact)
    if len(bindings) != len(args.artifact):
        parser.error("artifact labels must be unique")
    missing = sorted(set(REQUIRED_ARTIFACT_BINDINGS) - set(bindings))
    extra = sorted(set(bindings) - set(REQUIRED_ARTIFACT_BINDINGS))
    if missing or extra:
        parser.error(f"artifact binding mismatch; missing={missing}, extra={extra}")
    output = write_construction_source_receipt(
        args.output,
        repo_root=args.repo_root,
        artifact_bindings=bindings,
    )
    print(f"E26 construction source receipt: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
