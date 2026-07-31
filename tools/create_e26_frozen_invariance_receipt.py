#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from catena.lm.frozen_invariance import write_frozen_invariance_receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the structured live-HEAD, 556-source, and completed E00-E25 artifact "
            "invariance receipt required before E26a"
        )
    )
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_lock = yaml.safe_load(args.data_lock.read_text(encoding="utf-8"))
    if not isinstance(data_lock, dict):
        parser.error("--data-lock must contain a YAML mapping")
    output = write_frozen_invariance_receipt(
        args.output,
        data_lock=data_lock,
        baseline_manifest=args.baseline_manifest,
    )
    print(
        json.dumps(
            {
                "path": str(output.resolve(strict=True)),
                "scientific_evidence": False,
                "scientific_e26a_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
