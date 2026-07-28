#!/usr/bin/env python3
"""Convenience dispatcher for the canonical module entry points."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "e00": "e00_protocol_lock",
    "e01": "e01_local_controllability",
    "e01b": "e01b_constrained_behavioral_reachability",
    "e02": "e02_magnitude_factorization",
    "e03": "e03_granularity_orientation",
    "e04": "e04_functional_mediation",
    "e05": "e05_semantic_demand_inference",
    "e06": "e06_reusable_state_assimilation",
    "e07": "e07_transformer_boundary",
    "e08": "e08_claim_freeze",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    name = EXPERIMENTS[parsed.experiment]
    command = [
        sys.executable,
        "-m",
        f"experiments.{name}",
        "--config",
        str(ROOT / "configs" / f"{name}.yaml"),
        *parsed.args,
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
