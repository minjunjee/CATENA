from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", default="artifacts_dry_run")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    commands = [
        ["e00_protocol_lock"],
        ["e01_local_controllability"],
        ["e01b_constrained_behavioral_reachability"],
        ["e02_magnitude_factorization"],
        ["e03_granularity_orientation"],
        ["e04_functional_mediation"],
        ["e05_semantic_demand_inference"],
        ["e06_reusable_state_assimilation"],
        ["e07_transformer_boundary", "--mode", "mock"],
        ["e08_claim_freeze"],
    ]
    for command in commands:
        full = [
            sys.executable,
            "-m",
            f"experiments.{command[0]}",
            "--dry-run",
            "--device",
            args.device,
            "--artifact-root",
            args.artifact_root,
            *command[1:],
        ]
        print("+", " ".join(full), flush=True)
        subprocess.run(full, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
