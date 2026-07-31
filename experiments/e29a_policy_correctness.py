#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("e29a_policy_correctness", "configs/e29a_policy_correctness.yaml")
    )
