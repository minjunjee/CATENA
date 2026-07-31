#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("e28a_locality_oracle_pareto", "configs/e28a_locality_oracle_pareto.yaml")
    )
