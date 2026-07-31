#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("e29b_quality_cost_regime", "configs/e29b_quality_cost_regime.yaml")
    )
