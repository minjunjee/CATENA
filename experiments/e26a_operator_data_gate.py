#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("e26a_operator_data_gate", "configs/e26a_operator_data_gate.yaml")
    )
