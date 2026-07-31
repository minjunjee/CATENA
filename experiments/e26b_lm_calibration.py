#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(run_entrypoint("e26b_lm_calibration", "configs/e26b_calibration_lock.yaml"))
