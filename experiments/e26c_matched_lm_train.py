#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(run_entrypoint("e26c_matched_lm_train", "configs/e26c_main_train.yaml"))
