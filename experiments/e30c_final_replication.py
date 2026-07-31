#!/usr/bin/env python3
from catena.lm.experiment_driver import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(run_entrypoint("e30c_final_replication", "configs/e30c_replication.yaml"))
