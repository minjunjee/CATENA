#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from catena.lm.hashing import hash_mapping, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a prospective CATENA v8.1 protocol")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-fingerprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-question", required=True)
    parser.add_argument("--primary-estimand", required=True)
    parser.add_argument("--inference-unit", default="paired_training_seed")
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if payload["schema_version"] != "catena-v8.1":
        raise SystemExit("config schema_version must be catena-v8.1")
    registered_thresholds = {
        key: payload[key]
        for key in (
            "matching",
            "backend_gates",
            "data",
            "gate_population",
            "floor_gate",
            "throughput",
            "claim_gates",
            "mechanism_gates",
        )
        if key in payload
    }
    lock = {
        "schema_version": "catena-v8.1",
        "experiment": payload["experiment"],
        "stage": payload["stage"],
        "locked": True,
        "lock_utc": datetime.now(UTC).isoformat(),
        "source_hash": sha256_file(args.source_fingerprint),
        "config_hash": hash_mapping(payload),
        "primary_question": args.primary_question,
        "primary_estimand": args.primary_estimand,
        "inference_unit": args.inference_unit,
        "registered_dispositions": payload["registered_dispositions"],
        "thresholds": registered_thresholds,
        "full_config_snapshot": payload,
        "dependencies": payload.get("dependencies", [payload.get("dependency", {})]),
        "config_path": str(args.config.resolve()),
        "source_fingerprint_path": str(args.source_fingerprint.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
