#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catena.lm.stage2_protocol_lock import (
    Stage2ProtocolInputs,
    build_stage2_protocol_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable, acyclic E26a execution-source inventory and "
            "prospective Stage-2 protocol lock"
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock-utc", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--backend-candidate-lock", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--data-readiness", type=Path, required=True)
    parser.add_argument("--transaction-manifest", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--frozen-tree-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_stage2_protocol_lock(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        lock_utc=args.lock_utc,
        inputs=Stage2ProtocolInputs(
            config=args.config,
            calibration_config=args.calibration_config,
            backend_candidate_lock=args.backend_candidate_lock,
            tokenizer_manifest=args.tokenizer_manifest,
            corpus_manifest=args.corpus_manifest,
            data_lock=args.data_lock,
            data_readiness=args.data_readiness,
            transaction_manifest=args.transaction_manifest,
            schedule_manifest=args.schedule_manifest,
            frozen_tree_receipt=args.frozen_tree_receipt,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
