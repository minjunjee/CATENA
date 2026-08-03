#!/usr/bin/env python3
"""Write the immutable E26 Final prospective lock after source commit."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import write_json_strict
from catena.lm.e26_final_protocol import build_protocol_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/e26_final_gdn2_1p3b_transactional_transfer.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage3d-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite protocol lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_protocol_receipt(
        config_path=args.config,
        repo_root=args.repo_root,
        stage3d_report=args.stage3d_report,
    )
    write_json_strict(output, receipt)
    print(output.resolve())
    print(receipt["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
