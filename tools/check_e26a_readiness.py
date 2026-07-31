#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import write_json_strict
from catena.lm.readiness import validate_e26a_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation of all frozen E26a execution inputs"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--backend-manifest", required=True)
    parser.add_argument("--tokenizer-manifest", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite readiness receipt: {output}")
    readiness = validate_e26a_readiness(
        repo_root=args.repo_root,
        config_path=args.config,
        protocol_lock_path=args.protocol_lock,
        backend_manifest_path=args.backend_manifest,
        tokenizer_manifest_path=args.tokenizer_manifest,
        corpus_manifest_path=args.corpus_manifest,
    )
    write_json_strict(output, readiness.as_dict())
    print(f"E26a readiness: PASS ({readiness.readiness_sha256})")
    print(f"receipt: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
