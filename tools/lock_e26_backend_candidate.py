#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from catena.core.provenance_v61 import write_json_strict
from catena.lm.backend_lock import backend_candidate_lock_payload


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the immutable pre-audit E26 backend candidate lock"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/home/minjun_dev/CATENA_E26"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve(strict=True)
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Backend candidate lock requires a clean committed worktree")
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite backend candidate lock: {output}")
    resolved_output = output.parent.resolve(strict=True) / output.name
    if resolved_output == repo or repo in resolved_output.parents:
        raise ValueError("Backend candidate lock must live outside the execution-source worktree")
    config_path = args.config.expanduser().resolve(strict=True)
    config = _yaml_mapping(config_path)
    candidates = config.get("model_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(value, dict) for value in candidates)
    ):
        raise ValueError("E26a config lacks valid model_candidates")
    payload = backend_candidate_lock_payload(
        repo_root=repo,
        config_path=config_path,
        candidates=candidates,
    )
    write_json_strict(resolved_output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
