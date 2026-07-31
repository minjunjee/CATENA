#!/usr/bin/env python3
"""Verify that every file tracked at the frozen pre-E26 commit is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    )


def _base_entries(root: Path, base_commit: str) -> list[tuple[str, str]]:
    output = _git(root, "ls-tree", "-r", base_commit)
    entries: list[tuple[str, str]] = []
    for line in output.splitlines():
        metadata, relative = line.split("\t", 1)
        mode, object_type, _object_id = metadata.split()
        if object_type != "blob":
            continue
        entries.append((mode, relative))
    return entries


def _base_blob(root: Path, base_commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{base_commit}:{relative}"],
        cwd=root,
        stderr=subprocess.PIPE,
    )


def _aggregate(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    base_commit = _git(root, "rev-parse", args.base_commit).strip()
    rows: list[dict[str, Any]] = []
    changed: list[str] = []
    missing: list[str] = []
    for mode, relative in _base_entries(root, base_commit):
        path = root / relative
        is_expected_symlink = mode == "120000"
        if is_expected_symlink:
            exists = path.is_symlink()
        else:
            exists = path.is_file() and not path.is_symlink()
        if not exists:
            missing.append(relative)
            continue
        base_bytes = _base_blob(root, base_commit, relative)
        if is_expected_symlink:
            current_digest = hashlib.sha256(os.readlink(path).encode()).hexdigest()
        else:
            current_digest = _sha256(path)
        base_digest = hashlib.sha256(base_bytes).hexdigest()
        if current_digest != base_digest:
            changed.append(relative)
        rows.append(
            {
                "path": relative,
                "bytes": len(base_bytes),
                "sha256": base_digest,
            }
        )
    payload = {
        "schema_version": 1,
        "scope": "all_files_tracked_at_pre_e26_base_commit",
        "root": str(root),
        "base_commit": base_commit,
        "expected_files": len(rows) + len(missing),
        "observed_files": len(rows),
        "base_aggregate_sha256": _aggregate(rows),
        "missing": missing,
        "changed": changed,
        "passed": not missing and not changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
