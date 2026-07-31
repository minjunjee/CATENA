#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
        raise TypeError("JSON manifest must be a mapping of relative path to SHA-256")
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        records[relative.lstrip("* ")] = digest
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a frozen CATENA file hash manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    expected = load_manifest(args.manifest)
    mismatches = []
    missing = []
    for relative, digest in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed = sha256(path)
        if observed != digest:
            mismatches.append({"path": relative, "expected": digest, "observed": observed})
    report = {
        "root": str(root),
        "manifest": str(args.manifest.resolve()),
        "files_expected": len(expected),
        "missing": missing,
        "mismatches": mismatches,
        "passed": not missing and not mismatches,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
