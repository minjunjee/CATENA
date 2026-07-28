#!/usr/bin/env python3
"""Snapshot and verify immutable CATENA E00--E21 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EXPERIMENT = re.compile(r"^e(?P<number>\d{2})(?:[a-z]|_)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included_top_level(path: Path) -> bool:
    name = path.name
    return bool(
        path.is_file()
        and (
            (name.startswith("E") and name.endswith(".json"))
            or name
            in {
                "WORKFLOW_E00_E02_RESULTS_KO.md",
                "POSTCORE_E10_E16_RESULTS_SUMMARY_INDEX_KO.md",
                "POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md",
            }
        )
    )


def _included_experiment_directory(path: Path) -> bool:
    match = _EXPERIMENT.match(path.name)
    return bool(match and int(match.group("number")) <= 21)


def inventory(artifact_root: Path) -> dict[str, Any]:
    root = artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if _included_top_level(child):
            paths.append(child)
        elif child.is_dir() and not child.is_symlink() and _included_experiment_directory(child):
            paths.extend(
                path
                for path in sorted(child.rglob("*"))
                if path.is_file() and not path.is_symlink()
            )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update((f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n").encode())
    return {
        "schema_version": 1,
        "scope": "immutable_E00_through_E21_artifacts",
        "artifact_root": str(root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count": len(rows),
        "total_bytes": sum(path.stat().st_size for path in paths),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError(f"invalid immutable-artifact manifest: {path}")
    return payload


def compare(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    expected_rows = {str(row["path"]): row for row in expected["files"] if isinstance(row, dict)}
    observed_rows = {str(row["path"]): row for row in observed["files"] if isinstance(row, dict)}
    missing = sorted(set(expected_rows) - set(observed_rows))
    unexpected = sorted(set(observed_rows) - set(expected_rows))
    changed = sorted(
        path
        for path in set(expected_rows) & set(observed_rows)
        if (
            expected_rows[path].get("sha256") != observed_rows[path].get("sha256")
            or expected_rows[path].get("bytes") != observed_rows[path].get("bytes")
        )
    )
    passed = not missing and not unexpected and not changed
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "expected_file_count": len(expected_rows),
        "observed_file_count": len(observed_rows),
        "expected_aggregate_sha256": expected.get("aggregate_sha256"),
        "observed_aggregate_sha256": observed.get("aggregate_sha256"),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Post-E21 이전 artifact hash 검증",
        "",
        f"- Status: `{result['status']}`",
        f"- Expected files: `{result['expected_file_count']}`",
        f"- Observed files: `{result['observed_file_count']}`",
        f"- Expected aggregate: `{result['expected_aggregate_sha256']}`",
        f"- Observed aggregate: `{result['observed_aggregate_sha256']}`",
        f"- Missing: `{len(result['missing'])}`",
        f"- Unexpected: `{len(result['unexpected'])}`",
        f"- Changed: `{len(result['changed'])}`",
        "",
        "Scope는 canonical artifact root의 E00–E21 experiment directory와 "
        "top-level freeze/status JSON 및 기존 summary index다.",
        "",
    ]
    for label in ("missing", "unexpected", "changed"):
        if result[label]:
            lines.extend(
                [
                    f"## {label}",
                    "",
                    *(f"- `{item}`" for item in result[label]),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/artifacts"),
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    args = parser.parse_args()
    if (args.snapshot is None) == (args.baseline is None):
        parser.error("choose exactly one of --snapshot or --baseline")
    observed = inventory(args.artifact_root)
    if args.snapshot is not None:
        _write_json(args.snapshot, observed)
        print(
            f"[SNAPSHOT] {observed['file_count']} files "
            f"{observed['aggregate_sha256']} -> {args.snapshot}"
        )
        return 0
    expected = _read_manifest(args.baseline)
    result = compare(expected, observed)
    if args.report_json is not None:
        _write_json(args.report_json, result)
    if args.report_md is not None:
        _write_markdown(args.report_md, result)
    print(
        f"[{result['status']}] expected={result['expected_file_count']} "
        f"observed={result['observed_file_count']} "
        f"changed={len(result['changed'])}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
