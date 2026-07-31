#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import read_json_object_strict, sha256_file

_SCHEMA_FILES = {
    "protocol_lock.json": "protocol_lock.schema.json",
    "data_manifest.json": "data_manifest.schema.json",
    "model_manifest.json": "model_manifest.schema.json",
    "backend_manifest.json": "backend_manifest.schema.json",
    "report.json": "report.schema.json",
}

_REQUIRED_FILES = (
    "run_manifest.json",
    "protocol_lock.json",
    "data_manifest.json",
    "model_manifest.json",
    "backend_manifest.json",
    "training_metrics.jsonl",
    "evaluation_metrics.jsonl",
    "seed_effects.jsonl",
    "report.json",
    "RESULTS_SUMMARY_KO.md",
    "artifact_index.json",
)


def _validate_jsonl(path: Path) -> int:
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows += 1
    return rows


def validate_run(run_dir: Path, schema_root: Path) -> dict[str, Any]:
    try:
        import jsonschema
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Run-schema validation requires jsonschema; install the validation "
            "extra or expose it through PYTHONPATH"
        ) from error

    missing = [name for name in _REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run lacks required artifacts: {missing}")
    for artifact_name, schema_name in _SCHEMA_FILES.items():
        payload = read_json_object_strict(run_dir / artifact_name)
        schema = read_json_object_strict(schema_root / schema_name)
        jsonschema.Draft202012Validator(schema).validate(payload)

    report = read_json_object_strict(run_dir / "report.json")
    manifest = read_json_object_strict(run_dir / "run_manifest.json")
    if report.get("run_id") != run_dir.name or manifest.get("run_id") != run_dir.name:
        raise ValueError("Run ID differs across path/report/manifest")
    if report.get("experiment") != run_dir.parent.name:
        raise ValueError("Experiment differs across path and report")
    if manifest.get("report_sha256") != sha256_file(run_dir / "report.json"):
        raise ValueError("run_manifest.report_sha256 mismatch")
    if manifest.get("source_fingerprint_verified_at_completion") is not True:
        raise ValueError("Source fingerprint was not verified at completion")

    index = read_json_object_strict(run_dir / "artifact_index.json")
    for relative, descriptor in index.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"Invalid artifact index descriptor: {relative}")
        path = run_dir / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Indexed artifact missing or symlinked: {relative}")
        if descriptor.get("sha256") != sha256_file(path):
            raise ValueError(f"Artifact hash mismatch: {relative}")
        if descriptor.get("bytes") != path.stat().st_size:
            raise ValueError(f"Artifact byte count mismatch: {relative}")

    row_counts = {
        name: _validate_jsonl(run_dir / name)
        for name in (
            "training_metrics.jsonl",
            "evaluation_metrics.jsonl",
            "seed_effects.jsonl",
        )
    }
    return {
        "schema_version": "catena-v8.1",
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "experiment": run_dir.parent.name,
        "report_sha256": sha256_file(run_dir / "report.json"),
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "row_counts": row_counts,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one completed E26+ run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=Path("schemas/v8_1"),
    )
    args = parser.parse_args()
    result = validate_run(
        args.run_dir.expanduser().resolve(strict=True),
        args.schema_root.expanduser().resolve(strict=True),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
