from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from catena.core.config import load_config
from catena.core.io import (
    ensure_artifact_dir,
    environment_snapshot,
    file_sha256,
    write_json,
    write_latest_pointer,
)
from catena.core.provenance_v61 import (
    sha256_canonical_json,
    source_tree_fingerprint,
)
from catena.systems.device import resolve_device


def build_parser(experiment_id: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=experiment_id)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default=os.getenv("CATENA_ARTIFACT_ROOT", "artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def initialize_run(
    *,
    experiment_id: str,
    config_path: str,
    artifact_root: str,
    device_request: str,
    run_mode: str | None = None,
) -> tuple[dict[str, Any], Path, torch.device]:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    run_dir = ensure_artifact_dir(artifact_root, experiment_id)
    device = resolve_device(device_request)
    normalized_run_mode = "UNSPECIFIED" if run_mode is None else str(run_mode)
    if normalized_run_mode not in {"MAIN", "DRY_RUN", "UNSPECIFIED"}:
        raise ValueError(f"Unsupported run mode: {normalized_run_mode!r}")
    repo_root = Path(__file__).resolve().parents[1]
    source = source_tree_fingerprint(repo_root)
    resolved_config_text = yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=True,
    )
    (run_dir / "config.resolved.yaml").write_text(
        resolved_config_text,
        encoding="utf-8",
    )
    write_json(run_dir / "environment.json", environment_snapshot())
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": experiment_id,
            "run_id": run_dir.name,
            "run_mode": normalized_run_mode,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_path": str(resolved_config_path),
            "config_file_sha256": file_sha256(resolved_config_path),
            "resolved_config_sha256": sha256_canonical_json(config),
            "resolved_config_artifact_sha256": file_sha256(
                run_dir / "config.resolved.yaml"
            ),
            "source_fingerprint": source.as_dict(),
            "source_fingerprint_phase": "RUN_START",
            "device": str(device),
            "config": config,
        },
    )
    return config, run_dir, device


def finalize_run(
    *,
    experiment_id: str,
    artifact_root: str,
    run_dir: Path,
    report: dict[str, Any],
) -> None:
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected JSON object: {manifest_path}")
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["report_sha256"] = file_sha256(report_path)
    write_json(manifest_path, manifest)
    write_latest_pointer(artifact_root, experiment_id, run_dir)
