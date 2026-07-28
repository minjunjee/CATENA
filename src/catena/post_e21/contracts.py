"""Artifact and protocol contracts shared by E22--E25.

The helpers in this module are deliberately additive.  They wrap the existing
E18--E21 run writer without changing any frozen experiment implementation.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.io import file_sha256, write_json, write_jsonl
from catena.core.provenance_v61 import sha256_canonical_json


class PostE21ContractError(RuntimeError):
    """Raised when a Post-E21 protocol or artifact contract is violated."""


@dataclass(frozen=True, slots=True)
class ProtocolSnapshot:
    """Validated protocol-lock metadata copied into one immutable run."""

    path: Path
    sha256: str
    config_sha256: str
    payload: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PostE21ContractError(f"Protocol lock is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostE21ContractError(f"Cannot read protocol lock {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PostE21ContractError(f"Protocol lock must be a JSON object: {path}")
    return payload


def validate_protocol_lock(
    *,
    lock_path: str | Path,
    config_path: str | Path,
    experiment_id: str,
    repo_root: str | Path,
) -> ProtocolSnapshot:
    """Validate a prospective lock and every declared immutable file hash."""

    root = Path(repo_root).resolve(strict=True)
    lock = Path(lock_path).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    try:
        lock.relative_to(root)
        config.relative_to(root)
    except ValueError as error:
        raise PostE21ContractError("Protocol/config path escapes repository root") from error

    payload = _read_json_object(lock)
    if payload.get("schema_version") != 1:
        raise PostE21ContractError("Post-E21 protocol lock schema must be version 1")
    if payload.get("experiment_id") != experiment_id:
        raise PostE21ContractError(
            f"Protocol experiment mismatch: {payload.get('experiment_id')!r} != {experiment_id!r}"
        )
    if payload.get("protocol_frozen_before_main") is not True:
        raise PostE21ContractError("Protocol was not certified frozen before main evaluation")
    if payload.get("main_execution_started") is not False:
        raise PostE21ContractError("Static repository protocol lock must precede main execution")

    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise PostE21ContractError("Protocol lock lacks a non-empty file hash map")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise PostE21ContractError("Protocol file records must map strings to SHA-256 strings")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise PostE21ContractError(f"Locked path escapes repository: {relative}") from error
        if candidate.is_symlink() or not candidate.is_file():
            raise PostE21ContractError(f"Locked path is missing or unsafe: {relative}")
        if file_sha256(candidate) != expected:
            raise PostE21ContractError(f"Locked file changed: {relative}")

    relative_config = config.relative_to(root).as_posix()
    config_sha256 = file_sha256(config)
    if files.get(relative_config) != config_sha256:
        raise PostE21ContractError("Selected config is absent from, or differs from, its lock")
    return ProtocolSnapshot(
        path=lock,
        sha256=file_sha256(lock),
        config_sha256=config_sha256,
        payload=payload,
    )


def copy_protocol_snapshot(*, snapshot: ProtocolSnapshot, run_dir: str | Path) -> Path:
    """Copy the exact protocol bytes into a newly created run directory."""

    destination = Path(run_dir) / "protocol_lock.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite protocol snapshot: {destination}")
    shutil.copyfile(snapshot.path, destination)
    if file_sha256(destination) != snapshot.sha256:
        raise PostE21ContractError("Copied protocol lock does not match its source hash")
    return destination


def write_required_rows(
    *,
    run_dir: str | Path,
    raw_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    raw_filename: str = "raw_metrics.jsonl",
    seed_filename: str = "seed_metrics.jsonl",
) -> dict[str, Any]:
    """Write both required JSONL layers and return integrity descriptors."""

    directory = Path(run_dir)
    raw_path = directory / raw_filename
    seed_path = directory / seed_filename
    write_jsonl(raw_path, [dict(row) for row in raw_rows])
    write_jsonl(seed_path, [dict(row) for row in seed_rows])
    return {
        "raw": {
            "path": str(raw_path.resolve()),
            "rows": len(raw_rows),
            "sha256": file_sha256(raw_path),
        },
        "seed": {
            "path": str(seed_path.resolve()),
            "rows": len(seed_rows),
            "sha256": file_sha256(seed_path),
        },
    }


def write_data_manifest(
    *,
    run_dir: str | Path,
    payload: Mapping[str, Any],
    filename: str = "data_manifest.json",
) -> tuple[Path, str]:
    """Write and hash the outcome-independent data/namespace declaration."""

    manifest = dict(payload)
    data_sha256 = sha256_canonical_json(manifest)
    manifest["data_sha256"] = data_sha256
    path = Path(run_dir) / filename
    write_json(path, manifest)
    return path, data_sha256


def combined_checkpoint_sha256(checkpoint_hashes: Mapping[str, str]) -> str | None:
    """Return a deterministic aggregate hash, or ``None`` for non-training runs."""

    if not checkpoint_hashes:
        return None
    for name, value in checkpoint_hashes.items():
        if (
            not name
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PostE21ContractError(f"Invalid checkpoint hash for {name!r}: {value!r}")
    return sha256_canonical_json(dict(sorted(checkpoint_hashes.items())))


def report_contract_metadata(
    *,
    run_dir: str | Path,
    snapshot: ProtocolSnapshot,
    data_sha256: str,
    checkpoint_hashes: Mapping[str, str],
    evidence_tier: str,
    claim_eligible: bool,
) -> dict[str, Any]:
    """Build the mandatory common metadata included in every E22--E25 report."""

    manifest_path = Path(run_dir) / "run_manifest.json"
    manifest = _read_json_object(manifest_path)
    source = manifest.get("source_fingerprint")
    if not isinstance(source, dict):
        raise PostE21ContractError("Run manifest lacks source_fingerprint")
    if len(data_sha256) != 64:
        raise PostE21ContractError("data_sha256 must be a SHA-256 hex digest")
    combined = combined_checkpoint_sha256(checkpoint_hashes)
    return {
        "evidence_tier": evidence_tier,
        "claim_eligible": bool(claim_eligible),
        "scientific_evidence": bool(claim_eligible and evidence_tier == "OFFICIAL_OPERATOR"),
        "source_fingerprint": dict(source),
        "config_sha256": snapshot.config_sha256,
        "data_sha256": data_sha256,
        "checkpoint_sha256": combined,
        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
        "protocol_lock": {
            "source_path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "run_snapshot_path": str((Path(run_dir) / "protocol_lock.json").resolve()),
        },
    }
