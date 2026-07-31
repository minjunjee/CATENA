"""Bind Stage-2 data bytes to the exact builders and isolated tool environment."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

CRITICAL_TOOL_VERSIONS = {
    "huggingface_hub": "1.26.0",
    "numpy": "2.4.4",
    "pyarrow": "25.0.0",
    "tokenizers": "0.23.1",
}

CONSTRUCTION_SOURCE_FILES = (
    "configs/e26_data_lock_v1.yaml",
    "configs/e26_data_tooling_requirements.txt",
    "schemas/v8_1/construction_source_receipt.schema.json",
    "schemas/v8_1/general_corpus_manifest.schema.json",
    "schemas/v8_1/scientific_data_readiness_v2.schema.json",
    "schemas/v8_1/tokenizer_manifest.schema.json",
    "scripts/prepare_e26_data_v1.sh",
    "scripts/validate_e26_data_v1.sh",
    "src/catena/lm/construction_source.py",
    "src/catena/lm/data_lock.py",
    "src/catena/lm/data_readiness_v2.py",
    "src/catena/lm/fineweb_source.py",
    "src/catena/lm/general_corpus.py",
    "src/catena/lm/memmap_builder.py",
    "src/catena/lm/paired_stream.py",
    "src/catena/lm/parquet_documents.py",
    "src/catena/lm/schedule_manifest.py",
    "src/catena/lm/tokenizer.py",
    "src/catena/lm/tokenizer_builder.py",
    "src/catena/lm/transaction_data.py",
    "tools/audit_e26_near_duplicates.py",
    "tools/lock_e26_construction_source.py",
    "tools/prepare_e26_document_index.py",
    "tools/prepare_e26_memmaps.py",
    "tools/prepare_e26_schedule.py",
    "tools/prepare_e26_tokenizer.py",
    "tools/prepare_e26_transactions.py",
    "tools/resolve_e26_fineweb.py",
    "tools/validate_e26_data_v1.py",
)

REQUIRED_ARTIFACT_BINDINGS = (
    "data_lock",
    "source_inventory",
    "source_metadata",
    "download_receipt",
    "dedup_receipt",
    "tokenizer_manifest",
    "tokenizer_replay",
    "near_duplicate_audit",
    "memmap_receipt",
    "transaction_manifest",
    "schedule_manifest",
)


class ConstructionSourceError(RuntimeError):
    """Raised when exact construction provenance cannot be established."""


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_records(repo_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in CONSTRUCTION_SOURCE_FILES:
        path = (repo_root / relative).resolve(strict=True)
        if not path.is_file():
            raise ConstructionSourceError(f"Construction source is not a file: {path}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _artifact_records(
    bindings: dict[str, str | Path],
) -> list[dict[str, Any]]:
    if set(bindings) != set(REQUIRED_ARTIFACT_BINDINGS):
        missing = sorted(set(REQUIRED_ARTIFACT_BINDINGS) - set(bindings))
        extra = sorted(set(bindings) - set(REQUIRED_ARTIFACT_BINDINGS))
        raise ConstructionSourceError(
            f"Construction artifact binding mismatch; missing={missing}, extra={extra}"
        )
    records: list[dict[str, Any]] = []
    for label in REQUIRED_ARTIFACT_BINDINGS:
        path = Path(bindings[label]).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ConstructionSourceError(f"Bound artifact is not a file: {path}")
        records.append(
            {
                "label": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _pip_freeze() -> list[str]:
    result = subprocess.run(
        (sys.executable, "-m", "pip", "freeze", "--all"),
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def build_construction_source_receipt(
    *,
    repo_root: str | Path,
    artifact_bindings: dict[str, str | Path],
) -> dict[str, Any]:
    """Create a receipt without mutating the repository or any bound artifact."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    git_head = _git(root, "rev-parse", "HEAD")
    git_branch = _git(root, "branch", "--show-current")
    dirty_lines = tuple(
        line
        for line in _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines()
        if line
    )
    sources = _source_records(root)
    artifacts = _artifact_records(artifact_bindings)
    observed_versions = {
        package: importlib.metadata.version(package)
        for package in CRITICAL_TOOL_VERSIONS
    }
    if observed_versions != CRITICAL_TOOL_VERSIONS:
        raise ConstructionSourceError(
            f"Pinned construction tool mismatch: {observed_versions}"
        )
    freeze = _pip_freeze()
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-construction-source-v1",
        "manifest_type": "E26_CONSTRUCTION_SOURCE_RECEIPT",
        "scientific_evidence": False,
        "claim_ceiling": "SCIENTIFIC_INPUT_PROVENANCE_ONLY",
        "construction_contract_complete": True,
        "repo_root": str(root),
        "git_head": git_head,
        "git_branch": git_branch,
        "git_clean": not dirty_lines,
        "git_status_porcelain": list(dirty_lines),
        "builder_files": sources,
        "builder_source_sha256": sha256_canonical_json(sources),
        "artifact_bindings": artifacts,
        "artifact_binding_sha256": sha256_canonical_json(artifacts),
        "tool_environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "critical_versions": observed_versions,
            "pip_freeze": freeze,
            "pip_freeze_sha256": sha256_canonical_json(freeze),
        },
        "scientific_main_input_eligible": not dirty_lines,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def write_construction_source_receipt(
    path: str | Path,
    *,
    repo_root: str | Path,
    artifact_bindings: dict[str, str | Path],
) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite construction source receipt: {destination}"
        )
    payload = build_construction_source_receipt(
        repo_root=repo_root,
        artifact_bindings=artifact_bindings,
    )
    write_json_strict(destination, payload)
    return destination
