from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    DEFAULT_EXCLUDED_DIRECTORY_PREFIXES,
    SHA256_PATTERN,
    sha256_file,
)

E26_EXECUTION_SOURCE_SUFFIXES = frozenset(
    {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".txt", ".ini"}
)

E26_AUDIT_LOCKED_HASH_KEYS = frozenset(
    {
        "source_tree_sha256",
        "config_sha256",
        "calibration_config_sha256",
        "protocol_lock_sha256",
        "backend_candidate_lock_sha256",
        "data_readiness_sha256",
        "data_lock_sha256",
        "tokenizer_manifest_sha256",
        "corpus_manifest_sha256",
        "transaction_manifest_sha256",
        "validation_population_lock_sha256",
        "schedule_manifest_sha256",
        "frozen_tree_receipt_sha256",
    }
)


def validate_e26_audit_locked_hashes(
    locked_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Validate the exact acyclic Stage-2 binding shared by audit receipts."""

    observed = set(locked_hashes)
    if observed != E26_AUDIT_LOCKED_HASH_KEYS:
        missing = sorted(E26_AUDIT_LOCKED_HASH_KEYS - observed)
        extra = sorted(observed - E26_AUDIT_LOCKED_HASH_KEYS)
        raise ValueError(
            "E26 audit locked-hash fields differ from the registered contract: "
            f"missing={missing}, extra={extra}"
        )
    normalized = dict(sorted(locked_hashes.items()))
    invalid = [
        key
        for key, value in normalized.items()
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
    ]
    if invalid:
        raise ValueError(f"E26 audit hashes must be lowercase SHA-256: {invalid}")
    return normalized


def e26_execution_source_inventory(repo_root: str | Path) -> dict[str, Any]:
    """Fingerprint only executable E26 inputs, excluding result-report Markdown.

    The aggregate uses the same ``relative-path NUL file-bytes NUL`` encoding
    as the repository's generic source-tree fingerprint. Explicit rows make
    the scope auditable and prevent a later report-only commit from creating a
    circular dependency.
    """

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths: list[tuple[str, Path]] = []
    for current_root, directory_names, file_names in os.walk(root, topdown=True):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in DEFAULT_EXCLUDED_DIRECTORY_NAMES
            and not any(name.startswith(prefix) for prefix in DEFAULT_EXCLUDED_DIRECTORY_PREFIXES)
        )
        current = Path(current_root)
        for file_name in sorted(file_names):
            candidate = current / file_name
            if candidate.suffix.lower() not in E26_EXECUTION_SOURCE_SUFFIXES:
                continue
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"E26 execution source resolves outside repository: {candidate}"
                ) from error
            if resolved.is_file():
                paths.append((candidate.relative_to(root).as_posix(), resolved))
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for relative_path, resolved in sorted(paths):
        content = resolved.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        rows.append(
            {
                "path": relative_path,
                "bytes": len(content),
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "algorithm": "relative_path_nul_file_bytes_nul_v1",
        "suffixes": sorted(E26_EXECUTION_SOURCE_SUFFIXES),
        "files": len(rows),
        "rows": rows,
        "source_tree_sha256": digest.hexdigest(),
    }
