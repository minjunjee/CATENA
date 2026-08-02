#!/usr/bin/env python3
"""Create the prospective Stage-3C artifact binding consumed by Stage-3D.

The tool is intentionally narrow: it fingerprints the already-frozen raw
Stage-3C run and writes one new manifest outside that run.  It never modifies
the predecessor directory and it excludes the later human-readable summary
from the registered eleven-file raw payload.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)

REGISTERED_STAGE3C_RESULT_SHA256 = (
    "83fab26e7936654b664653776d501c3fdee6cb7f0ffd78c3d9682ed41d319b56"
)
REGISTERED_STAGE3C_STATUS_SHA256 = (
    "15b896a33e0fe286c80f2c204b7be2be0fbe6aaf8cdc512fafbd31040f8aabda"
)
REGISTERED_RAW_RUN_AGGREGATE_SHA256 = (
    "296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc"
)
REGISTERED_FAILURE_STATUS_SHA256 = (
    "dc7ed1837ccf022fe5110fdb44907c5e340391f0bcc5c92b7d5e26dcf2a95616"
)
REGISTERED_STAGE3C_DISPOSITION = (
    "BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE"
)

EXPECTED_RAW_FILES = (
    "d448_ctx4096.log",
    "d448_ctx4096_numerical.json",
    "d448_ctx4096_worker_spec.json",
    "d512_ctx2048.log",
    "d512_ctx2048_numerical.json",
    "d512_ctx2048_worker_spec.json",
    "d512_ctx4096.log",
    "d512_ctx4096_numerical.json",
    "d512_ctx4096_worker_spec.json",
    "failure_status.json",
    "source_lock.json",
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.PIPE
    ).strip()


def _row_aggregate(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def _registered_anchors(*, repo: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result_path = repo / "docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_RESULT_KO.md"
    status_path = repo / "docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_STATUS.json"
    if sha256_file(result_path) != REGISTERED_STAGE3C_RESULT_SHA256:
        raise ValueError("Registered Stage-3C result bytes changed")
    if sha256_file(status_path) != REGISTERED_STAGE3C_STATUS_SHA256:
        raise ValueError("Registered Stage-3C status bytes changed")
    status = read_json_object_strict(status_path)
    expected_status = {
        "stage3c_disposition": REGISTERED_STAGE3C_DISPOSITION,
        "raw_run_aggregate_sha256": REGISTERED_RAW_RUN_AGGREGATE_SHA256,
        "failure_status_sha256": REGISTERED_FAILURE_STATUS_SHA256,
        "scientific_e26a_started": False,
        "restart_audit_started": False,
        "resource_preflight_started": False,
    }
    if any(status.get(key) != value for key, value in expected_status.items()):
        raise ValueError("Registered Stage-3C status contract changed")
    failure_rows = [row for row in rows if row["path"] == "failure_status.json"]
    if len(failure_rows) != 1 or (
        failure_rows[0]["sha256"] != REGISTERED_FAILURE_STATUS_SHA256
    ):
        raise ValueError("Registered Stage-3C failure status artifact changed")
    return {
        "result": {
            "path": str(result_path.resolve()),
            "sha256": REGISTERED_STAGE3C_RESULT_SHA256,
        },
        "status": {
            "path": str(status_path.resolve()),
            "sha256": REGISTERED_STAGE3C_STATUS_SHA256,
        },
        "raw_run_aggregate_sha256": REGISTERED_RAW_RUN_AGGREGATE_SHA256,
        "failure_status_sha256": REGISTERED_FAILURE_STATUS_SHA256,
        "disposition": REGISTERED_STAGE3C_DISPOSITION,
    }


def build_manifest(
    *,
    repo_root: Path,
    artifact_root: Path,
    require_registered_anchors: bool = True,
) -> dict[str, Any]:
    repo = repo_root.expanduser().resolve(strict=True)
    root = artifact_root.expanduser().resolve(strict=True)
    if not repo.is_dir() or not root.is_dir() or root.is_symlink():
        raise ValueError("Repository and Stage-3C artifact root must be real directories")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Stage-3D manifest preparation requires a clean committed tree")

    observed = tuple(sorted(path.name for path in root.iterdir() if path.is_file()))
    allowed = tuple(sorted((*EXPECTED_RAW_FILES, "RESULTS_SUMMARY_KO.md")))
    if observed != allowed:
        raise ValueError(
            f"Stage-3C durable run file set changed: expected={allowed}, observed={observed}"
        )
    rows: list[dict[str, Any]] = []
    for relative in EXPECTED_RAW_FILES:
        path = (root / relative).resolve(strict=True)
        if path.parent != root or path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid Stage-3C raw artifact: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-stage3c-artifact-hash-manifest-v1",
        "manifest_type": "E26_STAGE3C_ARTIFACT_HASH_MANIFEST",
        "scientific_evidence": False,
        "predecessor_disposition": REGISTERED_STAGE3C_DISPOSITION,
        "artifact_root": str(root),
        "file_count": len(rows),
        "files": rows,
        "aggregate_algorithm": "path_nul_bytes_nul_sha256_newline_v1",
        "aggregate_sha256": _row_aggregate(rows),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "predecessor_mutated": False,
    }
    payload["registered_predecessor"] = (
        _registered_anchors(repo=repo, rows=rows)
        if require_registered_anchors
        else {"test_only_anchor_verification_skipped": True}
    )
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage3c-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    output.parent.resolve(strict=True)
    payload = build_manifest(
        repo_root=args.repo_root,
        artifact_root=args.stage3c_artifact_root,
    )
    write_json_strict(output, payload)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
