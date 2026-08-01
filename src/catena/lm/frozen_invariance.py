"""Structured live-source and completed E00--E25 evidence invariance for E26."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)


class FrozenInvarianceError(RuntimeError):
    """Raised when the frozen pre-E26 state cannot be verified exactly."""


_EXPERIMENT = re.compile(r"^e(?P<number>\d{2})(?:[a-z]|_)")
_TOP_LEVEL_EXPERIMENT = re.compile(r"^E(?P<number>\d{2})(?:[A-Z]|_)")
_BASE_ARTIFACT_SCOPE = "immutable_E00_through_E21_artifacts"
_COMPLETED_ARTIFACT_SCOPE = "immutable_E00_through_E25_completed_artifacts"
_POST_E21_TOP_LEVEL = {"POST_E21_WAVE1_STATUS.json"}
_LEGACY_TOP_LEVEL_SUMMARIES = {
    "WORKFLOW_E00_E02_RESULTS_KO.md",
    "POSTCORE_E10_E16_RESULTS_SUMMARY_INDEX_KO.md",
    "POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md",
}


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise FrozenInvarianceError(
            f"Git command failed in {root}: {' '.join(arguments)}"
        ) from error


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise FrozenInvarianceError(
            f"Cannot verify frozen source ancestry: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _base_entries(root: Path, base_commit: str) -> list[tuple[str, str]]:
    output = _git(root, "ls-tree", "-r", base_commit)
    entries: list[tuple[str, str]] = []
    for line in output.splitlines():
        metadata, relative = line.split("\t", 1)
        mode, object_type, _object_id = metadata.split()
        if object_type == "blob":
            entries.append((mode, relative))
    return entries


def _base_blob(root: Path, base_commit: str, relative: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{base_commit}:{relative}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise FrozenInvarianceError(
            f"Cannot read frozen source blob {base_commit}:{relative}"
        ) from error


def _row_aggregate(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def verify_pre_e26_source(
    *,
    live_repo: str | Path,
    expected_head: str,
    expected_file_count: int,
    expected_aggregate_sha256: str,
) -> dict[str, Any]:
    root = Path(live_repo).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise FrozenInvarianceError(f"Live repository is not a directory: {root}")
    observed_head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    entries = _base_entries(root, expected_head)
    rows: list[dict[str, Any]] = []
    changed: list[str] = []
    missing: list[str] = []
    for mode, relative in entries:
        path = root / relative
        expected_symlink = mode == "120000"
        exists = path.is_symlink() if expected_symlink else path.is_file() and not path.is_symlink()
        base_bytes = _base_blob(root, expected_head, relative)
        rows.append(
            {
                "path": relative,
                "bytes": len(base_bytes),
                "sha256": hashlib.sha256(base_bytes).hexdigest(),
            }
        )
        if not exists:
            missing.append(relative)
            continue
        observed_digest = (
            hashlib.sha256(os.readlink(path).encode()).hexdigest()
            if expected_symlink
            else sha256_file(path)
        )
        if observed_digest != rows[-1]["sha256"]:
            changed.append(relative)
    clean = status == ""
    head_matches = observed_head == expected_head
    expected_head_is_ancestor = _git_is_ancestor(root, expected_head, observed_head)
    aggregate = _row_aggregate(rows)
    passed = (
        clean
        and expected_head_is_ancestor
        and not missing
        and not changed
        and len(entries) == expected_file_count
        and aggregate == expected_aggregate_sha256
    )
    return {
        "scope": "all_files_tracked_at_pre_e26_base_commit",
        "root": str(root),
        "expected_head": expected_head,
        "observed_head": observed_head,
        "head_matches": head_matches,
        "expected_head_is_ancestor": expected_head_is_ancestor,
        "verification_mode": "ANCESTOR_COMMIT_BASE_BLOBS_BYTE_IDENTICAL",
        "git_clean": clean,
        "git_status_porcelain": status,
        "expected_files": len(entries),
        "observed_files": len(entries) - len(missing),
        "registered_file_count": expected_file_count,
        "registered_aggregate_sha256": expected_aggregate_sha256,
        "base_aggregate_sha256": aggregate,
        "missing": missing,
        "changed": changed,
        "passed": passed,
    }


def _included_top_level(
    path: Path,
    *,
    minimum_experiment: int,
    maximum_experiment: int,
) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    match = _TOP_LEVEL_EXPERIMENT.match(path.name)
    if match is not None and path.suffix == ".json":
        number = int(match.group("number"))
        return minimum_experiment <= number <= maximum_experiment
    if minimum_experiment <= 0 and path.name in _LEGACY_TOP_LEVEL_SUMMARIES:
        return True
    return minimum_experiment <= 22 <= maximum_experiment and path.name in _POST_E21_TOP_LEVEL


def _artifact_inventory(
    artifact_root: Path,
    *,
    minimum_experiment: int = 0,
    maximum_experiment: int = 25,
) -> dict[str, Any]:
    if minimum_experiment < 0 or maximum_experiment < minimum_experiment:
        raise FrozenInvarianceError("Invalid artifact experiment range")
    root = artifact_root.resolve(strict=True)
    paths: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        match = _EXPERIMENT.match(child.name)
        if _included_top_level(
            child,
            minimum_experiment=minimum_experiment,
            maximum_experiment=maximum_experiment,
        ):
            paths.append(child)
        elif (
            child.is_dir()
            and not child.is_symlink()
            and match is not None
            and minimum_experiment <= int(match.group("number")) <= maximum_experiment
        ):
            paths.extend(
                path
                for path in sorted(child.rglob("*"))
                if path.is_file() and not path.is_symlink()
            )
    rows: list[dict[str, Any]] = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "artifact_root": str(root),
        "minimum_experiment": minimum_experiment,
        "maximum_experiment": maximum_experiment,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "aggregate_sha256": _row_aggregate(rows),
        "files": rows,
    }


def _require_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or value < 0:
        raise FrozenInvarianceError(f"Completed artifact lock has invalid {key}")
    return value


def _require_sha(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FrozenInvarianceError(f"Completed artifact lock has invalid {key}")
    return value


def _resolve_relative_lock_path(lock_path: Path, relative: str) -> Path:
    candidate = (lock_path.parent / relative).resolve(strict=True)
    try:
        candidate.relative_to(lock_path.parent.resolve(strict=True))
    except ValueError as error:
        raise FrozenInvarianceError("Base artifact manifest escapes the lock directory") from error
    return candidate


def _namespace_rows(rows: list[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
    prefix = f"{namespace}/"
    return [row for row in rows if row["path"] == namespace or str(row["path"]).startswith(prefix)]


def verify_frozen_artifacts(
    *,
    baseline_manifest: str | Path,
    expected_file_count: int,
    expected_aggregate_sha256: str,
) -> dict[str, Any]:
    lock_path = Path(baseline_manifest).expanduser().resolve(strict=True)
    completed_lock = read_json_object_strict(lock_path)
    if (
        completed_lock.get("schema_version") != 1
        or completed_lock.get("scope") != _COMPLETED_ARTIFACT_SCOPE
    ):
        raise FrozenInvarianceError(
            "Frozen artifact lock must cover completed E00 through E25 evidence"
        )
    artifact_root_raw = completed_lock.get("artifact_root")
    base_reference = completed_lock.get("base_manifest")
    extension = completed_lock.get("extension")
    if (
        not isinstance(artifact_root_raw, str)
        or not isinstance(base_reference, Mapping)
        or not isinstance(extension, Mapping)
    ):
        raise FrozenInvarianceError("Completed artifact lock has an invalid structure")

    base_relative = base_reference.get("path")
    if not isinstance(base_relative, str) or not base_relative:
        raise FrozenInvarianceError("Completed artifact lock lacks the base manifest path")
    base_path = _resolve_relative_lock_path(lock_path, base_relative)
    if base_reference.get("scope") != _BASE_ARTIFACT_SCOPE:
        raise FrozenInvarianceError("Completed artifact lock has the wrong base scope")
    if sha256_file(base_path) != _require_sha(base_reference, "sha256"):
        raise FrozenInvarianceError("Frozen E00--E21 base manifest SHA-256 changed")
    base = read_json_object_strict(base_path)
    base_rows_raw = base.get("files")
    if not isinstance(base_rows_raw, list) or base.get("scope") != _BASE_ARTIFACT_SCOPE:
        raise FrozenInvarianceError("Frozen E00--E21 base manifest has an invalid structure")
    base_count = _require_int(base_reference, "file_count")
    base_aggregate = _require_sha(base_reference, "aggregate_sha256")
    if (
        base.get("file_count") != base_count
        or base.get("aggregate_sha256") != base_aggregate
        or base_count != expected_file_count
        or base_aggregate != expected_aggregate_sha256
    ):
        raise FrozenInvarianceError("Data-lock values differ from the E00--E21 base manifest")
    base_rows = {str(row["path"]): row for row in base_rows_raw if isinstance(row, Mapping)}
    if len(base_rows) != base_count or _row_aggregate(base_rows.values()) != base_aggregate:
        raise FrozenInvarianceError("Frozen E00--E21 base manifest rows are inconsistent")

    if (
        extension.get("experiment_min") != 22
        or extension.get("experiment_max") != 25
        or extension.get("include_top_level") != sorted(_POST_E21_TOP_LEVEL)
    ):
        raise FrozenInvarianceError("Completed artifact extension must lock E22--E25 exactly")
    namespace_locks_raw = extension.get("namespaces")
    if not isinstance(namespace_locks_raw, list) or not namespace_locks_raw:
        raise FrozenInvarianceError("Completed artifact extension lacks namespace locks")
    namespace_locks: dict[str, Mapping[str, Any]] = {}
    for row in namespace_locks_raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise FrozenInvarianceError("Malformed completed artifact namespace lock")
        namespace = str(row["path"])
        if namespace in namespace_locks or "/" in namespace or namespace in {".", ".."}:
            raise FrozenInvarianceError("Completed artifact namespace paths must be unique roots")
        namespace_locks[namespace] = row

    root = Path(artifact_root_raw).expanduser().resolve(strict=True)
    base_artifact_root = base.get("artifact_root")
    if (
        not isinstance(base_artifact_root, str)
        or Path(base_artifact_root).expanduser().resolve(strict=True) != root
    ):
        raise FrozenInvarianceError("Base and completed artifact locks use different roots")
    observed = _artifact_inventory(root, minimum_experiment=0, maximum_experiment=25)
    observed_base = _artifact_inventory(root, minimum_experiment=0, maximum_experiment=21)
    observed_extension = _artifact_inventory(root, minimum_experiment=22, maximum_experiment=25)
    observed_base_rows = {
        str(row["path"]): row for row in observed_base["files"] if isinstance(row, Mapping)
    }
    missing = sorted(set(base_rows) - set(observed_base_rows))
    unexpected = sorted(set(observed_base_rows) - set(base_rows))
    changed = sorted(
        path
        for path in set(base_rows) & set(observed_base_rows)
        if base_rows[path].get("bytes") != observed_base_rows[path].get("bytes")
        or base_rows[path].get("sha256") != observed_base_rows[path].get("sha256")
    )
    expected_namespaces = set(namespace_locks)
    observed_namespaces = {str(row["path"]).split("/", 1)[0] for row in observed_extension["files"]}
    missing_namespaces = sorted(expected_namespaces - observed_namespaces)
    unexpected_namespaces = sorted(observed_namespaces - expected_namespaces)
    changed_namespaces: list[str] = []
    for namespace in sorted(expected_namespaces & observed_namespaces):
        rows = _namespace_rows(observed_extension["files"], namespace)
        lock = namespace_locks[namespace]
        if (
            len(rows) != _require_int(lock, "file_count")
            or sum(int(row["bytes"]) for row in rows) != _require_int(lock, "total_bytes")
            or _row_aggregate(rows) != _require_sha(lock, "aggregate_sha256")
        ):
            changed_namespaces.append(namespace)

    extension_count = _require_int(extension, "file_count")
    extension_bytes = _require_int(extension, "total_bytes")
    extension_aggregate = _require_sha(extension, "aggregate_sha256")
    completed_count = _require_int(completed_lock, "file_count")
    completed_bytes = _require_int(completed_lock, "total_bytes")
    completed_aggregate = _require_sha(completed_lock, "aggregate_sha256")
    passed = (
        not missing
        and not unexpected
        and not changed
        and not missing_namespaces
        and not unexpected_namespaces
        and not changed_namespaces
        and observed_base["file_count"] == base_count
        and observed_base["aggregate_sha256"] == base_aggregate
        and observed_extension["file_count"] == extension_count
        and observed_extension["total_bytes"] == extension_bytes
        and observed_extension["aggregate_sha256"] == extension_aggregate
        and observed["file_count"] == completed_count
        and observed["total_bytes"] == completed_bytes
        and observed["aggregate_sha256"] == completed_aggregate
    )
    return {
        "scope": _COMPLETED_ARTIFACT_SCOPE,
        "artifact_root": str(root),
        "baseline_manifest": str(lock_path),
        "baseline_manifest_sha256": sha256_file(lock_path),
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256_file(base_path),
        "base_scope": _BASE_ARTIFACT_SCOPE,
        "registered_base_file_count": expected_file_count,
        "registered_base_aggregate_sha256": expected_aggregate_sha256,
        "expected_file_count": completed_count,
        "observed_file_count": int(observed["file_count"]),
        "registered_file_count": completed_count,
        "expected_aggregate_sha256": completed_aggregate,
        "observed_aggregate_sha256": str(observed["aggregate_sha256"]),
        "extension_file_count": extension_count,
        "extension_aggregate_sha256": extension_aggregate,
        "excluded_experiment_min": 26,
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
        "missing_namespaces": missing_namespaces,
        "unexpected_namespaces": unexpected_namespaces,
        "changed_namespaces": changed_namespaces,
        "passed": passed,
    }


def build_frozen_invariance_receipt(
    *,
    data_lock: Mapping[str, Any],
    baseline_manifest: str | Path,
) -> dict[str, Any]:
    repository = data_lock.get("repository")
    if not isinstance(repository, Mapping):
        raise FrozenInvarianceError("E26 data lock lacks repository settings")
    live_repo = repository.get("live_repo")
    expected_head = repository.get("expected_live_head")
    expected_source_files = repository.get("pre_e26_source_file_count")
    expected_source_aggregate = repository.get("pre_e26_source_aggregate_sha256")
    expected_artifact_files = repository.get("frozen_artifact_file_count")
    expected_aggregate = repository.get("frozen_artifact_aggregate_sha256")
    if not all(
        isinstance(value, str) and value for value in (live_repo, expected_head, expected_aggregate)
    ):
        raise FrozenInvarianceError("E26 data lock repository settings are incomplete")
    if (
        not isinstance(expected_source_files, int)
        or expected_source_files <= 0
        or not isinstance(expected_source_aggregate, str)
    ):
        raise FrozenInvarianceError("E26 data lock lacks the pre-E26 source count/aggregate")
    if not isinstance(expected_artifact_files, int) or expected_artifact_files <= 0:
        raise FrozenInvarianceError("E26 data lock lacks the frozen artifact file count")
    source = verify_pre_e26_source(
        live_repo=str(live_repo),
        expected_head=str(expected_head),
        expected_file_count=expected_source_files,
        expected_aggregate_sha256=expected_source_aggregate,
    )
    artifacts = verify_frozen_artifacts(
        baseline_manifest=baseline_manifest,
        expected_file_count=expected_artifact_files,
        expected_aggregate_sha256=str(expected_aggregate),
    )
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-frozen-invariance-v1",
        "manifest_type": "E26_FROZEN_INVARIANCE_RECEIPT",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "passed": source["passed"] is True and artifacts["passed"] is True,
        "live_repository": source,
        "frozen_artifacts": artifacts,
        "main_test_opened": False,
        "main_test_access_count": 0,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    return payload


def write_frozen_invariance_receipt(
    path: str | Path,
    *,
    data_lock: Mapping[str, Any],
    baseline_manifest: str | Path,
) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen invariance receipt: {destination}")
    write_json_strict(
        destination,
        build_frozen_invariance_receipt(
            data_lock=data_lock,
            baseline_manifest=baseline_manifest,
        ),
    )
    return destination


def validate_frozen_invariance_receipt(
    payload: Mapping[str, Any],
    *,
    data_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("manifest_type") != "E26_FROZEN_INVARIANCE_RECEIPT":
        raise FrozenInvarianceError("Frozen invariance receipt has the wrong type")
    frozen = payload.get("frozen_artifacts")
    if not isinstance(frozen, Mapping):
        raise FrozenInvarianceError("Frozen invariance receipt lacks artifact details")
    baseline_path = frozen.get("baseline_manifest")
    if not isinstance(baseline_path, str):
        raise FrozenInvarianceError("Frozen invariance receipt lacks baseline path")
    observed = build_frozen_invariance_receipt(
        data_lock=data_lock,
        baseline_manifest=baseline_path,
    )
    if dict(payload) != observed:
        raise FrozenInvarianceError(
            "Frozen invariance receipt differs from a live structured re-audit"
        )
    if observed["passed"] is not True:
        raise FrozenInvarianceError("Frozen source/artifact invariance audit failed")
    return observed
