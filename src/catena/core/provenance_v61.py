"""Strict, experiment-agnostic provenance helpers for CATENA v6.1.

This module intentionally does not encode claim gates or experiment ordering.  It
only establishes the filesystem, hashing, and completed-run integrity contract
that experiment-specific dependency policies can build on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

JsonObject = dict[str, Any]

DEFAULT_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".bib", ".md", ".py", ".tex", ".toml", ".yaml", ".yml"}
)
DEFAULT_EXCLUDED_DIRECTORY_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "artifacts",
        "artifacts_dry_run",
        "build",
        "cache",
        "caches",
        "dist",
        "venv",
    }
)
DEFAULT_EXCLUDED_DIRECTORY_PREFIXES: Final[tuple[str, ...]] = ("artifacts_", "artifacts-")
UTC_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")
SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceValidationError(RuntimeError):
    """Raised when a path or completed-run provenance contract is invalid."""


class StrictJSONError(ValueError):
    """Raised when input is outside the strict JSON data model."""


@dataclass(frozen=True)
class SourceTreeFingerprint:
    """A deterministic digest and the number of source files included in it."""

    sha256: str
    files: int

    def __post_init__(self) -> None:
        if not SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("Source fingerprint sha256 must be 64 lowercase hex characters.")
        if isinstance(self.files, bool) or not isinstance(self.files, int) or self.files < 0:
            raise ValueError("Source fingerprint files must be a non-negative integer.")

    def as_dict(self) -> dict[str, int | str]:
        return {"sha256": self.sha256, "files": self.files}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SourceTreeFingerprint:
        sha256 = payload.get("sha256")
        files = payload.get("files")
        if not isinstance(sha256, str):
            raise ProvenanceValidationError("source_fingerprint.sha256 must be a string.")
        if isinstance(files, bool) or not isinstance(files, int):
            raise ProvenanceValidationError("source_fingerprint.files must be an integer.")
        try:
            return cls(sha256=sha256, files=files)
        except ValueError as exc:
            raise ProvenanceValidationError(str(exc)) from exc


@dataclass(frozen=True)
class ManifestValidationRequirements:
    """Caller-selected policy for validating a generic completed-run manifest.

    The generic v6.1 execution contract uses ``run_mode`` plus
    ``eligibility: {"main": bool, "full": bool}``.  The meaning of a claim,
    the required predecessor experiment, and claim-specific report fields are
    deliberately left to experiment code.
    """

    expected_experiment_id: str | None = None
    accepted_schema_versions: frozenset[int] | None = None
    expected_source_sha256: str | None = None
    expected_source_files: int | None = None
    expected_run_mode: str | None = None
    require_main_eligible: bool = False
    require_full_eligible: bool = False
    allowed_statuses: frozenset[str] | None = frozenset({"PASS"})
    verify_config_file_sha256: bool = True
    require_source_verified_at_completion: bool = True
    require_dependencies_list: bool = True
    require_utc_run_id: bool = True

    def __post_init__(self) -> None:
        if self.expected_experiment_id is not None:
            _validate_path_component(self.expected_experiment_id, "expected_experiment_id")
        if self.accepted_schema_versions is not None:
            if not self.accepted_schema_versions:
                raise ValueError("accepted_schema_versions cannot be empty.")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in self.accepted_schema_versions
            ):
                raise ValueError("accepted_schema_versions must contain positive integers.")
        if self.expected_source_sha256 is not None and not SHA256_PATTERN.fullmatch(
            self.expected_source_sha256
        ):
            raise ValueError("expected_source_sha256 must be 64 lowercase hex characters.")
        if self.expected_source_files is not None and (
            isinstance(self.expected_source_files, bool)
            or not isinstance(self.expected_source_files, int)
            or self.expected_source_files < 0
        ):
            raise ValueError("expected_source_files must be a non-negative integer.")
        if self.expected_run_mode is not None and not self.expected_run_mode:
            raise ValueError("expected_run_mode cannot be empty.")
        if self.allowed_statuses is not None and not self.allowed_statuses:
            raise ValueError("allowed_statuses cannot be empty.")


@dataclass(frozen=True)
class ValidatedRun:
    """Integrity-checked evidence for one completed artifact run."""

    artifact_root: Path
    experiment_id: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    report_path: Path
    config_path: Path | None
    schema_version: int
    status: str
    run_mode: str
    main_eligible: bool
    full_eligible: bool
    source_fingerprint: SourceTreeFingerprint
    config_sha256: str
    config_file_sha256: str | None
    report_sha256: str
    manifest_sha256: str
    manifest: JsonObject
    report: JsonObject

    def dependency_record(self) -> JsonObject:
        """Return the immutable generic portion suitable for downstream lineage."""

        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "run_mode": self.run_mode,
            "eligibility": {
                "main": self.main_eligible,
                "full": self.full_eligible,
            },
            "source_fingerprint": self.source_fingerprint.as_dict(),
            "config_sha256": self.config_sha256,
            "config_file_sha256": self.config_file_sha256,
            "report_sha256": self.report_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def _validate_json_value(value: Any, path: str = "$", active: set[int] | None = None) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError(f"Non-finite number at {path}: {value!r}")
        return
    if active is None:
        active = set()
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise StrictJSONError(f"Circular list at {path}.")
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise StrictJSONError(f"Circular object at {path}.")
        active.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise StrictJSONError(
                        f"JSON object key at {path} must be a string, got {type(key).__name__}."
                    )
                _validate_json_value(item, f"{path}.{key}", active)
        finally:
            active.remove(identity)
        return
    raise StrictJSONError(f"Unsupported JSON value at {path}: {type(value).__name__}.")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> None:
    raise StrictJSONError(f"Non-standard JSON numeric constant: {token}.")


def dumps_json_strict(
    payload: Any,
    *,
    canonical: bool = False,
    indent: int | None = None,
) -> str:
    """Serialize strict JSON, rejecting NaN, infinity, non-string keys, and tuples."""

    _validate_json_value(payload)
    separators = (",", ":") if canonical else None
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=None if canonical else indent,
            separators=separators,
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StrictJSONError(f"Cannot serialize strict JSON: {exc}") from exc


def loads_json_strict(payload: str | bytes | bytearray) -> Any:
    """Parse RFC-compatible JSON and reject duplicate keys and NaN/infinity."""

    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise StrictJSONError(f"Cannot parse strict JSON: {exc}") from exc


def canonical_json_bytes(payload: Any) -> bytes:
    """Return UTF-8 canonical JSON bytes used by payload hashes."""

    try:
        return dumps_json_strict(payload, canonical=True).encode("utf-8")
    except UnicodeError as exc:
        raise StrictJSONError(f"Cannot encode strict JSON as UTF-8: {exc}") from exc


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    """Return a SHA-256 hex digest for bytes-like input."""

    return hashlib.sha256(bytes(payload)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a regular file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required file is missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_json(payload: Any) -> str:
    """Return a key-order-invariant SHA-256 digest of strict canonical JSON."""

    return sha256_bytes(canonical_json_bytes(payload))


def _atomic_write_text(path: Path, chunks: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def write_json_strict(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write strict JSON with a terminating newline."""

    serialized = dumps_json_strict(payload, indent=indent)
    _atomic_write_text(Path(path), (serialized, "\n"))


def read_json_strict(path: str | Path) -> Any:
    """Read one strict JSON value from a file."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {target}")
    try:
        return loads_json_strict(target.read_bytes())
    except StrictJSONError as exc:
        raise StrictJSONError(f"{target}: {exc}") from exc


def read_json_object_strict(path: str | Path) -> JsonObject:
    """Read a strict JSON object, rejecting arrays and scalar roots."""

    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise StrictJSONError(f"Expected a JSON object: {Path(path)}")
    return payload


def write_jsonl_strict(path: str | Path, rows: Iterable[Any]) -> None:
    """Atomically stream newline-delimited strict JSON."""

    def serialized_rows() -> Iterable[str]:
        for row_number, row in enumerate(rows, start=1):
            try:
                yield dumps_json_strict(row, canonical=True)
            except StrictJSONError as exc:
                raise StrictJSONError(f"JSONL row {row_number}: {exc}") from exc
            yield "\n"

    _atomic_write_text(Path(path), serialized_rows())


def read_jsonl_strict(path: str | Path) -> list[Any]:
    """Read newline-delimited strict JSON, rejecting empty lines."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required JSONL file is missing: {target}")
    rows: list[Any] = []
    with target.open("rb") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise StrictJSONError(f"{target}: empty JSONL row {row_number}.")
            try:
                rows.append(loads_json_strict(raw_line))
            except StrictJSONError as exc:
                raise StrictJSONError(f"{target}: JSONL row {row_number}: {exc}") from exc
    return rows


def _excluded_directory(
    name: str,
    excluded_names: frozenset[str],
    excluded_prefixes: tuple[str, ...],
) -> bool:
    return name in excluded_names or any(name.startswith(prefix) for prefix in excluded_prefixes)


def source_tree_fingerprint(
    repo_root: str | Path,
    *,
    included_suffixes: frozenset[str] = DEFAULT_SOURCE_SUFFIXES,
    excluded_directory_names: frozenset[str] = DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    excluded_directory_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_DIRECTORY_PREFIXES,
) -> SourceTreeFingerprint:
    """Fingerprint a source tree by relative path and content.

    Artifact trees, VCS metadata, virtual environments, and conventional cache
    directories are pruned before traversal.  Symlinked files that resolve
    outside ``repo_root`` are rejected rather than reading external state.
    """

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {root}")
    normalized_suffixes = frozenset(suffix.lower() for suffix in included_suffixes)
    files: list[tuple[str, Path]] = []

    for current_root, directory_names, file_names in os.walk(root, topdown=True):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _excluded_directory(
                name,
                excluded_directory_names,
                excluded_directory_prefixes,
            )
        )
        current = Path(current_root)
        for file_name in sorted(file_names):
            path = current / file_name
            if path.suffix.lower() not in normalized_suffixes:
                continue
            resolved = resolve_within_root(root, path, must_exist=True)
            if not resolved.is_file():
                continue
            files.append((path.relative_to(root).as_posix(), resolved))

    digest = hashlib.sha256()
    for relative_path, resolved_path in sorted(files):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved_path.read_bytes())
        digest.update(b"\0")
    return SourceTreeFingerprint(sha256=digest.hexdigest(), files=len(files))


def resolve_within_root(
    root: str | Path,
    candidate: str | Path,
    *,
    must_exist: bool = True,
    allow_root: bool = False,
) -> Path:
    """Resolve ``candidate`` and require it to remain below ``root``."""

    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Containment root is not a directory: {resolved_root}")
    raw_candidate = Path(candidate)
    if not raw_candidate.is_absolute():
        raw_candidate = resolved_root / raw_candidate
    try:
        resolved_candidate = raw_candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required contained path is missing: {raw_candidate}") from exc
    if resolved_candidate == resolved_root:
        if allow_root:
            return resolved_candidate
        raise ProvenanceValidationError(f"Expected a child path, got root itself: {resolved_root}")
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ProvenanceValidationError(
            f"Path escapes containment root: {resolved_candidate} (root: {resolved_root})"
        )
    return resolved_candidate


def _validate_path_component(value: str, field_name: str) -> None:
    component = Path(value)
    if (
        not value
        or value in {".", ".."}
        or component.is_absolute()
        or len(component.parts) != 1
        or component.name != value
    ):
        raise ValueError(f"{field_name} must be one non-special path component: {value!r}")


def resolve_latest_run(
    artifact_root: str | Path,
    experiment_id: str,
    *,
    pointer_name: str = "latest.json",
    require_utc_run_id: bool = True,
) -> Path:
    """Resolve an experiment's latest pointer to a contained direct-child run."""

    _validate_path_component(experiment_id, "experiment_id")
    _validate_path_component(pointer_name, "pointer_name")
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root}")
    experiment_root = resolve_within_root(root, experiment_id, must_exist=True)
    if not experiment_root.is_dir() or experiment_root.parent != root:
        raise ProvenanceValidationError(
            f"Experiment directory is not a direct child of artifact root: {experiment_root}"
        )
    raw_pointer_path = experiment_root / pointer_name
    if raw_pointer_path.is_symlink():
        raise ProvenanceValidationError(f"Latest pointer cannot be a symlink: {raw_pointer_path}")
    pointer_path = resolve_within_root(experiment_root, raw_pointer_path, must_exist=True)
    if pointer_path.parent != experiment_root:
        raise ProvenanceValidationError(
            f"Latest pointer is not directly inside experiment directory: {pointer_path}"
        )
    pointer = read_json_object_strict(pointer_path)
    raw_run_dir = pointer.get("run_dir")
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        raise ProvenanceValidationError(f"latest pointer lacks a string run_dir: {pointer_path}")
    candidate = Path(raw_run_dir)
    if not candidate.is_absolute():
        candidate = experiment_root / candidate
    run_dir = resolve_within_root(experiment_root, candidate, must_exist=True)
    if not run_dir.is_dir() or run_dir.parent != experiment_root:
        raise ProvenanceValidationError(
            f"Latest run is not a direct child of experiment directory: {run_dir}"
        )
    if require_utc_run_id and not UTC_RUN_ID_PATTERN.fullmatch(run_dir.name):
        raise ProvenanceValidationError(f"Latest run has an invalid UTC run ID: {run_dir.name!r}")
    return run_dir


def _required_string(payload: Mapping[str, Any], key: str, context: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ProvenanceValidationError(f"{context}: {key} must be a non-empty string.")
    return value


def _required_bool(payload: Mapping[str, Any], key: str, context: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ProvenanceValidationError(f"{context}: {key} must be a boolean.")
    return value


def _required_run_file(run_dir: Path, name: str) -> Path:
    raw_path = run_dir / name
    if raw_path.is_symlink():
        raise ProvenanceValidationError(f"Run metadata cannot be a symlink: {raw_path}")
    resolved = resolve_within_root(run_dir, raw_path, must_exist=True)
    if not resolved.is_file() or resolved.parent != run_dir:
        raise ProvenanceValidationError(f"Invalid run metadata file: {resolved}")
    return resolved


def _resolve_recorded_absolute_path(
    raw_path: str,
    *,
    field_name: str,
    context: Path,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        raise ProvenanceValidationError(f"{context}: {field_name} must be an absolute path.")
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ProvenanceValidationError(f"{context}: {field_name} does not exist: {path}") from exc


def _validate_sha256(value: Any, field_name: str, context: Path) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ProvenanceValidationError(
            f"{context}: {field_name} must be 64 lowercase hex characters."
        )
    return value


def validate_run_manifest(
    run_dir: str | Path,
    artifact_root: str | Path,
    *,
    requirements: ManifestValidationRequirements | None = None,
) -> ValidatedRun:
    """Validate one generic, completed v6.1 run and all recorded core hashes."""

    policy = requirements or ManifestValidationRequirements()
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {root}")
    resolved_run_dir = resolve_within_root(root, run_dir, must_exist=True)
    if not resolved_run_dir.is_dir():
        raise ProvenanceValidationError(f"Run path is not a directory: {resolved_run_dir}")
    if policy.require_utc_run_id and not UTC_RUN_ID_PATTERN.fullmatch(resolved_run_dir.name):
        raise ProvenanceValidationError(
            f"Run directory has an invalid UTC run ID: {resolved_run_dir.name!r}"
        )

    manifest_path = _required_run_file(resolved_run_dir, "run_manifest.json")
    report_path = _required_run_file(resolved_run_dir, "report.json")
    manifest = read_json_object_strict(manifest_path)
    report = read_json_object_strict(report_path)

    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        raise ProvenanceValidationError(
            f"{manifest_path}: schema_version must be a positive integer."
        )
    if (
        policy.accepted_schema_versions is not None
        and schema_version not in policy.accepted_schema_versions
    ):
        raise ProvenanceValidationError(
            f"{manifest_path}: unsupported schema_version {schema_version!r}."
        )

    experiment_id = _required_string(manifest, "experiment_id", manifest_path)
    try:
        _validate_path_component(experiment_id, "manifest experiment_id")
    except ValueError as exc:
        raise ProvenanceValidationError(str(exc)) from exc
    if policy.expected_experiment_id is not None and experiment_id != policy.expected_experiment_id:
        raise ProvenanceValidationError(
            f"{manifest_path}: experiment_id {experiment_id!r} does not match "
            f"{policy.expected_experiment_id!r}."
        )
    experiment_root = resolve_within_root(root, experiment_id, must_exist=True)
    if experiment_root.parent != root or resolved_run_dir.parent != experiment_root:
        raise ProvenanceValidationError(
            f"Run is not a direct child of its declared experiment directory: {resolved_run_dir}"
        )

    run_id = _required_string(manifest, "run_id", manifest_path)
    if run_id != resolved_run_dir.name:
        raise ProvenanceValidationError(
            f"{manifest_path}: run_id {run_id!r} does not match {resolved_run_dir.name!r}."
        )
    recorded_run_dir = _resolve_recorded_absolute_path(
        _required_string(manifest, "run_dir", manifest_path),
        field_name="run_dir",
        context=manifest_path,
    )
    if recorded_run_dir != resolved_run_dir:
        raise ProvenanceValidationError(f"{manifest_path}: recorded run_dir does not match.")
    recorded_root = _resolve_recorded_absolute_path(
        _required_string(manifest, "artifact_root", manifest_path),
        field_name="artifact_root",
        context=manifest_path,
    )
    if recorded_root != root:
        raise ProvenanceValidationError(f"{manifest_path}: recorded artifact_root does not match.")

    completed_at = manifest.get("completed_at_utc")
    if not isinstance(completed_at, str) or not completed_at:
        raise ProvenanceValidationError(f"{manifest_path}: completed_at_utc is required.")
    if policy.require_source_verified_at_completion and (
        manifest.get("source_fingerprint_verified_at_completion") is not True
    ):
        raise ProvenanceValidationError(
            f"{manifest_path}: source fingerprint was not verified at completion."
        )

    run_mode = _required_string(manifest, "run_mode", manifest_path)
    if policy.expected_run_mode is not None and run_mode != policy.expected_run_mode:
        raise ProvenanceValidationError(
            f"{manifest_path}: run_mode {run_mode!r} does not match {policy.expected_run_mode!r}."
        )
    eligibility = manifest.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ProvenanceValidationError(f"{manifest_path}: eligibility must be an object.")
    main_eligible = _required_bool(eligibility, "main", manifest_path)
    full_eligible = _required_bool(eligibility, "full", manifest_path)
    if policy.require_main_eligible and not main_eligible:
        raise ProvenanceValidationError(f"{manifest_path}: run is not main-eligible.")
    if policy.require_full_eligible and not full_eligible:
        raise ProvenanceValidationError(f"{manifest_path}: run is not full-eligible.")

    source_payload = manifest.get("source_fingerprint")
    if not isinstance(source_payload, dict):
        raise ProvenanceValidationError(f"{manifest_path}: source_fingerprint must be an object.")
    source = SourceTreeFingerprint.from_mapping(source_payload)
    if policy.expected_source_sha256 is not None and source.sha256 != policy.expected_source_sha256:
        raise ProvenanceValidationError(
            f"{manifest_path}: source fingerprint does not match expected source."
        )
    if policy.expected_source_files is not None and source.files != policy.expected_source_files:
        raise ProvenanceValidationError(
            f"{manifest_path}: source file count does not match expected source."
        )

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ProvenanceValidationError(f"{manifest_path}: config must be an object.")
    config_experiment_id = config.get("experiment_id")
    if config_experiment_id is not None and config_experiment_id != experiment_id:
        raise ProvenanceValidationError(
            f"{manifest_path}: config experiment_id does not match manifest."
        )
    config_sha256 = _validate_sha256(
        manifest.get("config_sha256"),
        "config_sha256",
        manifest_path,
    )
    if sha256_canonical_json(config) != config_sha256:
        raise ProvenanceValidationError(f"{manifest_path}: config payload SHA-256 mismatch.")

    config_path: Path | None = None
    config_file_sha256: str | None = None
    if policy.verify_config_file_sha256:
        config_path = _resolve_recorded_absolute_path(
            _required_string(manifest, "config_path", manifest_path),
            field_name="config_path",
            context=manifest_path,
        )
        if not config_path.is_file():
            raise ProvenanceValidationError(f"{manifest_path}: config_path is not a file.")
        config_file_sha256 = _validate_sha256(
            manifest.get("config_file_sha256"),
            "config_file_sha256",
            manifest_path,
        )
        if sha256_file(config_path) != config_file_sha256:
            raise ProvenanceValidationError(f"{manifest_path}: config file SHA-256 mismatch.")
    else:
        raw_config_path = manifest.get("config_path")
        if isinstance(raw_config_path, str) and Path(raw_config_path).is_absolute():
            try:
                config_path = Path(raw_config_path).resolve(strict=True)
            except FileNotFoundError:
                config_path = None
        raw_config_file_sha = manifest.get("config_file_sha256")
        if isinstance(raw_config_file_sha, str):
            config_file_sha256 = raw_config_file_sha

    report_sha256 = _validate_sha256(
        manifest.get("report_sha256"),
        "report_sha256",
        manifest_path,
    )
    if sha256_file(report_path) != report_sha256:
        raise ProvenanceValidationError(f"{manifest_path}: report SHA-256 mismatch.")

    if report.get("experiment_id") != experiment_id:
        raise ProvenanceValidationError(f"{report_path}: experiment_id does not match manifest.")
    if report.get("run_id") != run_id:
        raise ProvenanceValidationError(f"{report_path}: run_id does not match manifest.")
    status = _required_string(report, "status", report_path)
    if manifest.get("status") != status:
        raise ProvenanceValidationError(f"{manifest_path}: status does not match report.")
    if policy.allowed_statuses is not None and status not in policy.allowed_statuses:
        raise ProvenanceValidationError(
            f"{report_path}: status {status!r} is not allowed by validation policy."
        )
    report_run_mode = report.get("run_mode")
    if report_run_mode is not None and report_run_mode != run_mode:
        raise ProvenanceValidationError(f"{report_path}: run_mode does not match manifest.")
    report_eligibility = report.get("eligibility")
    if report_eligibility is not None and report_eligibility != eligibility:
        raise ProvenanceValidationError(f"{report_path}: eligibility does not match manifest.")
    report_source = report.get("source_fingerprint")
    if report_source is not None and report_source != source.as_dict():
        raise ProvenanceValidationError(
            f"{report_path}: source_fingerprint does not match manifest."
        )

    dependencies = manifest.get("dependencies")
    if policy.require_dependencies_list and not isinstance(dependencies, list):
        raise ProvenanceValidationError(f"{manifest_path}: dependencies must be a list.")

    return ValidatedRun(
        artifact_root=root,
        experiment_id=experiment_id,
        run_id=run_id,
        run_dir=resolved_run_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        config_path=config_path,
        schema_version=schema_version,
        status=status,
        run_mode=run_mode,
        main_eligible=main_eligible,
        full_eligible=full_eligible,
        source_fingerprint=source,
        config_sha256=config_sha256,
        config_file_sha256=config_file_sha256,
        report_sha256=report_sha256,
        manifest_sha256=sha256_file(manifest_path),
        manifest=manifest,
        report=report,
    )


def validate_latest_run(
    artifact_root: str | Path,
    experiment_id: str,
    *,
    requirements: ManifestValidationRequirements | None = None,
    pointer_name: str = "latest.json",
) -> ValidatedRun:
    """Resolve and validate the latest completed run for ``experiment_id``."""

    policy = requirements or ManifestValidationRequirements()
    if policy.expected_experiment_id is not None and policy.expected_experiment_id != experiment_id:
        raise ValueError(
            "requirements.expected_experiment_id conflicts with experiment_id argument."
        )
    policy = replace(policy, expected_experiment_id=experiment_id)
    run_dir = resolve_latest_run(
        artifact_root,
        experiment_id,
        pointer_name=pointer_name,
        require_utc_run_id=policy.require_utc_run_id,
    )
    return validate_run_manifest(run_dir, artifact_root, requirements=policy)
