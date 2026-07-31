from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from catena.core.provenance_v61 import (
    DEFAULT_SOURCE_SUFFIXES,
    dumps_json_strict,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    source_tree_fingerprint,
    write_json_strict,
)


class ArtifactContractError(RuntimeError):
    """Raised when an E26+ run would violate the immutable artifact contract."""


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _safe_child(parent: Path, name: str) -> Path:
    candidate = Path(name)
    if not name or candidate.is_absolute() or len(candidate.parts) != 1 or name in {".", ".."}:
        raise ArtifactContractError(f"Artifact name must be one path component: {name!r}")
    return parent / candidate


def write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON atomically using the repository provenance helper."""

    write_json_strict(path, _jsonable(value))


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append strict canonical JSONL and fsync the resulting file.

    Metric streams are the only intentionally appendable run artifacts.  Every
    row is serialized before opening the destination so a non-finite value
    cannot leave a partially appended batch.
    """

    serialized = [dumps_json_strict(dict(row), canonical=True) + "\n" for row in rows]
    if not serialized:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ArtifactContractError(f"Metric stream cannot be a symlink: {destination}")
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.writelines(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def git_fingerprint(cwd: str | Path) -> dict[str, Any]:
    root = Path(cwd)

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "UNAVAILABLE"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain=v1"),
    }


_E26_SOURCE_SUFFIXES = DEFAULT_SOURCE_SUFFIXES | frozenset({".json", ".sh"})


class ArtifactRun:
    """One immutable E26+ run directory.

    Dry runs are confined to a newly named direct child of ``/tmp``.  Any
    non-dry execution is confined to the canonical artifact root.  This class
    does not decide whether a scientific protocol may run; it makes an
    accidental alternate-root or reference-as-MAIN artifact impossible.
    """

    def __init__(
        self,
        *,
        experiment: str,
        artifact_root: str | Path,
        run_mode: str,
        dry_run: bool,
        canonical_artifact_root: str | Path = "/data/minjun_dev/CATENA/artifacts",
        source_root: str | Path | None = None,
    ) -> None:
        root = Path(artifact_root).expanduser().resolve()
        canonical = Path(canonical_artifact_root).expanduser().resolve()
        if dry_run:
            if not root.name.startswith("catena_e26_dry_") and not root.name.startswith(
                "catena_v81_packet_smoke_"
            ):
                raise ValueError(
                    "Dry-run artifact root must be a new /tmp direct child prefixed "
                    "catena_e26_dry_ or catena_v81_packet_smoke_"
                )
            if root.parent != Path("/tmp"):
                raise ValueError("Dry-run artifact root must be a direct child of /tmp")
            if root == canonical or root.is_relative_to(canonical):
                raise ValueError("Dry-run root cannot be inside canonical artifact root")
            if run_mode != "DRY_RUN":
                raise ValueError("dry_run=True requires run_mode=DRY_RUN")
        elif root != canonical:
            raise ArtifactContractError(
                f"Non-dry runs require canonical artifact root {canonical}, got {root}"
            )
        elif run_mode != "MAIN":
            raise ArtifactContractError("Non-dry E26+ runs require run_mode=MAIN")

        self.experiment = experiment
        self.run_mode = run_mode
        self.dry_run = dry_run
        self.root = root
        self.source_root = Path(source_root or Path.cwd()).resolve(strict=True)
        if not self.source_root.is_dir():
            raise NotADirectoryError(self.source_root)
        self._source_fingerprint = source_tree_fingerprint(
            self.source_root,
            included_suffixes=_E26_SOURCE_SUFFIXES,
        )
        self.run_id = utc_run_id()
        experiment_root = _safe_child(root, experiment)
        self.run_dir = _safe_child(experiment_root, self.run_id)
        if self.run_dir.exists():
            raise FileExistsError(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._finalized = False
        self._write_initial_manifest()

    @property
    def source_fingerprint(self) -> dict[str, Any]:
        return self._source_fingerprint.as_dict()

    def _write_initial_manifest(self) -> None:
        cudnn_version = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        manifest = {
            "schema_version": "catena-v8.1",
            "experiment": self.experiment,
            "run_id": self.run_id,
            "run_mode": self.run_mode,
            "scientific_evidence": False if self.dry_run else None,
            "evidence_tier": "NON_EVIDENCE_VALIDATION" if self.dry_run else None,
            "utc": datetime.now(UTC).isoformat(),
            "cwd": str(Path.cwd().resolve()),
            "source_root": str(self.source_root),
            "source_fingerprint": self._source_fingerprint.as_dict(),
            "source_fingerprint_verified_at_completion": False,
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git": git_fingerprint(self.source_root),
            "artifact_root_realpath": str(self.root),
        }
        write_json_strict(self.run_dir / "run_manifest.json", manifest)

    def write(self, name: str, value: Any) -> Path:
        if self._finalized:
            raise ArtifactContractError("Cannot add files after run finalization")
        path = _safe_child(self.run_dir, name)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite run artifact: {path}")
        write_json_strict(path, _jsonable(value))
        return path

    def append(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        if self._finalized:
            raise ArtifactContractError("Cannot append after run finalization")
        path = _safe_child(self.run_dir, name)
        append_jsonl(path, rows)
        return path

    def checkpoint_dir(self) -> Path:
        if self._finalized:
            raise ArtifactContractError("Cannot create checkpoints after run finalization")
        destination = self.run_dir / "checkpoints"
        if destination.is_symlink():
            raise ArtifactContractError(f"Checkpoint directory cannot be a symlink: {destination}")
        destination.mkdir(exist_ok=True)
        return destination

    def _validate_report_boundary(self, report: Mapping[str, Any]) -> None:
        if report.get("run_id") != self.run_id:
            raise ArtifactContractError("report.run_id does not match its run directory")
        if report.get("experiment") != self.experiment:
            raise ArtifactContractError("report.experiment does not match ArtifactRun")
        if report.get("run_mode") != self.run_mode:
            raise ArtifactContractError("report.run_mode does not match ArtifactRun")
        if self.dry_run and (
            report.get("scientific_evidence") is not False
            or report.get("evidence_tier") != "NON_EVIDENCE_VALIDATION"
        ):
            raise ArtifactContractError(
                "Dry-run report must remain NON_EVIDENCE_VALIDATION with scientific_evidence=false"
            )

    def finalize(self, report: dict[str, Any], summary_markdown: str) -> None:
        if self._finalized:
            raise ArtifactContractError("Run was already finalized")
        self._validate_report_boundary(report)
        completion_source = source_tree_fingerprint(
            self.source_root,
            included_suffixes=_E26_SOURCE_SUFFIXES,
        )
        source_unchanged = completion_source == self._source_fingerprint
        if not self.dry_run and not source_unchanged:
            raise ArtifactContractError(
                "Scientific source changed while the run was active; no final report was written"
            )
        report_path = self.run_dir / "report.json"
        summary_path = self.run_dir / "RESULTS_SUMMARY_KO.md"
        if report_path.exists() or summary_path.exists():
            raise FileExistsError("Refusing to overwrite existing final run artifacts")
        write_json_strict(report_path, report)
        summary_path.write_text(summary_markdown.rstrip() + "\n", encoding="utf-8")

        manifest_path = self.run_dir / "run_manifest.json"
        manifest = read_json_object_strict(manifest_path)
        manifest.update(
            {
                "completion_utc": datetime.now(UTC).isoformat(),
                "source_fingerprint_verified_at_completion": source_unchanged,
                "report_sha256": sha256_file(report_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )
        write_json_strict(manifest_path, manifest)

        artifacts: dict[str, dict[str, Any]] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_symlink():
                raise ArtifactContractError(f"Run artifacts cannot contain symlinks: {path}")
            if path.is_file():
                artifacts[str(path.relative_to(self.run_dir))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        index_path = self.run_dir / "artifact_index.json"
        write_json_strict(index_path, artifacts)
        latest = self.root / self.experiment / "latest.json"
        if latest.is_symlink():
            raise ArtifactContractError(f"Latest pointer cannot be a symlink: {latest}")
        write_json_strict(
            latest,
            {
                "schema_version": "catena-v8.1",
                "experiment": self.experiment,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "report_sha256": sha256_file(report_path),
                "artifact_index_sha256": sha256_file(index_path),
            },
        )
        self._finalized = True

    def artifact_hash(self) -> str:
        records = []
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                records.append(
                    {
                        "path": str(path.relative_to(self.run_dir)),
                        "sha256": sha256_file(path),
                    }
                )
        return sha256_canonical_json(records)
