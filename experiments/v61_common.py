from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from catena.core.config import load_config
from catena.core.io import ensure_artifact_dir
from catena.core.provenance_v61 import (
    ManifestValidationRequirements,
    ProvenanceValidationError,
    SourceTreeFingerprint,
    ValidatedRun,
    read_json_object_strict,
    resolve_latest_run,
    sha256_canonical_json,
    sha256_file,
    source_tree_fingerprint,
    validate_latest_run,
    write_json_strict,
)
from catena.systems.device import resolve_device

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class V61RunContext:
    experiment_id: str
    artifact_root: Path
    run_dir: Path
    run_id: str
    config_path: Path
    config: dict[str, Any]
    device_request: str
    device: torch.device
    run_mode: str
    source_fingerprint: SourceTreeFingerprint
    dependencies: list[dict[str, Any]]
    created_at_utc: str


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cwd": str(Path.cwd().resolve()),
    }


def initialize_v61_run(
    *,
    experiment_id: str,
    config_path: str,
    artifact_root: str,
    device_request: str,
    dry_run: bool,
    dependencies: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path, torch.device, V61RunContext]:
    """Create a new v6.1 run only after the caller has validated dependencies."""

    absolute_config = Path(config_path).resolve(strict=True)
    config = load_config(absolute_config)
    if config.get("experiment_id") != experiment_id:
        raise ValueError(
            f"Config experiment_id {config.get('experiment_id')!r} does not match "
            f"{experiment_id!r}."
        )
    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_artifact_dir(root, experiment_id).resolve(strict=True)
    device = resolve_device(device_request)
    source = source_tree_fingerprint(REPO_ROOT)
    run_mode = "dry_run" if dry_run else "main"
    created_at = _now_utc()
    context = V61RunContext(
        experiment_id=experiment_id,
        artifact_root=root,
        run_dir=run_dir,
        run_id=run_dir.name,
        config_path=absolute_config,
        config=config,
        device_request=device_request,
        device=device,
        run_mode=run_mode,
        source_fingerprint=source,
        dependencies=dependencies,
        created_at_utc=created_at,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "artifact_root": str(root),
        "created_at_utc": created_at,
        "completed_at_utc": None,
        "status": "RUNNING",
        "run_mode": run_mode,
        "eligibility": {"main": False, "full": False},
        "config": config,
        "config_path": str(absolute_config),
        "config_sha256": sha256_canonical_json(config),
        "config_file_sha256": sha256_file(absolute_config),
        "source_fingerprint": source.as_dict(),
        "source_fingerprint_verified_at_completion": False,
        "report_sha256": None,
        "dependencies": dependencies,
        "device": {
            "requested": device_request,
            "resolved": str(device),
        },
        "environment": _environment(),
        "command": [str(value) for value in sys.argv],
        "process": {"pid": os.getpid()},
        "scientific_evidence": False,
    }
    write_json_strict(run_dir / "run_manifest.json", manifest)
    return config, run_dir, device, context


def finalize_v61_run(
    *,
    context: V61RunContext,
    report: dict[str, Any],
    main_eligible: bool,
    full_eligible: bool,
) -> None:
    """Atomically finalize report, manifest, and latest pointer after source lock."""

    completion_source = source_tree_fingerprint(REPO_ROOT)
    if completion_source != context.source_fingerprint:
        raise ProvenanceValidationError(
            "Source tree changed while the experiment was running; the run remains "
            "incomplete and latest.json was not updated."
        )
    status = report.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("A completed report requires a nonempty status string.")
    eligibility = {
        "main": bool(main_eligible),
        "full": bool(full_eligible),
    }
    completed_report = {
        **report,
        "experiment_id": context.experiment_id,
        "run_id": context.run_id,
        "run_mode": context.run_mode,
        "eligibility": eligibility,
        "source_fingerprint": context.source_fingerprint.as_dict(),
        "scientific_evidence": False,
    }
    report_path = context.run_dir / "report.json"
    write_json_strict(report_path, completed_report)
    completed_at = _now_utc()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": context.experiment_id,
        "run_id": context.run_id,
        "run_dir": str(context.run_dir),
        "artifact_root": str(context.artifact_root),
        "created_at_utc": context.created_at_utc,
        "completed_at_utc": completed_at,
        "status": status,
        "run_mode": context.run_mode,
        "eligibility": eligibility,
        "config": context.config,
        "config_path": str(context.config_path),
        "config_sha256": sha256_canonical_json(context.config),
        "config_file_sha256": sha256_file(context.config_path),
        "source_fingerprint": context.source_fingerprint.as_dict(),
        "source_fingerprint_verified_at_completion": True,
        "report_sha256": sha256_file(report_path),
        "dependencies": context.dependencies,
        "device": {
            "requested": context.device_request,
            "resolved": str(context.device),
        },
        "environment": _environment(),
        "command": [str(value) for value in sys.argv],
        "process": {"pid": os.getpid()},
        "scientific_evidence": False,
    }
    write_json_strict(context.run_dir / "run_manifest.json", manifest)
    write_json_strict(
        context.artifact_root / context.experiment_id / "latest.json",
        {"run_dir": context.run_id},
    )


def _count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _validated_run_file(
    run_dir: Path,
    filename: object,
    expected_sha256: object,
) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ProvenanceValidationError(
            f"Artifact filename is not a safe direct child: {filename!r}."
        )
    path = run_dir / filename
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Required run artifact is missing: {path}.")
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise ProvenanceValidationError(f"Run artifact hash mismatch: {path}.")
    return path


def _geometry_grid_episode_count(
    grid: dict[str, Any],
    *,
    count_per_cell: int,
) -> int:
    cells = 1
    for key in (
        "num_associations",
        "key_correlations",
        "old_scales",
        "new_scales",
        "old_new_cosines",
    ):
        values = grid.get(key)
        if not isinstance(values, list) or not values:
            raise ProvenanceValidationError(f"Invalid geometry grid axis: {key}.")
        cells *= len(values)
    return cells * 4 * count_per_cell


def _expected_e01b_contract(config: dict[str, Any]) -> tuple[int, int]:
    seeds = config.get("seeds")
    data = config.get("data")
    if not isinstance(seeds, list) or not isinstance(data, dict):
        raise ProvenanceValidationError("E01b config lacks seeds/data contract.")
    train_grid = data.get("train_grid")
    test_grid = data.get("test_grid")
    if not isinstance(train_grid, dict) or not isinstance(test_grid, dict):
        raise ProvenanceValidationError("E01b config lacks train/test geometry grids.")
    train_per_condition = _geometry_grid_episode_count(
        train_grid,
        count_per_cell=int(data["train_count_per_cell"]),
    )
    test_per_condition = _geometry_grid_episode_count(
        test_grid,
        count_per_cell=int(data["test_count_per_cell"]),
    )
    condition_count = 2 * 2
    return (
        len(seeds) * condition_count * (train_per_condition + test_per_condition),
        len(seeds) * condition_count,
    )


def _expected_e02_contract(config: dict[str, Any]) -> tuple[int, int]:
    seeds = config.get("seeds")
    data = config.get("data")
    if not isinstance(seeds, list) or not isinstance(data, dict):
        raise ProvenanceValidationError("E02 config lacks seeds/data contract.")
    return (
        len(seeds) * 4 * int(data["test_count_per_operation"]),
        len(seeds) * 2,
    )


def _legacy_dependency_record(
    *,
    experiment_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
    evidence_role: str,
    main_full: bool,
) -> dict[str, Any]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ProvenanceValidationError(
            f"{run_dir}/run_manifest.json: config must be an object."
        )
    return {
        "schema_version": 0,
        "legacy_schema": "catena-v6.0-pilot",
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": report.get("status"),
        "run_mode": "main" if main_full else "dry_run",
        "eligibility": {"main": main_full, "full": main_full},
        "evidence_role": evidence_role,
        "config_sha256": sha256_canonical_json(config),
        "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "report_sha256": sha256_file(run_dir / "report.json"),
        "scientific_evidence": False,
    }


def validate_legacy_e00(
    artifact_root: str | Path,
    *,
    require_full: bool,
) -> dict[str, Any]:
    """Validate the latest E00 infrastructure lock against the current source."""

    root = Path(artifact_root).expanduser().resolve(strict=True)
    current_source = source_tree_fingerprint(REPO_ROOT)
    e00_dir = resolve_latest_run(root, "e00_protocol_lock")
    e00_manifest = read_json_object_strict(e00_dir / "run_manifest.json")
    e00_report = read_json_object_strict(e00_dir / "report.json")
    expected_e00_config = load_config(REPO_ROOT / "configs/e00_protocol_lock.yaml")
    if (
        e00_manifest.get("experiment_id") != "e00_protocol_lock"
        or e00_manifest.get("config") != expected_e00_config
    ):
        raise ProvenanceValidationError("E00 identity/config does not match v6.1 source.")
    if e00_report.get("status") != "PASS" or e00_report.get("failures") != []:
        raise ProvenanceValidationError("Latest E00 did not complete with zero failures.")
    source_payload = e00_report.get("source_fingerprint")
    if not isinstance(source_payload, dict):
        raise ProvenanceValidationError("E00 lacks a source fingerprint.")
    if (
        source_payload.get("sha256") != current_source.sha256
        or source_payload.get("files") != current_source.files
    ):
        raise ProvenanceValidationError(
            "Latest E00 source fingerprint does not match the current frozen source."
        )
    selected = expected_e00_config["environment"]["selected_gpu_indices"]
    gpu_checks = e00_report.get("gpu_bf16_checks")
    e00_full = bool(
        isinstance(gpu_checks, list)
        and len(gpu_checks) == len(selected)
        and e00_report.get("counts", {}).get("fail") == 0
    )
    if require_full and not e00_full:
        raise ProvenanceValidationError("E00 lacks the full selected-GPU parity contract.")
    return _legacy_dependency_record(
        experiment_id="e00_protocol_lock",
        run_dir=e00_dir,
        manifest=e00_manifest,
        report=e00_report,
        evidence_role="v6.1_infrastructure_lock_not_scientific_evidence",
        main_full=e00_full,
    )


def validate_legacy_e00_e01(
    artifact_root: str | Path,
    *,
    require_full: bool,
) -> list[dict[str, Any]]:
    """Validate the immutable v6.0 E00 infrastructure and E01 pilot artifacts."""

    root = Path(artifact_root).expanduser().resolve(strict=True)
    e00_record = validate_legacy_e00(root, require_full=require_full)
    records = [e00_record]
    e00_dir = Path(str(e00_record["run_dir"]))
    e01_dir = resolve_latest_run(root, "e01_local_controllability")
    if e01_dir.name <= e00_dir.name:
        raise ProvenanceValidationError("E01 pilot must have completed after the E00 lock.")
    e01_manifest = read_json_object_strict(e01_dir / "run_manifest.json")
    e01_report = read_json_object_strict(e01_dir / "report.json")
    expected_e01_config = load_config(
        REPO_ROOT / "configs/e01_local_controllability.yaml"
    )
    if (
        e01_manifest.get("experiment_id") != "e01_local_controllability"
        or e01_manifest.get("config") != expected_e01_config
    ):
        raise ProvenanceValidationError("E01 identity/config does not match v6.1 source.")
    if e01_report.get("status") not in {"PASS", "WARN"}:
        raise ProvenanceValidationError("Latest E01 pilot did not complete.")
    row_path = e01_dir / "episode_metrics.jsonl"
    row_count = _count_jsonl_rows(row_path)
    checkpoint_count = len(list(e01_dir.glob("seed*_*.pt")))
    e01_full = row_count == 3072 and checkpoint_count == 32
    e01_dry = row_count == 80 and checkpoint_count == 8
    if require_full and not e01_full:
        raise ProvenanceValidationError(
            f"E01 full-shape mismatch: rows={row_count}, checkpoints={checkpoint_count}."
        )
    if not require_full and not (e01_full or e01_dry):
        raise ProvenanceValidationError(
            f"E01 dry/full shape is invalid: rows={row_count}, "
            f"checkpoints={checkpoint_count}."
        )
    e01_record = _legacy_dependency_record(
        experiment_id="e01_local_controllability",
        run_dir=e01_dir,
        manifest=e01_manifest,
        report=e01_report,
        evidence_role="v6.0_pilot_nonconfirmatory_construct_diagnostic",
        main_full=e01_full,
    )
    e01_record["artifact_contract"] = {
        "episode_metrics_rows": row_count,
        "episode_metrics_sha256": sha256_file(row_path),
        "checkpoint_count": checkpoint_count,
    }
    records.append(e01_record)
    return records


def validate_e01b_dependency(
    artifact_root: str | Path,
    *,
    require_main_supported: bool,
) -> ValidatedRun:
    current_source = source_tree_fingerprint(REPO_ROOT)
    requirements = ManifestValidationRequirements(
        expected_experiment_id="e01b_constrained_behavioral_reachability",
        accepted_schema_versions=frozenset({SCHEMA_VERSION}),
        expected_source_sha256=current_source.sha256,
        expected_source_files=current_source.files,
        expected_run_mode="main" if require_main_supported else None,
        require_main_eligible=require_main_supported,
        require_full_eligible=require_main_supported,
    )
    validated = validate_latest_run(
        artifact_root,
        "e01b_constrained_behavioral_reachability",
        requirements=requirements,
    )
    primary = validated.report.get("primary")
    claim = validated.report.get("claim_gate")
    execution = validated.report.get("execution")
    if not isinstance(primary, dict) or not isinstance(claim, dict):
        raise ProvenanceValidationError("E01b report lacks its H1 claim contract.")
    expected_rows, expected_checkpoints = _expected_e01b_contract(
        validated.manifest["config"]
    )
    if require_main_supported and (
        primary.get("supported") is not True
        or claim.get("supported") is not True
        or not isinstance(execution, dict)
        or execution.get("dry_run") is not False
        or execution.get("row_count") != expected_rows
        or execution.get("checkpoint_count") != expected_checkpoints
    ):
        raise ProvenanceValidationError(
            "E02 main is blocked: E01b is not a supported full H1 artifact."
        )
    artifact = validated.report.get("artifacts", {}).get(
        "episode_geometry_metrics", {}
    )
    if not isinstance(artifact, dict):
        raise ProvenanceValidationError("E01b episode artifact contract is missing.")
    episode_path = _validated_run_file(
        validated.run_dir,
        artifact.get("path"),
        artifact.get("sha256"),
    )
    if artifact.get("rows") != _count_jsonl_rows(episode_path):
        raise ProvenanceValidationError("E01b episode artifact row count mismatch.")
    checkpoints = validated.report.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ProvenanceValidationError("E01b checkpoint contract is missing.")
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise ProvenanceValidationError("E01b checkpoint entry is invalid.")
        _validated_run_file(
            validated.run_dir,
            checkpoint.get("path"),
            checkpoint.get("file_sha256"),
        )
        if checkpoint.get("round_trip_match") is not True:
            raise ProvenanceValidationError("E01b checkpoint round-trip failed.")
    return validated


def validate_e02_dependency(
    artifact_root: str | Path,
    *,
    require_main_supported: bool,
) -> ValidatedRun:
    current_source = source_tree_fingerprint(REPO_ROOT)
    validated = validate_latest_run(
        artifact_root,
        "e02_magnitude_factorization",
        requirements=ManifestValidationRequirements(
            expected_experiment_id="e02_magnitude_factorization",
            accepted_schema_versions=frozenset({SCHEMA_VERSION}),
            expected_source_sha256=current_source.sha256,
            expected_source_files=current_source.files,
            expected_run_mode="main" if require_main_supported else None,
            require_main_eligible=require_main_supported,
            require_full_eligible=require_main_supported,
        ),
    )
    claim = validated.report.get("claim_gate")
    execution = validated.report.get("execution")
    expected_rows, expected_checkpoints = _expected_e02_contract(
        validated.manifest["config"]
    )
    if require_main_supported and (
        not isinstance(claim, dict)
        or claim.get("supported") is not True
        or not isinstance(execution, dict)
        or execution.get("dry_run") is not False
        or execution.get("row_count") != expected_rows
        or execution.get("strict_checkpoint_count") != expected_checkpoints
    ):
        raise ProvenanceValidationError(
            "E04 main is blocked: E02 is not a supported full H2 artifact."
        )
    artifact = validated.report.get("artifacts", {}).get("episode_metrics", {})
    if not isinstance(artifact, dict):
        raise ProvenanceValidationError("E02 episode artifact contract is missing.")
    episode_path = _validated_run_file(
        validated.run_dir,
        artifact.get("path"),
        artifact.get("sha256"),
    )
    if artifact.get("rows") != _count_jsonl_rows(episode_path):
        raise ProvenanceValidationError("E02 episode artifact row count mismatch.")
    checkpoint_contract = validated.report.get("strict_checkpoint_contract")
    if not isinstance(checkpoint_contract, dict):
        raise ProvenanceValidationError("E02 strict checkpoint contract is missing.")
    entries = checkpoint_contract.get("entries")
    if not isinstance(entries, list):
        raise ProvenanceValidationError("E02 checkpoint entries are missing.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProvenanceValidationError("E02 checkpoint entry is invalid.")
        _validated_run_file(
            validated.run_dir,
            entry.get("filename"),
            entry.get("sha256"),
        )
        if entry.get("round_trip_match") is not True:
            raise ProvenanceValidationError("E02 checkpoint round-trip failed.")
    if require_main_supported and (
        checkpoint_contract.get("main_contract_complete") is not True
        or len(entries) != expected_checkpoints
    ):
        raise ProvenanceValidationError("E02 lacks all 16 immutable strict checkpoints.")
    dependencies = validated.manifest.get("dependencies")
    if (
        not isinstance(dependencies, list)
        or not dependencies
        or not isinstance(dependencies[0], dict)
        or dependencies[0].get("experiment_id")
        != "e01b_constrained_behavioral_reachability"
    ):
        raise ProvenanceValidationError("E02 manifest lacks locked E01b lineage.")
    return validated
