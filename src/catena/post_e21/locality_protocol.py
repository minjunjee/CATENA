from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.io import file_sha256
from catena.post_e21.contracts import PostE21ContractError, ProtocolSnapshot

PARENT_LOCK_RELATIVE = "docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json"
EXPECTED_THRESHOLD_KEYS = frozenset(
    {
        "selective_gain",
        "maximum_nontarget_degradation",
        "retention_noninferiority",
        "minimum_seed_direction_fraction",
        "exact_sign_flip_alpha",
        "minimum_address_accuracy",
        "maximum_candidate_recovery_mse",
        "maximum_capable_affected_mse",
        "maximum_oracle_floor_mse",
        "minimum_verified_activity",
        "maximum_distractor_activity",
        "complete_paired_grid_and_provenance_required",
    }
)


@dataclass(frozen=True, slots=True)
class ParentThresholdContract:
    path: Path
    sha256: str
    thresholds: dict[str, float | bool]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PostE21ContractError(f"Missing or unsafe JSON dependency: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostE21ContractError(f"Cannot read JSON dependency {path}") from error
    if not isinstance(payload, dict):
        raise PostE21ContractError(f"Expected JSON object: {path}")
    return payload


def load_parent_threshold_contract(
    *,
    repo_root: str | Path,
    relative_path: str = PARENT_LOCK_RELATIVE,
) -> ParentThresholdContract:
    """Load every threshold from the immutable E21 lock at runtime."""

    root = Path(repo_root).resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PostE21ContractError("E21 threshold lock escapes repository") from error
    payload = _read_json_object(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_family") != "E21"
        or payload.get("protocol_frozen_before_any_e21_evaluation") is not True
    ):
        raise PostE21ContractError("E21 parent lock is not a prospective E21 lock")
    raw = payload.get("registered_thresholds")
    if not isinstance(raw, dict) or set(raw) != EXPECTED_THRESHOLD_KEYS:
        raise PostE21ContractError("E21 parent lock threshold key set changed or is incomplete")
    thresholds: dict[str, float | bool] = {}
    for key, value in raw.items():
        if key == "complete_paired_grid_and_provenance_required":
            if not isinstance(value, bool):
                raise PostE21ContractError(f"E21 threshold {key} is not boolean")
            thresholds[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PostE21ContractError(f"E21 threshold {key} is not numeric")
        thresholds[key] = float(value)
    return ParentThresholdContract(
        path=path,
        sha256=file_sha256(path),
        thresholds=thresholds,
    )


def validate_parent_binding(
    *,
    snapshot: ProtocolSnapshot,
    parent: ParentThresholdContract,
) -> None:
    """Require the static E22 lock to bind the exact parent and all values."""

    payload = snapshot.payload
    if payload.get("parent_e21_lock_path") != PARENT_LOCK_RELATIVE:
        raise PostE21ContractError("E22 lock declares the wrong E21 parent path")
    if payload.get("parent_e21_lock_sha256") != parent.sha256:
        raise PostE21ContractError("E22 lock parent E21 SHA-256 mismatch")
    inherited = payload.get("inherited_thresholds")
    if inherited != parent.thresholds:
        raise PostE21ContractError(
            "E22 lock threshold snapshot differs from dynamic E21 inheritance"
        )


def require_temp_dry_root(artifact_root: str | Path) -> Path:
    """Dry-runs are allowed only below /tmp and never in the canonical tree."""

    root = Path(artifact_root).resolve()
    temporary = Path("/tmp").resolve()
    try:
        root.relative_to(temporary)
    except ValueError as error:
        raise PostE21ContractError(
            "E22 dry-run artifact root must be a fresh path below /tmp"
        ) from error
    canonical = Path("/data/minjun_dev/CATENA/artifacts").resolve()
    if root == canonical or canonical in root.parents:
        raise PostE21ContractError("E22 dry-run cannot use canonical artifacts")
    return root


def validate_e21_freeze_dependency(
    *,
    freeze_path: str | Path,
    parent: ParentThresholdContract,
) -> dict[str, Any]:
    """Validate the explicit E21 outcome freeze required by E22a MAIN."""

    path = Path(freeze_path).resolve(strict=True)
    payload = _read_json_object(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_family") != "E21"
        or payload.get("immutable") is not True
        or payload.get("claim_status") != "NOT_SUPPORTED"
    ):
        raise PostE21ContractError("E22a requires the immutable E21 NOT_SUPPORTED freeze")
    locks = payload.get("locks_and_configs")
    source_lock = locks.get("source_protocol_lock") if isinstance(locks, dict) else None
    if not isinstance(source_lock, dict) or source_lock.get("sha256") != parent.sha256:
        raise PostE21ContractError("E21 freeze does not bind the E22 parent lock")
    repair = payload.get("repair_aggregate")
    if (
        not isinstance(repair, dict)
        or repair.get("execution_status") != "PASS"
        or repair.get("frozen_disposition") != "NOT_SUPPORTED"
    ):
        raise PostE21ContractError("E21 repaired aggregate disposition is invalid")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "claim_status": str(payload["claim_status"]),
        "repair_disposition": str(repair["frozen_disposition"]),
    }


def threshold_float(
    thresholds: Mapping[str, float | bool],
    key: str,
) -> float:
    value = thresholds[key]
    if isinstance(value, bool):
        raise TypeError(f"Threshold {key!r} is boolean, not numeric")
    return float(value)


def validate_selection_run_dependency(
    *,
    selection_run: str | Path,
    parent: ParentThresholdContract,
    expected_protocol_lock_sha256: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Validate the explicit E22a artifact that unlocks E22b."""

    run = Path(selection_run).resolve(strict=True)
    if not run.is_dir() or run.is_symlink():
        raise PostE21ContractError("E22b selection dependency is not a safe directory")
    manifest_path = run / "run_manifest.json"
    report_path = run / "report.json"
    selection_path = run / "selection_lock.json"
    raw_path = run / "raw_metrics.jsonl"
    seed_path = run / "seed_metrics.jsonl"
    scores_path = run / "selection_scores.jsonl"
    active_cells_path = run / "active_cell_metrics.jsonl"
    manifest = _read_json_object(manifest_path)
    report = _read_json_object(report_path)
    selection = _read_json_object(selection_path)
    required_mode = "DRY_RUN" if dry_run else "MAIN"
    required_selection_status = "DRY_RUN_SELECTED_NON_EVIDENCE" if dry_run else "SELECTED"
    if (
        manifest.get("experiment_id") != "e22a_locality_method_selection"
        or manifest.get("run_id") != run.name
        or manifest.get("run_mode") != required_mode
        or report.get("status") != "PASS"
        or report.get("run_mode") != required_mode
        or selection.get("experiment_id") != "e22a_locality_method_selection"
        or selection.get("run_id") != run.name
        or selection.get("run_mode") != required_mode
        or selection.get("selection_status") != required_selection_status
    ):
        raise PostE21ContractError("E22a selection phase identity/status mismatch")
    if file_sha256(report_path) != manifest.get("report_sha256"):
        raise PostE21ContractError("E22a report differs from its finalized manifest")
    artifacts = report.get("artifacts")
    selection_record = artifacts.get("selection_lock") if isinstance(artifacts, dict) else None
    if not isinstance(selection_record, dict) or selection_record.get("sha256") != file_sha256(
        selection_path
    ):
        raise PostE21ContractError("E22a selection lock hash is not report-bound")
    if (
        selection.get("parent_e21_lock_sha256") != parent.sha256
        or selection.get("e22a_protocol_lock_sha256") != expected_protocol_lock_sha256
        or bool(selection.get("main_confirmatory_unlock")) != (not dry_run)
    ):
        raise PostE21ContractError("E22a selection provenance/unlock mismatch")
    selected = selection.get("selected_method")
    baseline = selection.get("baseline_method")
    if (
        not isinstance(selected, dict)
        or not isinstance(baseline, dict)
        or selected.get("selection_eligible") is not True
        or selected.get("baseline") is not False
        or baseline.get("baseline") is not True
        or baseline.get("method_id") != "mean_retention"
    ):
        raise PostE21ContractError("E22a selected/baseline method contract is invalid")
    for path, key in (
        (raw_path, "raw_metrics_sha256"),
        (seed_path, "seed_metrics_sha256"),
        (scores_path, "selection_scores_sha256"),
        (active_cells_path, "active_cell_metrics_sha256"),
    ):
        if not path.is_file() or file_sha256(path) != selection.get(key):
            raise PostE21ContractError(f"E22a selection input changed: {path.name}")
    return {
        "run_dir": str(run),
        "run_id": run.name,
        "run_mode": required_mode,
        "report_sha256": file_sha256(report_path),
        "selection_lock_path": str(selection_path),
        "selection_lock_sha256": file_sha256(selection_path),
        "selected_method": selected,
        "baseline_method": baseline,
        "parent_e21_lock_sha256": parent.sha256,
        "e22a_protocol_lock_sha256": expected_protocol_lock_sha256,
    }
