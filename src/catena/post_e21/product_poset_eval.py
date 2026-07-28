"""Evaluation, dependency, and prospective-boundary logic for E23."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.io import file_sha256
from catena.core.provenance_v61 import sha256_canonical_json
from catena.data.controller_poset import (
    CANONICAL_CONTROLLERS,
    CONTROLLER_BY_ID,
    DEMAND_FAMILIES,
    PAIRWISE_DEMANDS,
    SINGLE_AXIS_DEMANDS,
    immediate_lower_covers,
    minimal_elements,
    required_controller,
    same_rank_incomparable_controllers,
    theory_adequate_controller_ids,
    theory_boundary_controller_ids,
    theory_minimal_controller_ids,
    theory_prediction_payload,
)
from catena.post_e21.contracts import (
    PostE21ContractError,
    ProtocolSnapshot,
)
from catena.post_e21.locality_data import (
    LocalityMethod,
    LocalityObjective,
    method_by_id,
    parse_locality_methods,
)

SAFE_E22_STATUS = "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
E22B_PROTOCOL_LOCK_SHA256 = "e19dfd26018e53d7ab601d1bd1b0e94c3bd922e1849c35cdcffec7ae38474598"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_E22A_CONFIG_PATH = _REPO_ROOT / "configs/e22a_locality_method_selection.yaml"
_LOCALITY_RISK_SCALE = 0.0005
NON_SAFE_E22_STATUSES = frozenset(
    {
        "CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED",
        "OVERREGULARIZED_LOCALITY_TRADEOFF",
        "NOT_SUPPORTED",
    }
)


@dataclass(frozen=True, slots=True)
class E22DependencyDecision:
    execution_status: str
    boundary_mode: str | None
    reason: str
    synthetic: bool
    report_path: str | None
    report_sha256: str | None
    protocol_lock_sha256: str | None
    evidence_tier: str | None
    dependency_claim_eligible: bool
    dependency_claim_status: str | None
    safe_locality_supported: bool
    locality_method: dict[str, Any]
    locality_risk_scale: float
    safe_objective_implemented: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class E18DependencyDecision:
    execution_status: str
    reason: str
    synthetic: bool
    freeze_path: str | None
    freeze_sha256: str | None
    report_path: str | None
    report_sha256: str | None
    claim_status: str | None
    evidence_tier: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class E23aScreenDependency:
    execution_status: str
    reason: str
    synthetic: bool
    report_path: str | None
    report_sha256: str | None
    e18_freeze_sha256: str | None
    outcomes_used_for_boundary: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _registered_locality_method(
    payload: Mapping[str, Any] | None,
    *,
    method_id: str | None = None,
) -> LocalityMethod:
    config = load_config(_E22A_CONFIG_PATH)
    methods = parse_locality_methods(config["methods"])
    selected_id = (
        method_id
        if method_id is not None
        else str(payload.get("method_id") if payload is not None else "")
    )
    method = method_by_id(methods, selected_id)
    if payload is not None and method.as_dict() != dict(payload):
        raise ValueError("E22 selected locality method differs from frozen grid")
    return method


def mean_locality_method_payload() -> dict[str, Any]:
    """Return the exact frozen E22 mean-retention baseline descriptor."""

    return _registered_locality_method(
        None,
        method_id="mean_retention",
    ).as_dict()


def resolve_e18b_freeze(
    *,
    freeze_path: str | Path | None,
    dry_run: bool,
) -> E18DependencyDecision:
    """Validate the immutable supported E18b freeze before any E23 MAIN."""

    if dry_run:
        # Deliberately do not inspect a canonical artifact during a non-evidence
        # dry-run.  The fixture only exercises dependency plumbing.
        return E18DependencyDecision(
            execution_status="PASS",
            reason="SYNTHETIC_DRY_RUN_E18B_DEPENDENCY_NON_EVIDENCE",
            synthetic=True,
            freeze_path=None,
            freeze_sha256=None,
            report_path=None,
            report_sha256=None,
            claim_status="SUPPORTED",
            evidence_tier="CONTROLLED_REFERENCE",
        )
    if freeze_path is None:
        return E18DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            reason="EXPLICIT_E18B_SUPPORTED_FREEZE_REQUIRED",
            synthetic=False,
            freeze_path=None,
            freeze_sha256=None,
            report_path=None,
            report_sha256=None,
            claim_status=None,
            evidence_tier=None,
        )
    source = Path(freeze_path).resolve()
    if not source.is_file() or source.is_symlink():
        return E18DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            reason="E18B_FREEZE_MISSING_OR_UNSAFE",
            synthetic=False,
            freeze_path=str(source),
            freeze_sha256=None,
            report_path=None,
            report_sha256=None,
            claim_status=None,
            evidence_tier=None,
        )
    try:
        freeze = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(freeze, dict):
            raise TypeError("freeze must be a JSON object")
        hashes = freeze["hashes"]
        if not isinstance(hashes, dict):
            raise TypeError("freeze.hashes must be a mapping")
        run_dir = Path(str(freeze["run_dir"])).resolve()
        report_path = run_dir / "report.json"
        if not report_path.is_file() or report_path.is_symlink():
            raise FileNotFoundError("frozen E18b report is missing or unsafe")
        report_sha = file_sha256(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise TypeError("E18b report must be a JSON object")
        valid = bool(
            freeze.get("schema_version") == 1
            and freeze.get("experiment_family") == "E18"
            and freeze.get("aggregate_experiment_id") == "e18b_sequence_control_lattice_aggregate"
            and freeze.get("execution_status") == "PASS"
            and freeze.get("claim_status") == "SUPPORTED"
            and freeze.get("evidence_tier") == "CONTROLLED_REFERENCE"
            and freeze.get("immutable") is True
            and hashes.get("report.json") == report_sha
            and report.get("status") == "PASS"
            and report.get("run_scope") == "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE_AGGREGATE"
            and isinstance(report.get("claim_gate"), dict)
            and report["claim_gate"].get("supported") is True
        )
        if not valid:
            raise ValueError("E18b freeze/report support contract did not pass")
    except (
        KeyError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return E18DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            reason=f"INVALID_E18B_FREEZE:{type(error).__name__}",
            synthetic=False,
            freeze_path=str(source),
            freeze_sha256=file_sha256(source),
            report_path=None,
            report_sha256=None,
            claim_status=None,
            evidence_tier=None,
        )
    return E18DependencyDecision(
        execution_status="PASS",
        reason="EXPLICIT_SUPPORTED_E18B_FREEZE_VALIDATED",
        synthetic=False,
        freeze_path=str(source),
        freeze_sha256=file_sha256(source),
        report_path=str(report_path),
        report_sha256=report_sha,
        claim_status="SUPPORTED",
        evidence_tier="CONTROLLED_REFERENCE",
    )


def resolve_e23a_screen_dependency(
    *,
    screen_run: str | Path | None,
    dry_run: bool,
    expected_e18_freeze_sha256: str | None,
) -> E23aScreenDependency:
    """Record screen provenance without allowing its outcomes to set a boundary."""

    if dry_run:
        return E23aScreenDependency(
            execution_status="PASS",
            reason="SYNTHETIC_DRY_RUN_E23A_SCREEN_NON_EVIDENCE",
            synthetic=True,
            report_path=None,
            report_sha256=None,
            e18_freeze_sha256=None,
            outcomes_used_for_boundary=False,
        )
    if screen_run is None:
        return E23aScreenDependency(
            execution_status="BLOCKED_DEPENDENCY",
            reason="EXPLICIT_COMPLETED_E23A_SCREEN_REQUIRED",
            synthetic=False,
            report_path=None,
            report_sha256=None,
            e18_freeze_sha256=None,
            outcomes_used_for_boundary=False,
        )
    source = Path(screen_run).resolve()
    report_path = source / "report.json" if source.is_dir() else source
    try:
        if not report_path.is_file() or report_path.is_symlink():
            raise FileNotFoundError("E23a report is missing or unsafe")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise TypeError("E23a report must be a JSON object")
        e18 = report.get("e18_dependency")
        claim_gate = report.get("claim_gate")
        valid = bool(
            report.get("experiment_id") == "e23a_product_poset_screen"
            and report.get("execution_status") == "PASS"
            and report.get("run_mode") == "MAIN"
            and report.get("phase") == "SCREEN"
            and isinstance(claim_gate, dict)
            and claim_gate.get("status") == "SCREEN_ONLY_NO_CONFIRMATORY_CLAIM"
            and claim_gate.get("supported") is False
            and isinstance(e18, dict)
            and e18.get("execution_status") == "PASS"
            and e18.get("freeze_sha256") == expected_e18_freeze_sha256
        )
        if not valid:
            raise ValueError("E23a screen provenance contract did not pass")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return E23aScreenDependency(
            execution_status="BLOCKED_DEPENDENCY",
            reason=f"INVALID_E23A_SCREEN:{type(error).__name__}",
            synthetic=False,
            report_path=str(report_path),
            report_sha256=(file_sha256(report_path) if report_path.is_file() else None),
            e18_freeze_sha256=None,
            outcomes_used_for_boundary=False,
        )
    return E23aScreenDependency(
        execution_status="PASS",
        reason="EXPLICIT_E23A_SCREEN_PROVENANCE_VALIDATED",
        synthetic=False,
        report_path=str(report_path),
        report_sha256=file_sha256(report_path),
        e18_freeze_sha256=expected_e18_freeze_sha256,
        outcomes_used_for_boundary=False,
    )


def _read_report(path: Path) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file() or report_path.is_symlink():
        raise FileNotFoundError(f"Completed E22b report is missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("E22b report must be a JSON object")
    return payload


def resolve_e22b_dependency(
    *,
    e22b_run: str | Path | None,
    dry_run: bool,
) -> E22DependencyDecision:
    """Choose E23 boundary without reading any E23 outcome."""

    mean_method = mean_locality_method_payload()
    if e22b_run is None and dry_run:
        return E22DependencyDecision(
            execution_status="PASS",
            boundary_mode="capacity_only",
            reason="SYNTHETIC_DRY_RUN_DEPENDENCY_NON_EVIDENCE",
            synthetic=True,
            report_path=None,
            report_sha256=None,
            protocol_lock_sha256=None,
            evidence_tier="CONTROLLED_REFERENCE",
            dependency_claim_eligible=False,
            dependency_claim_status="NOT_SUPPORTED",
            safe_locality_supported=False,
            locality_method=mean_method,
            locality_risk_scale=_LOCALITY_RISK_SCALE,
            safe_objective_implemented=True,
        )
    if e22b_run is None:
        return E22DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            boundary_mode=None,
            reason="EXPLICIT_COMPLETED_E22B_RUN_REQUIRED",
            synthetic=False,
            report_path=None,
            report_sha256=None,
            protocol_lock_sha256=None,
            evidence_tier=None,
            dependency_claim_eligible=False,
            dependency_claim_status=None,
            safe_locality_supported=False,
            locality_method=mean_method,
            locality_risk_scale=_LOCALITY_RISK_SCALE,
            safe_objective_implemented=False,
        )

    source = Path(e22b_run).resolve()
    report_path = source / "report.json" if source.is_dir() else source
    try:
        report = _read_report(source)
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        return E22DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            boundary_mode=None,
            reason=f"INVALID_OR_INCOMPLETE_E22B_REPORT:{type(error).__name__}",
            synthetic=False,
            report_path=str(report_path),
            report_sha256=None,
            protocol_lock_sha256=None,
            evidence_tier=None,
            dependency_claim_eligible=False,
            dependency_claim_status=None,
            safe_locality_supported=False,
            locality_method=mean_method,
            locality_risk_scale=_LOCALITY_RISK_SCALE,
            safe_objective_implemented=False,
        )

    execution = report.get("execution_status")
    tier = report.get("evidence_tier")
    claim_eligible = report.get("claim_eligible")
    claim_gate = report.get("claim_gate")
    protocol_lock = report.get("protocol_lock")
    if (
        execution != "PASS"
        or tier != "CONTROLLED_REFERENCE"
        or not isinstance(claim_eligible, bool)
        or not isinstance(claim_gate, dict)
        or not isinstance(protocol_lock, dict)
        or protocol_lock.get("sha256") != E22B_PROTOCOL_LOCK_SHA256
    ):
        return E22DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            boundary_mode=None,
            reason="E22B_REPORT_CONTRACT_NOT_SATISFIED",
            synthetic=False,
            report_path=str(report_path),
            report_sha256=file_sha256(report_path),
            protocol_lock_sha256=(
                str(protocol_lock.get("sha256"))
                if isinstance(protocol_lock, dict) and protocol_lock.get("sha256") is not None
                else None
            ),
            evidence_tier=str(tier) if tier is not None else None,
            dependency_claim_eligible=bool(claim_eligible),
            dependency_claim_status=None,
            safe_locality_supported=False,
            locality_method=mean_method,
            locality_risk_scale=_LOCALITY_RISK_SCALE,
            safe_objective_implemented=False,
        )
    status = claim_gate.get("status")
    safe = claim_gate.get("safe_locality_supported")
    if safe is None and isinstance(claim_gate.get("supported"), bool):
        safe = bool(status == SAFE_E22_STATUS and claim_gate.get("supported") is True)
    parent_e21 = report.get("parent_e21")
    phase_dependency = report.get("phase_dependency")
    try:
        if not isinstance(parent_e21, dict):
            raise TypeError("E22b parent provenance is missing")
        inherited = parent_e21.get("inherited_thresholds")
        if not isinstance(inherited, dict):
            raise TypeError("E22b inherited thresholds are missing")
        risk_scale = float(inherited["maximum_nontarget_degradation"])
        if risk_scale != _LOCALITY_RISK_SCALE:
            raise ValueError("E22b locality risk scale changed")
        if status == SAFE_E22_STATUS and safe is True:
            if not isinstance(phase_dependency, dict):
                raise TypeError("E22b selected-method provenance is missing")
            selected_payload = phase_dependency.get("selected_method")
            if not isinstance(selected_payload, dict):
                raise TypeError("E22b selected locality method is missing")
            selected_method = _registered_locality_method(selected_payload)
            locality_method = selected_method.as_dict()
            if locality_method["selection_eligible"] is not True:
                raise ValueError("E22b safe method was not selection eligible")
            if selected_method.objective not in {
                LocalityObjective.MEAN,
                LocalityObjective.CVAR,
                LocalityObjective.SMOOTH_MAX,
            }:
                return E22DependencyDecision(
                    execution_status="BLOCKED_DEPENDENCY",
                    boundary_mode=None,
                    reason=("E22B_SELECTED_LOCALITY_OBJECTIVE_NOT_FAITHFULLY_IMPLEMENTED"),
                    synthetic=False,
                    report_path=str(report_path),
                    report_sha256=file_sha256(report_path),
                    protocol_lock_sha256=E22B_PROTOCOL_LOCK_SHA256,
                    evidence_tier=str(tier),
                    dependency_claim_eligible=bool(claim_eligible),
                    dependency_claim_status=str(status),
                    safe_locality_supported=True,
                    locality_method=locality_method,
                    locality_risk_scale=risk_scale,
                    safe_objective_implemented=False,
                )
        else:
            locality_method = mean_method
    except (KeyError, TypeError, ValueError) as error:
        return E22DependencyDecision(
            execution_status="BLOCKED_DEPENDENCY",
            boundary_mode=None,
            reason=f"E22B_LOCALITY_OBJECTIVE_CONTRACT_FAILED:{type(error).__name__}",
            synthetic=False,
            report_path=str(report_path),
            report_sha256=file_sha256(report_path),
            protocol_lock_sha256=E22B_PROTOCOL_LOCK_SHA256,
            evidence_tier=str(tier),
            dependency_claim_eligible=bool(claim_eligible),
            dependency_claim_status=(status if isinstance(status, str) else None),
            safe_locality_supported=bool(safe),
            locality_method=mean_method,
            locality_risk_scale=_LOCALITY_RISK_SCALE,
            safe_objective_implemented=False,
        )
    if not isinstance(status, str) or not isinstance(safe, bool):
        reason = "E22B_CLAIM_GATE_CONTRACT_NOT_SATISFIED"
        mode = None
        output_status = "BLOCKED_DEPENDENCY"
    elif status == SAFE_E22_STATUS and safe:
        reason = "SAFE_E22B_STATUS_LOCKS_SAFE_MINIMALITY"
        mode = "safe_minimality"
        output_status = "PASS"
    elif status in NON_SAFE_E22_STATUSES and not safe:
        reason = "COMPLETED_NON_SAFE_E22B_STATUS_LOCKS_CAPACITY_ONLY"
        mode = "capacity_only"
        output_status = "PASS"
    else:
        reason = "INCONSISTENT_E22B_STATUS_AND_SAFE_FLAG"
        mode = None
        output_status = "BLOCKED_DEPENDENCY"
    return E22DependencyDecision(
        execution_status=output_status,
        boundary_mode=mode,
        reason=reason,
        synthetic=False,
        report_path=str(report_path),
        report_sha256=file_sha256(report_path),
        protocol_lock_sha256=E22B_PROTOCOL_LOCK_SHA256,
        evidence_tier=str(tier),
        dependency_claim_eligible=bool(claim_eligible),
        dependency_claim_status=status if isinstance(status, str) else None,
        safe_locality_supported=bool(safe),
        locality_method=locality_method,
        locality_risk_scale=risk_scale,
        safe_objective_implemented=True,
    )


def validate_theory_prediction_lock(
    *,
    snapshot: ProtocolSnapshot,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    theory = snapshot.payload.get("theory_prediction")
    if not isinstance(theory, dict):
        raise PostE21ContractError("E23 lock lacks theory_prediction")
    if "source_path" in theory:
        source = (snapshot.path.parent.parent / str(theory["source_path"])).resolve()
        try:
            source.relative_to(snapshot.path.parent.parent.resolve())
        except ValueError as error:
            raise PostE21ContractError("E23 theory lock path escapes the repository") from error
        if (
            not source.is_file()
            or source.is_symlink()
            or file_sha256(source) != theory.get("source_sha256")
        ):
            raise PostE21ContractError("E23 shared theory lock changed")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("experiment_id") != "e23_product_poset_theory"
            or payload.get("protocol_frozen_before_main") is not True
            or payload.get("main_execution_started") is not False
            or not isinstance(payload.get("theory_prediction"), dict)
        ):
            raise PostE21ContractError("E23 shared theory lock contract is invalid")
        theory = payload["theory_prediction"]
    evaluation = config["evaluation"]
    expected = theory_prediction_payload(
        affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
        retention_mse_tolerance=float(config["adequacy"]["retention_margin"]),
        locality_mse_tolerance=float(config["adequacy"]["locality_margin"]),
        intensities=[float(value) for value in config["intensities"]],
        updates=[int(value) for value in evaluation["updates"]],
        gap_events=[int(value) for value in evaluation["gap_events"]],
    )
    expected_sha = sha256_canonical_json(expected)
    if theory.get("payload") != expected:
        raise PostE21ContractError(
            "Locked E23 theory predictions differ from canonical poset theory"
        )
    if theory.get("sha256") != expected_sha:
        raise PostE21ContractError("Locked E23 theory prediction hash mismatch")
    return {
        "sha256": expected_sha,
        "poset_minimal_sets": expected["poset_minimal_sets"],
        "confirmatory_boundary_sets": expected["confirmatory_boundary_sets"],
    }


def expected_grid_size(
    *,
    seeds: Sequence[int],
    intensities: Sequence[float],
    updates: Sequence[int],
    gap_events: Sequence[int],
) -> int:
    return (
        len(seeds)
        * len(CANONICAL_CONTROLLERS)
        * len(DEMAND_FAMILIES)
        * len(intensities)
        * len(updates)
        * len(gap_events)
    )


def summarize_seed_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    intensities: Sequence[float],
    updates: Sequence[int],
    gap_events: Sequence[int],
    affected_mse_tolerance: float,
    target_margin: float,
    retention_margin: float,
    locality_margin: float,
    minimum_single_axis_exact_matches: int,
    minimum_pairwise_exact_matches: int,
    incomparable_direction_margin: float,
    maximal_simpler_degradation_margin: float,
    boundary_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply absolute adequacy, then compare poset-minimal sets to theory."""

    if boundary_mode not in {"capacity_only", "safe_minimality"}:
        raise ValueError("Unsupported E23 boundary mode")
    if minimum_single_axis_exact_matches != len(SINGLE_AXIS_DEMANDS):
        raise ValueError("E23 requires exact recovery of all single-axis demands")
    if not 0 <= minimum_pairwise_exact_matches <= len(PAIRWISE_DEMANDS):
        raise ValueError("invalid E23 pairwise match threshold")
    indexed: dict[tuple[int, str, float, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["demand_family"]),
            float(row["intensity"]),
            int(row["updates"]),
            int(row["gap_events"]),
            str(row["controller_id"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate E23 raw row: {key}")
        indexed[key] = row

    seed_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for seed in seeds:
        demand_results: dict[str, dict[str, Any]] = {}
        for demand in DEMAND_FAMILIES:
            target = required_controller(demand)
            target_id = target.controller_id
            boundary_ids = tuple(theory_boundary_controller_ids(demand))
            lower_ids = tuple(
                controller.controller_id for controller in immediate_lower_covers(target)
            )
            incomparable_ids = tuple(
                controller.controller_id
                for controller in same_rank_incomparable_controllers(target)
                if controller.controller_id in boundary_ids
            )
            controller_stats: dict[str, dict[str, Any]] = {}
            for controller_id in boundary_ids:
                controller_rows = [
                    indexed[
                        (
                            int(seed),
                            demand,
                            float(intensity),
                            int(update_count),
                            int(gap),
                            controller_id,
                        )
                    ]
                    for intensity in intensities
                    for update_count in updates
                    for gap in gap_events
                ]
                affected_values = [float(row["affected_mse"]) for row in controller_rows]
                retention_values = [float(row["retention_mse"]) for row in controller_rows]
                locality_values = [
                    float(row["active_nontarget_degradation"])
                    for row in controller_rows
                    if row["active_nontarget_degradation"] is not None
                ]
                maximum_affected = max(affected_values)
                maximum_retention = max(retention_values)
                maximum_locality = max(locality_values) if locality_values else None
                adequate = bool(
                    maximum_affected <= affected_mse_tolerance
                    and maximum_retention <= retention_margin
                    and (
                        boundary_mode == "capacity_only"
                        or (maximum_locality is not None and maximum_locality <= locality_margin)
                    )
                )
                controller_stats[controller_id] = {
                    "maximum_affected_mse": maximum_affected,
                    "mean_affected_mse": sum(affected_values) / len(affected_values),
                    "maximum_retention_mse": maximum_retention,
                    "maximum_active_nontarget_degradation": maximum_locality,
                    "absolute_adequate": adequate,
                }

            observed_adequate = tuple(
                controller_id
                for controller_id in boundary_ids
                if controller_stats[controller_id]["absolute_adequate"]
            )
            observed_minimal = tuple(
                controller.controller_id
                for controller in minimal_elements(
                    CONTROLLER_BY_ID[controller_id] for controller_id in observed_adequate
                )
            )
            predicted_minimal = theory_minimal_controller_ids(demand)
            predicted_adequate = tuple(
                controller_id
                for controller_id in boundary_ids
                if controller_id in theory_adequate_controller_ids(demand)
            )
            minimal_union = set(observed_minimal) | set(predicted_minimal)
            minimal_intersection = set(observed_minimal) & set(predicted_minimal)
            minimal_jaccard = (
                len(minimal_intersection) / len(minimal_union) if minimal_union else 1.0
            )
            false_adequate = tuple(sorted(set(observed_adequate) - set(predicted_adequate)))
            false_inadequate = tuple(sorted(set(predicted_adequate) - set(observed_adequate)))
            target_affected = float(controller_stats[target_id]["maximum_affected_mse"])
            predecessor_differences = tuple(
                float(controller_stats[controller_id]["maximum_affected_mse"]) - target_affected
                for controller_id in lower_ids
            )
            immediate_predecessor_failure = all(
                not controller_stats[controller_id]["absolute_adequate"]
                and difference >= target_margin
                for controller_id, difference in zip(
                    lower_ids,
                    predecessor_differences,
                    strict=True,
                )
            )
            target_mean = float(controller_stats[target_id]["mean_affected_mse"])
            incomparable_differences = tuple(
                float(controller_stats[controller_id]["mean_affected_mse"]) - target_mean
                for controller_id in incomparable_ids
            )
            incomparable_direction = all(
                difference > incomparable_direction_margin
                for difference in incomparable_differences
            )
            maximal_degradation: float | None = None
            maximal_guardrail: bool | None = None
            if demand in {*SINGLE_AXIS_DEMANDS, "preserve"}:
                maximal_degradation = (
                    float(controller_stats["c1111"]["maximum_affected_mse"]) - target_affected
                )
                maximal_guardrail = bool(
                    controller_stats["c1111"]["absolute_adequate"]
                    and maximal_degradation <= maximal_simpler_degradation_margin
                )
            result = {
                "seed": int(seed),
                "demand_family": demand,
                "boundary_controller_ids": list(boundary_ids),
                "predicted_adequate_controller_ids": list(predicted_adequate),
                "observed_adequate_controller_ids": list(observed_adequate),
                "predicted_minimal_controller_ids": list(predicted_minimal),
                "observed_minimal_controller_ids": list(observed_minimal),
                "prediction_exact_match": (observed_minimal == predicted_minimal),
                "minimal_set_jaccard": minimal_jaccard,
                "false_adequate_controller_ids": list(false_adequate),
                "false_inadequate_controller_ids": list(false_inadequate),
                "false_adequate_count": len(false_adequate),
                "false_inadequate_count": len(false_inadequate),
                "immediate_predecessor_ids": list(lower_ids),
                "minimum_immediate_predecessor_affected_gap": (
                    min(predecessor_differences) if predecessor_differences else None
                ),
                "immediate_predecessor_failure_passed": (immediate_predecessor_failure),
                "same_rank_incomparable_ids": list(incomparable_ids),
                "minimum_incomparable_affected_gap": (
                    min(incomparable_differences) if incomparable_differences else None
                ),
                "incomparable_crossover_direction_passed": (incomparable_direction),
                "maximal_simpler_affected_degradation": (maximal_degradation),
                "maximal_simpler_guardrail_passed": maximal_guardrail,
                "controller_adequacy": controller_stats,
            }
            demand_results[demand] = result
            cell_rows.append(result)

        single_matches = sum(
            bool(demand_results[demand]["prediction_exact_match"]) for demand in SINGLE_AXIS_DEMANDS
        )
        pairwise_matches = sum(
            bool(demand_results[demand]["prediction_exact_match"]) for demand in PAIRWISE_DEMANDS
        )
        preserve_match = bool(demand_results["preserve"]["prediction_exact_match"])
        predecessor_gate = all(
            bool(result["immediate_predecessor_failure_passed"])
            for result in demand_results.values()
        )
        incomparable_gate = all(
            bool(result["incomparable_crossover_direction_passed"])
            for result in demand_results.values()
        )
        maximal_gate = all(
            bool(demand_results[demand]["maximal_simpler_guardrail_passed"])
            for demand in (*SINGLE_AXIS_DEMANDS, "preserve")
        )
        primary_passed = bool(
            single_matches >= minimum_single_axis_exact_matches
            and pairwise_matches >= minimum_pairwise_exact_matches
            and preserve_match
            and predecessor_gate
            and incomparable_gate
            and maximal_gate
        )
        all_locality = [
            float(stats["maximum_active_nontarget_degradation"])
            for result in demand_results.values()
            for stats in result["controller_adequacy"].values()
            if stats["maximum_active_nontarget_degradation"] is not None
        ]
        maximum_locality = max(all_locality) if all_locality else None
        seed_rows.append(
            {
                "seed": int(seed),
                "boundary_mode": boundary_mode,
                "single_axis_exact_match_count": single_matches,
                "pairwise_exact_match_count": pairwise_matches,
                "preserve_exact_match": preserve_match,
                "minimum_minimal_set_jaccard": min(
                    float(result["minimal_set_jaccard"]) for result in demand_results.values()
                ),
                "false_adequate_count": sum(
                    int(result["false_adequate_count"]) for result in demand_results.values()
                ),
                "false_inadequate_count": sum(
                    int(result["false_inadequate_count"]) for result in demand_results.values()
                ),
                "immediate_predecessor_failure_passed": predecessor_gate,
                "incomparable_crossover_direction_passed": (incomparable_gate),
                "maximal_simpler_guardrail_passed": maximal_gate,
                "minimum_immediate_predecessor_gain": min(
                    float(result["minimum_immediate_predecessor_affected_gap"])
                    for result in demand_results.values()
                    if result["minimum_immediate_predecessor_affected_gap"] is not None
                ),
                "minimum_incomparable_affected_gap": min(
                    float(result["minimum_incomparable_affected_gap"])
                    for result in demand_results.values()
                    if result["minimum_incomparable_affected_gap"] is not None
                ),
                "maximum_maximal_simpler_degradation": max(
                    float(demand_results[demand]["maximal_simpler_affected_degradation"])
                    for demand in (*SINGLE_AXIS_DEMANDS, "preserve")
                ),
                "maximum_preserve_retention_mse": float(
                    demand_results["preserve"]["controller_adequacy"]["c0000"][
                        "maximum_retention_mse"
                    ]
                ),
                "maximum_active_nontarget_degradation": maximum_locality,
                "capacity_gate_passed": primary_passed,
                "safe_locality_gate_passed": (
                    None if boundary_mode == "capacity_only" else primary_passed
                ),
            }
        )
    assessment = {
        "boundary_mode": boundary_mode,
        "seed_count": len(seed_rows),
        "demand_count": len(cell_rows),
        "all_seed_capacity_gates_passed": all(
            bool(row["capacity_gate_passed"]) for row in seed_rows
        ),
        "all_seed_locality_gates_passed": (
            None
            if boundary_mode == "capacity_only"
            else all(bool(row["safe_locality_gate_passed"]) for row in seed_rows)
        ),
        "minimum_single_axis_exact_match_count": min(
            int(row["single_axis_exact_match_count"]) for row in seed_rows
        ),
        "minimum_pairwise_exact_match_count": min(
            int(row["pairwise_exact_match_count"]) for row in seed_rows
        ),
        "all_seed_preserve_exact_match": all(
            bool(row["preserve_exact_match"]) for row in seed_rows
        ),
        "minimum_minimal_set_jaccard": min(
            float(row["minimum_minimal_set_jaccard"]) for row in seed_rows
        ),
        "total_false_adequate_count": sum(int(row["false_adequate_count"]) for row in seed_rows),
        "total_false_inadequate_count": sum(
            int(row["false_inadequate_count"]) for row in seed_rows
        ),
        "minimum_immediate_predecessor_gain": min(
            float(row["minimum_immediate_predecessor_gain"]) for row in seed_rows
        ),
        "minimum_incomparable_affected_gap": min(
            float(row["minimum_incomparable_affected_gap"]) for row in seed_rows
        ),
        "maximum_maximal_simpler_degradation": max(
            float(row["maximum_maximal_simpler_degradation"]) for row in seed_rows
        ),
        "maximum_preserve_retention_mse": max(
            float(row["maximum_preserve_retention_mse"]) for row in seed_rows
        ),
        "maximum_active_nontarget_degradation": (
            None
            if boundary_mode == "capacity_only"
            else max(float(row["maximum_active_nontarget_degradation"]) for row in seed_rows)
        ),
        "capacity_supported": all(bool(row["capacity_gate_passed"]) for row in seed_rows),
        "safe_minimality_supported": (
            False
            if boundary_mode == "capacity_only"
            else all(
                bool(row["capacity_gate_passed"]) and bool(row["safe_locality_gate_passed"])
                for row in seed_rows
            )
        ),
    }
    return seed_rows, {"assessment": assessment, "cells": cell_rows}


def result_independent_boundary_manifest() -> dict[str, list[str]]:
    return {demand: list(theory_boundary_controller_ids(demand)) for demand in DEMAND_FAMILIES}


def ensure_finite_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite E23 metric {key!r}")
