from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ManifestValidationRequirements,
    ProvenanceValidationError,
    ValidatedRun,
    read_json_object_strict,
    read_jsonl_strict,
    sha256_canonical_json,
    sha256_file,
    validate_run_manifest,
    write_json_strict,
    write_jsonl_strict,
)
from catena.data.graded_operator_families import (
    GradedOperatorFamily,
    generate_graded_operator_family,
    tensor_sha256,
)
from catena.eval.jd_calibration import (
    AnalyticCandidate,
    CalibrationThresholds,
    RegretBin,
    assign_regret_bin,
    fit_jd_application_calibration,
    select_first_valid_candidates,
    validate_regret_bins,
    validate_selected_design,
)
from catena.systems.device import resolve_device
from catena.theory.joint_diagonalization import (
    _diagonal_approximation,
    _empirical_application_mse,
    _fit_shared_basis,
    _normalized_basis_regret,
)
from experiments.common import build_parser
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e03b_graded_jd_calibration"
DEFAULT_CONFIG = "configs/e03b_graded_jd_calibration.yaml"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
_ANALYTIC_PILOT_SHA256 = (
    "21d60d79738e5bd05034312ec630a694d6485f00be1bc824344088eebc9fe94c"
)
_FROZEN_PROTOCOL_SHA256 = (
    "f1665900785509ca97c814395aee081574078f26ae9d8dd0a90bae4a0a5a15e6"
)
_FROZEN_PROTOCOL_LOCK_SHA256 = (
    "dd548fa1d648f41310926a42f1919e387f1e13939c299f8edf430694967d1e10"
)
_PRESERVED_E03_RUNNER_SHA256 = (
    "bf45a0091bd989ca0efcf67e48547a23d5ab3c656885f75d5b66d84e2d5fbaf8"
)
_PRESERVED_JD_CORE_SHA256 = (
    "da3f01721b28d36d50464450b2dd790d0cac43e37d22cefd06d0fc4ff0f02e2e"
)
_EXPECTED_ALPHA_ANCHORS = (0.06, 0.12, 0.18, 0.26, 0.40, 0.55)
_REGISTERED_CONFIG_SHA256 = (
    "46315fbdc9ad01646206ec927f24e622fbcfedd1c8acdb22f70e18030566f8ec"
)
_REGISTERED_CONFIG_FILE_SHA256 = (
    "7e258f75aa406d058f748779ff5aeb23d0b93f19c9c5e1eb388df7578bdd7df9"
)

_EXPECTED_SOURCE_E03: dict[str, Any] = {
    "experiment_id": "e03_granularity_orientation",
    "run_id": "20260726T161535.271015Z",
    "schema_version": 1,
    "status": "PASS",
    "run_mode": "main",
    "manifest_sha256": (
        "be56f91b5fc3a5992a02dc2b13b70f82b0b6bd6c5a7e1231f0e0ddee6987f327"
    ),
    "report_sha256": (
        "ee0114f45d5facbc3ccdd0e3a0235531e1de078f29fd7a949420df0899fa98c0"
    ),
    "config_sha256": (
        "42f69bc5c4ab5097ccd90483582e1ef5fb94474d66dacc1b9a293425338a9589"
    ),
    "config_file_sha256": (
        "6daa0d633b9a0b1193c715ec754b7208eddf7b266f86b453e822932ca08fe5cb"
    ),
    "source_fingerprint_sha256": (
        "2be6b99e369e595ac8e65dffb5deeaa32bd3747559e7c530be29b6857638f09a"
    ),
    "source_fingerprint_files": 110,
}
_EXPECTED_SOURCE_SPLIT_REGISTRY: dict[str, Any] = {
    "main_seeds": [101, 211, 307, 401, 503, 601, 701, 809],
    "dry_run_seeds": [9001, 9002],
    "train_operators_per_family": 24,
    "heldout_operators_per_family": 8,
}
_EXPECTED_SOURCE_SPLIT_SHA256 = (
    "8b5fa4c13693e12a69afab8f27ae289f778efba0615988afa6c4c0f2bc67cdef"
)
_EXPECTED_CLAIM_STATUS_SHA256 = (
    "fb53ae74d4702aefad4a5503129552cfe34a0f75079bc0bbb9fd9b4a2deb26f6"
)
_EXPECTED_CLAIM_STATUS: dict[str, Any] = {
    "schema_version": 1,
    "experiment_id": "e03_granularity_orientation",
    "source_run": "20260726T161535.271015Z",
    "source_report_sha256": (
        "ee0114f45d5facbc3ccdd0e3a0235531e1de078f29fd7a949420df0899fa98c0"
    ),
    "execution_status": "PASS",
    "h3_categorical_geometry": {
        "status": "SUPPORTED",
        "four_practical_contrasts_passed": True,
        "absolute_geometry_gates_passed": True,
    },
    "h3_quantitative_calibration": {
        "status": "FAILED",
        "preregistered_gate_passed": False,
        "diagnosis": "PREDICTOR_RANGE_RESTRICTION",
    },
    "h3_full_claim_open": False,
    "e03b_status": "NOT_RUN",
    "scientific_evidence": False,
    "evidence_scope": {
        "evidence_tier": "CONTROLLED_REFERENCE",
        "official_backend_claim_eligible": False,
        "language_model_claim_eligible": False,
        "architecture_transfer_claim_eligible": False,
    },
}
_EXPECTED_BINS: tuple[tuple[str, float, float, bool], ...] = (
    ("bin_01", 1.0e-5, 2.5e-4, False),
    ("bin_02", 2.5e-4, 7.5e-4, False),
    ("bin_03", 7.5e-4, 1.5e-3, False),
    ("bin_04", 1.5e-3, 3.0e-3, False),
    ("bin_05", 3.0e-3, 4.5e-3, False),
    ("bin_06", 4.5e-3, 6.5e-3, True),
)


@dataclass(slots=True)
class _AnalyticRuntime:
    stream_index: int
    alpha_schedule_index: int
    replicate_index: int
    family: GradedOperatorFamily
    candidate: AnalyticCandidate
    basis: torch.Tensor
    basis_sha256: str
    train_analytic_regret: float
    heldout_operator_analytic_regrets: list[float]
    basis_diagnostics: dict[str, Any]
    identifiability: dict[str, Any]
    probe_seeds: list[int]


@dataclass(frozen=True, slots=True)
class _PreprobeLock:
    registry_path: Path
    registry_file_sha256: str
    analytic_audit_path: Path
    analytic_audit_sha256: str
    selection_path: Path
    selection_sha256: str
    metadata_path: Path
    metadata_sha256: str


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite.")
    return result


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _regret_bins(config: Mapping[str, Any]) -> tuple[RegretBin, ...]:
    selection = _mapping(config.get("selection"), "selection")
    raw_bins = selection.get("regret_bins")
    if not isinstance(raw_bins, list):
        raise ValueError("selection.regret_bins must be a list.")
    bins: list[RegretBin] = []
    for index, raw_bin in enumerate(raw_bins):
        item = _mapping(raw_bin, f"selection.regret_bins[{index}]")
        label = item.get("label")
        include_upper = item.get("include_upper")
        if not isinstance(label, str) or not isinstance(include_upper, bool):
            raise ValueError("Every regret bin requires label and include_upper.")
        bins.append(
            RegretBin(
                label=label,
                lower=_finite_float(item.get("lower"), f"{label}.lower"),
                upper=_finite_float(item.get("upper"), f"{label}.upper"),
                include_upper=include_upper,
            )
        )
    return validate_regret_bins(bins)


def _candidate_registry_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = dict(_mapping(config.get("candidate_registry"), "candidate_registry"))
    registry.pop("registry_sha256", None)
    return registry


def _stream_seed_set(stream: Mapping[str, Any], alpha_count: int) -> set[int]:
    start = _integer(stream.get("generation_seed_start"), "generation_seed_start")
    replicates = _positive_integer(stream.get("replicates_per_alpha"), "replicates_per_alpha")
    hard_max = _positive_integer(stream.get("hard_max_candidates"), "hard_max_candidates")
    if hard_max != alpha_count * replicates:
        raise ValueError("hard_max_candidates must equal alpha_count * replicates_per_alpha.")
    return set(range(start, start + hard_max))


def _validate_config(config: dict[str, Any]) -> None:
    if sha256_canonical_json(config) != _REGISTERED_CONFIG_SHA256:
        raise ValueError("E03b config payload does not match the canonical protocol hash.")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("The config experiment_id does not match E03b.")
    preserved_files = (
        (
            REPO_ROOT / "experiments/e03_granularity_orientation.py",
            _PRESERVED_E03_RUNNER_SHA256,
        ),
        (
            REPO_ROOT / "src/catena/theory/joint_diagonalization.py",
            _PRESERVED_JD_CORE_SHA256,
        ),
    )
    for path, expected_sha256 in preserved_files:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"Preserved original E03 source changed: {path}.")
    frozen_protocol_files = (
        (
            REPO_ROOT / "docs/E03B_PROTOCOL_PREREGISTRATION_FROZEN_KO.md",
            _FROZEN_PROTOCOL_SHA256,
        ),
        (
            REPO_ROOT / "docs/E03B_PROTOCOL_PREREGISTRATION_LOCK_KO.md",
            _FROZEN_PROTOCOL_LOCK_SHA256,
        ),
    )
    for path, expected_sha256 in frozen_protocol_files:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"Frozen E03b protocol changed: {path}.")

    source = _mapping(config.get("source_e03"), "source_e03")
    for key, expected in _EXPECTED_SOURCE_E03.items():
        if source.get(key) != expected:
            raise ValueError(f"source_e03.{key} does not match the immutable E03 pin.")
    if source.get("split_registry") != _EXPECTED_SOURCE_SPLIT_REGISTRY:
        raise ValueError("source_e03.split_registry does not match the original E03 split.")
    if source.get("split_registry_sha256") != _EXPECTED_SOURCE_SPLIT_SHA256:
        raise ValueError("source_e03.split_registry_sha256 is not the registered hash.")
    if sha256_canonical_json(source["split_registry"]) != _EXPECTED_SOURCE_SPLIT_SHA256:
        raise ValueError("source_e03.split_registry payload hash mismatch.")
    for artifact_name, expected_rows, expected_hash in (
        (
            "operator_family_metrics",
            24,
            "54efda16cc92da17e0a40472de0075b7fec252c3451fde682304f7f63fa29dc4",
        ),
        (
            "control_frontier",
            12,
            "59c1aba1bffa039b85a61aa51714951924869ad8164787361e3f982927c1f035",
        ),
    ):
        artifact = _mapping(source.get(artifact_name), f"source_e03.{artifact_name}")
        if artifact.get("path") != f"{artifact_name}.jsonl":
            raise ValueError(f"source_e03.{artifact_name}.path is not pinned.")
        if artifact.get("rows") != expected_rows or artifact.get("sha256") != expected_hash:
            raise ValueError(f"source_e03.{artifact_name} contract is not pinned.")
    claim_status = _mapping(
        source.get("claim_status_registry"),
        "source_e03.claim_status_registry",
    )
    expected_claim_status_pin = {
        "path": "E03_CLAIM_STATUS.json",
        "sha256": _EXPECTED_CLAIM_STATUS_SHA256,
        "schema_version": 1,
        "source_run": _EXPECTED_SOURCE_E03["run_id"],
        "source_report_sha256": _EXPECTED_SOURCE_E03["report_sha256"],
        "categorical_status": "SUPPORTED",
        "categorical_four_practical_contrasts_passed": True,
        "categorical_absolute_geometry_gates_passed": True,
        "quantitative_status": "FAILED",
        "quantitative_preregistered_gate_passed": False,
        "quantitative_diagnosis": "PREDICTOR_RANGE_RESTRICTION",
        "full_claim_open": False,
    }
    if claim_status != expected_claim_status_pin:
        raise ValueError("source_e03.claim_status_registry is not exactly pinned.")

    data = _mapping(config.get("data"), "data")
    if _positive_integer(data.get("dimension"), "data.dimension") != 32:
        raise ValueError("E03b is registered at dimension 32.")
    if _positive_integer(data.get("projector_rank"), "data.projector_rank") != 8:
        raise ValueError("E03b is registered at projector rank 8.")
    if (
        _positive_integer(
            data.get("train_operators_per_family"),
            "data.train_operators_per_family",
        )
        != 24
        or _positive_integer(
            data.get("heldout_operators_per_family"),
            "data.heldout_operators_per_family",
        )
        != 8
    ):
        raise ValueError("E03b requires the registered 24 train / 8 heldout split.")
    _positive_integer(
        data.get("probes_per_heldout_operator"),
        "data.probes_per_heldout_operator",
    )
    if not math.isclose(
        _finite_float(
            data.get("max_rotation_radians"),
            "data.max_rotation_radians",
            positive=True,
        ),
        math.pi,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("E03b max_rotation_radians must be pi.")

    estimator = _mapping(config.get("basis_estimator"), "basis_estimator")
    expected_estimator_values: dict[str, Any] = {
        "method": "e03_train_only_multirestart_qr_adam",
        "steps": 1000,
        "learning_rate": 0.03,
        "restarts": 4,
        "dry_run_steps": 250,
        "dry_run_restarts": 2,
        "optimizer_seed_offset": 30_000_000,
        "weight_seed": 73021,
        "weight_lower": 0.5,
        "weight_upper": 1.5,
        "weight_vector_sha256": (
            "829000d33992018bb98e92432ae9af26f25c5e2e848cced7390aed3260f7821c"
        ),
        "eigenvalue_order": "descending",
        "eigenvector_sign_rule": "largest_absolute_coordinate_positive",
        "require_unique_train_inclusion_signatures": True,
        "minimum_weighted_eigenvalue_gap": 1.0e-8,
        "maximum_zero_alpha_heldout_regret": 1.0e-10,
        "maximum_orthogonality_error": 1.0e-10,
        "maximum_comparator_regret_gap": 1.0e-12,
        "minimum_competitive_restarts": 2,
        "restart_consensus_absolute_tolerance": 1.0e-7,
        "restart_consensus_relative_tolerance": 1.0e-3,
        "maximum_optimizer_uncertainty": 1.0e-5,
        "saturation_heldout_uncertainty_floor": 4.61e-6,
        "minimum_bin_boundary_clearance": 1.0e-5,
        "bin_boundary_uncertainty_multiplier": 10.0,
    }
    for key, expected in expected_estimator_values.items():
        if estimator.get(key) != expected:
            raise ValueError(f"basis_estimator.{key} is not the registered value.")

    bins = _regret_bins(config)
    observed_bins = tuple(
        (item.label, item.lower, item.upper, item.include_upper) for item in bins
    )
    if observed_bins != _EXPECTED_BINS:
        raise ValueError("selection.regret_bins do not match the exact registered six bins.")
    selection = _mapping(config.get("selection"), "selection")
    if selection.get("policy") != "deterministic_first_valid_analytic_only":
        raise ValueError("selection.policy is not the analytic-only first-valid rule.")
    if _positive_integer(selection.get("families_per_bin"), "families_per_bin") != 8:
        raise ValueError("Main E03b requires exactly eight families per bin.")
    if (
        _positive_integer(
            selection.get("dry_run_families_per_bin"),
            "dry_run_families_per_bin",
        )
        != 1
    ):
        raise ValueError("Dry E03b requires exactly one family per bin.")
    if _finite_float(
        selection.get("minimum_nonzero_range"),
        "selection.minimum_nonzero_range",
        positive=True,
    ) != 0.004:
        raise ValueError("The registered nonzero analytic range is 0.004.")

    registry = _mapping(config.get("candidate_registry"), "candidate_registry")
    if (
        registry.get("version") != "e03b-graded-jd-candidates-v1"
        or registry.get("traversal") != "alpha_major_then_replicate"
    ):
        raise ValueError("candidate_registry identity/traversal is not registered.")
    raw_alphas = registry.get("alpha_schedule")
    if not isinstance(raw_alphas, list) or not raw_alphas:
        raise ValueError("candidate_registry.alpha_schedule must be nonempty.")
    alphas = [
        _finite_float(value, f"alpha_schedule[{index}]")
        for index, value in enumerate(raw_alphas)
    ]
    if len(set(alphas)) != len(alphas) or any(not 0.0 <= value <= 1.0 for value in alphas):
        raise ValueError("alpha_schedule must contain unique values in [0, 1].")
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("alpha_schedule must prospectively cover both endpoints.")
    main_seeds = _stream_seed_set(
        _mapping(registry.get("main"), "candidate_registry.main"),
        len(alphas),
    )
    dry_seeds = _stream_seed_set(
        _mapping(registry.get("dry_run"), "candidate_registry.dry_run"),
        len(alphas),
    )
    pilot = _mapping(
        registry.get("development_analytic_pilot"),
        "candidate_registry.development_analytic_pilot",
    )
    raw_pilot_seeds = pilot.get("generation_seeds")
    if not isinstance(raw_pilot_seeds, list) or not raw_pilot_seeds:
        raise ValueError("The analytic pilot seeds must be explicitly listed.")
    pilot_seeds = {
        _integer(value, "development_analytic_pilot.generation_seeds")
        for value in raw_pilot_seeds
    }
    if len(pilot_seeds) != len(raw_pilot_seeds):
        raise ValueError("The analytic pilot seed list contains duplicates.")
    if (
        pilot.get("empirical_probe_calls") != 0
        or pilot.get("excluded_from_candidate_selection") is not True
    ):
        raise ValueError("Development seeds must be analytic-only and excluded.")
    expected_pilot_fields: dict[str, Any] = {
        "artifact_path": "docs/E03B_ANALYTIC_PILOT_KO.md",
        "artifact_sha256": _ANALYTIC_PILOT_SHA256,
        "primary_budget": {
            "steps": 1000,
            "restarts": 4,
            "learning_rate": 0.03,
        },
        "saturation_budget": {
            "steps": 2000,
            "restarts": 8,
            "learning_rate": 0.03,
        },
        "maximum_observed_saturation_heldout_delta": 4.61e-6,
        "selected_alpha_anchors": list(_EXPECTED_ALPHA_ANCHORS),
    }
    for key, expected in expected_pilot_fields.items():
        if pilot.get(key) != expected:
            raise ValueError(f"development_analytic_pilot.{key} is not pinned.")
    if (
        float(estimator["saturation_heldout_uncertainty_floor"])
        != float(pilot["maximum_observed_saturation_heldout_delta"])
    ):
        raise ValueError("Pilot saturation drift is not the optimizer uncertainty floor.")
    inspected_pilot_seeds = pilot.get("inspected_generation_seeds")
    expected_inspected_pilot_seeds = [
        *range(300001, 300009),
        *range(900001, 900025),
    ]
    if (
        inspected_pilot_seeds != expected_inspected_pilot_seeds
        or not set(expected_inspected_pilot_seeds).issubset(pilot_seeds)
    ):
        raise ValueError("Inspected analytic-pilot seeds are not exactly registered.")
    pilot_path = REPO_ROOT / str(pilot["artifact_path"])
    if (
        pilot_path.is_symlink()
        or not pilot_path.is_file()
        or sha256_file(pilot_path) != _ANALYTIC_PILOT_SHA256
    ):
        raise ValueError("E03b analytic-pilot artifact hash mismatch.")
    if tuple(alphas[: len(_EXPECTED_ALPHA_ANCHORS)]) != _EXPECTED_ALPHA_ANCHORS:
        raise ValueError("The first six alpha values must be the frozen pilot anchors.")
    if main_seeds & dry_seeds or main_seeds & pilot_seeds or dry_seeds & pilot_seeds:
        raise ValueError("Main, dry, and analytic-pilot candidate seeds must be disjoint.")
    _positive_integer(registry.get("probe_seed_offset"), "probe_seed_offset")
    registered_registry_hash = registry.get("registry_sha256")
    if (
        not isinstance(registered_registry_hash, str)
        or sha256_canonical_json(_candidate_registry_payload(config))
        != registered_registry_hash
    ):
        raise ValueError("candidate_registry.registry_sha256 payload mismatch.")

    statistics = _mapping(config.get("statistics"), "statistics")
    exact_thresholds = {
        "minimum_empirical_prediction_r2": 0.99,
        "minimum_empirical_calibration_slope": 0.95,
        "maximum_empirical_calibration_slope": 1.05,
        "maximum_absolute_empirical_calibration_intercept": 1.0e-4,
    }
    for key, expected in exact_thresholds.items():
        if _finite_float(statistics.get(key), f"statistics.{key}", positive=True) != expected:
            raise ValueError(f"statistics.{key} changed from the E03 threshold.")
    runtime = _mapping(config.get("runtime"), "runtime")
    threads = _positive_integer(runtime.get("cpu_threads"), "runtime.cpu_threads")
    if threads != 1:
        raise ValueError("E03b CPU threads must match the one-thread analytic pilot.")
    evidence = _mapping(config.get("evidence_scope"), "evidence_scope")
    if (
        evidence.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or evidence.get("scientific_evidence") is not False
        or evidence.get("official_backend_claim_eligible") is not False
        or evidence.get("language_model_claim_eligible") is not False
        or evidence.get("architecture_transfer_claim_eligible") is not False
    ):
        raise ValueError("E03b must remain controlled-reference, non-scientific evidence.")


def _validate_registered_config_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    config = load_config(resolved)
    _validate_config(config)
    if sha256_file(resolved) != _REGISTERED_CONFIG_FILE_SHA256:
        raise ValueError("E03b config file bytes do not match the registered protocol hash.")
    return config


def _validate_reloaded_config(
    preview: Mapping[str, Any],
    reloaded: Mapping[str, Any],
) -> None:
    if reloaded != preview:
        raise ProvenanceValidationError("E03b config changed during initialization.")


def _validate_initialized_config(
    preview_path: Path,
    preview: Mapping[str, Any],
    initialized: Mapping[str, Any],
    context_path: Path,
) -> None:
    expected_path = preview_path.resolve(strict=True)
    observed_path = context_path.resolve(strict=True)
    if observed_path != expected_path:
        raise ProvenanceValidationError(
            "E03b resolved config path changed during initialization."
        )
    reread = _validate_registered_config_file(observed_path)
    _validate_reloaded_config(preview, reread)
    _validate_reloaded_config(reread, initialized)


def _validate_pinned_artifact(
    validated: ValidatedRun,
    contract: Mapping[str, Any],
) -> Path:
    raw_name = contract.get("path")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ProvenanceValidationError("Pinned E03 artifact filename is unsafe.")
    path = Path(validated.run_dir) / raw_name
    if path.is_symlink() or not path.is_file():
        raise ProvenanceValidationError(f"Pinned E03 artifact is missing: {path}.")
    if sha256_file(path) != contract.get("sha256"):
        raise ProvenanceValidationError(f"Pinned E03 artifact hash mismatch: {path}.")
    if _jsonl_row_count(path) != contract.get("rows"):
        raise ProvenanceValidationError(f"Pinned E03 artifact row count mismatch: {path}.")
    return path


def _validate_pinned_e03_source(
    artifact_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source = _mapping(config.get("source_e03"), "source_e03")
    root = Path(artifact_root).expanduser().resolve(strict=True)
    run_dir = root / str(source["experiment_id"]) / str(source["run_id"])
    validated = validate_run_manifest(
        run_dir,
        root,
        requirements=ManifestValidationRequirements(
            expected_experiment_id=str(source["experiment_id"]),
            accepted_schema_versions=frozenset({int(source["schema_version"])}),
            expected_run_mode="main",
            require_main_eligible=True,
            require_full_eligible=True,
            allowed_statuses=frozenset({"PASS"}),
        ),
    )
    observed = {
        "manifest_sha256": validated.manifest_sha256,
        "report_sha256": validated.report_sha256,
        "config_sha256": validated.config_sha256,
        "config_file_sha256": validated.config_file_sha256,
        "source_fingerprint_sha256": validated.source_fingerprint.sha256,
        "source_fingerprint_files": validated.source_fingerprint.files,
    }
    for key, value in observed.items():
        if value != source.get(key):
            raise ProvenanceValidationError(f"Pinned original E03 {key} mismatch.")
    original_config = _mapping(validated.manifest.get("config"), "original E03 config")
    original_data = _mapping(original_config.get("data"), "original E03 data")
    observed_split = {
        "main_seeds": original_config.get("seeds"),
        "dry_run_seeds": original_config.get("dry_run_seeds"),
        "train_operators_per_family": original_data.get("train_operators_per_family"),
        "heldout_operators_per_family": original_data.get("test_operators_per_family"),
    }
    if (
        observed_split != source.get("split_registry")
        or sha256_canonical_json(observed_split) != source.get("split_registry_sha256")
    ):
        raise ProvenanceValidationError("Pinned original E03 split registry mismatch.")
    claim = _mapping(validated.report.get("claim_gate"), "original E03 claim_gate")
    empirical = _mapping(
        validated.report.get("empirical_regret_prediction"),
        "original E03 empirical_regret_prediction",
    )
    execution = _mapping(validated.report.get("execution"), "original E03 execution")
    if (
        claim.get("evaluated") is not True
        or claim.get("supported") is not False
        or claim.get("empirical_regret_prediction_passed") is not False
        or empirical.get("passed") is not False
        or execution.get("main_execution_complete") is not True
        or execution.get("row_count") != 24
    ):
        raise ProvenanceValidationError(
            "Original E03 must remain complete with quantitative calibration failed."
        )
    metrics = _mapping(source.get("operator_family_metrics"), "source metrics")
    frontier = _mapping(source.get("control_frontier"), "source frontier")
    _validate_pinned_artifact(validated, metrics)
    _validate_pinned_artifact(validated, frontier)
    claim_status_contract = _mapping(
        source.get("claim_status_registry"),
        "source claim-status registry",
    )
    claim_status_path = root / str(claim_status_contract["path"])
    if (
        claim_status_path.is_symlink()
        or not claim_status_path.is_file()
        or sha256_file(claim_status_path) != _EXPECTED_CLAIM_STATUS_SHA256
    ):
        raise ProvenanceValidationError("Pinned additive E03 claim-status registry mismatch.")
    if read_json_object_strict(claim_status_path) != _EXPECTED_CLAIM_STATUS:
        raise ProvenanceValidationError(
            "Additive E03 claim-status categorical/quantitative/full fields changed."
        )
    record = validated.dependency_record()
    record.update(
        {
            "evidence_role": "immutable_original_e03_split_claim_source",
            "scientific_evidence": False,
            "original_categorical_geometry_status": "SUPPORTED",
            "original_quantitative_calibration_status": "FAILED",
            "original_quantitative_diagnosis": "PREDICTOR_RANGE_RESTRICTION",
            "original_full_h3_claim_open": False,
            "pinned_split_registry": source["split_registry"],
            "pinned_split_registry_sha256": source["split_registry_sha256"],
            "pinned_artifacts": {
                "operator_family_metrics": metrics,
                "control_frontier": frontier,
                "claim_status_registry": claim_status_contract,
            },
        }
    )
    return record


def _training_weights(
    estimator: Mapping[str, Any],
    train_count: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(estimator["weight_seed"]))
    lower = float(estimator["weight_lower"])
    upper = float(estimator["weight_upper"])
    weights = lower + (upper - lower) * torch.rand(
        train_count,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    if tensor_sha256(weights) != estimator["weight_vector_sha256"]:
        raise ProvenanceValidationError("Registered spectral weight-vector hash mismatch.")
    return weights


def _canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    result = basis.clone()
    for column in range(int(result.shape[1])):
        pivot = int(torch.argmax(torch.abs(result[:, column])).item())
        if float(result[pivot, column].item()) < 0.0:
            result[:, column] *= -1.0
    return result.contiguous()


def _fit_spectral_shared_basis(
    training_projectors: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    projectors = list(training_projectors)
    if not projectors or len(projectors) != len(weights):
        raise ValueError("The spectral estimator requires one weight per training projector.")
    aggregate = torch.zeros_like(projectors[0])
    for weight, projector in zip(weights, projectors, strict=True):
        aggregate = aggregate + weight * projector
    aggregate = aggregate / torch.sum(weights)
    eigenvalues, eigenvectors = torch.linalg.eigh(aggregate)
    order = torch.argsort(eigenvalues, descending=True)
    ordered_eigenvalues = eigenvalues[order]
    basis = _canonicalize_basis_signs(eigenvectors[:, order])
    identity = torch.eye(
        basis.shape[0],
        dtype=basis.dtype,
        device=basis.device,
    )
    orthogonality_error = float(
        torch.linalg.matrix_norm(basis.T @ basis - identity, ord="fro").item()
    )
    adjacent_gaps = torch.abs(ordered_eigenvalues[:-1] - ordered_eigenvalues[1:])
    return basis, {
        "method": "deterministic_weighted_training_eigh",
        "training_operator_count": len(projectors),
        "weight_vector_sha256": tensor_sha256(weights),
        "aggregate_sha256": tensor_sha256(aggregate),
        "basis_sha256": tensor_sha256(basis),
        "orthogonality_error": orthogonality_error,
        "minimum_adjacent_eigenvalue_gap": float(torch.min(adjacent_gaps).item()),
        "maximum_eigenvalue": float(torch.max(ordered_eigenvalues).item()),
        "minimum_eigenvalue": float(torch.min(ordered_eigenvalues).item()),
    }


def _fit_jd_shared_basis(
    training_projectors: Sequence[torch.Tensor],
    weights: torch.Tensor,
    estimator: Mapping[str, Any],
    *,
    optimizer_seed: int,
    dry_run: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit the E03 train-only JD objective; spectral output is comparator-only."""

    projectors = list(training_projectors)
    steps = int(estimator["dry_run_steps"] if dry_run else estimator["steps"])
    restarts = int(
        estimator["dry_run_restarts"] if dry_run else estimator["restarts"]
    )
    fit = _fit_shared_basis(
        projectors,
        steps=steps,
        learning_rate=float(estimator["learning_rate"]),
        restarts=restarts,
        seed=optimizer_seed,
    )
    spectral_basis, spectral_diagnostics = _fit_spectral_shared_basis(
        projectors,
        weights,
    )
    spectral_train_regret = float(
        _normalized_basis_regret(spectral_basis, projectors).item()
    )
    identity_or_spectral = min(fit.identity_regret, spectral_train_regret)
    nesting_tolerance = float(estimator["maximum_comparator_regret_gap"])
    ordered_restart_regrets = sorted(fit.restart_best_regrets)
    consensus_tolerance = max(
        float(estimator["restart_consensus_absolute_tolerance"]),
        float(estimator["restart_consensus_relative_tolerance"]) * fit.best_regret,
    )
    competitive_restart_count = sum(
        regret <= fit.best_regret + consensus_tolerance
        for regret in fit.restart_best_regrets
    )
    optimizer_uncertainty = (
        ordered_restart_regrets[1] - ordered_restart_regrets[0]
        if len(ordered_restart_regrets) >= 2
        else math.inf
    )
    identity = torch.eye(
        fit.basis.shape[0],
        dtype=fit.basis.dtype,
        device=fit.basis.device,
    )
    orthogonality_error = float(
        torch.linalg.matrix_norm(fit.basis.T @ fit.basis - identity, ord="fro").item()
    )
    finite = bool(
        torch.isfinite(fit.basis).all().item()
        and all(
            math.isfinite(value)
            for value in (
                fit.best_regret,
                fit.identity_regret,
                spectral_train_regret,
                optimizer_uncertainty,
                orthogonality_error,
                *fit.restart_initial_regrets,
                *fit.restart_best_regrets,
                *fit.restart_final_regrets,
            )
        )
    )
    adequacy_passed = bool(
        finite
        and orthogonality_error
        <= float(estimator["maximum_orthogonality_error"])
        and fit.best_regret <= identity_or_spectral + nesting_tolerance
        and competitive_restart_count
        >= int(estimator["minimum_competitive_restarts"])
        and optimizer_uncertainty
        <= float(estimator["maximum_optimizer_uncertainty"])
    )
    return fit.basis, {
        "method": "e03_train_only_multirestart_qr_adam",
        "estimand": "estimated_train_fit_out_of_sample_jd_regret",
        "training_operator_count": len(projectors),
        "optimizer_seed": optimizer_seed,
        "steps": steps,
        "learning_rate": float(estimator["learning_rate"]),
        "restarts": restarts,
        "best_train_regret": fit.best_regret,
        "best_restart": fit.best_restart,
        "best_step": fit.best_step,
        "identity_candidate_train_regret": fit.identity_regret,
        "spectral_comparator_train_regret": spectral_train_regret,
        "spectral_comparator_diagnostics": spectral_diagnostics,
        "comparator_nesting_tolerance": nesting_tolerance,
        "comparator_nesting_passed": (
            fit.best_regret <= identity_or_spectral + nesting_tolerance
        ),
        "restart_initial_regrets": fit.restart_initial_regrets,
        "restart_best_regrets": fit.restart_best_regrets,
        "restart_final_regrets": fit.restart_final_regrets,
        "restart_consensus_tolerance": consensus_tolerance,
        "competitive_restart_count": competitive_restart_count,
        "minimum_competitive_restarts": int(
            estimator["minimum_competitive_restarts"]
        ),
        "optimizer_uncertainty": optimizer_uncertainty,
        "maximum_optimizer_uncertainty": float(
            estimator["maximum_optimizer_uncertainty"]
        ),
        "orthogonality_error": orthogonality_error,
        "maximum_orthogonality_error": float(
            estimator["maximum_orthogonality_error"]
        ),
        "basis_sha256": tensor_sha256(fit.basis),
        "finite": finite,
        "adequacy_passed": adequacy_passed,
        "spectral_comparator_only": True,
        "spectral_fallback_used": False,
    }


def _operator_analytic_regrets(
    basis: torch.Tensor,
    projectors: Sequence[torch.Tensor],
) -> list[float]:
    values: list[float] = []
    for projector in projectors:
        approximation = _diagonal_approximation(projector, basis)
        values.append(float(torch.mean((projector - approximation) ** 2).item()))
    return values


def _train_identifiability(
    family: GradedOperatorFamily,
    weights: torch.Tensor,
    *,
    minimum_weighted_eigenvalue_gap: float,
    maximum_zero_alpha_regret: float,
) -> dict[str, Any]:
    signatures = {
        tuple(
            int(candidate.diagonal_mask[coordinate, coordinate].item())
            for candidate in family.train_candidates
        )
        for coordinate in range(family.spec.dim)
    }
    unique_signatures = len(signatures)
    zero_family = family.regenerate(0.0)
    zero_basis, zero_diagnostics = _fit_spectral_shared_basis(
        zero_family.projectors("train"),
        weights,
    )
    zero_regret = float(
        _normalized_basis_regret(
            zero_basis,
            list(zero_family.projectors("heldout")),
        ).item()
    )
    zero_gap = float(zero_diagnostics["minimum_adjacent_eigenvalue_gap"])
    passed = bool(
        unique_signatures == family.spec.dim
        and zero_gap >= minimum_weighted_eigenvalue_gap
        and zero_regret <= maximum_zero_alpha_regret
    )
    return {
        "passed": passed,
        "unique_train_inclusion_signatures": unique_signatures,
        "required_unique_train_inclusion_signatures": family.spec.dim,
        "zero_alpha_minimum_weighted_eigenvalue_gap": zero_gap,
        "minimum_required_weighted_eigenvalue_gap": minimum_weighted_eigenvalue_gap,
        "zero_alpha_heldout_regret": zero_regret,
        "maximum_zero_alpha_heldout_regret": maximum_zero_alpha_regret,
        "zero_alpha_basis_sha256": tensor_sha256(zero_basis),
        "zero_alpha_basis_diagnostics": zero_diagnostics,
    }


def _candidate_descriptors(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> Iterator[tuple[int, int, int, int, float]]:
    registry = _mapping(config.get("candidate_registry"), "candidate_registry")
    alphas = [float(value) for value in registry["alpha_schedule"]]
    stream_name = "dry_run" if dry_run else "main"
    stream = _mapping(registry.get(stream_name), f"candidate_registry.{stream_name}")
    seed = int(stream["generation_seed_start"])
    stream_index = 0
    for alpha_index, alpha in enumerate(alphas):
        for replicate_index in range(int(stream["replicates_per_alpha"])):
            yield stream_index, alpha_index, replicate_index, seed, alpha
            stream_index += 1
            seed += 1
    if stream_index != int(stream["hard_max_candidates"]):
        raise AssertionError("Candidate descriptor count violated the hard maximum.")


def _required_bin_boundary_clearance(
    estimator: Mapping[str, Any],
    basis_diagnostics: Mapping[str, Any],
) -> tuple[float, float]:
    uncertainty = max(
        float(basis_diagnostics["optimizer_uncertainty"]),
        float(estimator["saturation_heldout_uncertainty_floor"]),
    )
    required = max(
        float(estimator["minimum_bin_boundary_clearance"]),
        float(estimator["bin_boundary_uncertainty_multiplier"]) * uncertainty,
    )
    return required, uncertainty


def _scan_analytic_candidates(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    bins: Sequence[RegretBin],
    families_per_bin: int,
    weights: torch.Tensor,
) -> tuple[
    dict[str, list[AnalyticCandidate]],
    dict[str, _AnalyticRuntime],
    list[dict[str, Any]],
]:
    data = _mapping(config.get("data"), "data")
    estimator = _mapping(config.get("basis_estimator"), "basis_estimator")
    registry = _mapping(config.get("candidate_registry"), "candidate_registry")
    runtime_by_id: dict[str, _AnalyticRuntime] = {}
    audit_rows: list[dict[str, Any]] = []

    def stream() -> Iterator[AnalyticCandidate]:
        for stream_index, alpha_index, replicate_index, seed, alpha in _candidate_descriptors(
            config,
            dry_run=dry_run,
        ):
            family = generate_graded_operator_family(
                dim=int(data["dimension"]),
                rank=int(data["projector_rank"]),
                train_count=int(data["train_operators_per_family"]),
                heldout_count=int(data["heldout_operators_per_family"]),
                seed=seed,
                alpha=alpha,
                max_rotation_radians=float(data["max_rotation_radians"]),
            )
            identifiability = _train_identifiability(
                family,
                weights,
                minimum_weighted_eigenvalue_gap=float(
                    estimator["minimum_weighted_eigenvalue_gap"]
                ),
                maximum_zero_alpha_regret=float(
                    estimator["maximum_zero_alpha_heldout_regret"]
                ),
            )
            audit_base: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "stream_index": stream_index,
                "alpha_schedule_index": alpha_index,
                "replicate_index": replicate_index,
                "generation_seed": seed,
                "alpha": alpha,
                "base_candidate_id": family.base_candidate_id,
                "realization_id": family.realization_id,
                "construction_sha256": family.construction_sha256,
                "realization_sha256": family.realization_sha256,
                "base_basis_sha256": family.base_basis_sha256,
                "identifiability": identifiability,
            }
            if not bool(identifiability["passed"]):
                audit_rows.append(
                    {
                        **audit_base,
                        "eligible_for_binning": False,
                        "analytic_regret": None,
                        "assigned_bin": None,
                        "skip_reason": "train_construction_not_identifiable",
                    }
                )
                continue
            optimizer_seed = int(estimator["optimizer_seed_offset"]) + seed
            basis, basis_diagnostics = _fit_jd_shared_basis(
                family.projectors("train"),
                weights,
                estimator,
                optimizer_seed=optimizer_seed,
                dry_run=dry_run,
            )
            if not bool(basis_diagnostics["adequacy_passed"]):
                audit_rows.append(
                    {
                        **audit_base,
                        "eligible_for_binning": False,
                        "analytic_regret": None,
                        "assigned_bin": None,
                        "basis_sha256": tensor_sha256(basis),
                        "basis_diagnostics": basis_diagnostics,
                        "skip_reason": "jd_optimizer_adequacy_failed",
                    }
                )
                continue
            heldout = list(family.projectors("heldout"))
            operator_regrets = _operator_analytic_regrets(basis, heldout)
            analytic_regret = float(np.mean(operator_regrets))
            normalized_regret = float(
                _normalized_basis_regret(basis, heldout).item()
            )
            if not math.isclose(
                analytic_regret,
                normalized_regret,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise AssertionError("Per-operator and normalized analytic regret disagree.")
            assigned_bin = assign_regret_bin(analytic_regret, bins)
            matching_bin = next(
                (item for item in bins if item.label == assigned_bin),
                None,
            )
            boundary_clearance = (
                min(
                    analytic_regret - matching_bin.lower,
                    matching_bin.upper - analytic_regret,
                )
                if matching_bin is not None
                else 0.0
            )
            required_clearance, effective_optimizer_uncertainty = (
                _required_bin_boundary_clearance(
                    estimator,
                    basis_diagnostics,
                )
            )
            if assigned_bin is None or boundary_clearance < required_clearance:
                audit_rows.append(
                    {
                        **audit_base,
                        "eligible_for_binning": False,
                        "analytic_regret": analytic_regret,
                        "assigned_bin": assigned_bin,
                        "basis_sha256": tensor_sha256(basis),
                        "basis_diagnostics": basis_diagnostics,
                        "bin_boundary_clearance": boundary_clearance,
                        "required_bin_boundary_clearance": required_clearance,
                        "effective_optimizer_uncertainty": (
                            effective_optimizer_uncertainty
                        ),
                        "skip_reason": "jd_regret_not_robustly_inside_registered_bin",
                    }
                )
                continue
            candidate = AnalyticCandidate(
                candidate_id=family.realization_id,
                construction_sha256=family.realization_sha256,
                analytic_regret=analytic_regret,
                alpha=alpha,
                generation_seed=seed,
            )
            if candidate.candidate_id in runtime_by_id:
                raise ValueError(f"Duplicate candidate realization: {candidate.candidate_id}.")
            probe_seed_base = int(registry["probe_seed_offset"]) + seed * 100
            runtime = _AnalyticRuntime(
                stream_index=stream_index,
                alpha_schedule_index=alpha_index,
                replicate_index=replicate_index,
                family=family,
                candidate=candidate,
                basis=basis,
                basis_sha256=tensor_sha256(basis),
                train_analytic_regret=float(
                    _normalized_basis_regret(
                        basis,
                        list(family.projectors("train")),
                    ).item()
                ),
                heldout_operator_analytic_regrets=operator_regrets,
                basis_diagnostics=basis_diagnostics,
                identifiability=identifiability,
                probe_seeds=[
                    probe_seed_base + index
                    for index in range(len(family.heldout_candidates))
                ],
            )
            runtime_by_id[candidate.candidate_id] = runtime
            audit_rows.append(
                {
                    **audit_base,
                    "eligible_for_binning": True,
                    "analytic_regret": analytic_regret,
                    "assigned_bin": assigned_bin,
                    "basis_sha256": runtime.basis_sha256,
                    "basis_diagnostics": basis_diagnostics,
                    "bin_boundary_clearance": boundary_clearance,
                    "required_bin_boundary_clearance": required_clearance,
                    "effective_optimizer_uncertainty": (
                        effective_optimizer_uncertainty
                    ),
                    "skip_reason": None,
                }
            )
            yield candidate

    selected = select_first_valid_candidates(
        stream(),
        bins,
        families_per_bin=families_per_bin,
    )
    selected_lookup: dict[str, tuple[str, int]] = {}
    for regret_bin in bins:
        for within_bin_index, candidate in enumerate(selected[regret_bin.label]):
            selected_lookup[candidate.candidate_id] = (
                regret_bin.label,
                within_bin_index,
            )
    for row in audit_rows:
        identity = str(row["realization_id"])
        selection = selected_lookup.get(identity)
        row["selected"] = selection is not None
        row["selected_bin"] = selection[0] if selection is not None else None
        row["selected_within_bin_index"] = (
            selection[1] if selection is not None else None
        )
    return selected, runtime_by_id, audit_rows


def _selection_lock_rows(
    selected: Mapping[str, Sequence[AnalyticCandidate]],
    runtime_by_id: Mapping[str, _AnalyticRuntime],
    bins: Sequence[RegretBin],
    *,
    registry_sha256: str,
    probe_count: int,
) -> list[dict[str, Any]]:
    _positive_integer(probe_count, "probe_count")
    rows: list[dict[str, Any]] = []
    global_index = 0
    for bin_index, regret_bin in enumerate(bins):
        for within_bin_index, candidate in enumerate(selected[regret_bin.label]):
            runtime = runtime_by_id[candidate.candidate_id]
            current_basis_sha256 = tensor_sha256(runtime.basis)
            if current_basis_sha256 != runtime.basis_sha256:
                raise ProvenanceValidationError(
                    f"Runtime JD basis changed for {candidate.candidate_id}."
                )
            train_registry = [
                {
                    "candidate_id": operator.candidate_id,
                    "split_index": operator.split_index,
                    "base_sha256": operator.base_sha256,
                    "operator_sha256": operator.operator_sha256,
                    "projector_tensor_sha256": tensor_sha256(operator.projector),
                }
                for operator in runtime.family.train_candidates
            ]
            heldout_registry: list[dict[str, Any]] = []
            for operator, analytic, probe_seed in zip(
                runtime.family.heldout_candidates,
                runtime.heldout_operator_analytic_regrets,
                runtime.probe_seeds,
                strict=True,
            ):
                heldout_registry.append(
                    {
                        "candidate_id": operator.candidate_id,
                        "split_index": operator.split_index,
                        "base_sha256": operator.base_sha256,
                        "operator_sha256": operator.operator_sha256,
                        "projector_tensor_sha256": tensor_sha256(operator.projector),
                        "analytic_regret": analytic,
                        "probe_seed": probe_seed,
                        "probe_count": probe_count,
                    }
                )
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "selection_index": global_index,
                "bin_index": bin_index,
                "within_bin_index": within_bin_index,
                "bin": {
                    "label": regret_bin.label,
                    "lower": regret_bin.lower,
                    "upper": regret_bin.upper,
                    "include_upper": regret_bin.include_upper,
                },
                "candidate_registry_sha256": registry_sha256,
                "stream_index": runtime.stream_index,
                "alpha_schedule_index": runtime.alpha_schedule_index,
                "replicate_index": runtime.replicate_index,
                "generation_seed": candidate.generation_seed,
                "alpha": candidate.alpha,
                "rotation_magnitude_radians": (
                    runtime.family.rotation_magnitude_radians
                ),
                "candidate_id": candidate.candidate_id,
                "base_candidate_id": runtime.family.base_candidate_id,
                "candidate_construction_sha256": candidate.construction_sha256,
                "base_construction_sha256": runtime.family.construction_sha256,
                "realization_sha256": runtime.family.realization_sha256,
                "base_basis_sha256": runtime.family.base_basis_sha256,
                "jd_basis_sha256": current_basis_sha256,
                "jd_basis_values": runtime.basis.detach().cpu().tolist(),
                "basis_diagnostics": runtime.basis_diagnostics,
                "identifiability": runtime.identifiability,
                "train_operator_count": runtime.family.spec.train_count,
                "heldout_operator_count": runtime.family.spec.heldout_count,
                "train_operator_registry": train_registry,
                "train_analytic_regret": runtime.train_analytic_regret,
                "heldout_analytic_regret": candidate.analytic_regret,
                "heldout_operator_registry": heldout_registry,
            }
            row["row_payload_sha256"] = sha256_canonical_json(row)
            rows.append(row)
            global_index += 1
    return rows


def _write_registry_artifact(
    run_dir: Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, str]:
    registry_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection": config["selection"],
        "data_split": {
            "train_operators_per_family": config["data"]["train_operators_per_family"],
            "heldout_operators_per_family": config["data"][
                "heldout_operators_per_family"
            ],
        },
        "basis_estimator": config["basis_estimator"],
        "candidate_registry": _candidate_registry_payload(config),
        "candidate_registry_sha256": config["candidate_registry"]["registry_sha256"],
    }
    path = run_dir / "candidate_split_registry.json"
    write_json_strict(path, registry_payload)
    return (
        path,
        sha256_file(path),
        str(config["candidate_registry"]["registry_sha256"]),
    )


def _write_preprobe_lock(
    run_dir: Path,
    *,
    registry_path: Path,
    registry_file_sha256: str,
    registry_sha256: str,
    audit_rows: Sequence[dict[str, Any]],
    selection_rows: Sequence[dict[str, Any]],
    design_gate: Mapping[str, Any],
) -> _PreprobeLock:
    audit_path = run_dir / "analytic_candidate_audit.jsonl"
    selection_path = run_dir / "selection_lock.jsonl"
    write_jsonl_strict(audit_path, audit_rows)
    write_jsonl_strict(selection_path, selection_rows)
    audit_sha256 = sha256_file(audit_path)
    selection_sha256 = sha256_file(selection_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCKED_BEFORE_EMPIRICAL_PROBES",
        "candidate_registry": {
            "path": registry_path.name,
            "file_sha256": registry_file_sha256,
            "payload_sha256": registry_sha256,
        },
        "analytic_candidate_audit": {
            "path": audit_path.name,
            "rows": len(audit_rows),
            "sha256": audit_sha256,
        },
        "selection_lock": {
            "path": selection_path.name,
            "rows": len(selection_rows),
            "sha256": selection_sha256,
            "construction_sha256": [
                row["candidate_construction_sha256"] for row in selection_rows
            ],
            "basis_sha256": [
                row["jd_basis_sha256"] for row in selection_rows
            ],
        },
        "design_gate": dict(design_gate),
        "empirical_probe_phase_authorized": bool(design_gate["passed"]),
    }
    metadata_path = run_dir / "preprobe_lock.json"
    write_json_strict(metadata_path, metadata)
    lock = _PreprobeLock(
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        analytic_audit_path=audit_path,
        analytic_audit_sha256=audit_sha256,
        selection_path=selection_path,
        selection_sha256=selection_sha256,
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
    )
    _verify_preprobe_lock(lock, require_probe_authorization=False)
    return lock


def _verify_preprobe_lock(
    lock: _PreprobeLock,
    *,
    require_probe_authorization: bool,
) -> None:
    expected = (
        (lock.registry_path, lock.registry_file_sha256),
        (lock.analytic_audit_path, lock.analytic_audit_sha256),
        (lock.selection_path, lock.selection_sha256),
        (lock.metadata_path, lock.metadata_sha256),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise ProvenanceValidationError(f"Pre-probe lock changed or is missing: {path}.")
    metadata = read_json_object_strict(lock.metadata_path)
    if metadata.get("status") != "LOCKED_BEFORE_EMPIRICAL_PROBES":
        raise ProvenanceValidationError("Pre-probe lock status is invalid.")
    if (
        require_probe_authorization
        and metadata.get("empirical_probe_phase_authorized") is not True
    ):
        raise ProvenanceValidationError("Design gate did not authorize empirical probes.")


def _verify_runtime_against_selection_lock(
    lock: _PreprobeLock,
    selected: Mapping[str, Sequence[AnalyticCandidate]],
    runtime_by_id: Mapping[str, _AnalyticRuntime],
    bins: Sequence[RegretBin],
    *,
    probe_count: int,
) -> None:
    registry = read_json_object_strict(lock.registry_path)
    registry_sha256 = registry.get("candidate_registry_sha256")
    if not isinstance(registry_sha256, str):
        raise ProvenanceValidationError("Locked candidate registry hash is invalid.")
    locked_rows = read_jsonl_strict(lock.selection_path)
    expected_rows = _selection_lock_rows(
        selected,
        runtime_by_id,
        bins,
        registry_sha256=registry_sha256,
        probe_count=probe_count,
    )
    if locked_rows != expected_rows:
        raise ProvenanceValidationError(
            "Selection lock is not bound to the current runtime candidates and tensors."
        )


def _run_empirical_phase(
    selected: Mapping[str, Sequence[AnalyticCandidate]],
    runtime_by_id: Mapping[str, _AnalyticRuntime],
    bins: Sequence[RegretBin],
    *,
    preprobe_lock: _PreprobeLock,
    probe_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _verify_preprobe_lock(preprobe_lock, require_probe_authorization=True)
    _verify_runtime_against_selection_lock(
        preprobe_lock,
        selected,
        runtime_by_id,
        bins,
        probe_count=probe_count,
    )
    family_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    for regret_bin in bins:
        for candidate in selected[regret_bin.label]:
            runtime = runtime_by_id[candidate.candidate_id]
            empirical_values: list[float] = []
            for heldout, analytic, probe_seed in zip(
                runtime.family.heldout_candidates,
                runtime.heldout_operator_analytic_regrets,
                runtime.probe_seeds,
                strict=True,
            ):
                approximation = _diagonal_approximation(
                    heldout.projector,
                    runtime.basis,
                )
                empirical = _empirical_application_mse(
                    [heldout.projector],
                    [approximation],
                    probe_count=probe_count,
                    probe_seed=probe_seed,
                )
                empirical_values.append(empirical)
                operator_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "bin": regret_bin.label,
                        "candidate_id": candidate.candidate_id,
                        "base_candidate_id": runtime.family.base_candidate_id,
                        "generation_seed": candidate.generation_seed,
                        "alpha": candidate.alpha,
                        "heldout_candidate_id": heldout.candidate_id,
                        "heldout_split_index": heldout.split_index,
                        "heldout_operator_sha256": heldout.operator_sha256,
                        "jd_basis_sha256": runtime.basis_sha256,
                        "probe_seed": probe_seed,
                        "probe_count": probe_count,
                        "analytic_regret": analytic,
                        "empirical_application_error": empirical,
                    }
                )
            empirical_mean = float(np.mean(empirical_values))
            family_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "bin": regret_bin.label,
                    "candidate_id": candidate.candidate_id,
                    "base_candidate_id": runtime.family.base_candidate_id,
                    "candidate_construction_sha256": candidate.construction_sha256,
                    "base_construction_sha256": runtime.family.construction_sha256,
                    "jd_basis_sha256": runtime.basis_sha256,
                    "generation_seed": candidate.generation_seed,
                    "alpha": candidate.alpha,
                    "train_operator_count": runtime.family.spec.train_count,
                    "heldout_operator_count": runtime.family.spec.heldout_count,
                    "train_analytic_regret": runtime.train_analytic_regret,
                    "heldout_analytic_regret": candidate.analytic_regret,
                    "heldout_empirical_application_error": empirical_mean,
                    "probe_count_per_heldout_operator": probe_count,
                    "probe_seeds": runtime.probe_seeds,
                }
            )
    _verify_preprobe_lock(preprobe_lock, require_probe_authorization=True)
    _verify_runtime_against_selection_lock(
        preprobe_lock,
        selected,
        runtime_by_id,
        bins,
        probe_count=probe_count,
    )
    return family_rows, operator_rows


def _calibration_thresholds(config: Mapping[str, Any]) -> CalibrationThresholds:
    statistics = _mapping(config.get("statistics"), "statistics")
    return CalibrationThresholds(
        minimum_r2=float(statistics["minimum_empirical_prediction_r2"]),
        minimum_slope=float(statistics["minimum_empirical_calibration_slope"]),
        maximum_slope=float(statistics["maximum_empirical_calibration_slope"]),
        maximum_absolute_intercept=float(
            statistics["maximum_absolute_empirical_calibration_intercept"]
        ),
    )


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _execution_contract(
    *,
    dry_run: bool,
    design_passed: bool,
    registry_exhausted: bool,
    selected_count: int,
    selection_row_count: int,
    family_row_count: int,
    operator_row_count: int,
    expected_families: int,
    expected_operator_rows: int,
    metrics_finite: bool,
) -> dict[str, bool]:
    empirical_shape_complete = bool(
        design_passed
        and selected_count == expected_families
        and selection_row_count == expected_families
        and family_row_count == expected_families
        and operator_row_count == expected_operator_rows
        and metrics_finite
    )
    no_probe_design_failure_complete = bool(
        not design_passed
        and registry_exhausted
        and family_row_count == 0
        and operator_row_count == 0
    )
    protocol_execution_complete = bool(
        empirical_shape_complete or no_probe_design_failure_complete
    )
    main_execution_complete = bool(not dry_run and protocol_execution_complete)
    claim_evaluated = bool(main_execution_complete and empirical_shape_complete)
    return {
        "empirical_shape_complete": empirical_shape_complete,
        "no_probe_design_failure_complete": no_probe_design_failure_complete,
        "protocol_execution_complete": protocol_execution_complete,
        "main_execution_complete": main_execution_complete,
        "claim_evaluated": claim_evaluated,
    }


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    preview_path = Path(args.config).resolve(strict=True)
    preview = _validate_registered_config_file(preview_path)
    preflight_device = resolve_device(args.device)
    if preflight_device.type != "cpu":
        raise ValueError("E03b is a deterministic CPU theory experiment; use --device cpu.")
    runtime_config = _mapping(preview.get("runtime"), "runtime")
    torch.set_num_threads(int(runtime_config["cpu_threads"]))

    dependencies = [
        validate_legacy_e00(
            args.artifact_root,
            require_full=not args.dry_run,
        ),
        _validate_pinned_e03_source(args.artifact_root, preview),
    ]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    _validate_initialized_config(
        preview_path,
        preview,
        config,
        context.config_path,
    )
    if device.type != "cpu":
        raise AssertionError("E03b device changed after CPU preflight.")

    bins = _regret_bins(config)
    selection_config = _mapping(config.get("selection"), "selection")
    families_per_bin = int(
        selection_config[
            "dry_run_families_per_bin" if args.dry_run else "families_per_bin"
        ]
    )
    data = _mapping(config.get("data"), "data")
    estimator = _mapping(config.get("basis_estimator"), "basis_estimator")
    weights = _training_weights(
        estimator,
        int(data["train_operators_per_family"]),
    )
    registry_path, registry_file_sha256, registry_sha256 = _write_registry_artifact(
        run_dir,
        config,
    )

    started = time.perf_counter()
    selected, runtime_by_id, audit_rows = _scan_analytic_candidates(
        config,
        dry_run=bool(args.dry_run),
        bins=bins,
        families_per_bin=families_per_bin,
        weights=weights,
    )
    design_gate = validate_selected_design(
        selected,
        bins,
        families_per_bin=families_per_bin,
        minimum_nonzero_range=float(selection_config["minimum_nonzero_range"]),
    )
    selection_rows = _selection_lock_rows(
        selected,
        runtime_by_id,
        bins,
        registry_sha256=registry_sha256,
        probe_count=int(data["probes_per_heldout_operator"]),
    )
    preprobe_lock = _write_preprobe_lock(
        run_dir,
        registry_path=registry_path,
        registry_file_sha256=registry_file_sha256,
        registry_sha256=registry_sha256,
        audit_rows=audit_rows,
        selection_rows=selection_rows,
        design_gate=design_gate,
    )

    family_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] | None = None
    if bool(design_gate["passed"]):
        family_rows, operator_rows = _run_empirical_phase(
            selected,
            runtime_by_id,
            bins,
            preprobe_lock=preprobe_lock,
            probe_count=int(data["probes_per_heldout_operator"]),
        )
        calibration = fit_jd_application_calibration(
            np.asarray(
                [float(row["heldout_analytic_regret"]) for row in family_rows],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    float(row["heldout_empirical_application_error"])
                    for row in family_rows
                ],
                dtype=np.float64,
            ),
            thresholds=_calibration_thresholds(config),
        )

    family_path = run_dir / "family_calibration_metrics.jsonl"
    operator_path = run_dir / "heldout_operator_application_metrics.jsonl"
    write_jsonl_strict(family_path, family_rows)
    write_jsonl_strict(operator_path, operator_rows)

    expected_families = len(bins) * families_per_bin
    expected_operator_rows = expected_families * int(
        data["heldout_operators_per_family"]
    )
    registry = _mapping(config.get("candidate_registry"), "candidate_registry")
    stream_name = "dry_run" if args.dry_run else "main"
    stream_contract = _mapping(
        registry.get(stream_name),
        f"candidate_registry.{stream_name}",
    )
    registry_exhausted = len(audit_rows) == int(stream_contract["hard_max_candidates"])
    selected_count = sum(len(values) for values in selected.values())
    execution_contract = _execution_contract(
        dry_run=bool(args.dry_run),
        design_passed=bool(design_gate["passed"]),
        registry_exhausted=registry_exhausted,
        selected_count=selected_count,
        selection_row_count=len(selection_rows),
        family_row_count=len(family_rows),
        operator_row_count=len(operator_rows),
        expected_families=expected_families,
        expected_operator_rows=expected_operator_rows,
        metrics_finite=bool(_all_finite(family_rows) and _all_finite(operator_rows)),
    )
    empirical_shape_complete = execution_contract["empirical_shape_complete"]
    no_probe_design_failure_complete = execution_contract[
        "no_probe_design_failure_complete"
    ]
    protocol_execution_complete = execution_contract["protocol_execution_complete"]
    main_execution_complete = execution_contract["main_execution_complete"]
    claim_evaluated = execution_contract["claim_evaluated"]
    calibration_passed = bool(
        calibration is not None and calibration.get("passed") is True
    )
    supported = bool(claim_evaluated and calibration_passed)
    if supported:
        calibration_status = "SUPPORTED"
        calibration_reason = None
    elif claim_evaluated:
        calibration_status = "FAILED"
        calibration_reason = "PREREGISTERED_CALIBRATION_GATE_FAILED"
    elif args.dry_run:
        calibration_status = "NOT_EVALUATED"
        calibration_reason = "DRY_RUN"
    else:
        calibration_status = "NOT_EVALUATED"
        calibration_reason = "REGISTERED_DESIGN_OR_OPTIMIZATION_SUPPORT_INCOMPLETE"
    scanned_valid = sum(
        row.get("eligible_for_binning") is True for row in audit_rows
    )
    scanned_invalid = len(audit_rows) - scanned_valid

    report = {
        "status": "PASS",
        "execution": {
            "dry_run": bool(args.dry_run),
            "candidate_registry_sha256": registry_sha256,
            "analytic_candidates_scanned": len(audit_rows),
            "analytically_valid_candidates_scanned": scanned_valid,
            "analytically_invalid_candidates_skipped": scanned_invalid,
            "selected_family_count": selected_count,
            "expected_family_count": expected_families,
            "heldout_operator_metric_rows": len(operator_rows),
            "expected_heldout_operator_metric_rows": expected_operator_rows,
            "training_operators_per_family": int(
                data["train_operators_per_family"]
            ),
            "heldout_operators_per_family": int(
                data["heldout_operators_per_family"]
            ),
            "probes_per_heldout_operator": int(
                data["probes_per_heldout_operator"]
            ),
            "cpu_threads": torch.get_num_threads(),
            "spectral_comparator_weight_vector_sha256": tensor_sha256(weights),
            "candidate_registry_exhausted": registry_exhausted,
            "empirical_shape_complete": empirical_shape_complete,
            "no_probe_design_failure_complete": no_probe_design_failure_complete,
            "protocol_execution_complete": protocol_execution_complete,
            "main_execution_complete": main_execution_complete,
            "wall_seconds": time.perf_counter() - started,
        },
        "dependency_lineage": dependencies,
        "protocol_lineage": {
            "frozen_protocol": {
                "path": "docs/E03B_PROTOCOL_PREREGISTRATION_FROZEN_KO.md",
                "sha256": _FROZEN_PROTOCOL_SHA256,
            },
            "frozen_protocol_lock": {
                "path": "docs/E03B_PROTOCOL_PREREGISTRATION_LOCK_KO.md",
                "sha256": _FROZEN_PROTOCOL_LOCK_SHA256,
            },
            "analytic_pilot": {
                "path": "docs/E03B_ANALYTIC_PILOT_KO.md",
                "sha256": _ANALYTIC_PILOT_SHA256,
                "empirical_probe_calls": 0,
            },
            "preserved_original_e03_runner_sha256": (
                _PRESERVED_E03_RUNNER_SHA256
            ),
            "preserved_original_jd_core_sha256": _PRESERVED_JD_CORE_SHA256,
            "basis_estimator": dict(estimator),
        },
        "original_e03_disposition": {
            "run_id": _EXPECTED_SOURCE_E03["run_id"],
            "execution_completed": True,
            "categorical_geometry_status": "SUPPORTED",
            "quantitative_calibration_status": "FAILED",
            "quantitative_diagnosis": "PREDICTOR_RANGE_RESTRICTION",
            "full_h3_claim_open": False,
            "status": "immutable_split_claim_registry_not_relabelled",
        },
        "metric_definitions": {
            "analytic_regret": (
                "Heldout mean squared operator-entry reconstruction error in a "
                "fixed train-only multi-restart JD shared basis. This is an "
                "estimated train-fit out-of-sample JD regret, not a certified "
                "global minimum over heldout operators."
            ),
            "empirical_application_error": (
                "Heldout isotropic-probe application MSE with input-component "
                "variance 1/d; its expectation equals analytic regret."
            ),
            "basis_estimator": (
                "The original E03 QR/Adam multi-restart joint-diagonalization "
                "objective fitted on 24 training projectors only. A weighted "
                "eigendecomposition is used solely as an optimization comparator; "
                "no heldout probe or application error enters basis selection."
            ),
            "regression_unit": (
                "One equally weighted family mean after averaging its eight "
                "heldout operators; n=48 for main. Operator rows are not treated "
                "as independent regression observations."
            ),
        },
        "selection_design_gate": design_gate,
        "empirical_regret_prediction": calibration,
        "preprobe_lock": {
            "status": "LOCKED_BEFORE_EMPIRICAL_PROBES",
            "metadata_path": preprobe_lock.metadata_path.name,
            "metadata_sha256": preprobe_lock.metadata_sha256,
            "selection_path": preprobe_lock.selection_path.name,
            "selection_sha256": preprobe_lock.selection_sha256,
            "analytic_audit_path": preprobe_lock.analytic_audit_path.name,
            "analytic_audit_sha256": preprobe_lock.analytic_audit_sha256,
        },
        "artifacts": {
            "candidate_split_registry": {
                "path": registry_path.name,
                "sha256": registry_file_sha256,
            },
            "analytic_candidate_audit": {
                "path": preprobe_lock.analytic_audit_path.name,
                "rows": len(audit_rows),
                "sha256": preprobe_lock.analytic_audit_sha256,
            },
            "selection_lock": {
                "path": preprobe_lock.selection_path.name,
                "rows": len(selection_rows),
                "sha256": preprobe_lock.selection_sha256,
            },
            "preprobe_lock": {
                "path": preprobe_lock.metadata_path.name,
                "sha256": preprobe_lock.metadata_sha256,
            },
            "family_calibration_metrics": {
                "path": family_path.name,
                "rows": len(family_rows),
                "sha256": sha256_file(family_path),
            },
            "heldout_operator_application_metrics": {
                "path": operator_path.name,
                "rows": len(operator_rows),
                "sha256": sha256_file(operator_path),
            },
        },
        "claim_gate": {
            "evaluated": claim_evaluated,
            "supported": supported,
            "status": calibration_status,
            "reason": calibration_reason,
            "design_gate_passed": bool(design_gate["passed"]),
            "empirical_calibration_passed": calibration_passed,
            "allowed_claim": (
                "For the registered graded synthetic projector construction, "
                "estimated train-fit out-of-sample JD regret in the fixed learned "
                "shared basis "
                "calibrated to isotropic application error over the six locked bins."
                if supported
                else None
            ),
            "forbidden_claims": [
                (
                    "The numerical multi-restart estimator certifies the global "
                    "minimum of joint-diagonalization regret."
                ),
                (
                    "The original E03 rank-8 richer-control oracle is a "
                    "parameter-matched learned E03b controller."
                ),
                "This controlled reference establishes an official-backend claim.",
                "This controlled reference establishes a language-model claim.",
                "The original E03 artifact passed its empirical calibration gate.",
            ],
        },
        "limitations": [
            (
                "The predictor is estimated out-of-sample regret of one "
                "preregistered train-only multi-restart JD fit, not a certified "
                "global JD minimum."
            ),
            (
                "Rank 8 appears only in the original richer-control oracle and the "
                "fixed projector construction; E03b fits no rank-8 controller."
            ),
            (
                "Candidate selection is synthetic and analytic-only; empirical probes "
                "are applied only after the construction and basis lock."
            ),
        ],
        "evidence_scope": config["evidence_scope"],
    }
    _validate_initialized_config(
        preview_path,
        preview,
        config,
        context.config_path,
    )
    _verify_preprobe_lock(
        preprobe_lock,
        require_probe_authorization=bool(design_gate["passed"]),
    )
    if bool(design_gate["passed"]):
        _verify_runtime_against_selection_lock(
            preprobe_lock,
            selected,
            runtime_by_id,
            bins,
            probe_count=int(data["probes_per_heldout_operator"]),
        )
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=main_execution_complete,
        full_eligible=main_execution_complete,
    )
    print(
        f"[{EXPERIMENT_ID}] PASS: {run_dir} "
        f"(H3 calibration={calibration_status})"
    )


if __name__ == "__main__":
    main()
