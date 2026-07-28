"""Prospective protocol checks shared by the split E24 theory stresses."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from catena.core.io import file_sha256
from catena.post_e21.contracts import PostE21ContractError, ProtocolSnapshot

E24_PROTOCOL_VERSION = "E24_PREMAIN_V1"
E24_EVIDENCE_TIER = "CONTROLLED_THEORY_STRESS"
REGISTERED_SEEDS = (3101, 3119, 3137, 3163, 3181, 3203, 3221, 3251)
REGISTERED_RANKS = (1, 2, 4, 8, 16, 32)
REGISTERED_EXPONENTIAL_TAU = (1.5, 3.0, 6.0, 12.0)
REGISTERED_POWER_LAW_EXPONENT = (0.75, 1.0, 1.5, 2.0)
REGISTERED_BASE_RANKS = (2, 8, 16)
REGISTERED_NOISE = (0.0, 0.01, 0.05, 0.10)
REGISTERED_LAMBDAS = (0.25, 0.5, 0.75)
REGISTERED_HORIZONS = (1, 4, 8)
REGISTERED_READOUTS = ("linear", "fixed_nonlinear_mlp")
REGISTERED_DEMANDS = (
    "axis_commuting",
    "common_rotated_commuting",
    "noncommuting",
)
REGISTERED_CONTROLLERS = (
    "fixed_diagonal",
    "shared_basis_diagonal",
    "rank8",
    "full",
)
REGISTERED_E24B_NOISE_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "label": "clean_teacher",
        "target_noise_factor": 0.0,
        "teacher_noise_factor": 0.0,
    },
    {
        "label": "target_shift_0p01",
        "target_noise_factor": 0.01,
        "teacher_noise_factor": 0.0,
    },
    {
        "label": "target_shift_0p05",
        "target_noise_factor": 0.05,
        "teacher_noise_factor": 0.0,
    },
    {
        "label": "target_shift_0p10",
        "target_noise_factor": 0.10,
        "teacher_noise_factor": 0.0,
    },
    {
        "label": "teacher_corruption_0p01",
        "target_noise_factor": 0.0,
        "teacher_noise_factor": 0.01,
    },
    {
        "label": "teacher_corruption_0p05",
        "target_noise_factor": 0.0,
        "teacher_noise_factor": 0.05,
    },
    {
        "label": "teacher_corruption_0p10",
        "target_noise_factor": 0.0,
        "teacher_noise_factor": 0.10,
    },
    {
        "label": "matched_noise_0p01",
        "target_noise_factor": 0.01,
        "teacher_noise_factor": 0.01,
    },
    {
        "label": "matched_noise_0p05",
        "target_noise_factor": 0.05,
        "teacher_noise_factor": 0.05,
    },
    {
        "label": "matched_noise_0p10",
        "target_noise_factor": 0.10,
        "teacher_noise_factor": 0.10,
    },
)
REGISTERED_GEOMETRY_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "label": "baseline_geometry",
        "key_correlation": 0.0,
        "target_operator_norm": 1.0,
        "key_load_fraction": 0.5,
    },
    {
        "label": "unseen_key_correlation",
        "key_correlation": 0.75,
        "target_operator_norm": 1.0,
        "key_load_fraction": 0.5,
    },
    {
        "label": "unseen_operator_norm",
        "key_correlation": 0.0,
        "target_operator_norm": 1.5,
        "key_load_fraction": 0.5,
    },
    {
        "label": "unseen_key_load",
        "key_correlation": 0.0,
        "target_operator_norm": 1.0,
        "key_load_fraction": 1.0,
    },
)
REGISTERED_GEOMETRY_BLOCKS = (
    "baseline_geometry",
    "unseen_key_correlation",
    "unseen_operator_norm",
    "unseen_key_load",
)
REGISTERED_HOLDOUTS = (
    "demand_family",
    "controller_class",
    "geometry_block",
)
REGISTERED_PREDICTOR_FEATURES = (
    "log_teacher_linearized_behavioral_regret",
    "log_teacher_lipschitz_upper_bound",
    "log_teacher_operator_residual",
    "log_horizon",
    "target_noise_factor",
    "teacher_noise_factor",
    "readout_lambda",
    "key_correlation",
    "target_operator_norm",
    "key_load_fraction",
    "nonlinear_readout_indicator",
)
E24_REQUIRED_DEPENDENCIES: tuple[dict[str, Any], ...] = (
    {
        "anchor_id": "h1_behavioral_reachability",
        "experiment_id": "e01b_constrained_behavioral_reachability",
        "relative_report_path": (
            "e01b_constrained_behavioral_reachability/20260726T152354.081239Z/report.json"
        ),
        "report_sha256": ("8e1d16ca7763cec1e4e5b13d2b0f163f4015c8058ed7764871a6fbb5fa5ea6d6"),
        "expected_status": "PASS",
        "expected_claim_disposition": "SUPPORTED",
        "expected_fields": {
            "status": "PASS",
            "claim_gate.supported": True,
        },
    },
    {
        "anchor_id": "e10b_floor_aware_rank_scaling",
        "experiment_id": "e10b_floor_aware_rank_scaling",
        "relative_report_path": (
            "e10b_floor_aware_rank_scaling/20260727T190906.272784Z/report.json"
        ),
        "report_sha256": ("30f2f781bcc8528964602e0c66e1b61bb9d71a6ca5f964b833b2551c93b72484"),
        "expected_status": "PASS",
        "expected_claim_disposition": "SUPPORTED",
        "expected_fields": {
            "status": "PASS",
            "claim_gate.supported": True,
        },
    },
    {
        "anchor_id": "e11b_scale_normalized_coadaptation",
        "experiment_id": "e11b_scale_normalized_coadaptation",
        "relative_report_path": (
            "e11b_scale_normalized_coadaptation/20260727T183004.928280Z/report.json"
        ),
        "report_sha256": ("54015400029b3eae0367a1c12cb1dd717dee5a0568906f7d8e972c45bc4301b3"),
        "expected_status": "PASS",
        "expected_claim_disposition": "SUPPORTED",
        "expected_fields": {
            "status": "PASS",
            "claim_gate.supported": True,
        },
    },
)


class E24DependencyError(PostE21ContractError):
    """Future E24 main dependency failure with an explicit blocked status."""

    status = "BLOCKED_DEPENDENCY"


def _as_tuple(values: object, *, label: str) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(values)


def _require_exact(
    values: object,
    expected: Sequence[Any],
    *,
    label: str,
) -> None:
    observed = _as_tuple(values, label=label)
    if observed != tuple(expected):
        raise ValueError(f"{label} changed: {observed!r} != {tuple(expected)!r}")


def _require_finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return normalized


def _validate_dependency_contract(config: Mapping[str, Any]) -> None:
    dependencies = config.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ValueError("E24 immutable-evidence dependency section is missing")
    if (
        dependencies.get("canonical_artifact_root") != "/data/minjun_dev/CATENA/artifacts"
        or dependencies.get("dry_run_policy") != "EXPECTATIONS_ONLY_DO_NOT_READ_CANONICAL_ARTIFACTS"
        or dependencies.get("main_requires_all") is not True
        or dependencies.get("main_failure_status") != "BLOCKED_DEPENDENCY"
    ):
        raise ValueError("E24 dependency execution policy changed")
    _require_exact(
        dependencies.get("required"),
        E24_REQUIRED_DEPENDENCIES,
        label="dependencies.required",
    )


def _validate_common(config: Mapping[str, Any], *, experiment_id: str) -> None:
    if config.get("experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id} config experiment_id mismatch")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("E24 protocol section is missing")
    if (
        protocol.get("version") != E24_PROTOCOL_VERSION
        or protocol.get("protocol_frozen_before_main") is not True
        or protocol.get("main_execution_started") is not False
        or protocol.get("main_authorized_by_default") is not False
        or protocol.get("main_requires_explicit_allow_main") is not True
    ):
        raise ValueError("E24 pre-main protocol flags changed")
    evidence = config.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("E24 evidence section is missing")
    if (
        evidence.get("tier") != E24_EVIDENCE_TIER
        or evidence.get("dry_run_claim_eligible") is not False
        or evidence.get("official_backend_claim_eligible") is not False
        or evidence.get("pretrained_language_model_claim_eligible") is not False
    ):
        raise ValueError("E24 evidence boundary changed")
    _validate_dependency_contract(config)
    _require_exact(config.get("seeds"), REGISTERED_SEEDS, label="seeds")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("E24 execution section is missing")
    if (
        execution.get("dry_run_required_device_type") != "cpu"
        or execution.get("dry_run_artifact_root") != "fresh_subdirectory_below_tmp"
        or execution.get("main_requires_explicit_allow_main") is not True
        or execution.get("main_requires_explicit_dependency_root") is not True
        or execution.get("main_writes_forbidden_without_allow_main") is not True
    ):
        raise ValueError("E24 execution lock changed")
    reporting = config.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError("E24 reporting section is missing")
    if (
        reporting.get("results_summary_filename") != "RESULTS_SUMMARY_KO.md"
        or reporting.get("results_summary_language") != "ko"
        or int(reporting.get("maximum_lines", 0)) != 45
        or reporting.get("dry_run_status_label") != "DRY_RUN_NON_EVIDENCE"
        or reporting.get("main_status_label") != "MAIN_CONTROLLED_THEORY_STRESS"
    ):
        raise ValueError("E24 one-page dry-run summary contract changed")


def dependency_expectation_payload(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return frozen dependency expectations without touching artifact files."""

    _validate_dependency_contract(config)
    return {
        "validation_mode": "EXPECTATIONS_ONLY",
        "validation_status": "NOT_READ_DRY_RUN",
        "canonical_artifacts_read": False,
        "future_main_missing_or_mismatch_status": "BLOCKED_DEPENDENCY",
        "required": [
            {
                **anchor,
                "expected_fields": dict(anchor["expected_fields"]),
            }
            for anchor in E24_REQUIRED_DEPENDENCIES
        ],
    }


def _dotted_value(payload: Mapping[str, Any], dotted: str) -> object:
    current: object = payload
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted)
        current = current[component]
    return current


def _blocked_dependency(message: str) -> E24DependencyError:
    return E24DependencyError(f"BLOCKED_DEPENDENCY: {message}")


def validate_e24_main_dependencies(
    config: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate H1/E10b/E11b reports for a separately authorized future main."""

    _validate_dependency_contract(config)
    dependencies = config["dependencies"]
    if not isinstance(dependencies, Mapping):
        raise ValueError("E24 dependencies must be a mapping")
    selected_root = (
        dependencies["canonical_artifact_root"] if artifact_root is None else artifact_root
    )
    root = Path(selected_root).resolve()
    validated: list[dict[str, Any]] = []
    for anchor in E24_REQUIRED_DEPENDENCIES:
        relative = Path(str(anchor["relative_report_path"]))
        unresolved = root / relative
        if unresolved.is_symlink():
            raise _blocked_dependency(f"{anchor['anchor_id']} report path is a symlink")
        try:
            report_path = unresolved.resolve(strict=True)
            report_path.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise _blocked_dependency(
                f"{anchor['anchor_id']} report is missing or escapes artifact root"
            ) from error
        if not report_path.is_file() or report_path.is_symlink():
            raise _blocked_dependency(
                f"{anchor['anchor_id']} report is not a regular immutable file"
            )
        observed_hash = file_sha256(report_path)
        if observed_hash != anchor["report_sha256"]:
            raise _blocked_dependency(f"{anchor['anchor_id']} report SHA-256 mismatch")
        try:
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise _blocked_dependency(f"{anchor['anchor_id']} report is unreadable") from error
        if not isinstance(report, dict):
            raise _blocked_dependency(f"{anchor['anchor_id']} report is not a JSON object")
        expected_fields = anchor["expected_fields"]
        if not isinstance(expected_fields, Mapping):
            raise ValueError("E24 dependency expected_fields must be a mapping")
        for dotted, expected in expected_fields.items():
            try:
                observed = _dotted_value(report, str(dotted))
            except KeyError as error:
                raise _blocked_dependency(
                    f"{anchor['anchor_id']} lacks expected field {dotted}"
                ) from error
            if observed != expected:
                raise _blocked_dependency(f"{anchor['anchor_id']} field {dotted} changed")
        validated.append(
            {
                "anchor_id": anchor["anchor_id"],
                "experiment_id": anchor["experiment_id"],
                "report_path": str(report_path),
                "report_sha256": observed_hash,
                "status": anchor["expected_status"],
                "claim_disposition": anchor["expected_claim_disposition"],
            }
        )
    return {
        "validation_status": "PASS",
        "all_required_dependencies_validated": True,
        "dependencies": validated,
    }


def validate_e24a_config(config: Mapping[str, Any]) -> None:
    """Validate every registered E24a approximate-rank design field."""

    _validate_common(config, experiment_id="e24a_approximate_rank_stress")
    design = config.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("E24a design section is missing")
    dimension = int(design.get("dimension", 0))
    if dimension != 64 or dimension < max(REGISTERED_RANKS):
        raise ValueError("E24a dimension changed or cannot cover rank 32")
    _require_exact(
        design.get("controller_ranks"),
        REGISTERED_RANKS,
        label="design.controller_ranks",
    )
    spectra = design.get("spectra")
    if not isinstance(spectra, Mapping):
        raise ValueError("E24a spectra section is missing")
    _require_exact(
        spectra.get("exponential_tau"),
        REGISTERED_EXPONENTIAL_TAU,
        label="design.spectra.exponential_tau",
    )
    _require_exact(
        spectra.get("power_law_exponent"),
        REGISTERED_POWER_LAW_EXPONENT,
        label="design.spectra.power_law_exponent",
    )
    low_rank = spectra.get("low_rank_plus_noise")
    if not isinstance(low_rank, Mapping):
        raise ValueError("E24a low-rank-plus-noise section is missing")
    _require_exact(
        low_rank.get("base_ranks"),
        REGISTERED_BASE_RANKS,
        label="design.spectra.low_rank_plus_noise.base_ranks",
    )
    _require_exact(
        low_rank.get("tail_noise"),
        REGISTERED_NOISE,
        label="design.spectra.low_rank_plus_noise.tail_noise",
    )
    transfer = design.get("spectrum_family_transfer")
    if (
        not isinstance(transfer, Mapping)
        or transfer.get("fold_rule") != "leave_one_spectrum_family_out"
        or transfer.get("family_identity_feature_included") is not False
        or transfer.get("test_outcomes_used_for_training") is not False
        or transfer.get("predictions_written_before_test_outcome_join") is not True
        or transfer.get("shared_left_right_basis_within_seed") is not True
    ):
        raise ValueError("E24a OOD spectrum-family protocol changed")
    _require_exact(
        transfer.get("families"),
        ("exponential", "power_law", "low_rank_plus_noise"),
        label="design.spectrum_family_transfer.families",
    )
    learning = config.get("learning")
    if not isinstance(learning, Mapping):
        raise ValueError("E24a learning section is missing")
    if learning.get(
        "primary_estimator"
    ) != "descriptor_conditioned_low_rank_controller" or learning.get("reused_pattern") != (
        "E10/E10b LowRankOperatorController plus AdamW matrix training"
    ):
        raise ValueError("E24a learned-estimator contract changed")
    descriptor = learning.get("descriptor")
    if (
        not isinstance(descriptor, Mapping)
        or int(descriptor.get("dimension", 0)) != 16
        or descriptor.get("include_effective_rank") is not True
        or descriptor.get("include_stable_rank") is not True
        or descriptor.get("family_identity_feature_included") is not False
    ):
        raise ValueError("E24a descriptor contract changed")
    _require_exact(
        descriptor.get("log_singular_value_indices"),
        (0, 1, 2, 3, 7, 15, 31, 63),
        label="learning.descriptor.log_singular_value_indices",
    )
    _require_exact(
        descriptor.get("cumulative_energy_ranks"),
        REGISTERED_RANKS,
        label="learning.descriptor.cumulative_energy_ranks",
    )
    model = learning.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("class") != "LowRankOperatorController"
        or int(model.get("hidden_dim", 0)) != 64
    ):
        raise ValueError("E24a model contract changed")
    training = learning.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("optimizer") != "AdamW"
        or int(training.get("steps", 0)) != 800
        or int(training.get("batch_size", 0)) != 16
        or float(training.get("gradient_clip_norm", 0.0)) != 1.0
    ):
        raise ValueError("E24a optimizer schedule changed")
    _require_finite_positive(
        training.get("learning_rate"),
        label="learning.training.learning_rate",
    )
    _require_finite_positive(
        training.get("weight_decay"),
        label="learning.training.weight_decay",
    )
    evaluation = learning.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or int(evaluation.get("batch_size", 0)) != 64
        or evaluation.get("clean_population_operator") is not True
        or evaluation.get("test_outcome_join_after_prediction_artifact") is not True
    ):
        raise ValueError("E24a evaluation contract changed")
    diagnostic = learning.get("direct_empirical_svd_diagnostic")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("primary_result") is not False
        or int(diagnostic.get("observation_count", 0)) != 32
        or diagnostic.get("evaluation_target") != "clean_population_operator"
    ):
        raise ValueError("E24a direct diagnostic boundary changed")
    _require_finite_positive(
        diagnostic.get("relative_observation_noise"),
        label="learning.direct_empirical_svd_diagnostic.relative_noise",
    )
    epsilon = config.get("epsilon_minimal")
    if not isinstance(epsilon, Mapping):
        raise ValueError("E24a epsilon-minimal section is missing")
    epsilon_value = _require_finite_positive(
        epsilon.get("relative_residual_epsilon"),
        label="epsilon_minimal.relative_residual_epsilon",
    )
    if epsilon_value >= 1.0:
        raise ValueError("E24a epsilon must be below one")
    if epsilon.get("grid_only") is not True or epsilon.get("unresolved_is_not_a_match") is not True:
        raise ValueError("E24a epsilon-minimal decision rule changed")
    gate = config.get("claim_gate")
    if not isinstance(gate, Mapping) or gate.get("all_three_family_folds_required") is not True:
        raise ValueError("E24a OOD claim gate changed")
    _require_finite_positive(
        gate.get("minimum_ood_epsilon_rank_match_fraction"),
        label="claim_gate.minimum_ood_epsilon_rank_match_fraction",
    )
    _require_finite_positive(
        gate.get("maximum_mean_normalized_excess_over_oracle"),
        label="claim_gate.maximum_mean_normalized_excess_over_oracle",
    )
    dry = config.get("dry_run_overrides")
    if not isinstance(dry, Mapping):
        raise ValueError("E24a dry-run overrides are missing")
    if int(dry.get("seed", 0)) in REGISTERED_SEEDS:
        raise ValueError("E24a dry-run seed must be excluded from main")
    _require_exact(
        dry.get("learned_ranks"),
        (1, 8, 32),
        label="dry_run_overrides.learned_ranks",
    )
    if int(dry.get("training_steps", 0)) != 8:
        raise ValueError("E24a dry-run training steps changed")


def validate_e24b_config(config: Mapping[str, Any]) -> None:
    """Validate every registered E24b behavioral-attainability field."""

    _validate_common(config, experiment_id="e24b_behavioral_attainability_stress")
    design = config.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("E24b design section is missing")
    if int(design.get("dimension", 0)) != 16:
        raise ValueError("E24b dimension changed")
    if int(design.get("batch_size", 0)) != 128:
        raise ValueError("E24b batch size changed")
    _require_exact(
        design.get("noise_conditions"),
        REGISTERED_E24B_NOISE_CONDITIONS,
        label="design.noise_conditions",
    )
    _require_exact(
        design.get("readout_lambda"),
        REGISTERED_LAMBDAS,
        label="design.readout_lambda",
    )
    _require_exact(
        design.get("horizon"),
        REGISTERED_HORIZONS,
        label="design.horizon",
    )
    _require_exact(
        design.get("readout"),
        REGISTERED_READOUTS,
        label="design.readout",
    )
    _require_exact(
        design.get("demand_families"),
        REGISTERED_DEMANDS,
        label="design.demand_families",
    )
    _require_exact(
        design.get("controller_classes"),
        REGISTERED_CONTROLLERS,
        label="design.controller_classes",
    )
    _require_exact(
        design.get("geometry_profiles"),
        REGISTERED_GEOMETRY_PROFILES,
        label="design.geometry_profiles",
    )
    _require_exact(
        design.get("geometry_blocks"),
        REGISTERED_GEOMETRY_BLOCKS,
        label="design.geometry_blocks",
    )
    _require_exact(
        design.get("holdouts"),
        REGISTERED_HOLDOUTS,
        label="design.holdouts",
    )
    simulation = config.get("simulation")
    if not isinstance(simulation, Mapping):
        raise ValueError("E24b simulation section is missing")
    if (
        float(simulation.get("update_scale", -1.0)) != 0.20
        or float(simulation.get("recurrence_decay", -1.0)) != 0.55
        or float(simulation.get("affected_row_fraction", -1.0)) != 0.75
        or simulation.get("retained_rows_clean_zero") is not True
        or simulation.get("exact_temporal_input_whitening") is not True
        or simulation.get("minimum_batch_size") != "horizon * dimension"
        or simulation.get("controller_fit_source") != "noisy_teacher_only"
        or simulation.get("evaluation_target") != "clean_application_target_only"
        or simulation.get("geometry_randomness_paired_across_profiles") is not True
        or float(simulation.get("nonlinear_residual_scale", -1.0)) != 0.20
        or simulation.get("nonlinear_readout_definition")
        != ("deterministic block-local tied-orthogonal two-layer tanh residual MLP with bias")
    ):
        raise ValueError("E24b clean-target/teacher simulation contract changed")
    target_noise_definition = simulation.get("target_noise_definition")
    teacher_noise_definition = simulation.get("teacher_noise_definition")
    if (
        not isinstance(target_noise_definition, str)
        or "clean application target" not in target_noise_definition
        or not isinstance(teacher_noise_definition, str)
        or "clean application target" not in teacher_noise_definition
    ):
        raise ValueError("E24b target/teacher noise definitions changed")
    if (
        simulation.get("key_correlation_definition")
        != ("exact off-diagonal correlation of the temporally whitened input-key covariance")
        or simulation.get("key_load_definition")
        != (
            "fraction of input-key coordinates at unit scale; remaining coordinates have scale 0.25"
        )
        or simulation.get("operator_norm_definition")
        != ("Frobenius norm of each nominal clean target before target-noise realization")
    ):
        raise ValueError("E24b explicit input geometry definitions changed")
    weighting = simulation.get("readout_weighting")
    if (
        not isinstance(weighting, Mapping)
        or weighting.get("formula") != "lambda * affected_mse + (1 - lambda) * unaffected_mse"
        or weighting.get("row_blocks")
        != ("structural_affected_and_retained_state_rows_with_within_block_qr_whitening")
        or weighting.get("affected_block") != "first affected_row_fraction rows"
        or weighting.get("unaffected_block") != "remaining structurally retained rows"
        or weighting.get("component_mse_weighting") is not True
        or weighting.get("global_output_scaling") is not False
    ):
        raise ValueError("E24b affected/unaffected readout weighting changed")
    predictor = config.get("predictor")
    if not isinstance(predictor, Mapping):
        raise ValueError("E24b predictor section is missing")
    if (
        predictor.get("method") != "training_only_log_ridge_on_teacher_side_proxies_v2"
        or predictor.get("clean_target_features_forbidden") is not True
        or predictor.get("fold_membership_precomputed_without_outcomes") is not True
        or predictor.get("predictions_written_before_test_outcome_join") is not True
    ):
        raise ValueError("E24b outcome-independent predictor contract changed")
    _require_exact(
        predictor.get("feature_order"),
        REGISTERED_PREDICTOR_FEATURES,
        label="predictor.feature_order",
    )
    _require_finite_positive(predictor.get("ridge"), label="predictor.ridge")
    _require_finite_positive(
        predictor.get("outcome_floor"),
        label="predictor.outcome_floor",
    )
    optimization_gap = config.get("optimization_gap")
    if not isinstance(optimization_gap, Mapping):
        raise ValueError("E24b optimization-gap section is missing")
    if (
        optimization_gap.get("lower_bound_name")
        != "controller_specific_clean_target_analytic_behavioral_lower_bound"
        or optimization_gap.get("same_controller_class_required") is not True
        or optimization_gap.get("observed_application_outcome_used_in_bound") is not False
        or optimization_gap.get("bound_frozen_before_observed_application_outcome") is not True
        or optimization_gap.get("clean_oracle_attainable_error_reported_separately") is not True
        or optimization_gap.get("predictor_feature_excluded") is not True
        or optimization_gap.get("excess_definition")
        != (
            "observed_application_error minus controller-specific clean-target "
            "analytic behavioral lower bound"
        )
    ):
        raise ValueError("E24b optimization-gap contract changed")
    construction = optimization_gap.get("construction")
    if (
        not isinstance(construction, str)
        or "clean-target projection" not in construction
        or "same controller class" not in construction
    ):
        raise ValueError("E24b controller-specific bound construction changed")
    inference = config.get("inference")
    if not isinstance(inference, Mapping) or inference.get("upper_unit") != "seed":
        raise ValueError("E24b inference upper unit changed")
    bootstrap = inference.get("cluster_bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("enabled") is not True
        or int(bootstrap.get("replicates", 0)) != 2000
        or float(bootstrap.get("confidence_level", 0.0)) != 0.95
        or int(bootstrap.get("seed", 0)) != 24024
        or bootstrap.get("resample_unit") != "seed"
        or bootstrap.get("episode_row_resampling_forbidden") is not True
    ):
        raise ValueError("E24b seed-cluster bootstrap contract changed")
    metrics = config.get("oos_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("E24b OOS metrics section is missing")
    required = (
        "r2",
        "rmse",
        "mae",
        "pearson_r",
        "spearman_r",
        "calibration_slope",
        "calibration_intercept",
    )
    _require_exact(metrics.get("required"), required, label="oos_metrics.required")
    gates = metrics.get("gates")
    if not isinstance(gates, Mapping) or gates.get("all_holdout_axes_required") is not True:
        raise ValueError("E24b OOS gate section changed")
    _require_finite_positive(gates.get("minimum_r2"), label="oos minimum_r2")
    _require_finite_positive(
        gates.get("minimum_pearson_r"),
        label="oos minimum_pearson_r",
    )
    _require_finite_positive(
        gates.get("maximum_normalized_rmse"),
        label="oos maximum_normalized_rmse",
    )
    assessment = config.get("claim_assessment")
    if not isinstance(assessment, Mapping):
        raise ValueError("E24b claim-assessment section is missing")
    if (
        assessment.get("prediction_scale") != "log_behavioral_mse"
        or assessment.get("broad_pass_requires_overall_oos_gate") is not True
    ):
        raise ValueError("E24b claim-assessment scale or conjunction changed")
    broad = assessment.get("broad_subset")
    linear = assessment.get("linear_h1_subset")
    if (
        not isinstance(broad, Mapping)
        or broad.get("teacher_noise_factor") != "positive"
        or broad.get("readout") != "fixed_nonlinear_mlp"
        or broad.get("horizon") != "greater_than_one"
        or not isinstance(linear, Mapping)
        or linear.get("readout") != "linear"
        or int(linear.get("horizon", 0)) != 1
    ):
        raise ValueError("E24b registered claim subsets changed")
    _require_exact(
        assessment.get("decision_order"),
        (
            "BROAD_NOISY_NONLINEAR_MULTISTEP_PASS",
            "ONLY_LINEAR_H1_PASS",
            "CONSTRUCTION_ROBUST_PREDICTION_FAILURE",
        ),
        label="claim_assessment.decision_order",
    )
    dry = config.get("dry_run_overrides")
    if not isinstance(dry, Mapping):
        raise ValueError("E24b dry-run overrides are missing")
    if int(dry.get("seed", 0)) in REGISTERED_SEEDS:
        raise ValueError("E24b dry-run seed must be excluded from main")
    if (
        int(dry.get("dimension", 0)) != 16
        or int(dry.get("batch_size", 0)) != 64
        or int(dry.get("bootstrap_replicates", 0)) != 32
    ):
        raise ValueError("E24b dry-run structural overrides changed")
    _require_exact(
        dry.get("noise_conditions"),
        (
            "clean_teacher",
            "target_shift_0p05",
            "teacher_corruption_0p01",
            "teacher_corruption_0p10",
            "matched_noise_0p05",
        ),
        label="dry_run_overrides.noise_conditions",
    )
    _require_exact(
        dry.get("geometry_blocks"),
        REGISTERED_GEOMETRY_BLOCKS,
        label="dry_run_overrides.geometry_blocks",
    )
    reporting = config.get("reporting")
    if (
        not isinstance(reporting, Mapping)
        or reporting.get("family_level_scatter_unit") != "held_out_level_by_seed"
    ):
        raise ValueError("E24b family-level scatter contract changed")
    _require_exact(
        reporting.get("sensitivity_factors"),
        ("readout_lambda", "noise_condition", "horizon"),
        label="reporting.sensitivity_factors",
    )


def protocol_lock_path(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> Path:
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("E24 protocol section is missing")
    relative = protocol.get("lock_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("E24 protocol lock path is invalid")
    root = Path(repo_root).resolve(strict=True)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PostE21ContractError("E24 protocol lock escapes repository") from error
    return candidate


def validate_e24_snapshot(snapshot: ProtocolSnapshot) -> None:
    payload = snapshot.payload
    expected_dependencies = [
        {
            key: anchor[key]
            for key in (
                "anchor_id",
                "relative_report_path",
                "report_sha256",
                "expected_status",
                "expected_claim_disposition",
            )
        }
        for anchor in E24_REQUIRED_DEPENDENCIES
    ]
    if (
        payload.get("experiment_family") != "E24"
        or payload.get("protocol_version") != E24_PROTOCOL_VERSION
        or payload.get("main_authorized_by_default") is not False
        or payload.get("main_requires_explicit_allow_main") is not True
        or payload.get("dry_run_claim_eligible") is not False
        or payload.get("dry_run_dependency_validation")
        != "EXPECTATIONS_ONLY_DO_NOT_READ_CANONICAL_ARTIFACTS"
        or payload.get("future_main_dependency_failure_status") != "BLOCKED_DEPENDENCY"
        or payload.get("required_evidence_dependencies") != expected_dependencies
    ):
        raise PostE21ContractError("E24 lock does not preserve the pre-main boundary")


def require_temp_dry_root(artifact_root: str | Path) -> Path:
    """Require E24 dry-run artifacts to stay below /tmp."""

    root = Path(artifact_root).resolve()
    temporary = Path("/tmp").resolve()
    try:
        root.relative_to(temporary)
    except ValueError as error:
        raise PostE21ContractError("E24 dry-run artifact root must be below /tmp") from error
    canonical = Path("/data/minjun_dev/CATENA/artifacts").resolve()
    if root == canonical or canonical in root.parents:
        raise PostE21ContractError("E24 dry-run cannot use canonical artifacts")
    return root


def require_current_dry_run(
    *,
    dry_run: bool,
    artifact_root: str | Path,
    device: torch.device,
) -> Path:
    """Block all E24 main writes until a new prospective authorization exists."""

    if not dry_run:
        raise PostE21ContractError("E24 MAIN requires explicit --allow-main and --dependency-root")
    if device.type != "cpu":
        raise PostE21ContractError(f"E24 DRY_RUN requires CPU; got device type {device.type!r}")
    return require_temp_dry_root(artifact_root)


def select_e24_run_mode(
    *,
    config: Mapping[str, Any],
    dry_run: bool,
    allow_main: bool,
    dependency_root: str | Path | None,
    artifact_root: str | Path,
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    """Select dry/main explicitly and validate dependencies before any main write."""

    if dry_run and allow_main:
        raise PostE21ContractError("E24 --dry-run and --allow-main are mutually exclusive")
    if device.type != "cpu":
        raise PostE21ContractError(
            f"E24 deterministic implementation requires CPU; got {device.type!r}"
        )
    if dry_run:
        if dependency_root is not None:
            raise PostE21ContractError("E24 DRY_RUN must not receive or read --dependency-root")
        require_temp_dry_root(artifact_root)
        return "DRY_RUN", dependency_expectation_payload(config)
    if not allow_main:
        raise PostE21ContractError("E24 MAIN requires explicit --allow-main and --dependency-root")
    if dependency_root is None:
        raise PostE21ContractError("E24 --allow-main requires an explicit --dependency-root")
    dependencies = validate_e24_main_dependencies(
        config,
        artifact_root=dependency_root,
    )
    return "MAIN", dependencies
