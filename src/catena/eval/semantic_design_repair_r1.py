"""Prospective statistics for the one-shot E05a-R1 design repair.

This module is intentionally independent of the original E05a evaluator.  The
original four-seed outcomes are design history only and are never accepted as
part of the R1 sampling frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    FACTORIZED,
    ORACLE,
    SHARED,
    SemanticAnchorSeedMetrics,
)

E05A_R1_SEEDS = (1103, 2207, 3301, 4409, 5501, 6607, 7703, 8807)
_ORIGINAL_E05A_SEEDS = frozenset((101, 202, 303, 404))
E05A_R1_BOOTSTRAP_SEEDS = {
    "oracle_affected": 5301,
    "oracle_retention": 5302,
    "factorized_asymmetric_excess": 5303,
    "shared_asymmetric_headroom": 5304,
    "primary_gain": 5305,
    "factorized_preserve_excess": 5306,
    "shared_preserve_excess": 5307,
    "factorized_retention_excess": 5308,
    "shared_retention_excess": 5309,
    "retention_noninferiority": 5310,
    "shuffled_degradation": 5311,
    "wrong_address_degradation": 5312,
    "transaction_only_degradation": 5313,
    "state_only_degradation": 5314,
    "wrong_semantics_degradation": 5315,
}

_OPERATIONS = ("preserve", "add", "invalidate")
_ASYMMETRIC_OPERATIONS = ("add", "invalidate")
_CONTROL_BOOTSTRAP_KEYS = {
    "shuffled_fields": "shuffled_degradation",
    "wrong_address": "wrong_address_degradation",
    "transaction_only": "transaction_only_degradation",
    "state_only": "state_only_degradation",
    "wrong_semantics": "wrong_semantics_degradation",
}


@dataclass(frozen=True, slots=True)
class E05aR1Thresholds:
    """Frozen default thresholds for the prospective R1 design-validity gate."""

    positive_effect_sesoi: float = 0.001
    minimum_oracle_headroom: float = 0.001
    equivalence_margin: float = 0.0005
    retention_noninferiority_margin: float = 0.0005
    oracle_absolute_ceiling: float = 1e-8
    alpha: float = 0.05
    bootstrap_samples: int = 5000
    bootstrap_confidence: float = 0.95


_DEFAULT_THRESHOLDS = E05aR1Thresholds()


@dataclass(frozen=True, slots=True)
class _ValidatedSeed:
    operations: np.ndarray
    affected: Mapping[str, np.ndarray]
    retention: Mapping[str, np.ndarray]


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite")
    return result


def _validate_thresholds(thresholds: E05aR1Thresholds) -> E05aR1Thresholds:
    if not isinstance(thresholds, E05aR1Thresholds):
        raise TypeError("thresholds must be E05aR1Thresholds")
    _finite_positive(thresholds.positive_effect_sesoi, "positive_effect_sesoi")
    _finite_positive(
        thresholds.minimum_oracle_headroom, "minimum_oracle_headroom"
    )
    _finite_positive(thresholds.equivalence_margin, "equivalence_margin")
    _finite_positive(
        thresholds.retention_noninferiority_margin,
        "retention_noninferiority_margin",
    )
    _finite_positive(
        thresholds.oracle_absolute_ceiling, "oracle_absolute_ceiling"
    )
    alpha = _finite_positive(thresholds.alpha, "alpha")
    confidence = _finite_positive(
        thresholds.bootstrap_confidence, "bootstrap_confidence"
    )
    if alpha >= 0.5:
        raise ValueError("alpha must be below 0.5")
    if confidence >= 1.0:
        raise ValueError("bootstrap_confidence must be below 1")
    if (
        isinstance(thresholds.bootstrap_samples, bool)
        or not isinstance(thresholds.bootstrap_samples, int)
        or thresholds.bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    return thresholds


def _validate_fixed_seeds(fixed_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(fixed_seeds, tuple):
        raise TypeError("fixed_seeds must be an exact tuple")
    if len(fixed_seeds) != 8:
        raise ValueError("E05a-R1 requires exactly eight fixed seeds")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in fixed_seeds
    ):
        raise ValueError("fixed_seeds must contain nonnegative integers")
    if len(set(fixed_seeds)) != len(fixed_seeds):
        raise ValueError("fixed_seeds must be unique")
    reused = _ORIGINAL_E05A_SEEDS & set(fixed_seeds)
    if reused:
        raise ValueError(
            "fixed_seeds must not reuse original E05a seeds: "
            f"{sorted(reused)}"
        )
    return fixed_seeds


def _validate_bootstrap_seeds(
    bootstrap_seeds: Mapping[str, int] | None,
) -> dict[str, int]:
    selected = (
        E05A_R1_BOOTSTRAP_SEEDS
        if bootstrap_seeds is None
        else bootstrap_seeds
    )
    if not isinstance(selected, Mapping):
        raise TypeError("bootstrap_seeds must be a mapping")
    if set(selected) != set(E05A_R1_BOOTSTRAP_SEEDS):
        raise ValueError(
            "bootstrap_seeds keys differ from the frozen R1 protocol"
        )
    result: dict[str, int] = {}
    for key in E05A_R1_BOOTSTRAP_SEEDS:
        value = selected[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"bootstrap_seeds.{key} must be a nonnegative integer"
            )
        result[key] = value
    if len(set(result.values())) != len(result):
        raise ValueError("bootstrap_seeds must be unique")
    return result


def _string_vector(
    values: Sequence[str] | np.ndarray,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} rows")
    result = result.astype(str)
    if any(not item for item in result.tolist()):
        raise ValueError(f"{name} must not contain empty values")
    return result


def _metric_mapping(
    values: Mapping[str, Sequence[float] | np.ndarray],
    name: str,
    *,
    required: set[str],
    length: int,
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = required - set(values)
    if missing:
        raise ValueError(f"{name} is missing metrics: {sorted(missing)}")
    result: dict[str, np.ndarray] = {}
    for key, raw in values.items():
        vector = np.asarray(raw, dtype=np.float64)
        if vector.ndim != 1 or len(vector) != length:
            raise ValueError(f"{name}.{key} must contain exactly {length} rows")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name}.{key} contains non-finite values")
        if np.any(vector < 0.0):
            raise ValueError(f"{name}.{key} contains negative MSE")
        result[key] = vector
    return result


def _validate_data(
    data: Mapping[int, SemanticAnchorSeedMetrics],
    fixed_seeds: tuple[int, ...],
) -> dict[int, _ValidatedSeed]:
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in data):
        raise TypeError("data keys must be integer training seeds")
    if set(data) != set(fixed_seeds):
        raise ValueError(
            f"expected fixed seeds {list(fixed_seeds)}, got {sorted(data)}"
        )

    required_affected = {FACTORIZED, SHARED, ORACLE, *CONTROL_NAMES}
    required_retention = {FACTORIZED, SHARED, ORACLE}
    validated: dict[int, _ValidatedSeed] = {}
    all_episode_ids: set[str] = set()
    for seed in fixed_seeds:
        seed_data = data[seed]
        if not isinstance(seed_data, SemanticAnchorSeedMetrics):
            raise TypeError(
                f"data[{seed}] must be SemanticAnchorSeedMetrics"
            )
        episode_ids = _string_vector(
            seed_data.episode_ids, f"{seed}.episode_ids"
        )
        length = len(episode_ids)
        if len(set(episode_ids.tolist())) != length:
            raise ValueError(f"seed {seed} contains duplicate episode IDs")
        overlap = all_episode_ids & set(episode_ids.tolist())
        if overlap:
            raise ValueError("episode IDs must be disjoint across R1 seeds")
        all_episode_ids.update(episode_ids.tolist())
        _string_vector(seed_data.domains, f"{seed}.domains", length=length)
        _string_vector(seed_data.templates, f"{seed}.templates", length=length)
        operations = _string_vector(
            seed_data.operations, f"{seed}.operations", length=length
        )
        observed_operations = set(operations.tolist())
        if observed_operations != set(_OPERATIONS):
            raise ValueError(
                "R1 data must contain exactly PRESERVE, ADD, and INVALIDATE"
            )
        affected = _metric_mapping(
            seed_data.affected,
            f"{seed}.affected",
            required=required_affected,
            length=length,
        )
        retention = _metric_mapping(
            seed_data.retention,
            f"{seed}.retention",
            required=required_retention,
            length=length,
        )
        validated[seed] = _ValidatedSeed(
            operations=operations,
            affected=affected,
            retention=retention,
        )
    return validated


def _operation_equal_mean(
    values: np.ndarray,
    operations: np.ndarray,
    selected_operations: tuple[str, ...],
) -> float:
    means = []
    for operation in selected_operations:
        mask = operations == operation
        if not np.any(mask):
            raise ValueError(f"data lacks operation {operation}")
        means.append(float(values[mask].mean()))
    return float(np.mean(means))


def _seed_values(
    data: Mapping[int, _ValidatedSeed],
    fixed_seeds: tuple[int, ...],
    *,
    metric: str,
    lhs: str,
    rhs: str | None,
    operations: tuple[str, ...],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for seed in fixed_seeds:
        seed_data = data[seed]
        metric_map = (
            seed_data.affected if metric == "affected" else seed_data.retention
        )
        values = metric_map[lhs]
        if rhs is not None:
            values = values - metric_map[rhs]
        result[seed] = _operation_equal_mean(
            values, seed_data.operations, operations
        )
    return result


def _seed_cluster_interval(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    vector = np.asarray([values[seed] for seed in fixed_seeds], dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(vector), size=(samples, len(vector)))
    replicates = vector[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, (tail, 1.0 - tail))
    return {
        "estimate": float(vector.mean()),
        "ci95": [float(low), float(high)],
        "bootstrap_seed": bootstrap_seed,
        "seed_values": {
            str(seed): float(values[seed]) for seed in fixed_seeds
        },
    }


def _upper_gate(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    upper: float,
    thresholds: E05aR1Thresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    result = _seed_cluster_interval(
        values,
        fixed_seeds,
        samples=thresholds.bootstrap_samples,
        confidence=thresholds.bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    supported = bool(result["ci95"][1] <= upper)
    return {**result, "upper": upper, "supported": supported}


def _primary_gate(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    thresholds: E05aR1Thresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    result = _seed_cluster_interval(
        values,
        fixed_seeds,
        samples=thresholds.bootstrap_samples,
        confidence=thresholds.bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    raw = np.asarray([values[seed] for seed in fixed_seeds], dtype=np.float64)
    p_value = exact_sign_flip_test(raw, "greater")
    mean_meets_sesoi = bool(result["estimate"] >= thresholds.positive_effect_sesoi)
    ci_lower_above_zero = bool(result["ci95"][0] > 0.0)
    sign_flip_passed = bool(p_value <= thresholds.alpha)
    all_seed_positive = bool(np.all(raw > 0.0))
    return {
        **result,
        "sesoi": thresholds.positive_effect_sesoi,
        "mean_meets_sesoi": mean_meets_sesoi,
        "ci_lower_above_zero": ci_lower_above_zero,
        "exact_sign_flip": {
            "alternative": "greater",
            "null_shift": 0.0,
            "p": p_value,
            "alpha": thresholds.alpha,
            "passed": sign_flip_passed,
        },
        "all_seed_raw_direction_positive": all_seed_positive,
        "supported": bool(
            mean_meets_sesoi
            and ci_lower_above_zero
            and sign_flip_passed
            and all_seed_positive
        ),
    }


def _headroom_gate(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    thresholds: E05aR1Thresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    result = _seed_cluster_interval(
        values,
        fixed_seeds,
        samples=thresholds.bootstrap_samples,
        confidence=thresholds.bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    raw = np.asarray([values[seed] for seed in fixed_seeds], dtype=np.float64)
    all_seed_above = bool(np.all(raw > thresholds.minimum_oracle_headroom))
    ci_above = bool(result["ci95"][0] > thresholds.minimum_oracle_headroom)
    return {
        **result,
        "minimum": thresholds.minimum_oracle_headroom,
        "all_seed_means_above_minimum": all_seed_above,
        "ci_lower_above_minimum": ci_above,
        "supported": bool(all_seed_above and ci_above),
    }


def _retention_noninferiority_gate(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    thresholds: E05aR1Thresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    result = _seed_cluster_interval(
        values,
        fixed_seeds,
        samples=thresholds.bootstrap_samples,
        confidence=thresholds.bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    raw = np.asarray([values[seed] for seed in fixed_seeds], dtype=np.float64)
    shifted_p = exact_sign_flip_test(
        raw - thresholds.retention_noninferiority_margin, "less"
    )
    ci_passed = bool(
        result["ci95"][1] <= thresholds.retention_noninferiority_margin
    )
    sign_flip_passed = bool(shifted_p <= thresholds.alpha)
    return {
        **result,
        "upper": thresholds.retention_noninferiority_margin,
        "ci_upper_within_margin": ci_passed,
        "shifted_exact_sign_flip": {
            "alternative": "less",
            "null_shift": thresholds.retention_noninferiority_margin,
            "p": shifted_p,
            "alpha": thresholds.alpha,
            "passed": sign_flip_passed,
        },
        "supported": bool(ci_passed and sign_flip_passed),
    }


def _control_gate(
    values: Mapping[int, float],
    fixed_seeds: tuple[int, ...],
    *,
    thresholds: E05aR1Thresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    result = _seed_cluster_interval(
        values,
        fixed_seeds,
        samples=thresholds.bootstrap_samples,
        confidence=thresholds.bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    raw = np.asarray([values[seed] for seed in fixed_seeds], dtype=np.float64)
    shifted_p = exact_sign_flip_test(
        raw - thresholds.positive_effect_sesoi, "greater"
    )
    ci_passed = bool(result["ci95"][0] > thresholds.positive_effect_sesoi)
    sign_flip_passed = bool(shifted_p <= thresholds.alpha)
    all_seed_positive = bool(np.all(raw > 0.0))
    return {
        **result,
        "threshold": thresholds.positive_effect_sesoi,
        "ci_lower_above_threshold": ci_passed,
        "shifted_exact_sign_flip": {
            "alternative": "greater",
            "null_shift": thresholds.positive_effect_sesoi,
            "p": shifted_p,
            "alpha": thresholds.alpha,
            "passed": sign_flip_passed,
        },
        "all_seed_raw_direction_positive": all_seed_positive,
        "supported": bool(
            ci_passed and sign_flip_passed and all_seed_positive
        ),
    }


def _failure_reasons(
    *,
    oracle: Mapping[str, Mapping[str, object]],
    factorized_ai: Mapping[str, object],
    shared_headroom: Mapping[str, object],
    primary: Mapping[str, object],
    preserve: Mapping[str, Mapping[str, object]],
    retention: Mapping[str, Mapping[str, object]],
    controls: Mapping[str, Mapping[str, object]],
) -> list[str]:
    reasons: list[str] = []
    for metric, gate in oracle.items():
        if not gate["supported"]:
            reasons.append(f"ORACLE_{metric.upper()}_ABOVE_CEILING")
    if not factorized_ai["supported"]:
        reasons.append("FACTORIZED_AI_ORACLE_EXCESS_ABOVE_MARGIN")
    if not shared_headroom["supported"]:
        reasons.append("INSUFFICIENT_ORACLE_HEADROOM")
    if not primary["mean_meets_sesoi"]:
        reasons.append("PRIMARY_MEAN_BELOW_SESOI")
    if not primary["ci_lower_above_zero"]:
        reasons.append("PRIMARY_SEED_CI_NOT_ABOVE_ZERO")
    if not primary["exact_sign_flip"]["passed"]:
        reasons.append("PRIMARY_EXACT_SIGN_FLIP_FAILED")
    if not primary["all_seed_raw_direction_positive"]:
        reasons.append("PRIMARY_SEED_DIRECTION_INCONSISTENT")
    for model, gate in preserve.items():
        if not gate["supported"]:
            reasons.append(f"{model.upper()}_PRESERVE_EXCESS_ABOVE_MARGIN")
    for name, gate in retention.items():
        if not gate["supported"]:
            reasons.append(f"{name.upper()}_FAILED")
    for control, gate in controls.items():
        if not gate["supported"]:
            reasons.append(f"CONTROL_{control.upper()}_FAILED")
    return reasons


def evaluate_e05a_r1_design(
    data: Mapping[int, SemanticAnchorSeedMetrics],
    *,
    fixed_seeds: tuple[int, ...] = E05A_R1_SEEDS,
    thresholds: E05aR1Thresholds = _DEFAULT_THRESHOLDS,
    bootstrap_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate the prospective eight-seed E05a-R1 design repair."""

    fixed_seeds = _validate_fixed_seeds(fixed_seeds)
    thresholds = _validate_thresholds(thresholds)
    registered_bootstrap_seeds = _validate_bootstrap_seeds(bootstrap_seeds)
    validated = _validate_data(data, fixed_seeds)

    oracle = {
        metric: _upper_gate(
            _seed_values(
                validated,
                fixed_seeds,
                metric=metric,
                lhs=ORACLE,
                rhs=None,
                operations=_OPERATIONS,
            ),
            fixed_seeds,
            upper=thresholds.oracle_absolute_ceiling,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                f"oracle_{metric}"
            ],
        )
        for metric in ("affected", "retention")
    }
    factorized_ai = _upper_gate(
        _seed_values(
            validated,
            fixed_seeds,
            metric="affected",
            lhs=FACTORIZED,
            rhs=ORACLE,
            operations=_ASYMMETRIC_OPERATIONS,
        ),
        fixed_seeds,
        upper=thresholds.equivalence_margin,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds[
            "factorized_asymmetric_excess"
        ],
    )
    shared_headroom = _headroom_gate(
        _seed_values(
            validated,
            fixed_seeds,
            metric="affected",
            lhs=SHARED,
            rhs=ORACLE,
            operations=_ASYMMETRIC_OPERATIONS,
        ),
        fixed_seeds,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds[
            "shared_asymmetric_headroom"
        ],
    )
    primary = _primary_gate(
        _seed_values(
            validated,
            fixed_seeds,
            metric="affected",
            lhs=SHARED,
            rhs=FACTORIZED,
            operations=_ASYMMETRIC_OPERATIONS,
        ),
        fixed_seeds,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds["primary_gain"],
    )
    preserve = {
        model: _upper_gate(
            _seed_values(
                validated,
                fixed_seeds,
                metric="affected",
                lhs=model,
                rhs=ORACLE,
                operations=("preserve",),
            ),
            fixed_seeds,
            upper=thresholds.equivalence_margin,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                f"{model}_preserve_excess"
            ],
        )
        for model in (FACTORIZED, SHARED)
    }
    retention = {
        f"{model}_oracle_excess": _upper_gate(
            _seed_values(
                validated,
                fixed_seeds,
                metric="retention",
                lhs=model,
                rhs=ORACLE,
                operations=_OPERATIONS,
            ),
            fixed_seeds,
            upper=thresholds.equivalence_margin,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                f"{model}_retention_excess"
            ],
        )
        for model in (FACTORIZED, SHARED)
    }
    retention["factorized_shared_noninferiority"] = (
        _retention_noninferiority_gate(
            _seed_values(
                validated,
                fixed_seeds,
                metric="retention",
                lhs=FACTORIZED,
                rhs=SHARED,
                operations=_OPERATIONS,
            ),
            fixed_seeds,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                "retention_noninferiority"
            ],
        )
    )
    controls = {
        control: _control_gate(
            _seed_values(
                validated,
                fixed_seeds,
                metric="affected",
                lhs=control,
                rhs=FACTORIZED,
                operations=_ASYMMETRIC_OPERATIONS,
            ),
            fixed_seeds,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                _CONTROL_BOOTSTRAP_KEYS[control]
            ],
        )
        for control in CONTROL_NAMES
    }

    reasons = _failure_reasons(
        oracle=oracle,
        factorized_ai=factorized_ai,
        shared_headroom=shared_headroom,
        primary=primary,
        preserve=preserve,
        retention=retention,
        controls=controls,
    )
    go = not reasons
    return {
        "status": "GO" if go else "NO_GO",
        "go": go,
        "claim_evidence": False,
        "h5_claim_open": False,
        "h5_disposition": (
            "DESIGN_VALIDITY_GO_PENDING_HUMAN_AUDIT"
            if go
            else "TERMINATED_NOT_REFUTED"
        ),
        "e05b_execution_allowed": False,
        "diagnostic_reasons": reasons,
        "fixed_seeds": list(fixed_seeds),
        "original_e05a_outcomes_used_in_inference": False,
        "operation_weighting": {
            "primary_and_controls": ["add", "invalidate"],
            "primary_and_controls_equal_weight": True,
            "preserve_role": "NO_OP_GUARDRAIL_ONLY",
            "retention": ["preserve", "add", "invalidate"],
            "retention_equal_weight": True,
        },
        "bootstrap": {
            "method": "paired_training_seed_cluster_percentile",
            "unit": "training_seed_and_fresh_namespace_cluster",
            "two_sided": True,
            "seed_clusters_resampled": True,
            "episodes_resampled": False,
            "paired_conditions_within_cluster": True,
            "draws_per_replicate": 8,
            "samples": thresholds.bootstrap_samples,
            "confidence": thresholds.bootstrap_confidence,
            "seeds": registered_bootstrap_seeds,
        },
        "oracle_low": oracle,
        "factorized_ai_oracle_excess": factorized_ai,
        "shared_ai_oracle_headroom": shared_headroom,
        "primary_gain_shared_minus_factorized": primary,
        "preserve_no_op": preserve,
        "retention": retention,
        "controls": controls,
    }


__all__ = [
    "E05A_R1_BOOTSTRAP_SEEDS",
    "E05A_R1_SEEDS",
    "E05aR1Thresholds",
    "evaluate_e05a_r1_design",
]
