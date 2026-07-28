from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from catena.eval.seed_inference import exact_sign_flip_test
from catena.eval.statistics import Interval
from catena.eval.statistics_v61 import fixed_seed_operation_stratified_bootstrap

E05A_SEEDS = (101, 202, 303, 404)
E05B_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)

FACTORIZED = "factorized"
SHARED = "shared"
ORACLE = "oracle_demand"
CONTROL_NAMES = (
    "shuffled_fields",
    "wrong_address",
    "transaction_only",
    "state_only",
    "wrong_semantics",
)
Stratification = Literal["domain_template_operation", "domain_template"]
_CONTROL_BOOTSTRAP_KEYS = {
    "shuffled_fields": "shuffled_degradation",
    "wrong_address": "wrong_address_degradation",
    "transaction_only": "transaction_only_degradation",
    "state_only": "state_only_degradation",
    "wrong_semantics": "wrong_semantics_degradation",
}
E05A_BOOTSTRAP_SEEDS = {
    "oracle_affected": 5101,
    "oracle_retention": 5102,
    "factorized_excess_affected": 5103,
    "factorized_excess_retention": 5104,
    "shared_excess_affected": 5105,
    "shared_excess_retention": 5106,
    "seen_model_parity": 5107,
    "shuffled_degradation": 5108,
    "wrong_address_degradation": 5109,
    "transaction_only_degradation": 5110,
    "state_only_degradation": 5111,
    "wrong_semantics_degradation": 5112,
}
E05B_BOOTSTRAP_SEEDS = {
    "validation_parity": 5201,
    "oracle_affected": 5202,
    "oracle_retention": 5203,
    "shared_headroom": 5204,
    "primary_raw_improvement": 5205,
    "headroom_closure": 5206,
    "retention_noninferiority": 5207,
    "absolute_factorized_retention": 5208,
    "shuffled_degradation": 5209,
    "wrong_address_degradation": 5210,
    "transaction_only_degradation": 5211,
    "state_only_degradation": 5212,
    "wrong_semantics_degradation": 5213,
}


@dataclass(frozen=True, slots=True)
class SemanticAnchorThresholds:
    """Prospectively fixed statistical thresholds for the H5-lite anchor."""

    positive_effect_sesoi: float = 0.001
    minimum_oracle_headroom: float = 0.001
    headroom_fraction_sesoi: float = 0.10
    equivalence_margin: float = 0.0005
    retention_noninferiority_margin: float = 0.0005
    oracle_absolute_ceiling: float = 1e-8
    alpha: float = 0.05
    bootstrap_samples: int = 5000
    bootstrap_confidence: float = 0.95


_DEFAULT_THRESHOLDS = SemanticAnchorThresholds()


@dataclass(frozen=True, slots=True)
class SemanticAnchorSeedMetrics:
    """Episode-aligned behavioral metrics for one fixed training seed.

    Every model and intervention must be evaluated on the same ``episode_ids``.
    No filtering is performed by this module: a non-finite or misaligned input
    is rejected instead.
    """

    episode_ids: Sequence[str] | np.ndarray
    domains: Sequence[str] | np.ndarray
    templates: Sequence[str] | np.ndarray
    operations: Sequence[str] | np.ndarray
    affected: Mapping[str, Sequence[float] | np.ndarray]
    retention: Mapping[str, Sequence[float] | np.ndarray]


@dataclass(frozen=True, slots=True)
class _ValidatedSeedMetrics:
    episode_ids: np.ndarray
    strata: np.ndarray
    operations: np.ndarray
    affected: Mapping[str, np.ndarray]
    retention: Mapping[str, np.ndarray]


class OracleHeadroomError(RuntimeError):
    """The registered oracle-headroom denominator is not identifiable."""


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_thresholds(
    thresholds: SemanticAnchorThresholds,
) -> SemanticAnchorThresholds:
    positive = _finite_scalar(
        thresholds.positive_effect_sesoi, "positive_effect_sesoi"
    )
    headroom = _finite_scalar(
        thresholds.minimum_oracle_headroom, "minimum_oracle_headroom"
    )
    fraction = _finite_scalar(
        thresholds.headroom_fraction_sesoi, "headroom_fraction_sesoi"
    )
    equivalence = _finite_scalar(
        thresholds.equivalence_margin, "equivalence_margin"
    )
    retention = _finite_scalar(
        thresholds.retention_noninferiority_margin,
        "retention_noninferiority_margin",
    )
    oracle = _finite_scalar(
        thresholds.oracle_absolute_ceiling, "oracle_absolute_ceiling"
    )
    alpha = _finite_scalar(thresholds.alpha, "alpha")
    confidence = _finite_scalar(
        thresholds.bootstrap_confidence, "bootstrap_confidence"
    )
    if min(positive, headroom, fraction, equivalence, retention, oracle) <= 0.0:
        raise ValueError("All registered thresholds must be strictly positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and 0.5")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie strictly between zero and one")
    if (
        isinstance(thresholds.bootstrap_samples, bool)
        or not isinstance(thresholds.bootstrap_samples, int)
        or thresholds.bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    return thresholds


def _validate_bootstrap_seeds(
    values: Mapping[str, int] | None,
    expected: Mapping[str, int],
) -> dict[str, int]:
    selected = expected if values is None else values
    if not isinstance(selected, Mapping):
        raise TypeError("bootstrap_seeds must be a mapping")
    if set(selected) != set(expected):
        raise ValueError(
            "bootstrap_seeds keys differ from the frozen protocol: "
            f"expected {sorted(expected)}, got {sorted(selected)}"
        )
    result: dict[str, int] = {}
    for name in expected:
        value = selected[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"bootstrap_seeds.{name} must be a nonnegative integer")
        result[name] = value
    return result


def _string_vector(
    values: Sequence[str] | np.ndarray, name: str, expected_length: int | None = None
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(f"{name} has {len(array)} rows; expected {expected_length}")
    result = array.astype(str)
    if any(not value for value in result.tolist()):
        raise ValueError(f"{name} must not contain empty labels")
    return result


def _finite_vector(
    values: Sequence[float] | np.ndarray, name: str, expected_length: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(
            f"{name} must be a one-dimensional vector of length {expected_length}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_metric_mapping(
    values: Mapping[str, Sequence[float] | np.ndarray],
    name: str,
    expected_length: int,
    required: set[str],
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = required - set(values)
    if missing:
        raise ValueError(f"{name} is missing required metrics: {sorted(missing)}")
    result: dict[str, np.ndarray] = {}
    for key, vector in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings")
        metric = _finite_vector(vector, f"{name}.{key}", expected_length)
        if np.any(metric < 0.0):
            raise ValueError(f"{name}.{key} contains a negative MSE")
        result[key] = metric
    return result


def _validate_dataset(
    data: Mapping[int, SemanticAnchorSeedMetrics],
    *,
    expected_seeds: tuple[int, ...],
    required_affected: set[str],
    required_retention: set[str],
    stratification: Stratification,
) -> dict[int, _ValidatedSeedMetrics]:
    if stratification not in {"domain_template_operation", "domain_template"}:
        raise ValueError(f"Unknown semantic bootstrap stratification: {stratification}.")
    include_operation = stratification == "domain_template_operation"
    if not isinstance(data, Mapping):
        raise TypeError("seed metrics must be a mapping")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in data):
        raise TypeError("seed metric keys must be integers")
    observed_seeds = tuple(sorted(data))
    if observed_seeds != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"expected fixed seeds {list(expected_seeds)}, got {list(observed_seeds)}"
        )

    validated: dict[int, _ValidatedSeedMetrics] = {}
    reference_counts: dict[str, int] | None = None
    reference_length: int | None = None
    separator = "\x1f"
    for seed in expected_seeds:
        seed_data = data[seed]
        if not isinstance(seed_data, SemanticAnchorSeedMetrics):
            raise TypeError(
                f"seed metrics {seed} must be SemanticAnchorSeedMetrics"
            )
        episode_ids = _string_vector(seed_data.episode_ids, f"{seed}.episode_ids")
        length = len(episode_ids)
        if len(set(episode_ids.tolist())) != length:
            raise ValueError(f"seed {seed} contains duplicate episode_ids")
        domains = _string_vector(seed_data.domains, f"{seed}.domains", length)
        templates = _string_vector(seed_data.templates, f"{seed}.templates", length)
        operations = _string_vector(
            seed_data.operations, f"{seed}.operations", length
        )
        if any(
            separator in value
            for vector in (domains, templates, operations)
            for value in vector.tolist()
        ):
            raise ValueError("stratum labels contain the reserved separator")
        stratum_vectors = (
            (domains, templates, operations)
            if include_operation
            else (domains, templates)
        )
        strata = np.asarray(
            [
                separator.join(parts)
                for parts in zip(*stratum_vectors, strict=True)
            ],
            dtype=str,
        )
        labels, counts = np.unique(strata, return_counts=True)
        if len(set(counts.tolist())) != 1:
            stratum_name = (
                "domain×template×operation"
                if include_operation
                else "domain×template"
            )
            raise ValueError(
                f"seed {seed} {stratum_name} strata are not balanced"
            )
        count_map = {
            str(label): int(count)
            for label, count in zip(labels, counts, strict=True)
        }
        if reference_counts is None:
            reference_counts = count_map
            reference_length = length
        elif count_map != reference_counts or length != reference_length:
            raise ValueError(
                "Every fixed seed must have identical balanced stratum counts"
            )

        affected = _validated_metric_mapping(
            seed_data.affected,
            f"{seed}.affected",
            length,
            required_affected,
        )
        retention = _validated_metric_mapping(
            seed_data.retention,
            f"{seed}.retention",
            length,
            required_retention,
        )
        validated[seed] = _ValidatedSeedMetrics(
            episode_ids=episode_ids,
            strata=strata,
            operations=operations,
            affected=affected,
            retention=retention,
        )
    return validated


def _operation_subset(
    data: Mapping[int, _ValidatedSeedMetrics],
    operations: tuple[str, ...],
) -> dict[int, _ValidatedSeedMetrics]:
    allowed = set(operations)
    result: dict[int, _ValidatedSeedMetrics] = {}
    for seed, seed_data in data.items():
        observed = set(seed_data.operations.tolist())
        if not allowed <= observed:
            raise ValueError(
                f"seed {seed} lacks registered control operations "
                f"{sorted(allowed - observed)}"
            )
        mask = np.isin(seed_data.operations, list(operations))
        result[seed] = _ValidatedSeedMetrics(
            episode_ids=seed_data.episode_ids[mask],
            strata=seed_data.strata[mask],
            operations=seed_data.operations[mask],
            affected={
                name: vector[mask] for name, vector in seed_data.affected.items()
            },
            retention={
                name: vector[mask] for name, vector in seed_data.retention.items()
            },
        )
    return result


def exact_seed_tost(
    values: Sequence[float] | np.ndarray,
    *,
    margin: float,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Exact paired-seed TOST using sign-flip randomization."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
        raise ValueError("values must be a nonempty finite one-dimensional vector")
    margin_value = _finite_scalar(margin, "margin")
    alpha_value = _finite_scalar(alpha, "alpha")
    if margin_value <= 0.0:
        raise ValueError("margin must be strictly positive")
    if not 0.0 < alpha_value < 0.5:
        raise ValueError("alpha must lie strictly between zero and 0.5")
    lower_p = exact_sign_flip_test(vector + margin_value, "greater")
    upper_p = exact_sign_flip_test(vector - margin_value, "less")
    return {
        "lower_p": lower_p,
        "upper_p": upper_p,
        "alpha": alpha_value,
        "passed": bool(lower_p <= alpha_value and upper_p <= alpha_value),
    }


def _strata_by_seed(
    data: Mapping[int, _ValidatedSeedMetrics],
) -> dict[int, np.ndarray]:
    return {seed: seed_data.strata for seed, seed_data in data.items()}


def _seed_means(vectors: Mapping[int, np.ndarray]) -> dict[int, float]:
    return {seed: float(vectors[seed].mean()) for seed in sorted(vectors)}


def _paired_interval(
    data: Mapping[int, _ValidatedSeedMetrics],
    vectors: Mapping[int, np.ndarray],
    *,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
) -> Interval:
    def statistic(indices: Mapping[int, np.ndarray]) -> float:
        return float(
            np.mean(
                [
                    vectors[seed][indices[seed]].mean()
                    for seed in sorted(vectors)
                ]
            )
        )

    return fixed_seed_operation_stratified_bootstrap(
        _strata_by_seed(data),
        statistic,
        samples=thresholds.bootstrap_samples,
        seed=bootstrap_seed,
        confidence=thresholds.bootstrap_confidence,
    )


def _ratio_interval(
    data: Mapping[int, _ValidatedSeedMetrics],
    numerator: Mapping[int, np.ndarray],
    denominator: Mapping[int, np.ndarray],
    *,
    minimum_headroom: float,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
) -> Interval:
    def statistic(indices: Mapping[int, np.ndarray]) -> float:
        ratios: list[float] = []
        for seed in sorted(numerator):
            selected = indices[seed]
            denominator_mean = float(denominator[seed][selected].mean())
            if denominator_mean <= minimum_headroom:
                raise OracleHeadroomError(
                    f"seed {seed} oracle headroom {denominator_mean} is not "
                    f"above {minimum_headroom}"
                )
            ratios.append(
                float(numerator[seed][selected].mean()) / denominator_mean
            )
        return float(np.mean(ratios))

    return fixed_seed_operation_stratified_bootstrap(
        _strata_by_seed(data),
        statistic,
        samples=thresholds.bootstrap_samples,
        seed=bootstrap_seed,
        confidence=thresholds.bootstrap_confidence,
    )


def _interval_payload(interval: Interval) -> dict[str, object]:
    return {
        "estimate": interval.estimate,
        "ci95": [interval.low, interval.high],
    }


def _descriptive_interval_payload(
    data: Mapping[int, _ValidatedSeedMetrics],
    vectors: Mapping[int, np.ndarray],
    *,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
) -> dict[str, object]:
    interval = _paired_interval(
        data,
        vectors,
        thresholds=thresholds,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        **_interval_payload(interval),
        "bootstrap_seed": bootstrap_seed,
        "seed_values": {
            str(seed): value for seed, value in _seed_means(vectors).items()
        },
    }


def _positive_gate(
    data: Mapping[int, _ValidatedSeedMetrics],
    vectors: Mapping[int, np.ndarray],
    *,
    threshold: float,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
    require_seed_test: bool,
    require_all_seed_positive: bool,
) -> dict[str, object]:
    interval = _paired_interval(
        data,
        vectors,
        thresholds=thresholds,
        bootstrap_seed=bootstrap_seed,
    )
    seed_means = _seed_means(vectors)
    raw = np.asarray(list(seed_means.values()), dtype=np.float64)
    p_value = (
        exact_sign_flip_test(raw - threshold, "greater")
        if require_seed_test
        else None
    )
    seed_direction = bool(np.all(raw > 0.0))
    supported = bool(
        interval.low > threshold
        and (not require_all_seed_positive or seed_direction)
        and (p_value is None or p_value <= thresholds.alpha)
    )
    return {
        **_interval_payload(interval),
        "threshold": threshold,
        "bootstrap_seed": bootstrap_seed,
        "seed_values": {str(seed): value for seed, value in seed_means.items()},
        "all_seed_raw_direction_positive": seed_direction,
        "shifted_exact_sign_flip_p": p_value,
        "supported": supported,
    }


def _upper_gate(
    data: Mapping[int, _ValidatedSeedMetrics],
    vectors: Mapping[int, np.ndarray],
    *,
    upper: float,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
    require_seed_test: bool,
) -> dict[str, object]:
    interval = _paired_interval(
        data,
        vectors,
        thresholds=thresholds,
        bootstrap_seed=bootstrap_seed,
    )
    seed_means = _seed_means(vectors)
    raw = np.asarray(list(seed_means.values()), dtype=np.float64)
    p_value = (
        exact_sign_flip_test(raw - upper, "less")
        if require_seed_test
        else None
    )
    all_seed_within = bool(np.all(raw <= upper))
    return {
        **_interval_payload(interval),
        "upper": upper,
        "bootstrap_seed": bootstrap_seed,
        "seed_values": {str(seed): value for seed, value in seed_means.items()},
        "all_seed_means_within_upper": all_seed_within,
        "shifted_exact_sign_flip_p": p_value,
        "supported": bool(
            interval.high <= upper
            and (p_value is None or p_value <= thresholds.alpha)
        ),
    }


def _equivalence_gate(
    data: Mapping[int, _ValidatedSeedMetrics],
    vectors: Mapping[int, np.ndarray],
    *,
    margin: float,
    thresholds: SemanticAnchorThresholds,
    bootstrap_seed: int,
    require_exact_tost: bool,
) -> dict[str, object]:
    interval = _paired_interval(
        data,
        vectors,
        thresholds=thresholds,
        bootstrap_seed=bootstrap_seed,
    )
    seed_means = _seed_means(vectors)
    raw = np.asarray(list(seed_means.values()), dtype=np.float64)
    tost = exact_seed_tost(raw, margin=margin, alpha=thresholds.alpha)
    ci_within = bool(interval.low >= -margin and interval.high <= margin)
    all_seed_within = bool(np.all(np.abs(raw) <= margin))
    return {
        **_interval_payload(interval),
        "margin": margin,
        "bootstrap_seed": bootstrap_seed,
        "seed_values": {str(seed): value for seed, value in seed_means.items()},
        "ci_within_margin": ci_within,
        "all_seed_means_within_margin": all_seed_within,
        "exact_seed_tost": tost,
        "exact_tost_required": require_exact_tost,
        "supported": bool(
            ci_within
            and (not require_exact_tost or bool(tost["passed"]))
        ),
    }


def _metric_vectors(
    data: Mapping[int, _ValidatedSeedMetrics],
    *,
    metric: str,
    lhs: str,
    rhs: str | None = None,
) -> dict[int, np.ndarray]:
    if metric not in {"affected", "retention"}:
        raise ValueError("metric must be 'affected' or 'retention'")
    result: dict[int, np.ndarray] = {}
    for seed, seed_data in data.items():
        mapping = seed_data.affected if metric == "affected" else seed_data.retention
        result[seed] = (
            mapping[lhs].copy()
            if rhs is None
            else np.asarray(mapping[lhs] - mapping[rhs], dtype=np.float64)
        )
    return result


def _row_counts(
    data: Mapping[int, _ValidatedSeedMetrics],
) -> dict[str, int]:
    return {
        str(seed): int(len(seed_data.episode_ids))
        for seed, seed_data in sorted(data.items())
    }


def evaluate_e05a_go(
    data: Mapping[int, SemanticAnchorSeedMetrics],
    *,
    thresholds: SemanticAnchorThresholds = _DEFAULT_THRESHOLDS,
    bootstrap_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate the four-seed, seen-operation E05a design preflight."""

    thresholds = _validate_thresholds(thresholds)
    registered_bootstrap_seeds = _validate_bootstrap_seeds(
        bootstrap_seeds, E05A_BOOTSTRAP_SEEDS
    )
    required_models = {FACTORIZED, SHARED, ORACLE, *CONTROL_NAMES}
    validated = _validate_dataset(
        data,
        expected_seeds=E05A_SEEDS,
        required_affected=required_models,
        required_retention={FACTORIZED, SHARED, ORACLE},
        stratification="domain_template_operation",
    )
    control_data = _operation_subset(validated, ("add", "invalidate"))

    oracle = {
        "affected": _upper_gate(
            validated,
            _metric_vectors(validated, metric="affected", lhs=ORACLE),
            upper=thresholds.oracle_absolute_ceiling,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_affected"],
            require_seed_test=False,
        ),
        "retention": _upper_gate(
            validated,
            _metric_vectors(validated, metric="retention", lhs=ORACLE),
            upper=thresholds.oracle_absolute_ceiling,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_retention"],
            require_seed_test=False,
        ),
    }
    excess: dict[str, object] = {}
    for model in (FACTORIZED, SHARED):
        for metric in ("affected", "retention"):
            excess[f"{model}_{metric}"] = _upper_gate(
                validated,
                _metric_vectors(
                    validated, metric=metric, lhs=model, rhs=ORACLE
                ),
                upper=thresholds.equivalence_margin,
                thresholds=thresholds,
                bootstrap_seed=registered_bootstrap_seeds[
                    f"{model}_excess_{metric}"
                ],
                require_seed_test=False,
            )
    parity = {
        metric: _equivalence_gate(
            validated,
            _metric_vectors(
                validated, metric=metric, lhs=SHARED, rhs=FACTORIZED
            ),
            margin=thresholds.equivalence_margin,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["seen_model_parity"],
            require_exact_tost=False,
        )
        for metric in ("affected", "retention")
    }
    controls = {
        control: _positive_gate(
            control_data,
            _metric_vectors(
                control_data, metric="affected", lhs=control, rhs=FACTORIZED
            ),
            threshold=thresholds.positive_effect_sesoi,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                _CONTROL_BOOTSTRAP_KEYS[control]
            ],
            require_seed_test=False,
            require_all_seed_positive=True,
        )
        for control in CONTROL_NAMES
    }
    conjunction = bool(
        all(bool(gate["supported"]) for gate in oracle.values())
        and all(
            bool(gate["supported"])
            for gate in excess.values()
            if isinstance(gate, Mapping)
        )
        and bool(parity["affected"]["supported"])
        and all(bool(gate["supported"]) for gate in controls.values())
    )
    return {
        "status": "GO" if conjunction else "NO_GO",
        "claim_evidence": False,
        "fixed_seeds": list(E05A_SEEDS),
        "row_counts": _row_counts(validated),
        "bootstrap": {
            "unit": "episode_within_seed_domain_template_operation_stratum",
            "fixed_seeds_not_resampled": True,
            "paired_indices_across_models_and_controls": True,
            "samples": thresholds.bootstrap_samples,
            "confidence": thresholds.bootstrap_confidence,
            "seeds": registered_bootstrap_seeds,
        },
        "oracle_low": oracle,
        "oracle_excess": excess,
        "factorized_shared_parity": parity,
        "retention_parity_is_descriptive_only": True,
        "control_estimand": {
            "operations": ["add", "invalidate"],
            "operation_weighting": "equal",
            "preserve_in_control_conjunction": False,
            "row_counts": _row_counts(control_data),
        },
        "controls": controls,
        "go": conjunction,
    }


def evaluate_e05b_main(
    *,
    validation: Mapping[int, SemanticAnchorSeedMetrics],
    primary: Mapping[int, SemanticAnchorSeedMetrics],
    thresholds: SemanticAnchorThresholds = _DEFAULT_THRESHOLDS,
    bootstrap_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate the eight-seed held-out-SUPERSEDE confirmatory anchor."""

    validation_report = evaluate_e05b_validation(
        validation,
        thresholds=thresholds,
        bootstrap_seeds=bootstrap_seeds,
    )
    thresholds = _validate_thresholds(thresholds)
    registered_bootstrap_seeds = _validate_bootstrap_seeds(
        bootstrap_seeds, E05B_BOOTSTRAP_SEEDS
    )
    required_primary = {FACTORIZED, SHARED, ORACLE, *CONTROL_NAMES}
    primary_data = _validate_dataset(
        primary,
        expected_seeds=E05B_SEEDS,
        required_affected=required_primary,
        required_retention={FACTORIZED, SHARED, ORACLE},
        stratification="domain_template",
    )

    validation_gate = validation_report[
        "validation_factorized_shared_equivalence"
    ]

    oracle_gates = {
        "affected": _upper_gate(
            primary_data,
            _metric_vectors(primary_data, metric="affected", lhs=ORACLE),
            upper=thresholds.oracle_absolute_ceiling,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_affected"],
            require_seed_test=False,
        ),
        "retention": _upper_gate(
            primary_data,
            _metric_vectors(primary_data, metric="retention", lhs=ORACLE),
            upper=thresholds.oracle_absolute_ceiling,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_retention"],
            require_seed_test=False,
        ),
    }

    d_vectors = _metric_vectors(
        primary_data, metric="affected", lhs=SHARED, rhs=FACTORIZED
    )
    h_vectors = _metric_vectors(
        primary_data, metric="affected", lhs=SHARED, rhs=ORACLE
    )
    r_vectors = _metric_vectors(
        primary_data, metric="retention", lhs=FACTORIZED, rhs=SHARED
    )
    d_gate = _positive_gate(
        primary_data,
        d_vectors,
        threshold=thresholds.positive_effect_sesoi,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds["primary_raw_improvement"],
        require_seed_test=True,
        require_all_seed_positive=True,
    )
    h_gate = _positive_gate(
        primary_data,
        h_vectors,
        threshold=thresholds.minimum_oracle_headroom,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds["shared_headroom"],
        require_seed_test=True,
        require_all_seed_positive=False,
    )
    r_gate = _upper_gate(
        primary_data,
        r_vectors,
        upper=thresholds.retention_noninferiority_margin,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds["retention_noninferiority"],
        require_seed_test=True,
    )
    absolute_retention_gate = _upper_gate(
        primary_data,
        _metric_vectors(
            primary_data, metric="retention", lhs=FACTORIZED, rhs=ORACLE
        ),
        upper=thresholds.retention_noninferiority_margin,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds[
            "absolute_factorized_retention"
        ],
        require_seed_test=False,
    )
    controls = {
        control: _positive_gate(
            primary_data,
            _metric_vectors(
                primary_data, metric="affected", lhs=control, rhs=FACTORIZED
            ),
            threshold=thresholds.positive_effect_sesoi,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                _CONTROL_BOOTSTRAP_KEYS[control]
            ],
            require_seed_test=True,
            require_all_seed_positive=True,
        )
        for control in CONTROL_NAMES
    }

    h_seed_means = _seed_means(h_vectors)
    point_headroom_valid = all(
        value > thresholds.minimum_oracle_headroom
        for value in h_seed_means.values()
    )
    q_gate: dict[str, object]
    bootstrap_headroom_valid = False
    if not point_headroom_valid:
        q_gate = {
            "supported": False,
            "evaluable": False,
            "reason": "POINT_SEED_HEADROOM_AT_OR_BELOW_MINIMUM",
            "minimum_oracle_headroom": thresholds.minimum_oracle_headroom,
            "headroom_seed_values": {
                str(seed): value for seed, value in h_seed_means.items()
            },
        }
    else:
        try:
            q_interval = _ratio_interval(
                primary_data,
                d_vectors,
                h_vectors,
                minimum_headroom=thresholds.minimum_oracle_headroom,
                thresholds=thresholds,
                bootstrap_seed=registered_bootstrap_seeds["headroom_closure"],
            )
        except OracleHeadroomError as error:
            q_gate = {
                "supported": False,
                "evaluable": False,
                "reason": "BOOTSTRAP_SEED_HEADROOM_AT_OR_BELOW_MINIMUM",
                "detail": str(error),
                "minimum_oracle_headroom": thresholds.minimum_oracle_headroom,
                "headroom_seed_values": {
                    str(seed): value for seed, value in h_seed_means.items()
                },
            }
        else:
            bootstrap_headroom_valid = True
            q_seed_values = {
                seed: float(d_vectors[seed].mean()) / h_seed_means[seed]
                for seed in E05B_SEEDS
            }
            q_raw = np.asarray(list(q_seed_values.values()), dtype=np.float64)
            q_p = exact_sign_flip_test(
                q_raw - thresholds.headroom_fraction_sesoi, "greater"
            )
            q_gate = {
                **_interval_payload(q_interval),
                "evaluable": True,
                "threshold": thresholds.headroom_fraction_sesoi,
                "bootstrap_seed": registered_bootstrap_seeds[
                    "headroom_closure"
                ],
                "seed_values": {
                    str(seed): value for seed, value in q_seed_values.items()
                },
                "shifted_exact_sign_flip_p": q_p,
                "all_seed_raw_direction_positive": bool(np.all(q_raw > 0.0)),
                "supported": bool(
                    q_interval.low > thresholds.headroom_fraction_sesoi
                    and q_p <= thresholds.alpha
                ),
            }

    headroom_identifiable = bool(point_headroom_valid and bootstrap_headroom_valid)
    all_nonratio = bool(
        validation_gate["supported"]
        and all(bool(gate["supported"]) for gate in oracle_gates.values())
        and d_gate["supported"]
        and h_gate["supported"]
        and r_gate["supported"]
        and absolute_retention_gate["supported"]
        and all(bool(gate["supported"]) for gate in controls.values())
    )
    supported = bool(
        headroom_identifiable and all_nonratio and bool(q_gate["supported"])
    )
    if not headroom_identifiable:
        status = "INCONCLUSIVE_ORACLE_HEADROOM"
    elif supported:
        status = "SUPPORTED"
    else:
        status = "NOT_SUPPORTED"

    return {
        "status": status,
        "fixed_seeds": list(E05B_SEEDS),
        "row_counts": {
            "validation": validation_report["row_counts"],
            "primary": _row_counts(primary_data),
        },
        "no_outcome_based_exclusions": True,
        "bootstrap": {
            "unit": "episode_within_seed_domain_template_stratum",
            "fixed_seeds_not_resampled": True,
            "paired_indices_across_models_and_controls": True,
            "samples": thresholds.bootstrap_samples,
            "confidence": thresholds.bootstrap_confidence,
            "seeds": registered_bootstrap_seeds,
        },
        "validation_factorized_shared_equivalence": validation_gate,
        "oracle_low": oracle_gates,
        "D_shared_minus_factorized": d_gate,
        "H_shared_minus_oracle": h_gate,
        "Q_headroom_fraction_closed": q_gate,
        "R_factorized_minus_shared_retention": r_gate,
        "factorized_absolute_retention": absolute_retention_gate,
        "controls": controls,
        "headroom_identifiable": headroom_identifiable,
        "supported": supported,
    }


def evaluate_e05b_validation(
    validation: Mapping[int, SemanticAnchorSeedMetrics],
    *,
    thresholds: SemanticAnchorThresholds = _DEFAULT_THRESHOLDS,
    bootstrap_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate the sealed E05b validation without receiving primary data."""

    thresholds = _validate_thresholds(thresholds)
    registered_bootstrap_seeds = _validate_bootstrap_seeds(
        bootstrap_seeds, E05B_BOOTSTRAP_SEEDS
    )
    validation_data = _validate_dataset(
        validation,
        expected_seeds=E05B_SEEDS,
        required_affected={FACTORIZED, SHARED, ORACLE},
        required_retention={FACTORIZED, SHARED, ORACLE},
        stratification="domain_template",
    )
    validation_gate = _equivalence_gate(
        validation_data,
        _metric_vectors(
            validation_data, metric="affected", lhs=SHARED, rhs=FACTORIZED
        ),
        margin=thresholds.equivalence_margin,
        thresholds=thresholds,
        bootstrap_seed=registered_bootstrap_seeds["validation_parity"],
        require_exact_tost=True,
    )
    passed = bool(validation_gate["supported"])
    return {
        "status": "PASS" if passed else "NO_GO_MAIN_SEALED",
        "passed": passed,
        "fixed_seeds": list(E05B_SEEDS),
        "row_counts": _row_counts(validation_data),
        "primary_data_received": False,
        "bootstrap": {
            "unit": "episode_within_seed_domain_template_stratum",
            "fixed_seeds_not_resampled": True,
            "paired_indices_across_models": True,
            "samples": thresholds.bootstrap_samples,
            "confidence": thresholds.bootstrap_confidence,
            "seed": registered_bootstrap_seeds["validation_parity"],
        },
        "validation_factorized_shared_equivalence": validation_gate,
    }


def evaluate_e05b_secondary(
    data: Mapping[int, SemanticAnchorSeedMetrics],
    *,
    thresholds: SemanticAnchorThresholds = _DEFAULT_THRESHOLDS,
    bootstrap_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Report frozen E05b estimands on a non-confirmatory secondary split."""

    thresholds = _validate_thresholds(thresholds)
    registered_bootstrap_seeds = _validate_bootstrap_seeds(
        bootstrap_seeds, E05B_BOOTSTRAP_SEEDS
    )
    validated = _validate_dataset(
        data,
        expected_seeds=E05B_SEEDS,
        required_affected={FACTORIZED, SHARED, ORACLE},
        required_retention={FACTORIZED, SHARED, ORACLE},
        stratification="domain_template",
    )
    d_vectors = _metric_vectors(
        validated, metric="affected", lhs=SHARED, rhs=FACTORIZED
    )
    h_vectors = _metric_vectors(
        validated, metric="affected", lhs=SHARED, rhs=ORACLE
    )
    r_vectors = _metric_vectors(
        validated, metric="retention", lhs=FACTORIZED, rhs=SHARED
    )
    h_seed_means = _seed_means(h_vectors)
    point_headroom_valid = all(
        value > thresholds.minimum_oracle_headroom
        for value in h_seed_means.values()
    )
    if not point_headroom_valid:
        q_payload: dict[str, object] = {
            "evaluable": False,
            "reason": "POINT_SEED_HEADROOM_AT_OR_BELOW_MINIMUM",
            "minimum_oracle_headroom": thresholds.minimum_oracle_headroom,
            "headroom_seed_values": {
                str(seed): value for seed, value in h_seed_means.items()
            },
            "bootstrap_seed": registered_bootstrap_seeds["headroom_closure"],
        }
    else:
        try:
            q_interval = _ratio_interval(
                validated,
                d_vectors,
                h_vectors,
                minimum_headroom=thresholds.minimum_oracle_headroom,
                thresholds=thresholds,
                bootstrap_seed=registered_bootstrap_seeds["headroom_closure"],
            )
        except OracleHeadroomError as error:
            q_payload = {
                "evaluable": False,
                "reason": "BOOTSTRAP_SEED_HEADROOM_AT_OR_BELOW_MINIMUM",
                "detail": str(error),
                "minimum_oracle_headroom": thresholds.minimum_oracle_headroom,
                "headroom_seed_values": {
                    str(seed): value for seed, value in h_seed_means.items()
                },
                "bootstrap_seed": registered_bootstrap_seeds["headroom_closure"],
            }
        else:
            q_seed_values = {
                seed: float(d_vectors[seed].mean()) / h_seed_means[seed]
                for seed in E05B_SEEDS
            }
            q_payload = {
                **_interval_payload(q_interval),
                "evaluable": True,
                "bootstrap_seed": registered_bootstrap_seeds["headroom_closure"],
                "seed_values": {
                    str(seed): value for seed, value in q_seed_values.items()
                },
            }

    return {
        "status": "DESCRIPTIVE_ONLY",
        "descriptive_only": True,
        "contributes_to_primary_gate": False,
        "fixed_seeds": list(E05B_SEEDS),
        "row_counts": _row_counts(validated),
        "bootstrap": {
            "unit": "episode_within_seed_domain_template_stratum",
            "fixed_seeds_not_resampled": True,
            "paired_indices_across_models": True,
            "samples": thresholds.bootstrap_samples,
            "confidence": thresholds.bootstrap_confidence,
            "seeds": {
                "D_shared_minus_factorized": registered_bootstrap_seeds[
                    "primary_raw_improvement"
                ],
                "H_shared_minus_oracle": registered_bootstrap_seeds[
                    "shared_headroom"
                ],
                "Q_headroom_fraction_closed": registered_bootstrap_seeds[
                    "headroom_closure"
                ],
                "R_factorized_minus_shared_retention": (
                    registered_bootstrap_seeds["retention_noninferiority"]
                ),
                "oracle_affected": registered_bootstrap_seeds["oracle_affected"],
                "oracle_retention": registered_bootstrap_seeds["oracle_retention"],
                "factorized_retention_minus_oracle": (
                    registered_bootstrap_seeds["absolute_factorized_retention"]
                ),
            },
        },
        "D_shared_minus_factorized": _descriptive_interval_payload(
            validated,
            d_vectors,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                "primary_raw_improvement"
            ],
        ),
        "H_shared_minus_oracle": _descriptive_interval_payload(
            validated,
            h_vectors,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["shared_headroom"],
        ),
        "Q_headroom_fraction_closed": q_payload,
        "R_factorized_minus_shared_retention": _descriptive_interval_payload(
            validated,
            r_vectors,
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                "retention_noninferiority"
            ],
        ),
        "oracle_affected": _descriptive_interval_payload(
            validated,
            _metric_vectors(validated, metric="affected", lhs=ORACLE),
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_affected"],
        ),
        "oracle_retention": _descriptive_interval_payload(
            validated,
            _metric_vectors(validated, metric="retention", lhs=ORACLE),
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds["oracle_retention"],
        ),
        "factorized_retention_minus_oracle": _descriptive_interval_payload(
            validated,
            _metric_vectors(
                validated, metric="retention", lhs=FACTORIZED, rhs=ORACLE
            ),
            thresholds=thresholds,
            bootstrap_seed=registered_bootstrap_seeds[
                "absolute_factorized_retention"
            ],
        ),
        "controls": "NOT_EVALUATED_NO_FROZEN_SECONDARY_CONTROL_PAIRINGS",
    }


__all__ = [
    "CONTROL_NAMES",
    "E05A_BOOTSTRAP_SEEDS",
    "E05A_SEEDS",
    "E05B_BOOTSTRAP_SEEDS",
    "E05B_SEEDS",
    "FACTORIZED",
    "ORACLE",
    "SHARED",
    "OracleHeadroomError",
    "SemanticAnchorSeedMetrics",
    "SemanticAnchorThresholds",
    "evaluate_e05a_go",
    "evaluate_e05b_main",
    "evaluate_e05b_secondary",
    "evaluate_e05b_validation",
    "exact_seed_tost",
]
