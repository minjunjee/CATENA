"""Pure resource gates for the E26 Final official 1.3B preflight.

The functions in this module deliberately accept only systems measurements.
No loss, task score, or other scientific outcome can participate in token-
budget selection.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

TOKEN_BUDGET_CANDIDATES = (350_000_000, 500_000_000, 750_000_000, 1_000_000_000)
GPU_COUNT = 4
MAIN_RUN_COUNT = 10
WAVE_COUNT = 3
MIN_TOKENS_PER_SECOND_PER_GPU = 12_000.0
MEDIAN_UTILIZATION_PERCENT_MIN = 60.0
TIED_DUAL_THROUGHPUT_RATIO_MIN = 0.95
TIED_DUAL_THROUGHPUT_RATIO_MAX = 1.05
PEAK_VRAM_GIB_MAX = 92.0
JOINT_UNDERUTILIZATION_PERCENT = 50.0
JOINT_UNDERUTILIZATION_MEAN_POWER_WATTS = 150.0
PYTHON_LOOP_COUNT_MAX = 0
FALLBACK_COUNT_MAX = 0
MEASURED_STEPS_REQUIRED = 200
EVALUATION_ALLOWANCE_HOURS = 3.0
CONTINGENCY_FRACTION = 0.20
PROJECTED_TOTAL_HOURS_MAX = 36.0
REGISTERED_VARIANTS = ("tied", "dual")


class E26FinalResourceError(ValueError):
    """Raised when a resource request violates the prospective contract."""


@dataclass(frozen=True, slots=True)
class FinalResourcePolicy:
    token_budget_candidates: tuple[int, ...]
    gpu_count: int
    main_run_count: int
    wave_count: int
    min_tokens_per_second_per_gpu: float
    median_utilization_percent_min: float
    tied_dual_throughput_ratio_min: float
    tied_dual_throughput_ratio_max: float
    peak_vram_gib_max: float
    joint_underutilization_percent: float
    joint_underutilization_mean_power_watts: float
    python_loop_count_max: int
    fallback_count_max: int
    measured_steps_required: int
    evaluation_allowance_hours: float
    contingency_fraction: float
    projected_total_hours_max: float
    expected_kernel: str
    variants: tuple[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_budget_candidates"] = list(self.token_budget_candidates)
        payload["variants"] = list(self.variants)
        return payload


@dataclass(frozen=True, slots=True)
class SpeedObservation:
    gpu_index: int
    variant: str
    tokens_per_second_per_gpu: float
    median_utilization_percent: float
    mean_power_watts: float
    peak_vram_gib: float
    kernel: str
    python_loop_count: int
    fallback_count: int
    measured_steps: int
    finite_loss_steps: int
    finite_gradient_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeedValidation:
    passed: bool
    failures: tuple[str, ...]
    observation_count: int
    minimum_tokens_per_second_per_gpu: float
    tied_median_tokens_per_second_per_gpu: float
    dual_median_tokens_per_second_per_gpu: float
    tied_dual_throughput_ratio: float
    median_utilization_percent: float
    mean_power_watts: float
    peak_vram_gib: float
    joint_underutilization_blocked: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = list(self.failures)
        return payload


@dataclass(frozen=True, slots=True)
class BudgetProjection:
    token_budget: int
    single_run_training_hours: float
    training_wave_hours: float
    bridge_hours: float
    evaluation_allowance_hours: float
    subtotal_hours: float
    contingency_hours: float
    projected_total_hours: float
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenBudgetSelection:
    passed: bool
    disposition: str
    selected_token_budget: int | None
    selection_rule: str
    outcome_inputs_used: bool
    speed_validation: SpeedValidation
    projections: tuple[BudgetProjection, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "disposition": self.disposition,
            "selected_token_budget": self.selected_token_budget,
            "selection_rule": self.selection_rule,
            "outcome_inputs_used": self.outcome_inputs_used,
            "speed_validation": self.speed_validation.as_dict(),
            "projections": [row.as_dict() for row in self.projections],
        }


def registered_final_resource_policy(*, expected_kernel: str) -> FinalResourcePolicy:
    """Return the preregistered policy with an explicit official kernel ID."""

    return validate_registered_policy(
        FinalResourcePolicy(
            token_budget_candidates=TOKEN_BUDGET_CANDIDATES,
            gpu_count=GPU_COUNT,
            main_run_count=MAIN_RUN_COUNT,
            wave_count=WAVE_COUNT,
            min_tokens_per_second_per_gpu=MIN_TOKENS_PER_SECOND_PER_GPU,
            median_utilization_percent_min=MEDIAN_UTILIZATION_PERCENT_MIN,
            tied_dual_throughput_ratio_min=TIED_DUAL_THROUGHPUT_RATIO_MIN,
            tied_dual_throughput_ratio_max=TIED_DUAL_THROUGHPUT_RATIO_MAX,
            peak_vram_gib_max=PEAK_VRAM_GIB_MAX,
            joint_underutilization_percent=JOINT_UNDERUTILIZATION_PERCENT,
            joint_underutilization_mean_power_watts=(JOINT_UNDERUTILIZATION_MEAN_POWER_WATTS),
            python_loop_count_max=PYTHON_LOOP_COUNT_MAX,
            fallback_count_max=FALLBACK_COUNT_MAX,
            measured_steps_required=MEASURED_STEPS_REQUIRED,
            evaluation_allowance_hours=EVALUATION_ALLOWANCE_HOURS,
            contingency_fraction=CONTINGENCY_FRACTION,
            projected_total_hours_max=PROJECTED_TOTAL_HOURS_MAX,
            expected_kernel=expected_kernel,
            variants=REGISTERED_VARIANTS,
        )
    )


def validate_registered_policy(policy: FinalResourcePolicy) -> FinalResourcePolicy:
    """Reject any drift from the user-approved E26 Final resource policy."""

    integer_fields = (
        "gpu_count",
        "main_run_count",
        "wave_count",
        "python_loop_count_max",
        "fallback_count_max",
        "measured_steps_required",
    )
    numeric_fields = (
        "min_tokens_per_second_per_gpu",
        "median_utilization_percent_min",
        "tied_dual_throughput_ratio_min",
        "tied_dual_throughput_ratio_max",
        "peak_vram_gib_max",
        "joint_underutilization_percent",
        "joint_underutilization_mean_power_watts",
        "evaluation_allowance_hours",
        "contingency_fraction",
        "projected_total_hours_max",
    )
    for field in integer_fields:
        value = getattr(policy, field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise E26FinalResourceError(f"{field} must be an integer")
    for field in numeric_fields:
        _finite_number(getattr(policy, field), field)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in policy.token_budget_candidates
    ):
        raise E26FinalResourceError("token_budget_candidates must contain integers")
    if not all(isinstance(value, str) for value in policy.variants):
        raise E26FinalResourceError("variants must contain strings")

    expected: dict[str, Any] = {
        "token_budget_candidates": TOKEN_BUDGET_CANDIDATES,
        "gpu_count": GPU_COUNT,
        "main_run_count": MAIN_RUN_COUNT,
        "wave_count": WAVE_COUNT,
        "min_tokens_per_second_per_gpu": MIN_TOKENS_PER_SECOND_PER_GPU,
        "median_utilization_percent_min": MEDIAN_UTILIZATION_PERCENT_MIN,
        "tied_dual_throughput_ratio_min": TIED_DUAL_THROUGHPUT_RATIO_MIN,
        "tied_dual_throughput_ratio_max": TIED_DUAL_THROUGHPUT_RATIO_MAX,
        "peak_vram_gib_max": PEAK_VRAM_GIB_MAX,
        "joint_underutilization_percent": JOINT_UNDERUTILIZATION_PERCENT,
        "joint_underutilization_mean_power_watts": (JOINT_UNDERUTILIZATION_MEAN_POWER_WATTS),
        "python_loop_count_max": PYTHON_LOOP_COUNT_MAX,
        "fallback_count_max": FALLBACK_COUNT_MAX,
        "measured_steps_required": MEASURED_STEPS_REQUIRED,
        "evaluation_allowance_hours": EVALUATION_ALLOWANCE_HOURS,
        "contingency_fraction": CONTINGENCY_FRACTION,
        "projected_total_hours_max": PROJECTED_TOTAL_HOURS_MAX,
        "variants": REGISTERED_VARIANTS,
    }
    for field, value in expected.items():
        if getattr(policy, field) != value:
            raise E26FinalResourceError(
                f"E26 Final policy drift for {field}: {getattr(policy, field)!r} != {value!r}"
            )
    if not isinstance(policy.expected_kernel, str) or not policy.expected_kernel.strip():
        raise E26FinalResourceError("expected_kernel must be an explicit non-empty ID")
    if math.ceil(policy.main_run_count / policy.gpu_count) != policy.wave_count:
        raise E26FinalResourceError("4-GPU/3-wave topology is inconsistent")
    return policy


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E26FinalResourceError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise E26FinalResourceError(f"{field} must be finite")
    return converted


def speed_observation_from_mapping(payload: Mapping[str, Any]) -> SpeedObservation:
    """Parse a strict systems-only observation; unknown/outcome fields fail."""

    fields = tuple(SpeedObservation.__dataclass_fields__)
    unknown = sorted(set(payload).difference(fields))
    missing = sorted(set(fields).difference(payload))
    if unknown or missing:
        raise E26FinalResourceError(
            f"Speed observation fields differ; missing={missing}, unknown={unknown}"
        )
    integer_fields = (
        "gpu_index",
        "python_loop_count",
        "fallback_count",
        "measured_steps",
        "finite_loss_steps",
        "finite_gradient_steps",
    )
    integers: dict[str, int] = {}
    for field in integer_fields:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise E26FinalResourceError(f"{field} must be an integer")
        integers[field] = value
    variant = payload["variant"]
    kernel = payload["kernel"]
    if not isinstance(variant, str) or not isinstance(kernel, str):
        raise E26FinalResourceError("variant and kernel must be strings")
    return SpeedObservation(
        gpu_index=integers["gpu_index"],
        variant=variant,
        tokens_per_second_per_gpu=_finite_number(
            payload["tokens_per_second_per_gpu"], "tokens_per_second_per_gpu"
        ),
        median_utilization_percent=_finite_number(
            payload["median_utilization_percent"], "median_utilization_percent"
        ),
        mean_power_watts=_finite_number(payload["mean_power_watts"], "mean_power_watts"),
        peak_vram_gib=_finite_number(payload["peak_vram_gib"], "peak_vram_gib"),
        kernel=kernel,
        python_loop_count=integers["python_loop_count"],
        fallback_count=integers["fallback_count"],
        measured_steps=integers["measured_steps"],
        finite_loss_steps=integers["finite_loss_steps"],
        finite_gradient_steps=integers["finite_gradient_steps"],
    )


def validate_speed_preflight(
    observations: Sequence[SpeedObservation],
    *,
    policy: FinalResourcePolicy,
) -> SpeedValidation:
    """Evaluate only throughput, health, memory, and backend measurements."""

    policy = validate_registered_policy(policy)
    expected_cells = {
        (gpu_index, variant) for gpu_index in range(policy.gpu_count) for variant in policy.variants
    }
    observed_cells = [(row.gpu_index, row.variant) for row in observations]
    if len(observed_cells) != len(set(observed_cells)):
        raise E26FinalResourceError("Speed preflight contains duplicate GPU/variant cells")
    if set(observed_cells) != expected_cells:
        raise E26FinalResourceError(
            "Speed preflight must contain exactly one tied and dual row for each of 4 GPUs"
        )

    failures: list[str] = []
    for row in observations:
        integer_fields = (
            row.gpu_index,
            row.python_loop_count,
            row.fallback_count,
            row.measured_steps,
            row.finite_loss_steps,
            row.finite_gradient_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields
        ):
            raise E26FinalResourceError(
                "Speed observation integer counts must be non-negative integers"
            )
        numeric_nonnegative = (
            row.tokens_per_second_per_gpu,
            row.median_utilization_percent,
            row.mean_power_watts,
            row.peak_vram_gib,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_nonnegative):
            raise E26FinalResourceError("Speed observation contains invalid numeric values")
        if row.tokens_per_second_per_gpu <= 0.0:
            raise E26FinalResourceError("Per-GPU throughput must be positive")
        if row.median_utilization_percent > 100.0:
            raise E26FinalResourceError("GPU utilization cannot exceed 100 percent")
        cell = f"gpu{row.gpu_index}:{row.variant}"
        if row.kernel != policy.expected_kernel:
            failures.append(f"{cell}:KERNEL_MISMATCH")
        if row.python_loop_count > policy.python_loop_count_max:
            failures.append(f"{cell}:PYTHON_LOOP_PRESENT")
        if row.fallback_count > policy.fallback_count_max:
            failures.append(f"{cell}:FALLBACK_PRESENT")
        if row.measured_steps != policy.measured_steps_required:
            failures.append(f"{cell}:MEASURED_STEPS_NOT_200")
        if row.finite_loss_steps != policy.measured_steps_required:
            failures.append(f"{cell}:NONFINITE_LOSS_STEP")
        if row.finite_gradient_steps != policy.measured_steps_required:
            failures.append(f"{cell}:NONFINITE_GRADIENT_STEP")
        if row.peak_vram_gib > policy.peak_vram_gib_max:
            failures.append(f"{cell}:PEAK_VRAM_EXCEEDED")

    throughput = [row.tokens_per_second_per_gpu for row in observations]
    utilization = [row.median_utilization_percent for row in observations]
    power = [row.mean_power_watts for row in observations]
    minimum_throughput = min(throughput)
    median_utilization = float(statistics.median(utilization))
    mean_power = float(statistics.fmean(power))
    peak_vram = max(row.peak_vram_gib for row in observations)
    tied_median = float(
        statistics.median(
            row.tokens_per_second_per_gpu for row in observations if row.variant == "tied"
        )
    )
    dual_median = float(
        statistics.median(
            row.tokens_per_second_per_gpu for row in observations if row.variant == "dual"
        )
    )
    ratio = tied_median / dual_median if dual_median > 0.0 else math.inf
    joint_blocked = (
        median_utilization < policy.joint_underutilization_percent
        and mean_power < policy.joint_underutilization_mean_power_watts
    )
    if minimum_throughput < policy.min_tokens_per_second_per_gpu:
        failures.append("MINIMUM_PER_GPU_THROUGHPUT_BELOW_12000")
    if median_utilization < policy.median_utilization_percent_min:
        failures.append("MEDIAN_GPU_UTILIZATION_BELOW_60_PERCENT")
    if joint_blocked:
        failures.append("JOINT_UNDERUTILIZATION_AND_LOW_POWER")
    if not (
        policy.tied_dual_throughput_ratio_min <= ratio <= policy.tied_dual_throughput_ratio_max
    ):
        failures.append("TIED_DUAL_THROUGHPUT_RATIO_OUTSIDE_0_95_TO_1_05")

    return SpeedValidation(
        passed=not failures,
        failures=tuple(failures),
        observation_count=len(observations),
        minimum_tokens_per_second_per_gpu=minimum_throughput,
        tied_median_tokens_per_second_per_gpu=tied_median,
        dual_median_tokens_per_second_per_gpu=dual_median,
        tied_dual_throughput_ratio=ratio,
        median_utilization_percent=median_utilization,
        mean_power_watts=mean_power,
        peak_vram_gib=peak_vram,
        joint_underutilization_blocked=joint_blocked,
    )


def project_token_budgets(
    *,
    minimum_tokens_per_second_per_gpu: float,
    bridge_hours: float,
    policy: FinalResourcePolicy,
) -> tuple[BudgetProjection, ...]:
    """Project registered budgets using only conservative systems quantities."""

    policy = validate_registered_policy(policy)
    throughput = _finite_number(
        minimum_tokens_per_second_per_gpu, "minimum_tokens_per_second_per_gpu"
    )
    bridge = _finite_number(bridge_hours, "bridge_hours")
    if throughput <= 0.0 or bridge < 0.0:
        raise E26FinalResourceError("throughput must be positive and bridge_hours non-negative")
    rows: list[BudgetProjection] = []
    for budget in policy.token_budget_candidates:
        # Each model run occupies one GPU.  Four GPUs provide parallel runs and
        # are already represented by the preregistered three-wave schedule; do
        # not divide per-run time by GPU count a second time.
        single_run_hours = budget / throughput / 3600.0
        wave_hours = single_run_hours * policy.wave_count
        subtotal = wave_hours + bridge + policy.evaluation_allowance_hours
        contingency_hours = subtotal * policy.contingency_fraction
        total = subtotal + contingency_hours
        rows.append(
            BudgetProjection(
                token_budget=budget,
                single_run_training_hours=single_run_hours,
                training_wave_hours=wave_hours,
                bridge_hours=bridge,
                evaluation_allowance_hours=policy.evaluation_allowance_hours,
                subtotal_hours=subtotal,
                contingency_hours=contingency_hours,
                projected_total_hours=total,
                eligible=total <= policy.projected_total_hours_max,
            )
        )
    return tuple(rows)


def select_token_budget(
    observations: Sequence[SpeedObservation],
    *,
    bridge_hours: float,
    policy: FinalResourcePolicy,
) -> TokenBudgetSelection:
    """Select the largest time-eligible budget after all systems gates pass."""

    validation = validate_speed_preflight(observations, policy=policy)
    projections = project_token_budgets(
        minimum_tokens_per_second_per_gpu=validation.minimum_tokens_per_second_per_gpu,
        bridge_hours=bridge_hours,
        policy=policy,
    )
    eligible = [row.token_budget for row in projections if row.eligible]
    selected = max(eligible) if validation.passed and eligible else None
    if not validation.passed:
        disposition = "BLOCKED_SPEED_OR_SYSTEMS_GATE"
    elif selected is None:
        disposition = "BLOCKED_NO_TOKEN_BUDGET_WITHIN_36_HOURS"
    else:
        disposition = "TOKEN_BUDGET_SELECTED_NON_EVIDENCE"
    return TokenBudgetSelection(
        passed=selected is not None,
        disposition=disposition,
        selected_token_budget=selected,
        selection_rule=(
            "LARGEST_REGISTERED_BUDGET_WITH_PROJECTED_TOTAL_LE_36H_USING_"
            "MINIMUM_PER_GPU_THROUGHPUT_NO_OUTCOME_INPUTS"
        ),
        outcome_inputs_used=False,
        speed_validation=validation,
        projections=projections,
    )


_POLICY_FIELDS = tuple(FinalResourcePolicy.__dataclass_fields__)


def policy_from_mapping(payload: Mapping[str, Any]) -> FinalResourcePolicy:
    """Parse an explicit policy and require every registered field."""

    unknown = sorted(set(payload).difference(_POLICY_FIELDS))
    missing = sorted(set(_POLICY_FIELDS).difference(payload))
    if unknown or missing:
        raise E26FinalResourceError(
            f"Resource policy fields differ; missing={missing}, unknown={unknown}"
        )
    normalized = dict(payload)
    budgets = normalized["token_budget_candidates"]
    variants = normalized["variants"]
    if not isinstance(budgets, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in budgets
    ):
        raise E26FinalResourceError("token_budget_candidates must be an integer list")
    if not isinstance(variants, list) or not all(isinstance(value, str) for value in variants):
        raise E26FinalResourceError("variants must be a string list")
    integer_fields = (
        "gpu_count",
        "main_run_count",
        "wave_count",
        "python_loop_count_max",
        "fallback_count_max",
        "measured_steps_required",
    )
    numeric_fields = (
        "min_tokens_per_second_per_gpu",
        "median_utilization_percent_min",
        "tied_dual_throughput_ratio_min",
        "tied_dual_throughput_ratio_max",
        "peak_vram_gib_max",
        "joint_underutilization_percent",
        "joint_underutilization_mean_power_watts",
        "evaluation_allowance_hours",
        "contingency_fraction",
        "projected_total_hours_max",
    )
    for field in integer_fields:
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise E26FinalResourceError(f"{field} must be an integer")
    for field in numeric_fields:
        _finite_number(normalized[field], field)
    if not isinstance(normalized["expected_kernel"], str):
        raise E26FinalResourceError("expected_kernel must be a string")
    normalized["token_budget_candidates"] = tuple(budgets)
    normalized["variants"] = tuple(variants)
    try:
        policy = FinalResourcePolicy(**normalized)
    except TypeError as error:
        raise E26FinalResourceError(f"Malformed resource policy: {error}") from error
    return validate_registered_policy(policy)
