from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from catena.core.provenance_v61 import read_json_object_strict, sha256_canonical_json
from catena.lm.e26_final_resources import (
    E26FinalResourceError,
    FinalResourcePolicy,
    SpeedObservation,
    policy_from_mapping,
    project_token_budgets,
    registered_final_resource_policy,
    select_token_budget,
    speed_observation_from_mapping,
    validate_registered_policy,
    validate_speed_preflight,
)
from tools.select_e26_final_token_budget import (
    REQUEST_SCHEMA_VERSION,
    main,
    select_from_payload,
)

KERNEL = "official_gdn2_compiled_chunk_scan_v1"


def policy() -> FinalResourcePolicy:
    return registered_final_resource_policy(expected_kernel=KERNEL)


def replace_policy(
    original: FinalResourcePolicy, changes: Mapping[str, object]
) -> FinalResourcePolicy:
    return replace(original, **cast(dict[str, Any], dict(changes)))


def replace_observation(
    original: SpeedObservation, changes: Mapping[str, object]
) -> SpeedObservation:
    return replace(original, **cast(dict[str, Any], dict(changes)))


def observations(
    *,
    tied_tokens_per_second: float = 12_000.0,
    dual_tokens_per_second: float = 12_000.0,
    utilization: float = 60.0,
    power: float = 150.0,
    peak_vram: float = 92.0,
) -> list[SpeedObservation]:
    rows: list[SpeedObservation] = []
    for gpu_index in range(4):
        for variant, throughput in (
            ("tied", tied_tokens_per_second),
            ("dual", dual_tokens_per_second),
        ):
            rows.append(
                SpeedObservation(
                    gpu_index=gpu_index,
                    variant=variant,
                    tokens_per_second_per_gpu=throughput,
                    median_utilization_percent=utilization,
                    mean_power_watts=power,
                    peak_vram_gib=peak_vram,
                    kernel=KERNEL,
                    python_loop_count=0,
                    fallback_count=0,
                    measured_steps=200,
                    finite_loss_steps=200,
                    finite_gradient_steps=200,
                )
            )
    return rows


def request_payload(*, bridge_hours: float = 2.0) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "policy": policy().as_dict(),
        "bridge_hours": bridge_hours,
        "observations": [row.as_dict() for row in observations()],
    }


def test_registered_policy_contains_every_exact_resource_constant() -> None:
    observed = policy()
    assert observed.token_budget_candidates == (
        350_000_000,
        500_000_000,
        750_000_000,
        1_000_000_000,
    )
    assert (observed.gpu_count, observed.main_run_count, observed.wave_count) == (4, 10, 3)
    assert observed.min_tokens_per_second_per_gpu == 12_000.0
    assert observed.median_utilization_percent_min == 60.0
    assert (
        observed.tied_dual_throughput_ratio_min,
        observed.tied_dual_throughput_ratio_max,
    ) == (0.95, 1.05)
    assert observed.peak_vram_gib_max == 92.0
    assert (
        observed.joint_underutilization_percent,
        observed.joint_underutilization_mean_power_watts,
    ) == (50.0, 150.0)
    assert (observed.python_loop_count_max, observed.fallback_count_max) == (0, 0)
    assert observed.measured_steps_required == 200
    assert observed.evaluation_allowance_hours == 3.0
    assert observed.contingency_fraction == 0.20
    assert observed.projected_total_hours_max == 36.0
    assert observed.expected_kernel == KERNEL


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_count", 8),
        ("main_run_count", 8),
        ("wave_count", 2),
        ("min_tokens_per_second_per_gpu", 11_999.0),
        ("measured_steps_required", 199),
        ("evaluation_allowance_hours", 2.0),
        ("contingency_fraction", 0.10),
        ("projected_total_hours_max", 48.0),
    ],
)
def test_registered_policy_rejects_drift(field: str, value: object) -> None:
    with pytest.raises(E26FinalResourceError, match="policy drift"):
        validate_registered_policy(replace_policy(policy(), {field: value}))


def test_policy_mapping_is_explicit_and_strict() -> None:
    payload = policy().as_dict()
    assert policy_from_mapping(payload) == policy()
    with pytest.raises(E26FinalResourceError, match=r"unknown=\['accuracy'\]"):
        policy_from_mapping({**payload, "accuracy": 1.0})
    malformed = {**payload, "gpu_count": 4.0}
    with pytest.raises(E26FinalResourceError, match="gpu_count must be an integer"):
        policy_from_mapping(malformed)


def test_exact_speed_boundaries_pass_and_select_largest_time_eligible_budget() -> None:
    result = select_token_budget(observations(), bridge_hours=2.0, policy=policy())
    assert result.passed
    assert result.selected_token_budget == 350_000_000
    assert result.outcome_inputs_used is False
    assert result.speed_validation.minimum_tokens_per_second_per_gpu == 12_000.0
    assert result.speed_validation.median_utilization_percent == 60.0
    assert result.speed_validation.peak_vram_gib == 92.0
    assert result.speed_validation.tied_dual_throughput_ratio == 1.0
    assert not result.speed_validation.joint_underutilization_blocked
    assert result.projections[0].single_run_training_hours == pytest.approx(
        350_000_000 / 12_000.0 / 3600.0
    )
    assert result.projections[0].training_wave_hours == pytest.approx(
        result.projections[0].single_run_training_hours * 3
    )


def test_budget_projection_uses_minimum_per_gpu_throughput_and_largest_fit() -> None:
    rows = observations(tied_tokens_per_second=40_000.0, dual_tokens_per_second=40_000.0)
    rows[0] = replace(rows[0], tokens_per_second_per_gpu=30_000.0)
    result = select_token_budget(rows, bridge_hours=2.0, policy=policy())
    assert result.passed
    assert result.speed_validation.minimum_tokens_per_second_per_gpu == 30_000.0
    assert result.selected_token_budget == 750_000_000
    projections = {row.token_budget: row for row in result.projections}
    assert projections[750_000_000].eligible
    assert not projections[1_000_000_000].eligible
    assert projections[750_000_000].projected_total_hours == pytest.approx(31.0)


def test_high_per_gpu_throughput_selects_one_billion() -> None:
    result = select_token_budget(
        observations(tied_tokens_per_second=40_000.0, dual_tokens_per_second=40_000.0),
        bridge_hours=2.0,
        policy=policy(),
    )
    assert result.passed
    assert result.selected_token_budget == 1_000_000_000


def test_no_registered_budget_fits_large_bridge_time() -> None:
    result = select_token_budget(observations(), bridge_hours=30.0, policy=policy())
    assert not result.passed
    assert result.selected_token_budget is None
    assert result.disposition == "BLOCKED_NO_TOKEN_BUDGET_WITHIN_36_HOURS"


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ({"tokens_per_second_per_gpu": 11_999.0}, "MINIMUM_PER_GPU_THROUGHPUT"),
        ({"peak_vram_gib": 92.01}, "PEAK_VRAM_EXCEEDED"),
        ({"kernel": "reference_python"}, "KERNEL_MISMATCH"),
        ({"python_loop_count": 1}, "PYTHON_LOOP_PRESENT"),
        ({"fallback_count": 1}, "FALLBACK_PRESENT"),
        ({"measured_steps": 199}, "MEASURED_STEPS_NOT_200"),
        ({"finite_loss_steps": 199}, "NONFINITE_LOSS_STEP"),
        ({"finite_gradient_steps": 199}, "NONFINITE_GRADIENT_STEP"),
    ],
)
def test_each_cell_level_speed_gate_fails(mutation: dict[str, object], failure: str) -> None:
    rows = observations()
    rows[0] = replace_observation(rows[0], mutation)
    result = validate_speed_preflight(rows, policy=policy())
    assert not result.passed
    assert any(failure in item for item in result.failures)


def test_median_utilization_below_sixty_fails() -> None:
    result = validate_speed_preflight(observations(utilization=59.0), policy=policy())
    assert not result.passed
    assert "MEDIAN_GPU_UTILIZATION_BELOW_60_PERCENT" in result.failures


@pytest.mark.parametrize(
    ("tied", "dual", "passes"),
    [
        (12_350.0, 13_000.0, True),
        (12_600.0, 12_000.0, True),
        (12_349.0, 13_000.0, False),
        (12_601.0, 12_000.0, False),
    ],
)
def test_tied_dual_throughput_ratio_boundaries(tied: float, dual: float, passes: bool) -> None:
    result = validate_speed_preflight(
        observations(tied_tokens_per_second=tied, dual_tokens_per_second=dual),
        policy=policy(),
    )
    ratio_failure = "TIED_DUAL_THROUGHPUT_RATIO_OUTSIDE_0_95_TO_1_05"
    assert (ratio_failure not in result.failures) is passes


@pytest.mark.parametrize(
    ("utilization", "power", "joint_blocked"),
    [
        (49.0, 149.0, True),
        (49.0, 150.0, False),
        (50.0, 149.0, False),
    ],
)
def test_joint_underutilization_requires_both_strict_conditions(
    utilization: float, power: float, joint_blocked: bool
) -> None:
    result = validate_speed_preflight(
        observations(utilization=utilization, power=power), policy=policy()
    )
    assert result.joint_underutilization_blocked is joint_blocked
    assert ("JOINT_UNDERUTILIZATION_AND_LOW_POWER" in result.failures) is joint_blocked


def test_duplicate_or_missing_gpu_variant_cells_fail_closed() -> None:
    rows = observations()
    with pytest.raises(E26FinalResourceError, match="duplicate"):
        validate_speed_preflight([*rows, rows[0]], policy=policy())
    with pytest.raises(E26FinalResourceError, match="exactly one"):
        validate_speed_preflight(rows[:-1], policy=policy())


@pytest.mark.parametrize(
    "mutation",
    [
        {"tokens_per_second_per_gpu": math.nan},
        {"tokens_per_second_per_gpu": 0.0},
        {"mean_power_watts": math.inf},
        {"python_loop_count": -1},
        {"median_utilization_percent": 101.0},
    ],
)
def test_invalid_observation_values_raise(mutation: dict[str, object]) -> None:
    rows = observations()
    rows[0] = replace_observation(rows[0], mutation)
    with pytest.raises(E26FinalResourceError):
        validate_speed_preflight(rows, policy=policy())


def test_strict_observation_parser_rejects_outcome_fields() -> None:
    payload = observations()[0].as_dict()
    assert speed_observation_from_mapping(payload) == observations()[0]
    with pytest.raises(E26FinalResourceError, match=r"unknown=\['accuracy'\]"):
        speed_observation_from_mapping({**payload, "accuracy": 0.99})
    with pytest.raises(E26FinalResourceError, match="finite_loss_steps must be an integer"):
        speed_observation_from_mapping({**payload, "finite_loss_steps": 200.0})


def test_projection_rejects_nonfinite_or_negative_system_inputs() -> None:
    with pytest.raises(E26FinalResourceError):
        project_token_budgets(
            minimum_tokens_per_second_per_gpu=math.inf,
            bridge_hours=2.0,
            policy=policy(),
        )
    with pytest.raises(E26FinalResourceError):
        project_token_budgets(
            minimum_tokens_per_second_per_gpu=12_000.0,
            bridge_hours=-0.1,
            policy=policy(),
        )


def test_receipt_is_deterministic_non_evidence_and_hash_bound() -> None:
    payload = request_payload()
    first = select_from_payload(payload)
    second = select_from_payload(payload)
    assert first == second
    assert first["run_mode"] == "NON_EVIDENCE_SPEED_PREFLIGHT"
    assert first["scientific_evidence"] is False
    assert first["scientific_e26a_started"] is False
    assert first["outcome_inputs_used"] is False
    assert first["selection"]["selected_token_budget"] == 350_000_000
    claimed = first["receipt_sha256"]
    assert claimed == sha256_canonical_json(
        {key: value for key, value in first.items() if key != "receipt_sha256"}
    )


@pytest.mark.parametrize("outcome_field", ["accuracy", "loss", "did", "effect_size"])
def test_request_schema_rejects_every_scientific_outcome_input(outcome_field: str) -> None:
    with pytest.raises(E26FinalResourceError, match="unknown"):
        select_from_payload({**request_payload(), outcome_field: 0.5})


def test_cli_writes_new_receipt_and_returns_blocking_status(tmp_path: Any) -> None:
    from catena.core.provenance_v61 import write_json_strict

    passing_input = tmp_path / "pass.json"
    passing_output = tmp_path / "pass-receipt.json"
    write_json_strict(passing_input, request_payload())
    assert main(["--input", str(passing_input), "--output", str(passing_output)]) == 0
    assert read_json_object_strict(passing_output)["selection"]["passed"] is True

    blocked_input = tmp_path / "blocked.json"
    blocked_output = tmp_path / "blocked-receipt.json"
    write_json_strict(blocked_input, request_payload(bridge_hours=30.0))
    assert main(["--input", str(blocked_input), "--output", str(blocked_output)]) == 1
    assert read_json_object_strict(blocked_output)["selection"]["passed"] is False
    assert main(["--input", str(passing_input), "--output", str(passing_output)]) == 2
