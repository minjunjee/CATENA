"""Shared outcome-independent runner helpers for E23a and E23b."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from catena.core.io import file_sha256, write_json, write_jsonl
from catena.data.controller_poset import (
    CANONICAL_CONTROLLERS,
    CONTROLLER_AXES,
    DEMAND_FAMILIES,
    PAIRWISE_DEMANDS,
    SINGLE_AXIS_DEMANDS,
    missing_required_axes,
    theory_prediction_payload,
)
from catena.post_e21.locality_data import LocalityMethod, LocalityObjective
from catena.post_e21.product_poset_eval import (
    ensure_finite_rows,
    expected_grid_size,
    result_independent_boundary_manifest,
)
from catena.post_e21.product_poset_model import (
    MatchedProductPosetSequenceController,
    ProductPosetProbeConfig,
    product_poset_parameter_count,
    theoretical_affected_error,
)
from catena.post_e21.product_poset_training import (
    evaluate_product_poset_controller,
    train_product_poset_controller,
)
from catena.training.sequence_control_lattice import (
    state_dict_sha256,
)


@dataclass(slots=True)
class ProductPosetRunResult:
    rows: list[dict[str, Any]]
    training_rows: list[dict[str, Any]]
    checkpoint_hashes: dict[str, str]
    runtime: dict[str, Any]


def _locality_method_from_payload(
    payload: Mapping[str, Any],
) -> LocalityMethod:
    method = LocalityMethod(
        method_id=str(payload["method_id"]),
        objective=LocalityObjective(str(payload["objective"])),
        selection_eligible=bool(payload["selection_eligible"]),
        baseline=bool(payload["baseline"]),
        tail_fraction=(
            None if payload.get("tail_fraction") is None else float(payload["tail_fraction"])
        ),
        normalized_temperature=(
            None
            if payload.get("normalized_temperature") is None
            else float(payload["normalized_temperature"])
        ),
        active_fraction=(
            None if payload.get("active_fraction") is None else float(payload["active_fraction"])
        ),
    )
    if method.as_dict() != dict(payload):
        raise ValueError("E23 locality method payload is not canonical")
    return method


def validate_e23_config(
    config: Mapping[str, Any],
    *,
    experiment_id: str,
    expected_seed_count: int,
) -> None:
    if config.get("experiment_id") != experiment_id:
        raise ValueError("E23 config experiment identity mismatch")
    axes = tuple(str(value) for value in config["controller_axes"])
    if axes != CONTROLLER_AXES:
        raise ValueError("E23 controller-axis order is not canonical")
    demands = tuple(str(value) for value in config["demand_families"])
    if demands != DEMAND_FAMILIES:
        raise ValueError("E23 demand-family order is not canonical")
    if tuple(config["single_axis_demands"]) != SINGLE_AXIS_DEMANDS:
        raise ValueError("E23 single-axis demand set is not canonical")
    if tuple(config["pairwise_demands"]) != PAIRWISE_DEMANDS:
        raise ValueError("E23 pairwise demand set is not canonical")
    seeds = tuple(int(value) for value in config["seeds"])
    if len(seeds) != expected_seed_count or len(set(seeds)) != len(seeds):
        raise ValueError(f"E23 requires {expected_seed_count} unique registered seeds")
    intensities = tuple(float(value) for value in config["intensities"])
    if intensities != (0.25, 0.5, 1.0):
        raise ValueError("E23 intensity grid is not canonical")
    evaluation = config["evaluation"]
    if tuple(int(value) for value in evaluation["updates"]) != (1, 4, 8):
        raise ValueError("E23 update grid is not canonical")
    if tuple(int(value) for value in evaluation["gap_events"]) != (
        0,
        512,
        2048,
    ):
        raise ValueError("E23 gap grid is not canonical")
    if config["boundary_selection"]["rule"] != "theory_boundary_only_v1":
        raise ValueError("E23 boundary rule differs from the locked rule")
    if config["boundary_selection"]["result_independent"] is not True:
        raise ValueError("E23 boundary selection must be result-independent")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if (
        int(data["num_entities"]) != 32
        or int(data["value_dim"]) != 32
        or int(model["embedding_dim"]) != 128
        or int(model["hidden_dim"]) != 512
        or str(training["optimizer"]) != "AdamW"
        or training["maximal_parameter_surface"] is not True
        or training["paired_initialization"] is not True
        or training["paired_data_order"] is not True
    ):
        raise ValueError("E23 no longer matches the E18 learned-sequence surface")
    if int(training["steps"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("E23 training schedule must be positive")
    if int(evaluation["batches"]) <= 0 or int(evaluation["batch_size"]) <= 0:
        raise ValueError("E23 evaluation sampling must be positive")
    adequacy = config["adequacy"]
    affected_tolerance = float(adequacy["affected_mse_tolerance"])
    minimum_penalty = float(config["probe"]["missing_axis_floor"]) * min(intensities) ** 2
    if not 0 <= affected_tolerance < minimum_penalty:
        raise ValueError(
            "E23 affected tolerance does not identify the locked required-bit boundary"
        )
    if (
        int(adequacy["minimum_single_axis_exact_matches"]) != 4
        or int(adequacy["minimum_pairwise_exact_matches"]) != 5
        or float(adequacy["incomparable_direction_margin"]) != 0.0
        or float(adequacy["maximal_simpler_degradation_margin"]) != 0.0005
    ):
        raise ValueError("E23 primary adequacy gates differ from the frozen contract")


def probe_config(config: Mapping[str, Any]) -> ProductPosetProbeConfig:
    payload = config["probe"]
    return ProductPosetProbeConfig(
        missing_axis_floor=float(payload["missing_axis_floor"]),
        numerical_floor=float(payload["numerical_floor"]),
    )


def product_poset_runtime(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    training = config["training"]
    if not dry_run:
        return {
            "dry_run_reduced": False,
            "seeds": [int(value) for value in config["seeds"]],
            "intensities": [float(value) for value in config["intensities"]],
            "updates": [int(value) for value in evaluation["updates"]],
            "gap_events": [int(value) for value in evaluation["gap_events"]],
            "training_steps": int(training["steps"]),
            "training_batch_size": int(training["batch_size"]),
            "training_updates": int(training["updates"]),
            "training_gap_events": int(training["gap_events"]),
            "evaluation_batches": int(evaluation["batches"]),
            "evaluation_batch_size": int(evaluation["batch_size"]),
        }
    dry = config["dry_run"]
    return {
        "dry_run_reduced": True,
        "seeds": [int(value) for value in config["seeds"][: int(dry["seed_count"])]],
        "intensities": [float(config["intensities"][0])],
        "updates": [int(config["evaluation"]["updates"][0])],
        "gap_events": [int(config["evaluation"]["gap_events"][0])],
        "training_steps": int(dry["training_steps"]),
        "training_batch_size": int(dry["training_batch_size"]),
        "training_updates": int(dry["training_updates"]),
        "training_gap_events": int(dry["training_gap_events"]),
        "evaluation_batches": int(dry["evaluation_batches"]),
        "evaluation_batch_size": int(dry["evaluation_batch_size"]),
    }


def generate_product_poset_rows(
    config: Mapping[str, Any],
    *,
    boundary_mode: str,
    locality_method_payload: Mapping[str, Any],
    locality_risk_scale: float,
    device: torch.device,
    run_dir: Path,
    dry_run: bool,
) -> ProductPosetRunResult:
    """Train/evaluate all 16 projections; theory values are diagnostics only."""

    if boundary_mode not in {"capacity_only", "safe_minimality"}:
        raise ValueError("unsupported E23 boundary mode")
    runtime = product_poset_runtime(config, dry_run=dry_run)
    seeds = list(runtime["seeds"])
    intensities = list(runtime["intensities"])
    updates = list(runtime["updates"])
    gaps = list(runtime["gap_events"])
    theory_config = probe_config(config)
    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    locality_method = _locality_method_from_payload(locality_method_payload)
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=False, exist_ok=False)
    for seed in seeds:
        torch.manual_seed(int(training["initialization_seed_offset"]) + int(seed))
        template = MatchedProductPosetSequenceController(
            controller=CANONICAL_CONTROLLERS[-1],
            num_entities=int(data["num_entities"]),
            value_dim=int(data["value_dim"]),
            embedding_dim=int(model_config["embedding_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
        )
        initial_state = deepcopy(template.state_dict())
        initialization_sha256 = state_dict_sha256(initial_state)
        expected_parameter_count = product_poset_parameter_count(template)
        del template
        for controller in CANONICAL_CONTROLLERS:
            model = MatchedProductPosetSequenceController(
                controller=controller,
                num_entities=int(data["num_entities"]),
                value_dim=int(data["value_dim"]),
                embedding_dim=int(model_config["embedding_dim"]),
                hidden_dim=int(model_config["hidden_dim"]),
            )
            model.load_state_dict(initial_state, strict=True)
            if product_poset_parameter_count(model) != expected_parameter_count:
                raise RuntimeError("E23 maximal parameter surface changed")
            trace = train_product_poset_controller(
                model=model,
                demand_families=list(DEMAND_FAMILIES),
                intensities=[float(value) for value in config["intensities"]],
                steps=int(runtime["training_steps"]),
                batch_size=int(runtime["training_batch_size"]),
                num_entities=int(data["num_entities"]),
                value_dim=int(data["value_dim"]),
                updates=int(runtime["training_updates"]),
                gap_events=int(runtime["training_gap_events"]),
                learning_rate=float(training["learning_rate"]),
                retention_weight=float(training["retention_weight"]),
                locality_method=locality_method,
                locality_risk_scale=float(locality_risk_scale),
                device=device,
                seed=int(training["data_seed_offset"]) + int(seed),
            )
            checkpoint = checkpoints / f"{controller.controller_id}_seed{int(seed)}.pt"
            torch.save(model.state_dict(), checkpoint)
            checkpoint_sha256 = file_sha256(checkpoint)
            checkpoint_key = f"{controller.controller_id}_seed{int(seed)}"
            checkpoint_hashes[checkpoint_key] = checkpoint_sha256
            training_rows.append(
                {
                    "seed": int(seed),
                    "controller_id": controller.controller_id,
                    "controller_bits": "".join(str(bit) for bit in controller.bits),
                    "initialization_sha256": initialization_sha256,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "parameter_count": expected_parameter_count,
                    "optimizer": trace.optimizer,
                    "locality_method_id": trace.locality_method_id,
                    "locality_objective": trace.locality_objective,
                    "locality_risk_scale": float(locality_risk_scale),
                    "training_steps": int(runtime["training_steps"]),
                    "training_batch_size": int(runtime["training_batch_size"]),
                    "train_final_loss": trace.final_loss,
                    "train_best_loss": trace.best_loss,
                    "examples_per_second": trace.examples_per_second,
                    "peak_memory_bytes": trace.peak_memory_bytes,
                }
            )
            for demand in DEMAND_FAMILIES:
                boundary_ids = set(result_independent_boundary_manifest()[demand])
                demand_index = DEMAND_FAMILIES.index(demand)
                for intensity in intensities:
                    intensity_index = list(config["intensities"]).index(intensity)
                    evaluation_seed = (
                        int(config["evaluation"]["seed_offset"])
                        + 1_000_000 * int(seed)
                        + 10_000 * demand_index
                        + 100 * intensity_index
                    )
                    for update_count in updates:
                        for gap in gaps:
                            metrics = evaluate_product_poset_controller(
                                model=model,
                                demand_family=demand,
                                intensity=float(intensity),
                                batches=int(runtime["evaluation_batches"]),
                                batch_size=int(runtime["evaluation_batch_size"]),
                                num_entities=int(data["num_entities"]),
                                value_dim=int(data["value_dim"]),
                                updates=int(update_count),
                                gap_events=int(gap),
                                device=device,
                                seed=evaluation_seed,
                            )
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "controller_id": controller.controller_id,
                                    "controller_bits": "".join(str(bit) for bit in controller.bits),
                                    "controller_rank": controller.rank,
                                    "demand_family": demand,
                                    "intensity": float(intensity),
                                    "updates": int(update_count),
                                    "gap_events": int(gap),
                                    "boundary_mode": boundary_mode,
                                    "theoretical_affected_mse": (
                                        theoretical_affected_error(
                                            controller=controller,
                                            demand_family=demand,
                                            intensity=float(intensity),
                                            updates=int(update_count),
                                            gap_events=int(gap),
                                            config=theory_config,
                                        )
                                    ),
                                    **metrics,
                                    "active_nontarget_degradation": (
                                        float(metrics["worst_nontarget_mse"])
                                        if boundary_mode == "safe_minimality"
                                        else None
                                    ),
                                    "capacity_satisfied": not missing_required_axes(
                                        controller,
                                        demand,
                                    ),
                                    "confirmatory_boundary_member": (
                                        controller.controller_id in boundary_ids
                                    ),
                                    "checkpoint": str(checkpoint.resolve()),
                                    "checkpoint_sha256": checkpoint_sha256,
                                    "initialization_sha256": initialization_sha256,
                                    "parameter_count": expected_parameter_count,
                                    "optimizer": trace.optimizer,
                                    "locality_method_id": (trace.locality_method_id),
                                    "locality_objective": (trace.locality_objective),
                                    "locality_risk_scale": float(locality_risk_scale),
                                }
                            )
    expected = expected_grid_size(
        seeds=seeds,
        intensities=intensities,
        updates=updates,
        gap_events=gaps,
    )
    if len(rows) != expected:
        raise RuntimeError(f"E23 generated grid is incomplete: {len(rows)} != {expected}")
    ensure_finite_rows(rows)
    digest_groups: dict[tuple[int, str, float, int], set[str]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["demand_family"]),
            float(row["intensity"]),
            int(row["updates"]),
        )
        digest_groups.setdefault(key, set()).add(str(row["base_transaction_digest"]))
    if any(len(values) != 1 for values in digest_groups.values()):
        raise RuntimeError("E23 paired data changed across controller or gap conditions")
    return ProductPosetRunResult(
        rows=rows,
        training_rows=training_rows,
        checkpoint_hashes=checkpoint_hashes,
        runtime=runtime,
    )


def data_manifest_payload(
    config: Mapping[str, Any],
    *,
    phase: str,
    boundary_mode: str | None,
    dependency: Mapping[str, Any] | None,
    locality_method: Mapping[str, Any],
    locality_risk_scale: float,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "outcome_independent": True,
        "controller_axes": list(CONTROLLER_AXES),
        "controller_ids": [controller.controller_id for controller in CANONICAL_CONTROLLERS],
        "demand_families": list(DEMAND_FAMILIES),
        "intensities": [float(value) for value in config["intensities"]],
        "sequence_grid": {
            "updates": [int(value) for value in config["evaluation"]["updates"]],
            "gap_events": [int(value) for value in config["evaluation"]["gap_events"]],
        },
        "seeds": [int(value) for value in config["seeds"]],
        "namespace": str(config["namespace"]),
        "boundary_selection": {
            "rule": "theory_boundary_only_v1",
            "result_independent": True,
            "sets": result_independent_boundary_manifest(),
            "union_controller_ids": sorted(
                {
                    controller_id
                    for values in result_independent_boundary_manifest().values()
                    for controller_id in values
                }
            ),
        },
        "absolute_adequacy": dict(config["adequacy"]),
        "boundary_mode": boundary_mode,
        "dependency": None if dependency is None else dict(dependency),
        "locality_training": {
            "method": dict(locality_method),
            "risk_scale": float(locality_risk_scale),
            "safe_mode_uses_selected_e22_objective": (boundary_mode == "safe_minimality"),
            "capacity_mode_uses_mean_objective": (boundary_mode == "capacity_only"),
        },
        "learned_application": {
            "uses_e18_tensor_contract": True,
            "all_16_controller_projections": True,
            "same_maximal_parameter_surface": True,
            "paired_initialization_data_optimizer": True,
            "application_metric_is_not_theory_floor": True,
            "runtime": None if runtime is None else dict(runtime),
        },
    }


def write_theory_predictions(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    locked_sha256: str,
) -> dict[str, Any]:
    payload = theory_prediction_payload(
        affected_mse_tolerance=float(config["adequacy"]["affected_mse_tolerance"]),
        retention_mse_tolerance=float(config["adequacy"]["retention_margin"]),
        locality_mse_tolerance=float(config["adequacy"]["locality_margin"]),
        intensities=[float(value) for value in config["intensities"]],
        updates=[int(value) for value in config["evaluation"]["updates"]],
        gap_events=[int(value) for value in config["evaluation"]["gap_events"]],
    )
    artifact = {
        "schema_version": 1,
        "locked_before_outcomes": True,
        "locked_prediction_sha256": locked_sha256,
        "payload": payload,
    }
    path = run_dir / "theory_predictions.json"
    write_json(path, artifact)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def write_cell_rows(
    *,
    run_dir: Path,
    cell_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "poset_minimal_demands.jsonl"
    write_jsonl(path, [dict(row) for row in cell_rows])
    return {
        "path": str(path.resolve()),
        "rows": len(cell_rows),
        "sha256": file_sha256(path),
    }


def write_training_rows(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "product_poset_training_runs.jsonl"
    write_jsonl(path, [dict(row) for row in rows])
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "sha256": file_sha256(path),
    }


def results_summary_ko(
    *,
    phase: str,
    run_mode: str,
    status: str,
    boundary_mode: str | None,
    assessment: Mapping[str, Any] | None,
    dependency_reason: str | None,
) -> str:
    lines = [
        f"# E23 {phase} 결과 요약",
        "",
        f"- 실행: **{run_mode} / {status}**",
        f"- boundary mode: `{boundary_mode or 'NONE'}`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "- application metric: `LEARNED_REPEATED_SEQUENCE`",
        "- 16 controller는 동일 E18-compatible maximal parameter surface 사용",
        "- dry-run과 screen은 confirmatory claim에 사용할 수 없음",
    ]
    if dependency_reason is not None:
        lines.append(f"- E22b dependency: `{dependency_reason}`")
    if assessment is not None:
        lines.extend(
            (
                "",
                "## Poset 진단",
                "",
                f"- seed 수: `{int(assessment['seed_count'])}`",
                f"- demand 판정 수: `{int(assessment['demand_count'])}`",
                "- single-axis exact match 최소: "
                f"`{int(assessment['minimum_single_axis_exact_match_count'])}/4`",
                "- pairwise exact match 최소: "
                f"`{int(assessment['minimum_pairwise_exact_match_count'])}/6`",
                "- minimal-set Jaccard 최소: "
                f"`{float(assessment['minimum_minimal_set_jaccard']):.6g}`",
                "- false adequate / inadequate: "
                f"`{int(assessment['total_false_adequate_count'])}` / "
                f"`{int(assessment['total_false_inadequate_count'])}`",
                "- 최소 immediate-predecessor gain: "
                f"`{float(assessment['minimum_immediate_predecessor_gain']):.6g}`",
                "- 최소 incomparable direction gap: "
                f"`{float(assessment['minimum_incomparable_affected_gap']):.6g}`",
                "- maximal simpler-demand degradation 최대: "
                f"`{float(assessment['maximum_maximal_simpler_degradation']):.6g}`",
                f"- capacity gate: `{'PASS' if assessment['capacity_supported'] else 'FAIL'}`",
            )
        )
        if boundary_mode == "safe_minimality":
            lines.append(
                "- safe-locality gate: "
                f"`{'PASS' if assessment['safe_minimality_supported'] else 'FAIL'}`"
            )
    lines.extend(
        (
            "",
            "## 해석 경계",
            "",
            "- learned 4-bit controlled sequence poset의 capacity/minimality만 평가한다.",
            "- oracle address/candidate, explicit demand, verified-bit 경계를 유지한다.",
            "- E22b가 비안전이면 locality claim은 계산·개방하지 않는다.",
            "- semantic/NL/LM/agent/official-backend/runtime claim은 닫혀 있다.",
            "",
        )
    )
    if len(lines) > 55:
        raise RuntimeError("E23 summary exceeds one-page contract")
    return "\n".join(lines)
