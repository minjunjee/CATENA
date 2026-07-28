from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from catena.core.config import load_config
from catena.core.io import file_sha256
from catena.post_e21.locality_data import (
    LocalityMethod,
    LocalityObjective,
    method_by_id,
    parse_locality_methods,
)
from catena.post_e21.locality_eval import (
    assess_locality_confirmatory,
    build_active_cell_rows,
    compute_locality_seed_summaries,
    select_locality_method,
    validate_paired_metric_grid,
)
from catena.post_e21.locality_models import (
    LocalityStructuredSequenceController,
    ProtectedLocalityDiagnosticController,
)
from catena.post_e21.locality_protocol import (
    EXPECTED_THRESHOLD_KEYS,
    load_parent_threshold_contract,
    require_temp_dry_root,
)
from catena.post_e21.locality_runner import (
    run_locality_method_grid,
    runtime_locality_config,
)
from catena.post_e21.locality_training import (
    locality_retention_risk,
    normalized_smooth_max,
    upper_tail_mean,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parent_thresholds_are_loaded_dynamically_from_e21_lock() -> None:
    contract = load_parent_threshold_contract(repo_root=REPO_ROOT)
    assert set(contract.thresholds) == EXPECTED_THRESHOLD_KEYS
    assert contract.sha256 == file_sha256(contract.path)
    assert contract.thresholds["selective_gain"] == pytest.approx(0.001)
    for config_name in (
        "configs/e22a_locality_method_selection.yaml",
        "configs/e22b_active_path_locality.yaml",
    ):
        config = load_config(REPO_ROOT / config_name)
        assert "thresholds" not in config
        assert "claim_gate" not in config
        entrypoint = REPO_ROOT / "experiments" / f"{Path(config_name).stem}.py"
        assert '"maximum_nontarget_degradation"' in entrypoint.read_text(encoding="utf-8")
    e22b = load_config(REPO_ROOT / "configs/e22b_active_path_locality.yaml")
    assert set(e22b["registered_statuses"]) == {
        "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
        "CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED",
        "OVERREGULARIZED_LOCALITY_TRADEOFF",
        "NOT_SUPPORTED",
    }


def test_registered_method_grid_is_complete_and_protected_is_not_selectable() -> None:
    config = load_config(REPO_ROOT / "configs/e22a_locality_method_selection.yaml")
    methods = parse_locality_methods(config["methods"])
    assert len(methods) == 11
    assert {method.objective for method in methods} == set(LocalityObjective)
    protected = [
        method for method in methods if method.objective is LocalityObjective.PROTECTED_DIAGNOSTIC
    ]
    assert len(protected) == 1
    assert protected[0].selection_eligible is False
    assert sum(method.baseline for method in methods) == 1


def test_tail_and_smoothmax_risks_have_expected_order() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 4.0])
    assert float(upper_tail_mean(values, 0.25)) == pytest.approx(4.0)
    assert float(upper_tail_mean(values, 0.50)) == pytest.approx(3.0)
    smooth = normalized_smooth_max(
        values,
        normalized_temperature=0.5,
        risk_scale=1.0,
    )
    assert float(values.mean()) < float(smooth) < float(values.max())
    cvar = LocalityMethod(
        method_id="cvar",
        objective=LocalityObjective.CVAR,
        selection_eligible=True,
        baseline=False,
        tail_fraction=0.5,
    )
    assert float(locality_retention_risk(values, method=cvar, risk_scale=1.0)) == pytest.approx(3.0)


def test_sparse_route_is_hard_topk_forward_and_soft_backward() -> None:
    from catena.models.structured_sequence_localization import (
        StructuredSequenceFreedom,
    )

    controller = LocalityStructuredSequenceController(
        freedom=StructuredSequenceFreedom.FULL,
        slots=8,
        identifier_dim=8,
        value_dim=8,
        hidden_dim=16,
        address_temperature=0.2,
        active_fraction=0.25,
    )
    logits = torch.linspace(-1.0, 1.0, 8, requires_grad=True)
    weights = torch.softmax(logits, dim=-1).unsqueeze(0)
    routed, mask = controller._apply_route(weights)
    assert int(mask.sum()) == 2
    assert int((routed.detach() > 0.0).sum()) == 2
    assert float(routed.detach().sum()) == pytest.approx(1.0)
    routed.square().sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_protected_diagnostic_projects_every_non_target_readout_direction() -> None:
    from catena.data.structured_sequence_localization import (
        StructuredTransferDemand,
        generate_structured_sequence_transfer_batch,
        make_structured_identifier_codebook,
    )
    from catena.models.structured_sequence_localization import (
        StructuredSequenceFreedom as Freedom,
    )

    slots = 8
    controller = ProtectedLocalityDiagnosticController(
        freedom=Freedom.FULL,
        slots=slots,
        identifier_dim=slots,
        value_dim=slots,
        hidden_dim=16,
        address_temperature=0.2,
    )
    batch = generate_structured_sequence_transfer_batch(
        family=StructuredTransferDemand.ADDRESS_DECOUPLING,
        batch_size=2,
        slots=slots,
        value_dim=slots,
        updates=1,
        gap_events=2,
        state_scale=0.5,
        identifier_codebook=make_structured_identifier_codebook(
            slots=slots,
            code_dim=slots,
            seed=7,
        ),
        seed=11,
        base_namespace="e22-projection-test-base",
        distractor_namespace="e22-projection-test-distractor",
        device=torch.device("cpu"),
    )
    sequence_length = int(batch.update_mask.shape[1])
    update_delta = torch.ones(2, slots, slots)
    route_mask = torch.ones(2, 2, slots, dtype=torch.bool)
    for time_index in range(sequence_length):
        projected, applied = controller._project_event_update(
            batch=batch,
            time_index=time_index,
            update_delta=update_delta,
            route_mask=route_mask,
        )
        if bool(batch.update_mask[:, time_index].any()):
            allowed = (
                torch.nn.functional.one_hot(
                    batch.erase_addresses[:, time_index],
                    num_classes=slots,
                ).bool()
                | torch.nn.functional.one_hot(
                    batch.write_addresses[:, time_index],
                    num_classes=slots,
                ).bool()
            )
            assert torch.equal(projected.ne(0).any(dim=-1), allowed)
            assert torch.equal(applied.any(dim=1), allowed)
        else:
            assert int(projected.count_nonzero()) == 0
            assert int(applied.count_nonzero()) == 0


def _synthetic_rows(
    *,
    seeds: list[int],
    method_ids: list[str],
) -> list[dict[str, Any]]:
    conditions = [
        "A_oracle_address_oracle_candidate",
        "B_learned_address_oracle_candidate",
        "C_oracle_address_state_read_candidate",
        "D_learned_address_state_read_candidate",
    ]
    families = [
        "magnitude_factorization",
        "value_granularity",
        "address_decoupling",
        "state_conditioning",
    ]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method_id in method_ids:
            locality_harm = {
                "mean_retention": 0.0004,
                "cvar_010": 0.0001,
                "protected": 0.00005,
            }[method_id]
            support = {"mean_retention": 4.0, "cvar_010": 2.0, "protected": 1.0}[method_id]
            for variant in ("base", "separate_address", "state_aware", "full"):
                has_separate = variant in {"separate_address", "full"}
                has_state = variant in {"state_aware", "full"}
                for condition in conditions:
                    for family in families:
                        for updates in (1, 8):
                            for gap in (0, 2048):
                                affected = 0.0002
                                target = False
                                if (
                                    condition == "B_learned_address_oracle_candidate"
                                    and family == "address_decoupling"
                                ):
                                    target = True
                                    affected = 0.0002 if has_separate else 0.002
                                elif (
                                    condition == "C_oracle_address_state_read_candidate"
                                    and family == "state_conditioning"
                                ):
                                    target = True
                                    affected = 0.0002 if has_state else 0.002
                                elif (
                                    condition == "D_learned_address_state_read_candidate"
                                    and family == "address_decoupling"
                                ):
                                    target = True
                                    affected = 0.0002 if variant == "full" else 0.002
                                if not target:
                                    active = (
                                        condition
                                        in {
                                            "B_learned_address_oracle_candidate",
                                            "D_learned_address_state_read_candidate",
                                        }
                                        and has_separate
                                    ) or (
                                        condition
                                        in {
                                            "C_oracle_address_state_read_candidate",
                                            "D_learned_address_state_read_candidate",
                                        }
                                        and has_state
                                    )
                                    if active:
                                        affected += locality_harm
                                rows.append(
                                    {
                                        "seed": seed,
                                        "method_id": method_id,
                                        "variant": variant,
                                        "condition": condition,
                                        "demand_family": family,
                                        "updates": updates,
                                        "gap_events": gap,
                                        "base_transaction_digest": (
                                            f"{seed}:{condition}:{family}:{updates}:{gap}"
                                        ),
                                        "checkpoint_sha256": "a" * 64,
                                        "affected_mse": affected,
                                        "retention_mse": 0.0001,
                                        "address_accuracy": 1.0,
                                        "candidate_recovery_mse": 0.0001,
                                        "verified_activity_mean": 0.99,
                                        "distractor_activity_mean": 0.01,
                                        "raw_route_mask_sha256": "b" * 64,
                                        "active_route_mask_sha256": "c" * 64,
                                        "raw_route_support_size": support,
                                        "raw_route_support_fraction": support / 64.0,
                                        "active_route_support_size": support,
                                        "active_route_support_fraction": support / 64.0,
                                        "active_event_fraction": 1.0,
                                        "post_mask_update_rms": 0.1,
                                        "predicted_update_rms": 0.1,
                                        "target_update_rms": 0.1,
                                        "update_compute_units": support * 32.0,
                                    }
                                )
    return rows


def test_grid_selection_and_confirmatory_assessment() -> None:
    seeds = [1, 2, 3]
    method_ids = ["mean_retention", "cvar_010", "protected"]
    rows = _synthetic_rows(seeds=seeds, method_ids=method_ids)
    validate_paired_metric_grid(
        rows,
        seeds=seeds,
        methods=method_ids,
        variants=["base", "separate_address", "state_aware", "full"],
        conditions=[
            "A_oracle_address_oracle_candidate",
            "B_learned_address_oracle_candidate",
            "C_oracle_address_state_read_candidate",
            "D_learned_address_state_read_candidate",
        ],
        demand_families=[
            "magnitude_factorization",
            "value_granularity",
            "address_decoupling",
            "state_conditioning",
        ],
        updates_grid=[1, 8],
        gaps_grid=[0, 2048],
    )
    summaries = compute_locality_seed_summaries(
        rows,
        seeds=seeds,
        method_ids=method_ids,
        updates_grid=[1, 8],
        gaps_grid=[0, 2048],
        demand_families=[
            "magnitude_factorization",
            "value_granularity",
            "address_decoupling",
            "state_conditioning",
        ],
        stress_updates=8,
        stress_gap_events=2048,
    )
    thresholds = load_parent_threshold_contract(repo_root=REPO_ROOT).thresholds
    methods = [
        LocalityMethod(
            method_id="mean_retention",
            objective=LocalityObjective.MEAN,
            selection_eligible=False,
            baseline=True,
        ),
        LocalityMethod(
            method_id="cvar_010",
            objective=LocalityObjective.CVAR,
            selection_eligible=True,
            baseline=False,
            tail_fraction=0.1,
        ),
        LocalityMethod(
            method_id="protected",
            objective=LocalityObjective.PROTECTED_DIAGNOSTIC,
            selection_eligible=False,
            baseline=False,
        ),
    ]
    selection = select_locality_method(
        summaries,
        methods=methods,
        thresholds=thresholds,
        dry_run=False,
    )
    assert selection["status"] == "SELECTED"
    assert selection["selected_method_id"] == "cvar_010"

    # E22a selection was preregistered on recovery + primary retention.
    # Absolute capable/address/candidate quantities are diagnostics here and
    # become hard guardrails only in E22b.
    diagnostic_capacity_failure = deepcopy(summaries)
    for row in diagnostic_capacity_failure:
        if row["method_id"] == "cvar_010":
            row["maximum_capable_affected_mse"] = 0.01
    selection_with_capacity_diagnostic = select_locality_method(
        diagnostic_capacity_failure,
        methods=methods,
        thresholds=thresholds,
        dry_run=False,
    )
    assert selection_with_capacity_diagnostic["selected_method_id"] == "cvar_010"
    selected_score = next(
        row
        for row in selection_with_capacity_diagnostic["method_summaries"]
        if row["method_id"] == "cvar_010"
    )
    assert selected_score["capacity_gate_passed"] is False
    assert selected_score["hard_gate_passed"] is True

    diagnostic_direction_failure = deepcopy(summaries)
    cvar_rows = [row for row in diagnostic_direction_failure if row["method_id"] == "cvar_010"]
    cvar_rows[0]["b_separate_address_gain"] = -0.0002
    cvar_rows[0]["b_stress_gain"] = -0.0002
    direction_diagnostic_selection = select_locality_method(
        diagnostic_direction_failure,
        methods=methods,
        thresholds=thresholds,
        dry_run=False,
    )
    assert direction_diagnostic_selection["selected_method_id"] == "cvar_010"
    direction_score = next(
        row
        for row in direction_diagnostic_selection["method_summaries"]
        if row["method_id"] == "cvar_010"
    )
    assert direction_score["recovery_gate_passed"] is True
    assert direction_score["seed_and_stress_direction_diagnostic_passed"] is False
    assert direction_score["hard_gate_passed"] is True

    confirm_seeds = [11, 12, 13, 14, 15, 16, 17, 18]
    confirm_rows = _synthetic_rows(
        seeds=confirm_seeds,
        method_ids=["mean_retention", "cvar_010"],
    )
    confirm_summaries = compute_locality_seed_summaries(
        confirm_rows,
        seeds=confirm_seeds,
        method_ids=["mean_retention", "cvar_010"],
        updates_grid=[1, 8],
        gaps_grid=[0, 2048],
        demand_families=[
            "magnitude_factorization",
            "value_granularity",
            "address_decoupling",
            "state_conditioning",
        ],
        stress_updates=8,
        stress_gap_events=2048,
    )
    assessment = assess_locality_confirmatory(
        confirm_summaries,
        selected_method_id="cvar_010",
        baseline_method_id="mean_retention",
        required_seeds=confirm_seeds,
        thresholds=thresholds,
        dry_run=False,
    )
    assert assessment["status"] == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
    assert assessment["selected_vs_mean_locality"]["sign_flip_p"] == pytest.approx(1.0 / 256.0)
    assert build_active_cell_rows(confirm_rows)

    capacity_only = deepcopy(confirm_summaries)
    for row in capacity_only:
        row["maximum_nontarget_degradation"] = 0.0008 if row["method_id"] == "cvar_010" else 0.0012
    assert (
        assess_locality_confirmatory(
            capacity_only,
            selected_method_id="cvar_010",
            baseline_method_id="mean_retention",
            required_seeds=confirm_seeds,
            thresholds=thresholds,
            dry_run=False,
        )["status"]
        == "CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED"
    )

    locality_only = deepcopy(confirm_summaries)
    for row in locality_only:
        if row["method_id"] == "cvar_010":
            for key in (
                "b_separate_address_gain",
                "c_state_read_gain",
                "d_full_only_gain",
            ):
                row[key] = 0.0002
    assert (
        assess_locality_confirmatory(
            locality_only,
            selected_method_id="cvar_010",
            baseline_method_id="mean_retention",
            required_seeds=confirm_seeds,
            thresholds=thresholds,
            dry_run=False,
        )["status"]
        == "OVERREGULARIZED_LOCALITY_TRADEOFF"
    )

    neither = deepcopy(locality_only)
    for row in neither:
        if row["method_id"] == "cvar_010":
            row["maximum_nontarget_degradation"] = 0.0008
    assert (
        assess_locality_confirmatory(
            neither,
            selected_method_id="cvar_010",
            baseline_method_id="mean_retention",
            required_seeds=confirm_seeds,
            thresholds=thresholds,
            dry_run=False,
        )["status"]
        == "NOT_SUPPORTED"
    )


def test_dry_run_root_is_restricted_to_tmp(tmp_path: Path) -> None:
    assert require_temp_dry_root(tmp_path) == tmp_path.resolve()
    with pytest.raises(Exception, match="below /tmp"):
        require_temp_dry_root(REPO_ROOT / "artifacts")


def test_tiny_training_and_evaluation_grid_is_paired(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs/e22a_locality_method_selection.yaml")
    runtime = runtime_locality_config(config, dry_run=True)
    registered = parse_locality_methods(config["methods"])
    methods = [
        method_by_id(registered, "mean_retention"),
        method_by_id(registered, "protected_projection_diagnostic"),
    ]
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    rows, hashes, metadata = run_locality_method_grid(
        runtime=runtime,
        methods=methods,
        seeds=[7],
        run_dir=tmp_path,
        device=torch.device("cpu"),
        parent_lock_sha256=parent.sha256,
        protocol_lock_sha256="0" * 64,
        risk_scale=float(parent.thresholds["maximum_nontarget_degradation"]),
    )
    assert len(rows) == 256
    assert len(hashes) == 8
    assert len(metadata["initialization_hashes"]) == 1
    validate_paired_metric_grid(
        rows,
        seeds=[7],
        methods=[method.method_id for method in methods],
        variants=list(runtime["model"]["variants"]),
        conditions=list(runtime["conditions"]),
        demand_families=list(runtime["demand_families"]),
        updates_grid=list(runtime["evaluation"]["updates"]),
        gaps_grid=list(runtime["evaluation"]["gap_events"]),
    )
