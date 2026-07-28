from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file
from catena.data.semantic_transactions_v61 import SemanticNamespaceRegistry
from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    SemanticAnchorSeedMetrics,
)
from catena.eval.semantic_design_repair_r1 import (
    E05A_R1_BOOTSTRAP_SEEDS,
    E05A_R1_SEEDS,
    evaluate_e05a_r1_design,
)
from experiments import e05a_r1_common as common
from experiments.e05a_r1_common import (
    original_e05a_dependency_record,
    validate_frozen_r1_protocol,
    validate_original_e05a_dependency,
)
from experiments.e05a_r1_semantic_design_repair import (
    _access_manifest,
    _static_design_gates,
    _thresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def test_frozen_hashes_and_original_no_go_dependency_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = validate_frozen_r1_protocol()
    assert sha256_file(common.R1_CONFIG_PATH) == common.PINNED_R1_CONFIG_FILE_SHA256
    assert (
        sha256_canonical_json(config)
        == common.PINNED_R1_CONFIG_CANONICAL_SHA256
    )
    assert (
        sha256_file(common.R1_PROTOCOL_PATH)
        == common.PINNED_R1_PROTOCOL_SHA256
    )
    assert (
        sha256_file(common.R1_PROTOCOL_LOCK_PATH)
        == common.PINNED_R1_PROTOCOL_LOCK_SHA256
    )

    original = validate_original_e05a_dependency(ARTIFACT_ROOT)
    dependency = original_e05a_dependency_record(original)
    assert original.report["e05a_design_status"] == "NO_GO"
    assert original.report["full_h5_lite_claim_open"] is False
    assert dependency["e05a_original_status"] == "NO_GO"
    assert dependency["h5_claim_open"] is False
    assert dependency["rows_reused_in_r1_inference"] == 0
    assert (
        dependency["artifact_freeze_sha256"]
        == common.PINNED_ORIGINAL_E05A_FREEZE_SHA256
    )
    assert (
        dependency["claim_status_sha256"]
        == common.PINNED_ORIGINAL_E05A_CLAIM_SHA256
    )

    monkeypatch.setattr(
        common,
        "PINNED_R1_CONFIG_FILE_SHA256",
        "0" * 64,
    )
    with pytest.raises(Exception, match="hash mismatch"):
        validate_frozen_r1_protocol()


def test_config_bootstrap_keys_thresholds_and_access_manifest_match_evaluator() -> None:
    config = validate_frozen_r1_protocol()
    configured_bootstrap = {
        str(key): int(value)
        for key, value in config["statistics"]["bootstrap_seeds"].items()
    }
    assert configured_bootstrap == E05A_R1_BOOTSTRAP_SEEDS
    assert len(set(configured_bootstrap.values())) == len(configured_bootstrap)

    thresholds = _thresholds(config)
    assert thresholds.positive_effect_sesoi == pytest.approx(0.001)
    assert thresholds.minimum_oracle_headroom == pytest.approx(0.001)
    assert thresholds.equivalence_margin == pytest.approx(0.0005)
    assert thresholds.retention_noninferiority_margin == pytest.approx(0.0005)
    assert thresholds.oracle_absolute_ceiling == pytest.approx(1e-8)
    assert thresholds.bootstrap_samples == 5000

    access = _access_manifest(config)
    assert access["forbidden_access_test_passed"] is True
    assert access["gate_record_fields_match_frozen_set"] is True
    assert access["forbidden_field_overlap"] == []
    assert access["encoder_parameters"] == [
        "self",
        "record",
        "mask_semantics",
    ]
    assert access["state_read_in_gate_encoder"] is False
    assert access["address_in_gate_encoder"] is False
    assert access["incoming_value_in_gate_encoder"] is False
    assert access["state_read_used_only_for_erase_candidate"] is True
    assert access["learned_gate_supervision"] is False
    assert access["target_state_weight"] == 0.0


@pytest.mark.parametrize(
    (
        "dry_run",
        "seed_count",
        "train_count_per_cell",
        "validation_count_per_cell",
        "registry_rows",
        "pairing_rows",
        "balance_rows",
        "metric_rows",
    ),
    [
        (True, 1, 66, 44, 1_320, 528, 24, 4_224),
        (False, 8, 330, 154, 46_464, 14_784, 192, 118_272),
    ],
)
def test_dry_and_main_namespaces_have_only_r1_rows_and_exact_counts(
    dry_run: bool,
    seed_count: int,
    train_count_per_cell: int,
    validation_count_per_cell: int,
    registry_rows: int,
    pairing_rows: int,
    balance_rows: int,
    metric_rows: int,
) -> None:
    config = validate_frozen_r1_protocol()
    registry = SemanticNamespaceRegistry.from_config(
        config["namespace"],
        dry_run=dry_run,
    )
    assert set(registry.names) == {"r1_train", "r1_validation"}
    assert registry.integer_root == 6_000_000_000_000

    cells_per_seed = (
        len(config["r1"]["operations"])
        * len(config["r1"]["seen_domains"])
        * len(config["r1"]["seen_templates"])
    )
    expected_train = seed_count * cells_per_seed * train_count_per_cell
    expected_validation = (
        seed_count * cells_per_seed * validation_count_per_cell
    )
    assert expected_train + expected_validation == registry_rows
    assert expected_validation == pairing_rows
    assert seed_count * cells_per_seed * 2 == balance_rows
    assert expected_validation * (3 + len(CONTROL_NAMES)) == metric_rows
    assert train_count_per_cell % 11 == 0
    assert validation_count_per_cell % 11 == 0
    assert cells_per_seed * train_count_per_cell < registry.seed_stride
    assert cells_per_seed * validation_count_per_cell < registry.seed_stride

    largest_numeric_seed = registry.numeric_seed(
        "r1_validation",
        seed_slot=seed_count - 1,
        index=cells_per_seed * validation_count_per_cell - 1,
    )
    assert largest_numeric_seed < int(
        config["namespace"]["audit_pool_root_reserved_not_opened"]
    )
    assert largest_numeric_seed < int(
        config["namespace"]["e05b_r1_root_reserved_not_opened"]
    )


def _passing_static_inputs() -> tuple[
    dict[str, object],
    SemanticNamespaceRegistry,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    config = validate_frozen_r1_protocol()
    registry = SemanticNamespaceRegistry.from_config(
        config["namespace"],
        dry_run=False,
    )
    rows = [
        {
            "namespace_name": "r1_train",
            "split": "r1_train",
            "numeric_seed": registry.numeric_seed(
                "r1_train", seed_slot=0, index=0
            ),
            "checkpoint_seed": E05A_R1_SEEDS[0],
            "operation_private": "add",
        },
        {
            "namespace_name": "r1_validation",
            "split": "r1_validation",
            "numeric_seed": registry.numeric_seed(
                "r1_validation", seed_slot=0, index=0
            ),
            "checkpoint_seed": E05A_R1_SEEDS[0],
            "operation_private": "invalidate",
        },
    ]
    raw_rows = [{"operation": "add"}, {"operation": "invalidate"}]
    budgets = [
        {
            "parameter_count": 10,
            "dense_multiply_adds_per_example": 20,
            "common_initialization": True,
            "common_schedule": True,
        }
    ]
    return config, registry, rows, raw_rows, budgets


def _static_gates(
    *,
    registry_rows: list[dict[str, object]] | None = None,
    raw_rows: list[dict[str, object]] | None = None,
    access: bool = True,
    balance: bool = True,
    mismatch: float = 0.0,
    budgets: list[dict[str, object]] | None = None,
) -> dict[str, bool]:
    config, registry, default_rows, default_raw, default_budgets = (
        _passing_static_inputs()
    )
    return _static_design_gates(
        config=config,
        access_manifest={"forbidden_access_test_passed": access},
        namespace_registry=registry,
        registry_rows=default_rows if registry_rows is None else registry_rows,
        raw_rows=default_raw if raw_rows is None else raw_rows,
        balance_passed=balance,
        max_norm_mismatch=mismatch,
        budget_records=default_budgets if budgets is None else budgets,
    )


def test_static_gate_polarity_and_materialization_boundaries_are_data_derived() -> None:
    assert all(_static_gates().values())
    assert _static_gates(access=False)["forbidden_access"] is False
    assert _static_gates(balance=False)["write_false_strata_balanced"] is False
    assert (
        _static_gates(mismatch=1.000001e-6)["wrong_address_norm_matched"]
        is False
    )
    assert (
        _static_gates(budgets=[])["parameter_and_compute_budget_match"]
        is False
    )
    assert (
        _static_gates(raw_rows=[{"operation": "supersede"}])[
            "supersede_absent"
        ]
        is False
    )

    config, _, rows, _, _ = _passing_static_inputs()
    reused = [dict(row) for row in rows]
    reused[0]["checkpoint_seed"] = 101
    assert (
        _static_gates(registry_rows=reused)["original_e05a_rows_not_reused"]
        is False
    )

    audit_materialized = [dict(row) for row in rows]
    audit_materialized[1]["numeric_seed"] = int(
        config["namespace"]["audit_pool_root_reserved_not_opened"]
    )
    assert (
        _static_gates(registry_rows=audit_materialized)[
            "no_audit_or_e05b_rows_generated"
        ]
        is False
    )

    downstream_namespace = [dict(row) for row in rows]
    downstream_namespace[1]["namespace_name"] = "e05b_r1_primary"
    downstream_namespace[1]["split"] = "e05b_r1_primary"
    gates = _static_gates(registry_rows=downstream_namespace)
    assert gates["namespace_only_r1_train_validation"] is False
    assert gates["no_audit_or_e05b_rows_generated"] is False


def _passing_seed_metrics(seed: int) -> SemanticAnchorSeedMetrics:
    operations = np.asarray(["preserve", "add", "invalidate"])
    factorized = np.asarray([0.0001, 0.0002, 0.0002])
    shared = np.asarray([0.0001, 0.0022, 0.0022])
    oracle = np.zeros(3, dtype=np.float64)
    affected: dict[str, np.ndarray] = {
        "factorized": factorized,
        "shared": shared,
        "oracle_demand": oracle,
    }
    for control in CONTROL_NAMES:
        affected[control] = factorized + np.asarray([0.0, 0.003, 0.003])
    return SemanticAnchorSeedMetrics(
        episode_ids=np.asarray(
            [f"r1-contract-{seed}-{index}" for index in range(3)]
        ),
        domains=np.asarray(["api", "api", "api"]),
        templates=np.asarray(["record", "record", "record"]),
        operations=operations,
        affected=affected,
        retention={
            "factorized": np.full(3, 0.0001),
            "shared": np.full(3, 0.0001),
            "oracle_demand": oracle,
        },
    )


def test_frozen_statistical_and_static_conjunction_has_a_reachable_go_state() -> None:
    config = validate_frozen_r1_protocol()
    statistical = evaluate_e05a_r1_design(
        {
            seed: _passing_seed_metrics(seed)
            for seed in E05A_R1_SEEDS
        },
        fixed_seeds=E05A_R1_SEEDS,
        thresholds=_thresholds(config),
        bootstrap_seeds={
            str(key): int(value)
            for key, value in config["statistics"]["bootstrap_seeds"].items()
        },
    )
    static = _static_gates()

    assert statistical["go"] is True
    assert statistical["diagnostic_reasons"] == []
    assert all(static.values())
    assert statistical["go"] and all(static.values())


def test_runner_source_has_no_audit_pool_or_e05b_registry_generator() -> None:
    source = (
        REPO_ROOT / "experiments/e05a_r1_semantic_design_repair.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "naturalization_audit_items",
        "_generate_e05b",
        "build_e05b",
        "e05a_r1_semantic_audit_pool_lock import",
        "e05b_r1_registry_lock import",
    ):
        assert forbidden not in source
