from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catena.core.provenance_v61 import (
    ProvenanceValidationError,
    sha256_file,
    write_json_strict,
    write_jsonl_strict,
)
from catena.core.schema import Operation
from catena.data.semantic_controls_r1 import build_control_pairing_registry_r1
from catena.data.semantic_controls_v61 import (
    ControlPairingRegistry,
    SemanticControl,
)
from catena.data.semantic_registry_v61 import semantic_example_registry_row
from catena.data.semantic_transactions_r1 import (
    R1_WRITE_FALSE_STRATA,
    R1_WRITE_TRUE_STRATUM,
    build_balanced_semantic_examples_r1,
    classify_r1_write_stratum,
)
from catena.data.semantic_transactions_v61 import (
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    assert_disjoint_semantic_vocabularies,
    semantic_vocabularies,
)
from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    SemanticAnchorSeedMetrics,
)
from catena.eval.semantic_design_repair_r1 import (
    E05aR1Thresholds,
    evaluate_e05a_r1_design,
)
from catena.models.semantic_controllers_v61 import MatchedSemanticControllerV61
from catena.models.semantic_encoder_r1 import (
    R1_DAY_SCALE,
    R1_FEATURE_DIM,
    R1_FEATURE_NAMES,
    R1_VERSION_SCALE,
    RelationalSemanticEncoderR1,
    RelationalSemanticRecord,
)
from catena.training.semantic_probe_r1 import (
    evaluate_semantic_model_r1,
    train_matched_semantic_pair_r1,
)
from catena.training.semantic_probe_v61 import (
    BatchedVisibleUpdateContext,
    SemanticTrainingConfigV61,
    seed_metrics_from_condition_arrays,
)
from experiments.common import build_parser
from experiments.e05a_r1_common import (
    PINNED_R1_CONFIG_CANONICAL_SHA256,
    PINNED_R1_CONFIG_FILE_SHA256,
    PINNED_R1_PROTOCOL_LOCK_SHA256,
    PINNED_R1_PROTOCOL_SHA256,
    R1_CONFIG_PATH,
    original_e05a_dependency_record,
    validate_frozen_r1_protocol,
    validate_original_e05a_dependency,
    validate_r1_config_path,
)
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e05a_r1_semantic_design_repair"
DEFAULT_CONFIG = "configs/e05a_r1_semantic_design_repair.yaml"

_CONDITION_TO_CONTROL = {
    "shuffled_fields": SemanticControl.SHUFFLED_FIELDS,
    "wrong_address": SemanticControl.WRONG_ADDRESS,
    "transaction_only": SemanticControl.TRANSACTION_ONLY,
    "state_only": SemanticControl.STATE_ONLY,
    "wrong_semantics": SemanticControl.WRONG_SEMANTICS,
}


def _operation_values(values: Sequence[object]) -> tuple[Operation, ...]:
    return tuple(Operation(str(value)) for value in values)


def _memory_spec(config: Mapping[str, Any]) -> SemanticMemorySpec:
    memory = config["memory"]
    if memory["orthonormal_keys"] is not True:
        raise ProvenanceValidationError("E05a-R1 requires orthonormal keys.")
    if memory["oracle_address"] is not True:
        raise ProvenanceValidationError("E05a-R1 requires the oracle address.")
    if memory["candidate_source"] != "visible_state_read_at_visible_address":
        raise ProvenanceValidationError("E05a-R1 candidate source changed.")
    if memory["gate_receives_state_read"] is not False:
        raise ProvenanceValidationError("E05a-R1 gate must not receive state read.")
    if memory["dtype"] != "float32":
        raise ProvenanceValidationError("E05a-R1 frozen dtype is float32.")
    return SemanticMemorySpec(
        num_associations=int(memory["num_associations"]),
        key_dim=int(memory["key_dim"]),
        value_dim=int(memory["value_dim"]),
        dtype=torch.float32,
    )


def _training_config(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> SemanticTrainingConfigV61:
    training = config["training"]
    return SemanticTrainingConfigV61(
        steps=int(training["dry_steps"] if dry_run else training["steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        affected_read_weight=float(training["affected_read_weight"]),
        unaffected_retention_weight=float(
            training["unaffected_retention_weight"]
        ),
        target_state_weight=float(training["target_state_weight"]),
    )


def _thresholds(config: Mapping[str, Any]) -> E05aR1Thresholds:
    statistics = config["statistics"]
    return E05aR1Thresholds(
        positive_effect_sesoi=float(statistics["positive_effect_sesoi"]),
        minimum_oracle_headroom=float(statistics["positive_effect_sesoi"]),
        equivalence_margin=float(statistics["equivalence_margin"]),
        retention_noninferiority_margin=float(
            statistics["retention_noninferiority_margin"]
        ),
        oracle_absolute_ceiling=float(statistics["oracle_absolute_ceiling"]),
        alpha=float(statistics["alpha"]),
        bootstrap_samples=int(statistics["bootstrap_samples"]),
        bootstrap_confidence=float(statistics["bootstrap_confidence"]),
    )


def _descriptor(path: Path, *, rows: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "filename": path.name,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _attach_condition(
    rows: Sequence[dict[str, object]],
    *,
    seed: int,
    condition: str,
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "checkpoint_seed": seed,
            "split": "r1_validation",
            "condition": condition,
            **row,
        }
        for row in rows
    ]


def _operation_summary(
    examples: Sequence[SemanticExample],
    rows: Sequence[Mapping[str, object]],
    affected: np.ndarray,
    retention: np.ndarray,
) -> dict[str, object]:
    operations = np.asarray([example.operation.value for example in examples])
    result: dict[str, object] = {
        "affected_read_mse": float(affected.mean()),
        "unaffected_retention_mse": float(retention.mean()),
        "by_operation": {},
    }
    by_operation: dict[str, object] = {}
    for operation in ("preserve", "add", "invalidate"):
        mask = operations == operation
        selected_rows = [
            row for row, keep in zip(rows, mask.tolist(), strict=True) if keep
        ]
        by_operation[operation] = {
            "rows": int(mask.sum()),
            "affected_read_mse": float(affected[mask].mean()),
            "unaffected_retention_mse": float(retention[mask].mean()),
            "mean_applied_erase": float(
                np.mean([float(row["applied_erase"]) for row in selected_rows])
            ),
            "mean_applied_write": float(
                np.mean([float(row["applied_write"]) for row in selected_rows])
            ),
        }
    result["by_operation"] = by_operation
    return result


def _evaluate_seed(
    *,
    seed: int,
    validation: Sequence[SemanticExample],
    encoder: RelationalSemanticEncoderR1,
    pairing_registry: ControlPairingRegistry,
    factorized: MatchedSemanticControllerV61,
    shared: MatchedSemanticControllerV61,
    device: torch.device,
) -> tuple[
    SemanticAnchorSeedMetrics,
    list[dict[str, object]],
    dict[str, object],
]:
    raw_rows: list[dict[str, object]] = []
    affected: dict[str, np.ndarray] = {}
    retention: dict[str, np.ndarray] = {}
    summaries: dict[str, object] = {}

    for name, model in (("factorized", factorized), ("shared", shared)):
        rows, affected_values, retention_values = evaluate_semantic_model_r1(
            model,
            validation,
            encoder=encoder,
            control=SemanticControl.FULL,
            pairing_registry=None,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(_attach_condition(rows, seed=seed, condition=name))
        affected[name] = affected_values
        retention[name] = retention_values
        summaries[name] = _operation_summary(
            validation,
            rows,
            affected_values,
            retention_values,
        )

    oracle_rows, oracle_affected, oracle_retention = evaluate_semantic_model_r1(
        None,
        validation,
        encoder=encoder,
        control=SemanticControl.FULL,
        pairing_registry=None,
        oracle_demand=True,
        device=device,
    )
    raw_rows.extend(
        _attach_condition(
            oracle_rows,
            seed=seed,
            condition="oracle_demand",
        )
    )
    affected["oracle_demand"] = oracle_affected
    retention["oracle_demand"] = oracle_retention
    summaries["oracle_demand"] = _operation_summary(
        validation,
        oracle_rows,
        oracle_affected,
        oracle_retention,
    )

    for condition in CONTROL_NAMES:
        rows, affected_values, retention_values = evaluate_semantic_model_r1(
            factorized,
            validation,
            encoder=encoder,
            control=_CONDITION_TO_CONTROL[condition],
            pairing_registry=pairing_registry,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(
            _attach_condition(rows, seed=seed, condition=condition)
        )
        affected[condition] = affected_values
        summaries[condition] = _operation_summary(
            validation,
            rows,
            affected_values,
            retention_values,
        )

    return (
        seed_metrics_from_condition_arrays(
            validation,
            affected=affected,
            retention=retention,
        ),
        raw_rows,
        summaries,
    )


def _equal_operation_mean(
    examples: Sequence[SemanticExample],
    values: np.ndarray,
    operations: tuple[str, ...],
) -> float:
    labels = np.asarray([example.operation.value for example in examples])
    return float(
        np.mean([float(values[labels == operation].mean()) for operation in operations])
    )


def _seed_effect_row(
    *,
    seed: int,
    validation: Sequence[SemanticExample],
    metrics: SemanticAnchorSeedMetrics,
    budget: Mapping[str, object],
) -> dict[str, object]:
    affected = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in metrics.affected.items()
    }
    asymmetric = ("add", "invalidate")
    factorized_ai = _equal_operation_mean(
        validation,
        affected["factorized"],
        asymmetric,
    )
    shared_ai = _equal_operation_mean(
        validation,
        affected["shared"],
        asymmetric,
    )
    oracle_ai = _equal_operation_mean(
        validation,
        affected["oracle_demand"],
        asymmetric,
    )
    return {
        "schema_version": 1,
        "checkpoint_seed": seed,
        **budget,
        "factorized_ai_affected_mse": factorized_ai,
        "shared_ai_affected_mse": shared_ai,
        "oracle_ai_affected_mse": oracle_ai,
        "shared_minus_factorized_ai_gain": shared_ai - factorized_ai,
        **{
            f"{condition}_minus_factorized_ai_degradation": (
                _equal_operation_mean(
                    validation,
                    affected[condition],
                    asymmetric,
                )
                - factorized_ai
            )
            for condition in CONTROL_NAMES
        },
    }


def _access_manifest(config: Mapping[str, Any]) -> dict[str, object]:
    model_fields = tuple(RelationalSemanticRecord.__annotations__)
    expected_fields = tuple(
        str(value) for value in config["semantic_schema"]["gate_encoder_fields"]
    )
    forbidden = set(
        str(value)
        for value in config["semantic_schema"]["gate_forbidden_sources"]
    )
    encoder_parameters = tuple(
        inspect.signature(RelationalSemanticEncoderR1.encode).parameters
    )
    model_forward_parameters = tuple(
        inspect.signature(MatchedSemanticControllerV61.forward).parameters
    )
    context_fields = tuple(BatchedVisibleUpdateContext.__dataclass_fields__)
    safe = bool(
        set(model_fields) == set(expected_fields)
        and not (set(model_fields) & forbidden)
        and encoder_parameters == ("self", "record", "mask_semantics")
        and model_forward_parameters == ("self", "features")
        and R1_FEATURE_DIM == 6
        and float(config["feature_encoder"]["day_scale"]) == R1_DAY_SCALE
        and float(config["feature_encoder"]["version_scale"]) == R1_VERSION_SCALE
        and tuple(config["feature_encoder"]["ordered_features"])
        == R1_FEATURE_NAMES
        and "features" in context_fields
        and "visible_state" in context_fields
        and "visible_address" in context_fields
        and "incoming_value" in context_fields
    )
    return {
        "gate_record_fields": list(model_fields),
        "gate_record_fields_match_frozen_set": set(model_fields)
        == set(expected_fields),
        "forbidden_field_overlap": sorted(set(model_fields) & forbidden),
        "encoder_parameters": list(encoder_parameters),
        "encoder_input_dim": R1_FEATURE_DIM,
        "encoder_feature_names": list(R1_FEATURE_NAMES),
        "model_forward_parameters": list(model_forward_parameters),
        "visible_update_context_fields": list(context_fields),
        "state_read_in_gate_encoder": False,
        "address_in_gate_encoder": False,
        "incoming_value_in_gate_encoder": False,
        "state_read_used_only_for_erase_candidate": True,
        "learned_gate_supervision": False,
        "target_state_weight": float(config["training"]["target_state_weight"]),
        "forbidden_access_test_passed": safe,
    }


def _balance_audit(
    examples: Sequence[SemanticExample],
    *,
    split: str,
    checkpoint_seed: int,
    count_per_cell: int,
) -> tuple[list[dict[str, object]], bool]:
    cell_counts: dict[tuple[str, str, str], Counter[str]] = {}
    for example in examples:
        key = (example.operation.value, example.domain, example.template)
        counts = cell_counts.setdefault(key, Counter())
        counts[classify_r1_write_stratum(example.safe_record).key] += 1

    rows: list[dict[str, object]] = []
    passed = True
    false_keys = {stratum.key for stratum in R1_WRITE_FALSE_STRATA}
    for (operation, domain, template), counts in sorted(cell_counts.items()):
        if operation == "add":
            expected = {R1_WRITE_TRUE_STRATUM.key: count_per_cell}
        else:
            per_stratum = count_per_cell // len(R1_WRITE_FALSE_STRATA)
            expected = {key: per_stratum for key in false_keys}
        cell_passed = dict(counts) == expected
        passed = passed and cell_passed
        rows.append(
            {
                "schema_version": 1,
                "split": split,
                "checkpoint_seed": checkpoint_seed,
                "operation": operation,
                "domain": domain,
                "template": template,
                "observed": dict(sorted(counts.items())),
                "expected": dict(sorted(expected.items())),
                "passed": cell_passed,
            }
        )
    return rows, passed


def _merge_vocabulary(
    target: dict[str, dict[str, set[str]]],
    name: str,
    examples: Sequence[SemanticExample],
) -> None:
    target[name] = {
        key: set(values)
        for key, values in semantic_vocabularies(examples).items()
    }


def _assert_vocab_disjoint(
    vocabularies: Mapping[str, Mapping[str, set[str]]],
) -> None:
    names = sorted(vocabularies)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for kind in ("entity", "old_value", "new_value"):
                if vocabularies[left][kind] & vocabularies[right][kind]:
                    raise ProvenanceValidationError(
                        f"R1 {kind} vocabulary overlaps: {left} vs {right}."
                    )


def _static_design_gates(
    *,
    config: Mapping[str, Any],
    access_manifest: Mapping[str, object],
    namespace_registry: SemanticNamespaceRegistry,
    registry_rows: Sequence[Mapping[str, object]],
    raw_rows: Sequence[Mapping[str, object]],
    balance_passed: bool,
    max_norm_mismatch: float,
    budget_records: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    """Derive every static gate from materialized R1 records and contracts."""

    expected_names = {"r1_train", "r1_validation"}
    observed_names = {str(row.get("namespace_name", "")) for row in registry_rows}
    observed_splits = {str(row.get("split", "")) for row in registry_rows}
    numeric_seeds = [int(row["numeric_seed"]) for row in registry_rows]
    checkpoint_seeds = {
        int(row["checkpoint_seed"]) for row in registry_rows
    }
    namespace_config = config["namespace"]
    prior_max = int(namespace_config["forbid_overlap_with_prior_numeric_seed_max"])
    downstream_floor = min(
        int(namespace_config["audit_pool_root_reserved_not_opened"]),
        int(namespace_config["e05b_r1_root_reserved_not_opened"]),
    )
    boundary_tokens = ("audit", "e05b")

    budget_match = bool(
        budget_records
        and len({row["parameter_count"] for row in budget_records}) == 1
        and len(
            {
                row["dense_multiply_adds_per_example"]
                for row in budget_records
            }
        )
        == 1
        and all(bool(row["common_initialization"]) for row in budget_records)
        and all(bool(row["common_schedule"]) for row in budget_records)
    )
    namespace_only_r1 = bool(
        registry_rows
        and set(namespace_registry.names) == expected_names
        and observed_names == expected_names
        and observed_splits == expected_names
    )
    original_rows_not_reused = bool(
        numeric_seeds
        and min(numeric_seeds) > prior_max
        and checkpoint_seeds.isdisjoint({101, 202, 303, 404})
    )
    no_downstream_rows = bool(
        numeric_seeds
        and max(numeric_seeds) < downstream_floor
        and not any(
            token in name.lower()
            for name in observed_names | observed_splits
            for token in boundary_tokens
        )
    )
    supersede_absent = bool(
        all(
            row.get("operation") != Operation.SUPERSEDE.value
            for row in raw_rows
        )
        and all(
            row.get("operation_private") != Operation.SUPERSEDE.value
            for row in registry_rows
        )
    )
    return {
        "forbidden_access": bool(
            access_manifest["forbidden_access_test_passed"]
        ),
        "namespace_only_r1_train_validation": namespace_only_r1,
        "fresh_vocabularies_disjoint": True,
        "write_false_strata_balanced": bool(balance_passed),
        "wrong_address_norm_matched": (
            max_norm_mismatch
            <= float(config["controls"]["wrong_address_norm_tolerance"])
        ),
        "parameter_and_compute_budget_match": budget_match,
        "supersede_absent": supersede_absent,
        "original_e05a_rows_not_reused": original_rows_not_reused,
        "no_audit_or_e05b_rows_generated": no_downstream_rows,
    }


def _summary_markdown(
    *,
    dry_run: bool,
    design_status: str,
    statistical_report: Mapping[str, object] | None,
    static_gates: Mapping[str, bool],
    seed_effect_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# E05a-R1 Semantic Design Repair 결과",
        "",
        "- Execution: `PASS`",
        f"- Design status: `{design_status}`",
        "- Original E05a: `NO_GO` (immutable)",
        "- H5 claim open: `false`",
        "- Evidence tier: `CONTROLLED_REFERENCE`",
        "",
        "## 개발·검증 특기사항",
        "",
        (
            "- Gate input은 6개 raw relational feature로 제한되며 "
            "state/address/value를 받지 않는다."
        ),
        (
            "- `PRESERVE`와 `INVALIDATE`의 각 cell은 가능한 write-false "
            "stratum 11개를 동일 수로 포함한다."
        ),
        "- 원본 E05a의 네 seed와 episode는 R1 추론에 재사용하지 않았다.",
        "- R1 run은 human-audit item이나 E05b-R1 row를 생성하지 않는다.",
        "",
        "## Static gates",
        "",
        "| Gate | Pass |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | `{str(value).lower()}` |"
        for name, value in static_gates.items()
    )
    lines.extend(
        [
            "",
            "## Seed별 primary 결과",
            "",
            "| Seed | Factorized A/I MSE | Shared A/I MSE | Shared−Factorized |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in seed_effect_rows:
        lines.append(
            "| {seed} | {factorized:.8f} | {shared:.8f} | {gain:.8f} |".format(
                seed=row["checkpoint_seed"],
                factorized=float(row["factorized_ai_affected_mse"]),
                shared=float(row["shared_ai_affected_mse"]),
                gain=float(row["shared_minus_factorized_ai_gain"]),
            )
        )
    if dry_run:
        lines.extend(
            [
                "",
                "Dry-run은 구현 검증만 수행했으며 통계 판정을 하지 않았다.",
            ]
        )
    elif statistical_report is not None:
        primary = statistical_report["primary_gain_shared_minus_factorized"]
        reasons = statistical_report["diagnostic_reasons"]
        lines.extend(
            [
                "",
                "## 사전등록 판정",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Mean primary gain | {float(primary['estimate']):.8f} |",
                (
                    "| Seed-cluster 95% CI | "
                    f"[{float(primary['ci95'][0]):.8f}, "
                    f"{float(primary['ci95'][1]):.8f}] |"
                ),
                (
                    "| Exact sign-flip p | "
                    f"{float(primary['exact_sign_flip']['p']):.8f} |"
                ),
                f"| 8/8 positive | `{str(primary['all_seed_raw_direction_positive']).lower()}` |",
                "",
                "Failure reasons: "
                + (
                    ", ".join(f"`{reason}`" for reason in reasons)
                    if reasons
                    else "`none`"
                ),
                "",
            ]
        )
        if design_status == "GO":
            lines.append(
                "R1은 design-validity `GO`다. H5는 아직 열리지 않으며 "
                "별도 human-audit pool lock이 다음 단계다."
            )
        else:
            lines.append(
                "R1은 `NO_GO`다. 이는 H5 반증이 아니라 design validity "
                "미확립이며, 사전등록대로 이번 제출에서 H5를 종료한다."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    validate_r1_config_path(args.config)
    frozen_config = validate_frozen_r1_protocol()
    original_e05a = validate_original_e05a_dependency(args.artifact_root)
    e00 = validate_legacy_e00(args.artifact_root, require_full=True)
    dependencies = [
        e00,
        original_e05a_dependency_record(original_e05a),
    ]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=args.dry_run,
        dependencies=dependencies,
    )
    if config != frozen_config:
        raise ProvenanceValidationError("Runtime R1 config differs from frozen config.")

    torch.set_num_threads(1)
    memory_spec = _memory_spec(config)
    encoder = RelationalSemanticEncoderR1()
    training_config = _training_config(config, dry_run=args.dry_run)
    namespace_registry = SemanticNamespaceRegistry.from_config(
        config["namespace"],
        dry_run=args.dry_run,
    )
    r1 = config["r1"]
    seeds = (
        [int(r1["dry_seed"])]
        if args.dry_run
        else [int(value) for value in r1["seeds"]]
    )
    train_count = int(
        r1[
            "dry_train_count_per_cell"
            if args.dry_run
            else "train_count_per_cell"
        ]
    )
    validation_count = int(
        r1[
            "dry_validation_count_per_cell"
            if args.dry_run
            else "validation_count_per_cell"
        ]
    )
    operations = _operation_values(r1["operations"])
    domains = tuple(str(value) for value in r1["seen_domains"])
    templates = tuple(str(value) for value in r1["seen_templates"])

    access_manifest = _access_manifest(config)
    access_path = run_dir / "r1_semantic_access_manifest.json"
    write_json_strict(access_path, access_manifest)

    seed_metrics: dict[int, SemanticAnchorSeedMetrics] = {}
    raw_rows: list[dict[str, object]] = []
    seed_effect_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    pairing_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    seed_summaries: dict[str, object] = {}
    vocabularies: dict[str, dict[str, set[str]]] = {}
    budget_records: list[dict[str, object]] = []
    balance_passed = True
    max_norm_mismatch = 0.0

    for seed_slot, seed in enumerate(seeds):
        train = build_balanced_semantic_examples_r1(
            namespace_registry=namespace_registry,
            namespace_name="r1_train",
            checkpoint_seed=seed,
            seed_slot=seed_slot,
            operations=operations,
            domains=domains,
            templates=templates,
            count_per_cell=train_count,
            memory_spec=memory_spec,
        )
        validation = build_balanced_semantic_examples_r1(
            namespace_registry=namespace_registry,
            namespace_name="r1_validation",
            checkpoint_seed=seed,
            seed_slot=seed_slot,
            operations=operations,
            domains=domains,
            templates=templates,
            count_per_cell=validation_count,
            memory_spec=memory_spec,
        )
        assert_disjoint_semantic_vocabularies(
            {"r1_train": train, "r1_validation": validation}
        )
        _merge_vocabulary(vocabularies, f"seed{seed}_train", train)
        _merge_vocabulary(vocabularies, f"seed{seed}_validation", validation)
        registry_rows.extend(
            semantic_example_registry_row(
                example,
                split=split,
                seed_slot=seed_slot,
            )
            for split, examples in (
                ("r1_train", train),
                ("r1_validation", validation),
            )
            for example in examples
        )
        for split, examples, count in (
            ("r1_train", train, train_count),
            ("r1_validation", validation, validation_count),
        ):
            local_rows, local_passed = _balance_audit(
                examples,
                split=split,
                checkpoint_seed=seed,
                count_per_cell=count,
            )
            balance_rows.extend(local_rows)
            balance_passed = balance_passed and local_passed

        pairings = build_control_pairing_registry_r1(
            validation,
            norm_tolerance=float(
                config["controls"]["wrong_address_norm_tolerance"]
            ),
        )
        pairing_rows.extend(
            {
                "schema_version": 1,
                "checkpoint_seed": seed,
                **row,
            }
            for row in pairings.to_rows()
        )
        max_norm_mismatch = max(
            max_norm_mismatch,
            max(
                pairing.maximum_wrong_address_norm_mismatch
                for pairing in pairings.pairings
            ),
        )

        training = train_matched_semantic_pair_r1(
            train,
            encoder=encoder,
            hidden_dim=int(config["model"]["path_hidden_dim"]),
            config=training_config,
            seed=seed,
            device=device,
        )
        checkpoints: dict[str, object] = {}
        for name, model in (
            ("factorized", training.factorized),
            ("shared", training.shared),
        ):
            checkpoint_path = run_dir / f"seed{seed}_{name}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoints[name] = _descriptor(checkpoint_path)

        metrics, condition_rows, condition_summary = _evaluate_seed(
            seed=seed,
            validation=validation,
            encoder=encoder,
            pairing_registry=pairings,
            factorized=training.factorized,
            shared=training.shared,
            device=device,
        )
        seed_metrics[seed] = metrics
        raw_rows.extend(condition_rows)
        budget = {
            "parameter_count": training.parameter_count,
            "dense_multiply_adds_per_example": (
                training.dense_multiply_adds_per_example
            ),
            "initial_state_sha256": training.initial_state_sha256,
            "schedule_sha256": training.schedule_sha256,
            "steps": training_config.steps,
            "batch_size": training_config.batch_size,
            "common_initialization": True,
            "common_schedule": True,
        }
        budget_records.append({"checkpoint_seed": seed, **budget})
        seed_effect_rows.append(
            _seed_effect_row(
                seed=seed,
                validation=validation,
                metrics=metrics,
                budget=budget,
            )
        )
        seed_summaries[str(seed)] = {
            "training_final_loss": dict(training.final_loss),
            "conditions": condition_summary,
            "budget": budget,
            "checkpoints": checkpoints,
        }

    _assert_vocab_disjoint(vocabularies)
    registry_path = run_dir / "r1_namespace_registry.jsonl"
    pairing_path = run_dir / "r1_control_pairings.jsonl"
    balance_path = run_dir / "r1_stratum_balance.jsonl"
    raw_path = run_dir / "r1_semantic_metrics.jsonl"
    effects_path = run_dir / "r1_seed_effects.jsonl"
    write_jsonl_strict(registry_path, registry_rows)
    write_jsonl_strict(pairing_path, pairing_rows)
    write_jsonl_strict(balance_path, balance_rows)
    write_jsonl_strict(raw_path, raw_rows)
    write_jsonl_strict(effects_path, seed_effect_rows)

    statistical_report: dict[str, object] | None = None
    if not args.dry_run:
        statistical_report = evaluate_e05a_r1_design(
            seed_metrics,
            fixed_seeds=tuple(seeds),
            thresholds=_thresholds(config),
            bootstrap_seeds={
                str(key): int(value)
                for key, value in config["statistics"]["bootstrap_seeds"].items()
            },
        )

    static_gates = _static_design_gates(
        config=config,
        access_manifest=access_manifest,
        namespace_registry=namespace_registry,
        registry_rows=registry_rows,
        raw_rows=raw_rows,
        balance_passed=balance_passed,
        max_norm_mismatch=max_norm_mismatch,
        budget_records=budget_records,
    )
    design_go = bool(
        not args.dry_run
        and statistical_report is not None
        and statistical_report["go"]
        and all(static_gates.values())
    )
    design_status = (
        "NOT_EVALUATED_DRY_RUN"
        if args.dry_run
        else ("GO" if design_go else "NO_GO")
    )
    static_failure_reasons = [
        f"STATIC_{name.upper()}_FAILED"
        for name, passed in static_gates.items()
        if not passed
    ]

    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        _summary_markdown(
            dry_run=args.dry_run,
            design_status=design_status,
            statistical_report=statistical_report,
            static_gates=static_gates,
            seed_effect_rows=seed_effect_rows,
        ),
        encoding="utf-8",
    )

    artifacts: dict[str, dict[str, object]] = {
        "semantic_access_manifest": _descriptor(access_path),
        "namespace_registry": _descriptor(
            registry_path,
            rows=len(registry_rows),
        ),
        "control_pairings": _descriptor(
            pairing_path,
            rows=len(pairing_rows),
        ),
        "stratum_balance": _descriptor(
            balance_path,
            rows=len(balance_rows),
        ),
        "semantic_metrics": _descriptor(raw_path, rows=len(raw_rows)),
        "seed_effects": _descriptor(
            effects_path,
            rows=len(seed_effect_rows),
        ),
        "results_summary_ko": _descriptor(summary_path),
    }
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "e05a_r1_design_status": design_status,
        "original_e05a_status": "NO_GO",
        "h5_status": (
            "NOT_TESTED_PENDING_HUMAN_AUDIT"
            if design_go
            else (
                "NOT_EVALUATED_DRY_RUN"
                if args.dry_run
                else "TERMINATED_NOT_REFUTED"
            )
        ),
        "h5_claim_open": False,
        "protocol_lock": {
            "protocol_sha256": PINNED_R1_PROTOCOL_SHA256,
            "protocol_lock_sha256": PINNED_R1_PROTOCOL_LOCK_SHA256,
            "config_canonical_sha256": (
                PINNED_R1_CONFIG_CANONICAL_SHA256
            ),
            "config_file_sha256": PINNED_R1_CONFIG_FILE_SHA256,
            "config_path": str(R1_CONFIG_PATH),
        },
        "original_e05a_dependency": {
            "run_id": original_e05a.run_id,
            "status": "NO_GO",
            "rows_reused_in_r1_inference": 0,
        },
        "access_contract": access_manifest,
        "static_design_gates": static_gates,
        "static_failure_reasons": static_failure_reasons,
        "statistical_design_gates": statistical_report,
        "seed_summaries": seed_summaries,
        "namespace": {
            "dry_run": args.dry_run,
            "integer_root": namespace_registry.integer_root,
            "opened_names": list(namespace_registry.names),
            "audit_pool_root_opened": False,
            "e05b_r1_root_opened": False,
            "vocabulary_intersections": 0,
        },
        "human_audit": {
            "pool_generated": False,
            "pool_generation_eligible": design_go,
            "status": "NOT_GENERATED",
            "separate_experiment_required": True,
            "ai_agent_review_substitution_allowed": False,
        },
        "claim_gate": {
            "e05a_r1_is_h5_claim_evidence": False,
            "h5_claim_open": False,
            "e05b_r1_execution_allowed": False,
            "r1_no_go_terminates_h5_for_this_submission": (
                not args.dry_run and not design_go
            ),
        },
        "artifacts": artifacts,
        "evidence_scope": {
            "evidence_tier": "CONTROLLED_REFERENCE",
            "scientific_evidence": False,
            "official_backend_claim_eligible": False,
            "language_model_claim_eligible": False,
            "architecture_transfer_claim_eligible": False,
        },
    }
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=not args.dry_run,
        full_eligible=not args.dry_run,
    )
    print(f"[{EXPERIMENT_ID}] PASS ({design_status}): {run_dir}")


if __name__ == "__main__":
    main()
