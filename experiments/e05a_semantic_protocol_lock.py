from __future__ import annotations

import csv
import hashlib
import inspect
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
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
from catena.data.semantic_controls_v61 import (
    ControlPairingRegistry,
    SemanticControl,
    build_control_pairing_registry,
)
from catena.data.semantic_registry_v61 import (
    audit_id,
    render_naturalized_record,
    semantic_example_from_registry_row,
    semantic_example_registry_row,
    semantic_registry_row_from_design,
    validate_semantic_registry_rows,
)
from catena.data.semantic_transactions_v61 import (
    SafeSemanticRecord,
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    assert_disjoint_semantic_vocabularies,
    build_balanced_semantic_examples,
    semantic_vocabularies,
)
from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    SemanticAnchorSeedMetrics,
    SemanticAnchorThresholds,
    evaluate_e05a_go,
)
from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
)
from catena.models.semantic_encoder_v61 import (
    FrozenSemanticFieldEncoderV61,
    SemanticFeatureConfigV61,
)
from catena.training.semantic_probe_v61 import (
    BatchedVisibleUpdateContext,
    SemanticTrainingConfigV61,
    evaluate_semantic_model,
    seed_metrics_from_condition_arrays,
    train_matched_semantic_pair,
)
from experiments.common import build_parser
from experiments.e05_common_v61 import (
    E05A_CONFIG_PATH,
    PINNED_E05A_CONFIG_CANONICAL_SHA256,
    PINNED_E05A_CONFIG_FILE_SHA256,
    PINNED_E05B_CONFIG_CANONICAL_SHA256,
    PINNED_E05B_CONFIG_FILE_SHA256,
    PINNED_PROTOCOL_LOCK_SHA256,
    PINNED_PROTOCOL_SHA256,
    validate_frozen_e04_dependency,
    validate_frozen_e05_protocol,
)
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e05a_semantic_protocol_lock"
DEFAULT_CONFIG = "configs/e05a_semantic_protocol_lock.yaml"

_CONDITION_TO_CONTROL = {
    "shuffled_fields": SemanticControl.SHUFFLED_FIELDS,
    "wrong_address": SemanticControl.WRONG_ADDRESS,
    "transaction_only": SemanticControl.TRANSACTION_ONLY,
    "state_only": SemanticControl.STATE_ONLY,
    "wrong_semantics": SemanticControl.WRONG_SEMANTICS,
}
_E05B_SPLIT_CONTRACT = {
    "train": ("e05b_train", "e05b_train"),
    "sealed_validation": ("e05b_validation", "e05b_validation"),
    "primary": ("e05b_primary", "e05b_primary"),
    "heldout_paraphrase": ("e05b_paraphrase", "e05b_paraphrase"),
    "heldout_domain": ("e05b_domain", "e05b_domain"),
    "combined_stress": ("e05b_combined", "e05b_combined"),
}
_AUDIT_SPLITS = (
    "primary",
    "heldout_paraphrase",
    "heldout_domain",
    "combined_stress",
)


def _operation_values(values: Sequence[object]) -> tuple[Operation, ...]:
    return tuple(Operation(str(value)) for value in values)


def _memory_spec(config: Mapping[str, Any]) -> SemanticMemorySpec:
    memory = config["memory"]
    if memory["orthonormal_keys"] is not True:
        raise ProvenanceValidationError("E05 requires orthonormal keys.")
    if memory["candidate_source"] != "visible_state_read_at_visible_address":
        raise ProvenanceValidationError("E05 visible candidate source changed.")
    if memory["dtype"] != "float32":
        raise ProvenanceValidationError("E05 frozen dtype is float32.")
    return SemanticMemorySpec(
        num_associations=int(memory["num_associations"]),
        key_dim=int(memory["key_dim"]),
        value_dim=int(memory["value_dim"]),
        dtype=torch.float32,
    )


def _encoder(
    config: Mapping[str, Any],
    memory_spec: SemanticMemorySpec,
) -> FrozenSemanticFieldEncoderV61:
    feature = config["feature_encoder"]
    return FrozenSemanticFieldEncoderV61(
        SemanticFeatureConfigV61(
            categorical_fields=tuple(str(value) for value in feature["categorical_fields"]),
            numeric_fields=tuple(str(value) for value in feature["numeric_fields"]),
            categorical_bins_per_field=int(feature["categorical_bins_per_field"]),
            version_scale=float(feature["version_scale"]),
            day_scale=float(feature["day_scale"]),
            state_read_dim=memory_spec.value_dim,
        )
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


def _thresholds(config: Mapping[str, Any]) -> SemanticAnchorThresholds:
    statistics = config["statistics"]
    return SemanticAnchorThresholds(
        positive_effect_sesoi=float(statistics["positive_effect_sesoi"]),
        minimum_oracle_headroom=0.001,
        headroom_fraction_sesoi=0.10,
        equivalence_margin=float(statistics["equivalence_margin"]),
        retention_noninferiority_margin=float(statistics["equivalence_margin"]),
        oracle_absolute_ceiling=float(statistics["oracle_absolute_ceiling"]),
        alpha=float(statistics["alpha"]),
        bootstrap_samples=int(statistics["bootstrap_samples"]),
        bootstrap_confidence=float(statistics["bootstrap_confidence"]),
    )


def _validate_frozen_config_path(config_path: str) -> None:
    resolved = Path(config_path).resolve(strict=True)
    if resolved != E05A_CONFIG_PATH.resolve(strict=True):
        raise ProvenanceValidationError("E05a requires the frozen default config path.")


def _access_manifest(
    config: Mapping[str, Any],
) -> dict[str, object]:
    safe_fields = tuple(field.name for field in fields(SafeSemanticRecord))
    allowed = tuple(str(value) for value in config["semantic_schema"]["allowed_structured_fields"])
    forbidden = set(
        str(value) for value in config["semantic_schema"]["forbidden_model_fields"]
    )
    public_context_fields = tuple(
        field.name for field in fields(BatchedVisibleUpdateContext)
    )
    model_forward = tuple(
        inspect.signature(MatchedSemanticControllerV61.forward).parameters
    )
    safe = bool(
        safe_fields == allowed
        and not (set(safe_fields) & forbidden)
        and set(public_context_fields)
        == {
            "features",
            "visible_state",
            "visible_address",
            "incoming_value",
            "erase_candidate_scale",
            "write_candidate_scale",
        }
        and model_forward == ("self", "features")
    )
    return {
        "safe_record_fields": list(safe_fields),
        "allowed_fields_match_exactly": safe_fields == allowed,
        "forbidden_field_overlap": sorted(set(safe_fields) & forbidden),
        "model_forward_parameters": list(model_forward),
        "batched_visible_update_context_fields": list(public_context_fields),
        "private_target_or_demand_in_public_update_context": bool(
            {"target", "target_state", "operation", "demand", "old_value"}
            & set(public_context_fields)
        ),
        "operation_and_demand_used_only_for_oracle_scoring": True,
        "learned_training_gate_supervision": False,
        "target_state_weight": float(config["training"]["target_state_weight"]),
        "forbidden_access_test_passed": safe,
    }


def _count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _descriptor(path: Path, *, rows: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "filename": path.name,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        payload["rows"] = rows
    return payload


def _attach_condition(
    rows: Iterable[dict[str, object]],
    *,
    seed: int,
    condition: str,
    split: str,
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "seed": seed,
            "condition": condition,
            "split": split,
            **row,
        }
        for row in rows
    ]


def _evaluate_seed(
    *,
    seed: int,
    validation: Sequence[SemanticExample],
    encoder: FrozenSemanticFieldEncoderV61,
    pairing_registry: ControlPairingRegistry,
    factorized: MatchedSemanticControllerV61,
    shared: MatchedSemanticControllerV61,
    device: torch.device,
) -> tuple[
    SemanticAnchorSeedMetrics,
    list[dict[str, object]],
    dict[str, dict[str, float]],
]:
    raw_rows: list[dict[str, object]] = []
    affected: dict[str, np.ndarray] = {}
    retention: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}

    learned_conditions = {
        "factorized": factorized,
        "shared": shared,
    }
    for name, model in learned_conditions.items():
        rows, affected_values, retention_values = evaluate_semantic_model(
            model,
            validation,
            encoder=encoder,
            control=SemanticControl.FULL,
            pairing_registry=None,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(
            _attach_condition(rows, seed=seed, condition=name, split="pilot_validation")
        )
        affected[name] = affected_values
        retention[name] = retention_values
        summaries[name] = {
            "affected_read_mse": float(affected_values.mean()),
            "unaffected_retention_mse": float(retention_values.mean()),
        }

    oracle_rows, oracle_affected, oracle_retention = evaluate_semantic_model(
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
            split="pilot_validation",
        )
    )
    affected["oracle_demand"] = oracle_affected
    retention["oracle_demand"] = oracle_retention
    summaries["oracle_demand"] = {
        "affected_read_mse": float(oracle_affected.mean()),
        "unaffected_retention_mse": float(oracle_retention.mean()),
    }

    for condition in CONTROL_NAMES:
        control = _CONDITION_TO_CONTROL[condition]
        rows, affected_values, retention_values = evaluate_semantic_model(
            factorized,
            validation,
            encoder=encoder,
            control=control,
            pairing_registry=pairing_registry,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(
            _attach_condition(
                rows,
                seed=seed,
                condition=condition,
                split="pilot_validation",
            )
        )
        affected[condition] = affected_values
        summaries[condition] = {
            "affected_read_mse": float(affected_values.mean()),
            "unaffected_retention_mse": float(retention_values.mean()),
        }

    return (
        seed_metrics_from_condition_arrays(
            validation,
            affected=affected,
            retention=retention,
        ),
        raw_rows,
        summaries,
    )


def _update_vocab_sets(
    target: dict[str, dict[str, set[str]]],
    name: str,
    examples: Sequence[SemanticExample],
) -> None:
    local = semantic_vocabularies(examples)
    destination = target.setdefault(
        name,
        {"entity": set(), "old_value": set(), "new_value": set()},
    )
    for vocabulary in destination:
        destination[vocabulary].update(local[vocabulary])


def _assert_disjoint_vocab_sets(
    vocabularies: Mapping[str, Mapping[str, set[str]]],
) -> None:
    names = sorted(vocabularies)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for vocabulary in ("entity", "old_value", "new_value"):
                overlap = vocabularies[left][vocabulary] & vocabularies[right][vocabulary]
                if overlap:
                    raise ValueError(
                        f"{vocabulary} vocabulary overlaps between {left} and {right}."
                    )


def _vocab_payload(
    vocabularies: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split, groups in sorted(vocabularies.items()):
        result[split] = {
            name: {
                "count": len(values),
                "sha256": hashlib.sha256(
                    "\n".join(sorted(values)).encode("utf-8")
                ).hexdigest(),
            }
            for name, values in sorted(groups.items())
        }
    return result


def _registry_rows_from_contract(
    *,
    namespace_registry: SemanticNamespaceRegistry,
    namespace_name: str,
    split: str,
    seeds: Sequence[int],
    operations: Sequence[Operation],
    domains: Sequence[str],
    templates: Sequence[str],
    count_per_cell: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_slot, seed in enumerate(seeds):
        index = 0
        for operation in operations:
            for domain in domains:
                for template in templates:
                    for _ in range(count_per_cell):
                        numeric_seed = namespace_registry.numeric_seed(
                            namespace_name,
                            seed_slot=seed_slot,
                            index=index,
                        )
                        rows.append(
                            semantic_registry_row_from_design(
                                namespace_name=namespace_name,
                                split=split,
                                numeric_seed=numeric_seed,
                                checkpoint_seed=int(seed),
                                seed_slot=seed_slot,
                                operation=operation,
                                domain=domain,
                                template_surface=template,
                            )
                        )
                        index += 1
    return rows


def _vocab_from_registry_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    result = {"entity": set(), "old_value": set(), "new_value": set()}
    for row in rows:
        safe = row["safe_record"]
        if not isinstance(safe, dict):
            raise TypeError("Registry safe_record must be an object.")
        result["entity"].add(str(safe["entity_description"]))
        result["old_value"].add(str(row["old_value_token_private"]))
        result["new_value"].add(str(safe["incoming_value_token"]))
    return result


def _select_audit_rows(
    candidates: Sequence[Mapping[str, object]],
    *,
    split: str,
    count: int,
) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in candidates:
        safe = row["safe_record"]
        if not isinstance(safe, dict):
            raise TypeError("Audit candidate lacks a safe record.")
        grouped[(str(safe["domain"]), str(row["operation_private"]))].append(row)
    if not grouped:
        raise ValueError(f"No audit candidates for {split}.")
    ordered_groups = sorted(grouped)
    for group in ordered_groups:
        grouped[group].sort(
            key=lambda row: hashlib.sha256(
                f"{split}\0{row['example_id']}".encode()
            ).hexdigest()
        )
    base, remainder = divmod(count, len(ordered_groups))
    selected: list[Mapping[str, object]] = []
    for index, group in enumerate(ordered_groups):
        quota = base + int(index < remainder)
        if len(grouped[group]) < quota:
            raise ValueError(f"Audit stratum {group} is undersized.")
        selected.extend(grouped[group][:quota])
    if len(selected) != count:
        raise AssertionError("Audit sampler returned the wrong count.")
    return selected


def _write_audit_files(
    run_dir: Path,
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    count_per_split: int,
) -> dict[str, dict[str, object]]:
    selected: list[tuple[str, Mapping[str, object]]] = []
    for split in _AUDIT_SPLITS:
        selected.extend(
            (split, row)
            for row in _select_audit_rows(
                candidates[split],
                split=split,
                count=count_per_split,
            )
        )
    items_path = run_dir / "naturalization_audit_items.csv"
    reviewer_a_path = run_dir / "reviewer_a_template.csv"
    reviewer_b_path = run_dir / "reviewer_b_template.csv"
    adjudication_path = run_dir / "adjudication_template.csv"
    with items_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audit_id",
                "split",
                "domain",
                "intended_operation",
                "example_id",
                "structured_record_json",
                "naturalized_text",
            ]
        )
        for split, row in selected:
            safe_payload = row["safe_record"]
            if not isinstance(safe_payload, dict):
                raise TypeError("Audit row safe_record must be an object.")
            record = SafeSemanticRecord(**safe_payload)
            writer.writerow(
                [
                    audit_id(split, str(row["example_id"])),
                    split,
                    record.domain,
                    row["operation_private"],
                    row["example_id"],
                    json.dumps(
                        safe_payload,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    render_naturalized_record(record),
                ]
            )
    audit_ids = [
        audit_id(split, str(row["example_id"])) for split, row in selected
    ]
    for path in (reviewer_a_path, reviewer_b_path):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["audit_id", "meaning_preserved", "answer_leakage", "notes"]
            )
            writer.writerows([[value, "", "", ""] for value in audit_ids])
    with adjudication_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audit_id",
                "adjudicated_meaning_preserved",
                "adjudicated_answer_leakage",
                "notes",
            ]
        )
        writer.writerows([[value, "", "", ""] for value in audit_ids])
    return {
        "naturalization_audit_items": _descriptor(
            items_path,
            rows=len(selected),
        ),
        "reviewer_a_template": _descriptor(reviewer_a_path, rows=len(selected)),
        "reviewer_b_template": _descriptor(reviewer_b_path, rows=len(selected)),
        "adjudication_template": _descriptor(
            adjudication_path,
            rows=len(selected),
        ),
    }


def _generate_e05b_seal(
    *,
    run_dir: Path,
    e05a_config: Mapping[str, Any],
    e05b_config: Mapping[str, Any],
    namespace_registry: SemanticNamespaceRegistry,
    memory_spec: SemanticMemorySpec,
    existing_vocabularies: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    seeds = [int(value) for value in e05b_config["seeds"]]
    artifact_registry: dict[str, dict[str, object]] = {}
    audit_candidates: dict[str, list[Mapping[str, object]]] = {}
    split_paths: dict[str, Path] = {}
    split_rows: dict[str, list[dict[str, object]]] = {}

    for split_name, (namespace_name, artifact_stem) in _E05B_SPLIT_CONTRACT.items():
        contract = e05b_config["data"][split_name]
        operations = _operation_values(contract["operations"])
        domains = tuple(str(value) for value in contract["domains"])
        templates = tuple(str(value) for value in contract["templates"])
        count_per_cell = int(contract["count_per_cell"])
        rows = _registry_rows_from_contract(
            namespace_registry=namespace_registry,
            namespace_name=namespace_name,
            split=split_name,
            seeds=seeds,
            operations=operations,
            domains=domains,
            templates=templates,
            count_per_cell=count_per_cell,
        )
        expected_per_seed = (
            len(operations) * len(domains) * len(templates) * count_per_cell
        )
        validate_semantic_registry_rows(
            rows,
            expected_split=split_name,
            expected_seeds=seeds,
            expected_rows_per_seed=expected_per_seed,
            expected_namespace_name=namespace_name,
            expected_seed_slots={
                int(seed): seed_slot for seed_slot, seed in enumerate(seeds)
            },
            namespace_registry=namespace_registry,
            memory_spec=memory_spec,
            reconstruct=False,
        )
        path = run_dir / f"{artifact_stem}_registry.jsonl"
        write_jsonl_strict(path, rows)
        if _count_jsonl(path) != len(rows):
            raise AssertionError("E05b registry row count changed after writing.")
        artifact_registry[f"{artifact_stem}_registry"] = _descriptor(
            path,
            rows=len(rows),
        )
        split_paths[split_name] = path
        split_rows[split_name] = rows
        existing_vocabularies[f"e05b_{split_name}"] = _vocab_from_registry_rows(rows)
        if split_name in _AUDIT_SPLITS:
            audit_candidates[split_name] = rows

    _assert_disjoint_vocab_sets(existing_vocabularies)

    control_rows: list[dict[str, object]] = []
    validation_rows = split_rows["sealed_validation"]
    primary_rows = split_rows["primary"]
    for seed in seeds:
        validation_examples = [
            semantic_example_from_registry_row(row, memory_spec=memory_spec)
            for row in validation_rows
            if row["checkpoint_seed"] == seed
        ]
        primary_examples = [
            semantic_example_from_registry_row(row, memory_spec=memory_spec)
            for row in primary_rows
            if row["checkpoint_seed"] == seed
        ]
        pairings = build_control_pairing_registry(
            primary_examples,
            semantic_donors=validation_examples,
            norm_tolerance=float(
                e05b_config["controls"]["wrong_address_norm_tolerance"]
            ),
        )
        control_rows.extend(
            {"checkpoint_seed": seed, **row} for row in pairings.to_rows()
        )
    control_path = run_dir / "e05b_primary_control_pairings.jsonl"
    write_jsonl_strict(control_path, control_rows)
    artifact_registry["e05b_primary_control_pairings"] = _descriptor(
        control_path,
        rows=len(control_rows),
    )

    audit_artifacts = _write_audit_files(
        run_dir,
        audit_candidates,
        count_per_split=int(
            e05a_config["e05b_registry"]["audit_items_per_split"]
        ),
    )
    artifact_registry.update(audit_artifacts)

    vocab_path = run_dir / "semantic_vocab_registry.json"
    vocab_payload = _vocab_payload(existing_vocabularies)
    write_json_strict(vocab_path, vocab_payload)
    artifact_registry["semantic_vocab_registry"] = _descriptor(vocab_path)

    seal = {
        "schema_version": 1,
        "protocol_sha256": PINNED_PROTOCOL_SHA256,
        "protocol_lock_sha256": PINNED_PROTOCOL_LOCK_SHA256,
        "e05a_config": {
            "canonical_sha256": PINNED_E05A_CONFIG_CANONICAL_SHA256,
            "file_sha256": PINNED_E05A_CONFIG_FILE_SHA256,
        },
        "e05b_config": {
            "canonical_sha256": PINNED_E05B_CONFIG_CANONICAL_SHA256,
            "file_sha256": PINNED_E05B_CONFIG_FILE_SHA256,
        },
        "registry_artifacts": artifact_registry,
        "namespace_root": namespace_registry.integer_root,
        "vocabulary_intersections": 0,
        "control_mappings_use_outcomes": False,
        "human_audit_status": "PENDING",
        "e05b_training_started": False,
        "main_registry_sealed": True,
    }
    seal_path = run_dir / "e05b_registry_seal.json"
    write_json_strict(seal_path, seal)
    artifact_registry["e05b_registry_seal"] = _descriptor(seal_path)
    return artifact_registry, seal


def _results_summary(
    *,
    dry_run: bool,
    design_status: str,
    statistical_report: Mapping[str, Any] | None,
    seed_summaries: Mapping[str, Any],
    e05b_sealed: bool,
) -> str:
    lines = [
        "# E05a Semantic Protocol and Leakage Lock — 결과 요약",
        "",
        "## 판정",
        "",
        "- execution_status: `PASS`",
        f"- run_mode: `{'dry_run' if dry_run else 'main'}`",
        f"- e05a_design_status: `{design_status}`",
        "- H5 claim opened: `false`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "",
        "E05a는 semantic claim evidence가 아니라 E05b 실행 전 design-validity "
        "gate다.",
        "",
        "## Seed별 개발 지표",
        "",
        "| Seed | Factorized affected | Shared affected | Oracle affected |",
        "|---:|---:|---:|---:|",
    ]
    for seed, summary in sorted(seed_summaries.items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {seed} | "
            f"{summary['conditions']['factorized']['affected_read_mse']:.8g} | "
            f"{summary['conditions']['shared']['affected_read_mse']:.8g} | "
            f"{summary['conditions']['oracle_demand']['affected_read_mse']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## 특기사항",
            "",
            "| 항목 | 기록 |",
            "|---|---|",
            "| Legacy E05 | v6.1 evidence에서 사용하지 않음 |",
            "| Old value | lexical token은 숨기고 oracle address의 state read만 허용 |",
            "| Training | affected + unaffected retention, state-MSE weight 0 |",
            "| Controls | outcome-independent mapping; hidden oracle candidate 재사용 없음 |",
            f"| E05b registry | {'sealed' if e05b_sealed else 'not generated'} |",
            "| Human audit | 두 사람의 독립 review 전까지 E05b training 금지 |",
        ]
    )
    if statistical_report is not None:
        lines.extend(
            [
                "",
                "## Registered gate",
                "",
                f"- statistical GO: `{statistical_report['go']}`",
                "- Control estimand: ADD/INVALIDATE equal-weight; PRESERVE descriptive.",
            ]
        )
    lines.extend(
        [
            "",
            "## 연구 흐름에서의 짧은 해석",
            "",
            "H1–H4 controlled core 이후 semantic demand-composition anchor를 시작하기 "
            "위한 leakage·control validity만 판정했다. 이 결과 자체는 semantic "
            "understanding을 입증하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    _validate_frozen_config_path(args.config)
    frozen_config, e05b_config = validate_frozen_e05_protocol()
    e04 = validate_frozen_e04_dependency(args.artifact_root)
    e00 = validate_legacy_e00(args.artifact_root, require_full=True)
    dependencies = [
        e00,
        e04.dependency_record(),
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
        raise ProvenanceValidationError("Runtime E05a config differs from frozen config.")

    torch.set_num_threads(1)
    memory_spec = _memory_spec(config)
    encoder = _encoder(config, memory_spec)
    training_config = _training_config(config, dry_run=args.dry_run)
    namespace_registry = SemanticNamespaceRegistry.from_config(
        config["namespace"],
        dry_run=args.dry_run,
    )
    pilot = config["pilot"]
    seeds = (
        [int(pilot["dry_seed"])]
        if args.dry_run
        else [int(value) for value in pilot["seeds"]]
    )
    train_count_per_cell = int(
        pilot[
            "dry_train_count_per_cell"
            if args.dry_run
            else "train_count_per_cell"
        ]
    )
    validation_count_per_cell = int(
        pilot[
            "dry_validation_count_per_cell"
            if args.dry_run
            else "validation_count_per_cell"
        ]
    )
    operations = _operation_values(pilot["operations"])
    domains = tuple(str(value) for value in pilot["seen_domains"])
    templates = tuple(str(value) for value in pilot["seen_templates"])

    access_manifest = _access_manifest(config)
    access_path = run_dir / "semantic_access_manifest.json"
    write_json_strict(access_path, access_manifest)

    seed_metrics: dict[int, SemanticAnchorSeedMetrics] = {}
    raw_rows: list[dict[str, object]] = []
    seed_effect_rows: list[dict[str, object]] = []
    pilot_registry_rows: list[dict[str, object]] = []
    pairing_rows: list[dict[str, object]] = []
    seed_summaries: dict[str, object] = {}
    vocabularies: dict[str, dict[str, set[str]]] = {}
    budget_records: list[dict[str, object]] = []

    for seed_slot, seed in enumerate(seeds):
        train = build_balanced_semantic_examples(
            namespace_registry=namespace_registry,
            namespace_name="pilot_train",
            checkpoint_seed=seed,
            seed_slot=seed_slot,
            operations=operations,
            domains=domains,
            templates=templates,
            count_per_cell=train_count_per_cell,
            memory_spec=memory_spec,
        )
        validation = build_balanced_semantic_examples(
            namespace_registry=namespace_registry,
            namespace_name="pilot_validation",
            checkpoint_seed=seed,
            seed_slot=seed_slot,
            operations=operations,
            domains=domains,
            templates=templates,
            count_per_cell=validation_count_per_cell,
            memory_spec=memory_spec,
        )
        assert_disjoint_semantic_vocabularies({"train": train, "validation": validation})
        _update_vocab_sets(vocabularies, "e05a_pilot_train", train)
        _update_vocab_sets(vocabularies, "e05a_pilot_validation", validation)
        pilot_registry_rows.extend(
            semantic_example_registry_row(
                example,
                split=split,
                seed_slot=seed_slot,
            )
            for split, examples in (
                ("pilot_train", train),
                ("pilot_validation", validation),
            )
            for example in examples
        )
        pairings = build_control_pairing_registry(
            validation,
            norm_tolerance=float(config["controls"]["wrong_address_norm_tolerance"]),
        )
        pairing_rows.extend(
            {"checkpoint_seed": seed, **row} for row in pairings.to_rows()
        )
        training = train_matched_semantic_pair(
            train,
            encoder=encoder,
            hidden_dim=int(config["model"]["path_hidden_dim"]),
            config=training_config,
            seed=seed,
            device=device,
        )
        checkpoint_descriptors: dict[str, object] = {}
        for name, model in (
            ("factorized", training.factorized),
            ("shared", training.shared),
        ):
            path = run_dir / f"seed{seed}_{name}.pt"
            torch.save(model.state_dict(), path)
            checkpoint_descriptors[name] = _descriptor(path)

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
            "seed": seed,
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
        budget_records.append(budget)
        seed_effect_rows.append(
            {
                "schema_version": 1,
                **budget,
                "shared_minus_factorized_affected": (
                    condition_summary["shared"]["affected_read_mse"]
                    - condition_summary["factorized"]["affected_read_mse"]
                ),
                **{
                    f"{name}_minus_full_affected": (
                        condition_summary[name]["affected_read_mse"]
                        - condition_summary["factorized"]["affected_read_mse"]
                    )
                    for name in CONTROL_NAMES
                },
            }
        )
        seed_summaries[str(seed)] = {
            "training_final_loss": dict(training.final_loss),
            "conditions": condition_summary,
            "budget": budget,
            "checkpoints": checkpoint_descriptors,
        }

    _assert_disjoint_vocab_sets(vocabularies)
    pilot_registry_path = run_dir / "pilot_namespace_registry.jsonl"
    pairing_path = run_dir / "pilot_control_pairings.jsonl"
    raw_path = run_dir / "pilot_semantic_metrics.jsonl"
    effects_path = run_dir / "pilot_seed_effects.jsonl"
    write_jsonl_strict(pilot_registry_path, pilot_registry_rows)
    write_jsonl_strict(pairing_path, pairing_rows)
    write_jsonl_strict(raw_path, raw_rows)
    write_jsonl_strict(effects_path, seed_effect_rows)

    statistical_report: dict[str, object] | None = None
    if not args.dry_run:
        statistical_report = evaluate_e05a_go(
            seed_metrics,
            thresholds=_thresholds(config),
            bootstrap_seeds={
                str(key): int(value)
                for key, value in config["statistics"]["bootstrap_seeds"].items()
            },
        )

    budget_match = bool(
        budget_records
        and len({row["parameter_count"] for row in budget_records}) == 1
        and len(
            {row["dense_multiply_adds_per_example"] for row in budget_records}
        )
        == 1
        and all(row["common_initialization"] for row in budget_records)
        and all(row["common_schedule"] for row in budget_records)
    )
    static_gates = {
        "forbidden_access": bool(
            access_manifest["forbidden_access_test_passed"]
        ),
        "namespace_integrity": True,
        "parameter_and_budget_match": budget_match,
        "supercede_absent_from_e05a": all(
            row["operation"] != Operation.SUPERSEDE.value for row in raw_rows
        ),
    }
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

    e05b_artifacts: dict[str, dict[str, object]] = {}
    e05b_seal: dict[str, object] | None = None
    if design_go:
        main_namespace = SemanticNamespaceRegistry.from_config(
            config["namespace"],
            dry_run=False,
        )
        e05b_artifacts, e05b_seal = _generate_e05b_seal(
            run_dir=run_dir,
            e05a_config=config,
            e05b_config=e05b_config,
            namespace_registry=main_namespace,
            memory_spec=memory_spec,
            existing_vocabularies=vocabularies,
        )

    summary_text = _results_summary(
        dry_run=args.dry_run,
        design_status=design_status,
        statistical_report=statistical_report,
        seed_summaries=seed_summaries,
        e05b_sealed=e05b_seal is not None,
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    artifacts: dict[str, dict[str, object]] = {
        "semantic_access_manifest": _descriptor(access_path),
        "pilot_namespace_registry": _descriptor(
            pilot_registry_path,
            rows=len(pilot_registry_rows),
        ),
        "pilot_control_pairings": _descriptor(
            pairing_path,
            rows=len(pairing_rows),
        ),
        "pilot_semantic_metrics": _descriptor(raw_path, rows=len(raw_rows)),
        "pilot_seed_effects": _descriptor(
            effects_path,
            rows=len(seed_effect_rows),
        ),
        "results_summary_ko": _descriptor(summary_path),
        **e05b_artifacts,
    }
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "e05a_design_status": design_status,
        "full_h5_lite_claim_open": False,
        "protocol_lock": {
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "protocol_lock_sha256": PINNED_PROTOCOL_LOCK_SHA256,
            "e05a_config_canonical_sha256": (
                PINNED_E05A_CONFIG_CANONICAL_SHA256
            ),
            "e05a_config_file_sha256": PINNED_E05A_CONFIG_FILE_SHA256,
            "e05b_config_canonical_sha256": (
                PINNED_E05B_CONFIG_CANONICAL_SHA256
            ),
            "e05b_config_file_sha256": PINNED_E05B_CONFIG_FILE_SHA256,
        },
        "dependency_disposition": {
            "e04_run_id": e04.run_dir.name,
            "e04_full_h4_claim_open": True,
            "original_e02_confirmatory_status": "INCONCLUSIVE",
            "e02b_prospective_repair_status": "SUPPORTED",
        },
        "access_contract": access_manifest,
        "static_design_gates": static_gates,
        "statistical_design_gates": statistical_report,
        "seed_summaries": seed_summaries,
        "namespace": {
            "dry_run": args.dry_run,
            "opened_names": list(namespace_registry.names),
            "e05b_registry_generated": e05b_seal is not None,
            "vocabulary_intersections": 0,
        },
        "human_audit": {
            "status": "PENDING" if e05b_seal is not None else "NOT_GENERATED",
            "required_before_e05b_training": True,
            "ai_agent_review_substitution_allowed": False,
        },
        "artifacts": artifacts,
        "claim_gate": {
            "e05a_is_claim_evidence": False,
            "e05b_training_eligible_after_human_audit": design_go,
            "h5_lite_open": False,
        },
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
