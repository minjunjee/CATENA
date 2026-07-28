from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json
from catena.eval.postcore_metrics import exact_sign_flip
from catena.eval.statistics import equivalence_within, paired_bootstrap
from catena.post_e21.contracts import (
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.text_anchor import (
    FrozenHashNgramEncoder,
    MatchedTextTransactionController,
    TextController,
    evaluate_text_controller,
    matched_parameter_count,
    oracle_rows,
    summarize_rows,
    train_text_controller,
)
from catena.post_e21.text_transactions import (
    MagnitudeOperation,
    TextDemand,
    TextSplit,
    TextTransaction,
    build_text_transactions,
    shuffled_texts,
    tensor_sha256,
    visible_registry_rows,
    wrong_entity_texts,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e25b_text_transaction_anchor"
DEFAULT_CONFIG = "configs/e25b_text_transaction_anchor.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E25B_TEXT_TRANSACTION_LOCK.json"

_AUDIT_FIELDS = (
    "audit_id",
    "example_id",
    "minimal_pair_id",
    "state_counterpair_id",
    "split",
    "text",
    "direct_fact_query",
    "derived_action_query",
    "old_rule_query",
    "unaffected_query",
    "private_demand_for_audit",
    "private_erase_entity",
    "private_write_entity",
    "private_domain",
    "private_template_index",
    "private_evidence_index",
    "private_active_state",
    "private_magnitude_operation",
    "private_old_value_label",
    "private_new_value_label",
    "private_effective_day",
    "private_coordinate_mask",
    "private_current_state_sha256",
    "private_old_value_sha256",
    "private_new_value_sha256",
    "private_target_state_sha256",
    "gold_direct_fact_entity",
    "gold_direct_fact_answer",
    "gold_direct_fact_vector_sha256",
    "gold_direct_fact_vector_values",
    "gold_derived_action",
    "gold_derived_action_rule",
    "gold_old_rule_status",
    "gold_old_rule_status_definition",
    "gold_unaffected_retention",
    "reviewer_a_semantic_preservation",
    "reviewer_a_operation_leakage",
    "reviewer_a_entity_ambiguity",
    "reviewer_a_old_value_leakage",
    "reviewer_a_gold_consistency",
    "reviewer_b_semantic_preservation",
    "reviewer_b_operation_leakage",
    "reviewer_b_entity_ambiguity",
    "reviewer_b_old_value_leakage",
    "reviewer_b_gold_consistency",
    "adjudication_semantic_preservation",
    "adjudication_operation_leakage",
    "adjudication_entity_ambiguity",
    "adjudication_old_value_leakage",
    "adjudication_gold_consistency",
    "notes",
)

_AUDIT_REVIEW_FIELDS = tuple(
    field
    for field in _AUDIT_FIELDS
    if field.startswith(("reviewer_", "adjudication_")) or field == "notes"
)
_AUDIT_IMMUTABLE_FIELDS = tuple(
    field for field in _AUDIT_FIELDS if field not in _AUDIT_REVIEW_FIELDS
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-audit", action="store_true")
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--audit-population-lock", type=Path)
    return parser


def _initialize_e25b_run(
    *,
    config_path: str,
    artifact_root: str,
    device_request: str,
    run_mode: str,
) -> tuple[dict[str, Any], Path, torch.device]:
    common_mode = "UNSPECIFIED" if run_mode == "AUDIT_PREPARATION" else run_mode
    initialized = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=config_path,
        artifact_root=artifact_root,
        device_request=device_request,
        run_mode=common_mode,
    )
    if run_mode == "AUDIT_PREPARATION":
        manifest_path = initialized[1] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("run_mode") != "UNSPECIFIED":
            raise RuntimeError("unexpected common manifest before audit-mode specialization")
        manifest["run_mode"] = "AUDIT_PREPARATION"
        write_json(manifest_path, manifest)
    return initialized


def _runtime_config(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    runtime = cast(dict[str, Any], json.loads(json.dumps(config)))
    if not dry_run:
        return runtime
    runtime["seeds"] = [int(config["seeds"][0])]
    runtime["data"]["slots"] = 8
    runtime["data"]["value_dim"] = 8
    runtime["data"]["train_examples_per_demand"] = 4
    runtime["data"]["validation_examples_per_demand"] = 2
    runtime["data"]["test_examples_per_demand"] = 4
    runtime["encoder"]["output_dim"] = 24
    runtime["encoder"]["buckets"] = 128
    runtime["model"]["hidden_dim"] = 24
    runtime["training"]["steps"] = 2
    runtime["training"]["batch_size"] = 4
    runtime["evaluation"]["batch_size"] = 8
    return runtime


def _namespace_seed(config: Mapping[str, Any], split: TextSplit, seed: int) -> int:
    root = int(config["namespaces"]["seed_root"])
    split_index = list(TextSplit).index(split)
    return root + 10_000_000 * split_index + int(seed)


def _examples(
    *,
    config: Mapping[str, Any],
    split: TextSplit,
    seed: int,
    count: int,
) -> list[TextTransaction]:
    return build_text_transactions(
        split=split,
        demand_families=[TextDemand(str(value)) for value in config["data"]["demand_families"]],
        count_per_demand=count,
        slots=int(config["data"]["slots"]),
        value_dim=int(config["data"]["value_dim"]),
        namespace_seed=_namespace_seed(config, split, seed),
        semantic_value_seed=int(config["namespaces"]["semantic_value_seed"]),
        blacklist=[str(value) for value in config["audit"]["lexical_blacklist"]],
    )


def _audit_examples(config: Mapping[str, Any]) -> list[TextTransaction]:
    splits = [TextSplit(str(value)) for value in config["audit"]["splits"]]
    required = int(config["audit"]["items"])
    counts = {
        TextSplit(str(split)): int(count)
        for split, count in config["audit"]["items_per_demand_by_split"].items()
    }
    if set(counts) != set(splits):
        raise ValueError("audit per-split counts do not cover the registered splits")
    rows: list[TextTransaction] = []
    for split in splits:
        rows.extend(
            _examples(
                config=config,
                split=split,
                seed=int(config["audit"]["population_seed"]),
                count=counts[split],
            )
        )
    if len(rows) != required:
        raise RuntimeError("audit generator did not create the registered population")
    return rows


def _audit_population_rows(
    examples: Sequence[TextTransaction],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, example in enumerate(examples):
        direct_index = (
            example.write_index if example.demand is TextDemand.ADDRESS else example.erase_index
        )
        rows.append(
            {
                "audit_id": f"E25B-{index + 1:04d}-{example.example_id}",
                "example_id": example.example_id,
                "minimal_pair_id": example.minimal_pair_id,
                "state_counterpair_id": example.state_counterpair_id,
                "split": example.split.value,
                "text": example.text,
                "direct_fact_query": example.query_direct,
                "derived_action_query": example.query_derived,
                "old_rule_query": example.query_old_rule,
                "unaffected_query": example.query_unaffected,
                "private_demand_for_audit": example.demand.value,
                "private_erase_entity": example.entity,
                "private_write_entity": example.other_entity,
                "private_domain": example.domain,
                "private_template_index": str(example.template_index),
                "private_evidence_index": str(example.evidence_index),
                "private_active_state": "ACTIVE" if example.active else "INACTIVE",
                "private_magnitude_operation": example.magnitude_operation,
                "private_old_value_label": example.old_value_label,
                "private_new_value_label": example.new_value_label,
                "private_effective_day": str(example.day),
                "private_coordinate_mask": "".join(
                    str(int(value)) for value in example.coordinate_mask.tolist()
                ),
                "private_current_state_sha256": tensor_sha256(example.state),
                "private_old_value_sha256": tensor_sha256(example.old_value),
                "private_new_value_sha256": tensor_sha256(example.new_value),
                "private_target_state_sha256": tensor_sha256(example.target_state),
                "gold_direct_fact_entity": example.direct_fact_entity,
                "gold_direct_fact_answer": example.direct_fact_answer,
                "gold_direct_fact_vector_sha256": tensor_sha256(example.target_state[direct_index]),
                "gold_direct_fact_vector_values": "|".join(
                    f"{float(value):.17g}" for value in example.target_state[direct_index].tolist()
                ),
                "gold_derived_action": example.derived_action_label,
                "gold_derived_action_rule": example.derived_action_rule,
                "gold_old_rule_status": example.old_rule_status.value,
                "gold_old_rule_status_definition": (
                    "FULL=all prior rule content remains; "
                    "PARTIAL=only a registered coordinate subset remains; "
                    "NONE=no prior rule content remains"
                ),
                "gold_unaffected_retention": "UNCHANGED",
            }
        )
    _validate_audit_population_structure(rows)
    return rows


def _audit_population_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    payload = [{field: str(row[field]) for field in _AUDIT_IMMUTABLE_FIELDS} for row in rows]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_audit_population_structure(rows: Sequence[Mapping[str, str]]) -> None:
    audit_ids = [str(row["audit_id"]) for row in rows]
    example_ids = [str(row["example_id"]) for row in rows]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("audit_id values must be unique")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("audit example_id values must be unique")
    member_keys = [
        (
            str(row["split"]),
            str(row["minimal_pair_id"]),
            str(row["private_demand_for_audit"]),
        )
        for row in rows
    ]
    if len(member_keys) != len(set(member_keys)):
        raise ValueError("minimal-pair demand members must be unique")
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["split"]), str(row["minimal_pair_id"])),
            [],
        ).append(row)
    expected_demands = {demand.value for demand in TextDemand}
    shared_fields = (
        "private_current_state_sha256",
        "private_old_value_sha256",
        "private_new_value_sha256",
        "private_effective_day",
        "private_old_value_label",
        "private_new_value_label",
    )
    for key, group in grouped.items():
        demands = {str(row["private_demand_for_audit"]) for row in group}
        if len(group) != len(TextDemand) or demands != expected_demands:
            raise ValueError(f"audit minimal pair is incomplete: {key}")
        for field in shared_fields:
            if len({str(row[field]) for row in group}) != 1:
                raise ValueError(f"audit minimal pair does not share {field}: {key}")

    counterpairs: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        counterpairs.setdefault(str(row["state_counterpair_id"]), []).append(row)
    counterpair_shared_fields = (
        "split",
        "private_demand_for_audit",
        "text",
        "private_erase_entity",
        "private_write_entity",
        "private_domain",
        "private_template_index",
        "private_evidence_index",
        "private_magnitude_operation",
        "private_new_value_label",
        "private_effective_day",
    )
    for counterpair_id, group in counterpairs.items():
        # The locked 300-item population is not divisible by eight (four
        # demands × two private-state branches).  It contains 296 rows in
        # complete counterpairs and four deterministically locked singleton
        # tails; the latter remain ID-bound but are not treated as pair audits.
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise ValueError(f"state counterpair is incomplete: {counterpair_id}")
        if {str(row["private_active_state"]) for row in group} != {
            "ACTIVE",
            "INACTIVE",
        }:
            raise ValueError(f"state counterpair lacks both branches: {counterpair_id}")
        for field in counterpair_shared_fields:
            if len({str(row[field]) for row in group}) != 1:
                raise ValueError(
                    f"state counterpair leaks private branch through {field}: {counterpair_id}"
                )
        if len({str(row["private_current_state_sha256"]) for row in group}) != 2:
            raise ValueError(f"state counterpair must differ in private state: {counterpair_id}")


def _write_csv(
    *,
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fieldnames})


def _write_audit_artifacts(
    *,
    run_dir: Path,
    examples: Sequence[TextTransaction],
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    rows = _audit_population_rows(examples)
    if len(rows) != int(config["audit"]["items"]):
        raise ValueError("audit population is not the registered size")
    items_path = run_dir / str(config["audit"]["items_filename"])
    review_template_path = run_dir / str(config["audit"]["review_template_filename"])
    population_lock_path = run_dir / str(config["audit"]["population_lock_filename"])
    _write_csv(path=items_path, rows=rows, fieldnames=_AUDIT_IMMUTABLE_FIELDS)
    _write_csv(path=review_template_path, rows=rows, fieldnames=_AUDIT_FIELDS)
    population_sha256 = _audit_population_sha256(rows)
    population_lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": str(config["protocol"]["protocol_id"]),
        "items_filename": items_path.name,
        "review_template_filename": review_template_path.name,
        "review_work_filename": str(config["audit"]["review_work_filename"]),
        "population_lock_filename": population_lock_path.name,
        "rows": len(rows),
        "immutable_fields": list(_AUDIT_IMMUTABLE_FIELDS),
        "population_sha256": population_sha256,
        "items_csv_sha256": file_sha256(items_path),
        "review_template_sha256": file_sha256(review_template_path),
        "config_sha256": file_sha256(config_path),
    }
    population_lock_path.write_text(
        json.dumps(population_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "items": {
            "path": str(items_path.resolve()),
            "sha256": file_sha256(items_path),
            "rows": len(rows),
        },
        "review_template": {
            "path": str(review_template_path.resolve()),
            "sha256": file_sha256(review_template_path),
            "rows": len(rows),
        },
        "population_lock": {
            "path": str(population_lock_path.resolve()),
            "sha256": file_sha256(population_lock_path),
            "population_sha256": population_sha256,
        },
    }


def _validate_human_audit(
    path: Path,
    *,
    population_lock_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"reviewed audit CSV is missing or unsafe: {path}")
    if not population_lock_path.is_file() or population_lock_path.is_symlink():
        raise ValueError("audit population lock is missing or unsafe")
    expected_names = {
        "items": str(config["audit"]["items_filename"]),
        "template": str(config["audit"]["review_template_filename"]),
        "work": str(config["audit"]["review_work_filename"]),
        "lock": str(config["audit"]["population_lock_filename"]),
    }
    if path.name != expected_names["work"]:
        raise ValueError("review work CSV does not use the registered filename")
    if population_lock_path.name != expected_names["lock"]:
        raise ValueError("audit population lock does not use the registered filename")
    preparation_dir = population_lock_path.resolve().parent
    try:
        path.resolve().relative_to(preparation_dir)
    except ValueError:
        pass
    else:
        raise ValueError("review work CSV must be outside the immutable preparation directory tree")
    population_lock = json.loads(population_lock_path.read_text(encoding="utf-8"))
    if not isinstance(population_lock, dict):
        raise ValueError("audit population lock must be a JSON object")
    items_path = population_lock_path.parent / expected_names["items"]
    if not items_path.is_file() or items_path.is_symlink():
        raise ValueError("locked audit items CSV is missing or unsafe")
    review_template_path = population_lock_path.parent / expected_names["template"]
    if not review_template_path.is_file() or review_template_path.is_symlink():
        raise ValueError("locked audit review template is missing or unsafe")
    expected_population = _audit_population_rows(_audit_examples(config))
    expected_population_sha256 = _audit_population_sha256(expected_population)
    with items_path.open("r", encoding="utf-8", newline="") as handle:
        items_reader = csv.DictReader(handle)
        if tuple(items_reader.fieldnames or ()) != _AUDIT_IMMUTABLE_FIELDS:
            raise ValueError("locked audit items columns do not match the registered schema")
        prepared_population = list(items_reader)
    _validate_audit_population_structure(prepared_population)
    if prepared_population != expected_population:
        raise ValueError("locked audit items differ from the registered population")
    if _audit_population_sha256(prepared_population) != expected_population_sha256:
        raise ValueError("locked audit items population hash mismatch")
    with review_template_path.open("r", encoding="utf-8", newline="") as handle:
        template_reader = csv.DictReader(handle)
        if tuple(template_reader.fieldnames or ()) != _AUDIT_FIELDS:
            raise ValueError("locked review template columns do not match the schema")
        template_rows = list(template_reader)
    template_immutable = [
        {field: str(row.get(field, "")) for field in _AUDIT_IMMUTABLE_FIELDS}
        for row in template_rows
    ]
    if template_immutable != expected_population:
        raise ValueError("locked review template differs from the registered population")
    if any(str(row.get(field, "")) for row in template_rows for field in _AUDIT_REVIEW_FIELDS):
        raise ValueError("immutable review template already contains reviewer edits")
    expected_lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": str(config["protocol"]["protocol_id"]),
        "items_filename": expected_names["items"],
        "review_template_filename": expected_names["template"],
        "review_work_filename": expected_names["work"],
        "population_lock_filename": expected_names["lock"],
        "rows": int(config["audit"]["items"]),
        "immutable_fields": list(_AUDIT_IMMUTABLE_FIELDS),
        "population_sha256": expected_population_sha256,
        "items_csv_sha256": file_sha256(items_path),
        "review_template_sha256": file_sha256(review_template_path),
        "config_sha256": file_sha256(config_path),
    }
    for key, value in expected_lock.items():
        if population_lock.get(key) != value:
            raise ValueError(f"audit population lock mismatch: {key}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _AUDIT_FIELDS:
            raise ValueError("reviewed audit columns do not match the locked schema")
        rows = list(reader)
    required = int(config["audit"]["items"])
    if len(rows) != required:
        raise ValueError(f"reviewed audit must contain exactly {required} rows")
    immutable_rows = [
        {field: str(row.get(field, "")) for field in _AUDIT_IMMUTABLE_FIELDS} for row in rows
    ]
    _validate_audit_population_structure(immutable_rows)
    if immutable_rows != expected_population:
        raise ValueError("reviewed audit population differs from the prepared population")
    if _audit_population_sha256(immutable_rows) != expected_population_sha256:
        raise ValueError("reviewed audit population hash mismatch")
    judgments = (
        "semantic_preservation",
        "operation_leakage",
        "entity_ambiguity",
        "old_value_leakage",
        "gold_consistency",
    )
    agreed = 0
    total = 0
    unresolved: list[str] = []
    critical_failures: list[str] = []
    for row in rows:
        for judgment in judgments:
            left = str(row.get(f"reviewer_a_{judgment}", "")).strip().upper()
            right = str(row.get(f"reviewer_b_{judgment}", "")).strip().upper()
            if left not in {"PASS", "FAIL"} or right not in {"PASS", "FAIL"}:
                raise ValueError("each reviewer judgment must be PASS or FAIL")
            total += 1
            agreed += int(left == right)
            adjudication = str(row.get(f"adjudication_{judgment}", "")).strip().upper()
            if left != right and adjudication not in {"PASS", "FAIL"}:
                unresolved.append(str(row.get("audit_id", "")))
            final = left if left == right else adjudication
            if final == "FAIL":
                critical_failures.append(f"{row.get('audit_id', '')}:{judgment}")
    agreement = agreed / total
    passed = bool(
        agreement >= float(config["audit"]["minimum_agreement"])
        and not unresolved
        and not critical_failures
    )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "population_lock_path": str(population_lock_path.resolve()),
        "population_lock_sha256": file_sha256(population_lock_path),
        "population_sha256": expected_population_sha256,
        "items_csv_sha256": file_sha256(items_path),
        "rows": len(rows),
        "agreement": agreement,
        "unresolved": sorted(set(unresolved)),
        "critical_failures": critical_failures,
        "passed": passed,
    }


def _assessment(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "DRY_RUN_NON_EVIDENCE",
            "supported": False,
            "reason": "Development namespace and reduced runtime only.",
        }
    lookup = {
        (
            int(row["seed"]),
            str(row["split"]),
            str(row["demand_family"]),
            str(row["magnitude_operation"]),
            str(row["condition"]),
        ): row
        for row in seed_rows
    }
    contrasts = {
        "value": ("dual", "diagonal"),
        "address": ("diagonal", "separate_address"),
        "state_conditioning": ("separate_address", "state_aware"),
    }
    seeds = [int(value) for value in config["seeds"]]
    effects: dict[str, Any] = {}
    all_interactions = True
    for demand, (simpler, richer) in contrasts.items():
        values = [
            float(
                lookup[(seed, "primary", demand, "not_applicable", simpler)][
                    "affected_correction_mse"
                ]
            )
            - float(
                lookup[(seed, "primary", demand, "not_applicable", richer)][
                    "affected_correction_mse"
                ]
            )
            for seed in seeds
        ]
        mean = sum(values) / len(values)
        p_value = exact_sign_flip(values, alternative="greater")
        passed = bool(
            mean >= float(config["statistics"]["selective_gain_sesoi"])
            and all(value > 0.0 for value in values)
            and p_value <= float(config["statistics"]["alpha"])
        )
        effects[demand] = {
            "simpler": simpler,
            "richer": richer,
            "seed_effects": values,
            "mean": mean,
            "exact_sign_flip_p": p_value,
            "passed": passed,
        }
        all_interactions = bool(all_interactions and passed)

    asymmetric_operations: dict[str, Any] = {}
    asymmetric_seed_means: list[float] = []
    operation_effects: dict[str, list[float]] = {}
    for operation in (
        MagnitudeOperation.ADD.value,
        MagnitudeOperation.INVALIDATE.value,
    ):
        values = [
            float(
                lookup[
                    (
                        seed,
                        "primary",
                        TextDemand.MAGNITUDE.value,
                        operation,
                        TextController.TIED.value,
                    )
                ]["affected_correction_mse"]
            )
            - float(
                lookup[
                    (
                        seed,
                        "primary",
                        TextDemand.MAGNITUDE.value,
                        operation,
                        TextController.DUAL.value,
                    )
                ]["affected_correction_mse"]
            )
            for seed in seeds
        ]
        operation_effects[operation] = values
        mean = sum(values) / len(values)
        p_value = exact_sign_flip(values, alternative="greater")
        passed = bool(
            mean >= float(config["statistics"]["asymmetric_magnitude_gain_sesoi"])
            and all(value > 0.0 for value in values)
            and p_value <= float(config["statistics"]["alpha"])
        )
        asymmetric_operations[operation] = {
            "seed_effects": values,
            "mean": mean,
            "exact_sign_flip_p": p_value,
            "passed": passed,
        }
    asymmetric_seed_means = [
        0.5
        * (
            operation_effects[MagnitudeOperation.ADD.value][index]
            + operation_effects[MagnitudeOperation.INVALIDATE.value][index]
        )
        for index in range(len(seeds))
    ]
    asymmetric_mean = sum(asymmetric_seed_means) / len(asymmetric_seed_means)
    asymmetric_p = exact_sign_flip(asymmetric_seed_means, alternative="greater")
    asymmetric_supported = bool(
        all(row["passed"] for row in asymmetric_operations.values())
        and asymmetric_mean >= float(config["statistics"]["asymmetric_magnitude_gain_sesoi"])
        and all(value > 0.0 for value in asymmetric_seed_means)
        and asymmetric_p <= float(config["statistics"]["alpha"])
    )
    asymmetric_gate = {
        "operations": asymmetric_operations,
        "seed_effects_equal_weight_add_invalidate": asymmetric_seed_means,
        "mean": asymmetric_mean,
        "exact_sign_flip_p": asymmetric_p,
        "passed": asymmetric_supported,
    }

    supersede_tied = np.asarray(
        [
            float(
                lookup[
                    (
                        seed,
                        "primary",
                        TextDemand.MAGNITUDE.value,
                        MagnitudeOperation.SUPERSEDE.value,
                        TextController.TIED.value,
                    )
                ]["affected_correction_mse"]
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )
    supersede_dual = np.asarray(
        [
            float(
                lookup[
                    (
                        seed,
                        "primary",
                        TextDemand.MAGNITUDE.value,
                        MagnitudeOperation.SUPERSEDE.value,
                        TextController.DUAL.value,
                    )
                ]["affected_correction_mse"]
            )
            for seed in seeds
        ],
        dtype=np.float64,
    )
    supersede_interval = paired_bootstrap(
        supersede_dual,
        supersede_tied,
        samples=int(config["statistics"]["equivalence_bootstrap_samples"]),
        seed=int(config["statistics"]["equivalence_bootstrap_seed"]),
        confidence=float(config["statistics"]["equivalence_bootstrap_confidence"]),
    )
    supersede_margin = float(config["statistics"]["supersede_absolute_equivalence_margin"])
    supersede_tied_floor = float(supersede_tied.max())
    supersede_dual_floor = float(supersede_dual.max())
    supersede_gate = {
        "estimand": "dual_minus_tied_affected_mse",
        "seed_effects": (supersede_dual - supersede_tied).tolist(),
        "estimate": supersede_interval.estimate,
        "ci95": [supersede_interval.low, supersede_interval.high],
        "equivalence_margin": supersede_margin,
        "equivalence_passed": equivalence_within(
            supersede_interval,
            supersede_margin,
        ),
        "tied_floor": supersede_tied_floor,
        "dual_floor": supersede_dual_floor,
        "tied_floor_passed": supersede_tied_floor
        <= float(config["statistics"]["supersede_tied_floor_ceiling"]),
        "dual_floor_passed": supersede_dual_floor
        <= float(config["statistics"]["supersede_dual_floor_ceiling"]),
    }
    supersede_gate["passed"] = bool(
        supersede_gate["equivalence_passed"]
        and supersede_gate["tied_floor_passed"]
        and supersede_gate["dual_floor_passed"]
    )
    all_interactions = bool(
        all_interactions and asymmetric_gate["passed"] and supersede_gate["passed"]
    )

    oracle_floor = max(
        float(row["affected_correction_mse"])
        for row in seed_rows
        if row["condition"] == "oracle_demand"
    )
    shared_encoder_controller_floor = max(
        float(row["affected_correction_mse"])
        for row in seed_rows
        if row["condition"] == TextController.STATE_AWARE.value and row["split"] == "primary"
    )
    minimum_state_aware_address_accuracy = min(
        min(
            float(row["erase_address_accuracy"]),
            float(row["write_address_accuracy"]),
        )
        for row in seed_rows
        if row["condition"] == TextController.STATE_AWARE.value and row["split"] == "primary"
    )
    retention = max(
        float(row["unaffected_retention_mse"])
        for row in seed_rows
        if row["condition"] in {variant.value for variant in TextController}
    )
    full_by_key = {
        (
            int(row["seed"]),
            str(row["split"]),
            str(row["demand_family"]),
            str(row["magnitude_operation"]),
        ): float(row["affected_correction_mse"])
        for row in seed_rows
        if row["condition"] == TextController.STATE_AWARE.value
    }
    control_degradations: dict[str, float] = {}
    for condition in ("shuffled_text", "wrong_entity", "transaction_only_zero_state", "state_only"):
        values = [
            float(row["affected_correction_mse"])
            - full_by_key[
                (
                    int(row["seed"]),
                    str(row["split"]),
                    str(row["demand_family"]),
                    str(row["magnitude_operation"]),
                )
            ]
            for row in seed_rows
            if row["condition"] == condition
        ]
        control_degradations[condition] = sum(values) / len(values)
    controls_passed = all(
        value >= float(config["statistics"]["minimum_control_degradation"])
        for value in control_degradations.values()
    )
    full_recovery = [
        float(row["oracle_headroom_normalized_recovery"])
        for row in seed_rows
        if row["condition"] == TextController.STATE_AWARE.value
        and row["oracle_headroom_identifiable"] is True
    ]
    supported = bool(
        all_interactions
        and oracle_floor <= float(config["statistics"]["oracle_floor_ceiling"])
        and shared_encoder_controller_floor
        <= float(config["statistics"]["shared_encoder_controller_floor_ceiling"])
        and minimum_state_aware_address_accuracy
        >= float(config["statistics"]["minimum_state_aware_address_accuracy"])
        and retention <= float(config["statistics"]["retention_noninferiority"])
        and controls_passed
    )
    return {
        "status": (
            "SUPPORTED_SHARED_TEXT_ARCHITECTURE_DEMAND_ANCHOR" if supported else "NOT_SUPPORTED"
        ),
        "supported": supported,
        "interaction_effects": effects,
        "magnitude_asymmetric_gain": asymmetric_gate,
        "magnitude_supersede_composition_equivalence": supersede_gate,
        "oracle_floor": oracle_floor,
        "shared_encoder_controller_floor": shared_encoder_controller_floor,
        "minimum_state_aware_address_accuracy": minimum_state_aware_address_accuracy,
        "maximum_retention_mse": retention,
        "control_degradations": control_degradations,
        "controls_passed": controls_passed,
        "state_aware_oracle_headroom_normalized_recovery_mean": (
            sum(full_recovery) / len(full_recovery) if full_recovery else None
        ),
    }


def _summary_text(
    *,
    run_mode: str,
    assessment: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> str:
    audit_status: object = (
        "PENDING"
        if assessment.get("status") == "AUDIT_PENDING"
        else (audit["passed"] if audit else "NOT_REQUIRED")
    )
    lines = [
        "# E25b Shared-Text-Encoder Transaction Anchor 결과 요약",
        "",
        f"- Run mode: `{run_mode}`",
        f"- Status: `{assessment['status']}`",
        "- Evidence tier: `CONTROLLED_REFERENCE`",
        "- Encoder: frozen deterministic hash-ngram; controller 간 완전 공유",
        "- Claim boundary: text-form controlled transaction anchor only",
        f"- Human audit: `{audit_status}`",
    ]
    effects = assessment.get("interaction_effects")
    if isinstance(effects, Mapping):
        lines.extend(
            [
                "",
                "## Primary controller interaction",
                "",
                "| Demand | Mean gain | Exact p | Gate |",
                "|---|---:|---:|---|",
            ]
        )
        for demand, row in effects.items():
            lines.append(
                f"| `{demand}` | {float(row['mean']):.6g} | "
                f"{float(row['exact_sign_flip_p']):.6g} | "
                f"{'PASS' if row['passed'] else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                f"- Oracle floor: `{float(assessment['oracle_floor']):.6g}`",
                "- Shared-encoder/controller floor: "
                f"`{float(assessment['shared_encoder_controller_floor']):.6g}`",
                "- Minimum state-aware address accuracy: "
                f"`{float(assessment['minimum_state_aware_address_accuracy']):.6g}`",
                f"- Maximum retention MSE: `{float(assessment['maximum_retention_mse']):.6g}`",
                "- State-aware oracle-headroom recovery: "
                f"`{assessment['state_aware_oracle_headroom_normalized_recovery_mean']}`",
            ]
        )
    asymmetric = assessment.get("magnitude_asymmetric_gain")
    supersede = assessment.get("magnitude_supersede_composition_equivalence")
    if isinstance(asymmetric, Mapping) and isinstance(supersede, Mapping):
        lines.extend(
            [
                "",
                "## Magnitude protocol",
                "",
                "- Held-out ADD/INVALIDATE tied-minus-dual gain: "
                f"`{float(asymmetric['mean']):.6g}` "
                f"(`{'PASS' if asymmetric['passed'] else 'FAIL'}`)",
                "- Held-out SUPERSEDE dual-minus-tied equivalence CI: "
                f"`[{float(supersede['ci95'][0]):.6g}, "
                f"{float(supersede['ci95'][1]):.6g}]` "
                f"(`{'PASS' if supersede['passed'] else 'FAIL'}`)",
            ]
        )
    lines.extend(
        [
            "",
            (
                "DRY_RUN/AUDIT_PREPARATION은 scientific evidence가 아니며 "
                "official/LM/agent claim을 열지 않는다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_summary(
    *,
    run_dir: Path,
    run_mode: str,
    assessment: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = run_dir / "RESULTS_SUMMARY_KO.md"
    path.write_text(
        _summary_text(run_mode=run_mode, assessment=assessment, audit=audit),
        encoding="utf-8",
    )
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > 45:
        raise RuntimeError("E25b results summary exceeds the one-page contract")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "line_count": line_count,
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E25b config experiment_id mismatch")
    if int(config["encoder"]["visible_candidate_decoder"]["semantic_value_seed"]) != int(
        config["namespaces"]["semantic_value_seed"]
    ):
        raise ValueError("visible candidate decoder seed differs from data semantic seed")
    snapshot = validate_protocol_lock(
        lock_path=LOCK_PATH,
        config_path=args.config,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    run_mode = (
        "DRY_RUN" if args.dry_run else ("AUDIT_PREPARATION" if args.prepare_audit else "MAIN")
    )
    stage = "AUDIT_PREPARATION" if args.prepare_audit else "TRAIN_EVAL"
    if args.dry_run and args.prepare_audit:
        raise ValueError("--dry-run and --prepare-audit are mutually exclusive")
    if run_mode == "MAIN" and args.audit_csv is None:
        raise ValueError("E25b MAIN requires --audit-csv with two completed reviews")
    if run_mode == "MAIN" and args.audit_population_lock is None:
        raise ValueError("E25b MAIN requires --audit-population-lock from audit preparation")
    # Audit validation is deliberately completed before initialize_run.  A
    # malformed or unbound review therefore cannot create a MAIN directory,
    # manifest, or latest.json pointer.
    audit = (
        _validate_human_audit(
            args.audit_csv,
            population_lock_path=args.audit_population_lock,
            config=config,
            config_path=Path(args.config),
        )
        if run_mode == "MAIN"
        else None
    )
    if audit is not None and not audit["passed"]:
        raise ValueError("E25b human audit did not pass its locked gate")
    runtime = _runtime_config(config, dry_run=args.dry_run)
    _, run_dir, device = _initialize_e25b_run(
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode=run_mode,
    )
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)

    if args.prepare_audit:
        audit_examples = _audit_examples(config)
        audit_artifacts = _write_audit_artifacts(
            run_dir=run_dir,
            examples=audit_examples,
            config=config,
            config_path=Path(args.config),
        )
        audit_registry = visible_registry_rows(audit_examples)
        _, data_sha256 = write_data_manifest(
            run_dir=run_dir,
            payload={
                "stage": "AUDIT_PREPARATION",
                "registry": audit_registry,
            },
        )
        artifacts = write_required_rows(
            run_dir=run_dir,
            raw_rows=[],
            seed_rows=[],
            raw_filename=str(config["artifacts"]["raw_metrics_filename"]),
            seed_filename=str(config["artifacts"]["seed_metrics_filename"]),
        )
        assessment = {
            "status": "AUDIT_PENDING",
            "supported": False,
            "audit_artifacts": audit_artifacts,
        }
        metadata = report_contract_metadata(
            run_dir=run_dir,
            snapshot=snapshot,
            data_sha256=data_sha256,
            checkpoint_hashes={},
            evidence_tier="CONTROLLED_REFERENCE",
            claim_eligible=False,
        )
        summary = _write_summary(
            run_dir=run_dir,
            run_mode=run_mode,
            assessment=assessment,
            audit=None,
        )
        report = {
            "experiment_id": EXPERIMENT_ID,
            "execution_status": "PASS",
            "run_mode": run_mode,
            "stage": stage,
            **metadata,
            "artifacts": artifacts,
            "results_summary": summary,
            "claim_gate": assessment,
        }
        finalize_run(
            experiment_id=EXPERIMENT_ID,
            artifact_root=args.artifact_root,
            run_dir=run_dir,
            report=report,
        )
        print(f"[{EXPERIMENT_ID}] AUDIT_PENDING: {run_dir}")
        return

    seeds = [int(value) for value in runtime["seeds"]]
    variants = [TextController(str(value)) for value in runtime["model"]["variants"]]
    raw_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    encoder_hashes: set[str] = set()
    for seed in seeds:
        train_split = TextSplit.DEVELOPMENT if args.dry_run else TextSplit.TRAIN
        train_examples = _examples(
            config=runtime,
            split=train_split,
            seed=seed,
            count=int(runtime["data"]["train_examples_per_demand"]),
        )
        eval_splits = (
            [TextSplit.DEVELOPMENT]
            if args.dry_run
            else [
                TextSplit.PRIMARY,
                TextSplit.PARAPHRASE,
                TextSplit.IDENTIFIER,
                TextSplit.DOMAIN,
                TextSplit.COMBINED,
            ]
        )
        evaluations = {
            split: _examples(
                config=runtime,
                split=split,
                seed=seed,
                count=int(runtime["data"]["test_examples_per_demand"]),
            )
            for split in eval_splits
        }
        registry.extend(visible_registry_rows(train_examples))
        for examples in evaluations.values():
            registry.extend(visible_registry_rows(examples))

        trained: dict[TextController, MatchedTextTransactionController] = {}
        shared_encoder = FrozenHashNgramEncoder(
            output_dim=int(runtime["encoder"]["output_dim"]),
            buckets=int(runtime["encoder"]["buckets"]),
            ngram_min=int(runtime["encoder"]["ngram_min"]),
            ngram_max=int(runtime["encoder"]["ngram_max"]),
            seed=int(runtime["encoder"]["seed"]),
        )
        encoder_hashes.add(shared_encoder.fingerprint())
        for variant in variants:
            torch.manual_seed(100_000 + seed)
            model = MatchedTextTransactionController(
                variant=variant,
                encoder=shared_encoder,
                slots=int(runtime["data"]["slots"]),
                value_dim=int(runtime["data"]["value_dim"]),
                hidden_dim=int(runtime["model"]["hidden_dim"]),
                semantic_value_seed=int(runtime["namespaces"]["semantic_value_seed"]),
            )
            parameter_counts[variant.value] = matched_parameter_count(model)
            train_text_controller(
                model,
                train_examples,
                steps=int(runtime["training"]["steps"]),
                batch_size=int(runtime["training"]["batch_size"]),
                learning_rate=float(runtime["training"]["learning_rate"]),
                weight_decay=float(runtime["training"]["weight_decay"]),
                affected_weight=float(runtime["training"]["affected_weight"]),
                retention_weight=float(runtime["training"]["retention_weight"]),
                gradient_clip_norm=float(runtime["training"]["gradient_clip_norm"]),
                seed=seed,
                device=device,
            )
            checkpoint_path = run_dir / f"checkpoint_seed{seed}_{variant.value}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_hashes[f"{seed}:{variant.value}"] = file_sha256(checkpoint_path)
            trained[variant] = model
            for examples in evaluations.values():
                raw_rows.extend(
                    evaluate_text_controller(
                        model,
                        examples,
                        seed=seed,
                        device=device,
                        batch_size=int(runtime["evaluation"]["batch_size"]),
                        accuracy_mse_threshold=float(
                            runtime["evaluation"]["accuracy_mse_threshold"]
                        ),
                    )
                )

        full_model = trained[TextController.STATE_AWARE]
        for examples in evaluations.values():
            raw_rows.extend(oracle_rows(examples, seed=seed))
            shuffled = shuffled_texts(examples)
            wrong_entity = wrong_entity_texts(examples)
            for condition, options in (
                ("shuffled_text", {"text_overrides": shuffled}),
                ("wrong_entity", {"text_overrides": wrong_entity}),
                ("transaction_only_zero_state", {"zero_state": True}),
                ("state_only", {"state_only": True}),
            ):
                raw_rows.extend(
                    evaluate_text_controller(
                        full_model,
                        examples,
                        seed=seed,
                        device=device,
                        batch_size=int(runtime["evaluation"]["batch_size"]),
                        accuracy_mse_threshold=float(
                            runtime["evaluation"]["accuracy_mse_threshold"]
                        ),
                        condition=condition,
                        **options,
                    )
                )

    if len(encoder_hashes) != 1 or len(set(parameter_counts.values())) != 1:
        raise RuntimeError("E25b shared encoder or matched parameter surface contract failed")
    seed_rows = summarize_rows(
        raw_rows,
        minimum_identifiable_oracle_headroom=float(
            runtime["statistics"]["minimum_identifiable_oracle_headroom"]
        ),
    )
    _, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload={
            "run_mode": run_mode,
            "namespaces": dict(runtime["namespaces"]),
            "visible_registry": registry,
            "encoder_sha256": next(iter(encoder_hashes)),
            "visible_candidate_decoder": dict(runtime["encoder"]["visible_candidate_decoder"]),
            "human_audit": audit,
        },
    )
    artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=raw_rows,
        seed_rows=seed_rows,
        raw_filename=str(runtime["artifacts"]["raw_metrics_filename"]),
        seed_filename=str(runtime["artifacts"]["seed_metrics_filename"]),
    )
    assessment = _assessment(seed_rows, config=runtime, dry_run=args.dry_run)
    claim_eligible = bool(run_mode == "MAIN" and assessment["supported"])
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=claim_eligible,
    )
    summary = _write_summary(
        run_dir=run_dir,
        run_mode=run_mode,
        assessment=assessment,
        audit=audit,
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PASS",
        "run_mode": run_mode,
        "stage": stage,
        **metadata,
        "shared_encoder_sha256": next(iter(encoder_hashes)),
        "visible_candidate_decoder": dict(runtime["encoder"]["visible_candidate_decoder"]),
        "parameter_counts": parameter_counts,
        "human_audit": audit,
        "artifacts": artifacts,
        "results_summary": summary,
        "claim_gate": {
            **assessment,
            "allowed_claim": (
                "A shared frozen text representation preserves registered "
                "controller-freedom interactions on controlled held-out "
                "transaction forms and identifiers."
            ),
            "forbidden_claim": (
                "Free-form language generation, pretrained-LM, agent, official "
                "backend, or production-system transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {assessment['status']}: {run_dir}")


if __name__ == "__main__":
    main()
