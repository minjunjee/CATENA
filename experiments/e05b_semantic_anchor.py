from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catena.core.provenance_v61 import (
    ProvenanceValidationError,
    read_json_object_strict,
    read_jsonl_strict,
    sha256_file,
    write_jsonl_strict,
)
from catena.data.semantic_controls_v61 import (
    SemanticControl,
    build_control_pairing_registry,
)
from catena.data.semantic_registry_v61 import (
    semantic_example_from_registry_row,
    validate_semantic_registry_rows,
)
from catena.data.semantic_transactions_v61 import (
    SemanticExample,
    SemanticNamespaceRegistry,
)
from catena.eval.semantic_anchor_v61 import (
    CONTROL_NAMES,
    SemanticAnchorSeedMetrics,
    SemanticAnchorThresholds,
    evaluate_e05b_main,
    evaluate_e05b_secondary,
    evaluate_e05b_validation,
)
from catena.training.semantic_probe_v61 import (
    evaluate_semantic_model,
    seed_metrics_from_condition_arrays,
    train_matched_semantic_pair,
)
from experiments.common import build_parser
from experiments.e05_common_v61 import (
    E05B_CONFIG_PATH,
    PINNED_E05A_CONFIG_CANONICAL_SHA256,
    PINNED_E05A_CONFIG_FILE_SHA256,
    PINNED_E05B_CONFIG_CANONICAL_SHA256,
    PINNED_E05B_CONFIG_FILE_SHA256,
    PINNED_PROTOCOL_LOCK_SHA256,
    PINNED_PROTOCOL_SHA256,
    validate_completed_e05a_human_audit,
    validate_completed_e05a_run,
    validate_frozen_e05_protocol,
)
from experiments.e05a_semantic_protocol_lock import (
    _CONDITION_TO_CONTROL,
    _attach_condition,
    _descriptor,
    _encoder,
    _memory_spec,
    _training_config,
)
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e05b_semantic_anchor"
DEFAULT_CONFIG = "configs/e05b_semantic_anchor.yaml"


def _assert_jsonl_rows(path: Path, expected_rows: int) -> None:
    observed = sum(1 for line in path.read_bytes().splitlines() if line.strip())
    if observed != expected_rows:
        raise ProvenanceValidationError(
            f"{path} has {observed} rows; expected {expected_rows}."
        )


def _validate_config_contract(
    config_path: str,
    *,
    frozen_e05a: Mapping[str, Any],
    frozen_e05b: Mapping[str, Any],
) -> None:
    if Path(config_path).resolve(strict=True) != E05B_CONFIG_PATH.resolve(strict=True):
        raise ProvenanceValidationError("E05b requires the frozen default config.")
    shared_training = {
        key: value
        for key, value in frozen_e05a["training"].items()
        if key != "dry_steps"
    }
    if any(
        frozen_e05b["training"].get(key) != value
        for key, value in shared_training.items()
    ) or (
        frozen_e05b["training"].get("validation_retraining_or_selection")
        != "forbidden"
    ):
        raise ProvenanceValidationError("E05a/E05b fixed training contracts differ.")
    if frozen_e05b["memory"] != frozen_e05a["memory"]:
        raise ProvenanceValidationError("E05a/E05b memory contracts differ.")
    if frozen_e05b["model"] != frozen_e05a["model"]:
        raise ProvenanceValidationError("E05a/E05b model contracts differ.")


def _thresholds(config: Mapping[str, Any]) -> SemanticAnchorThresholds:
    statistics = config["statistics"]
    return SemanticAnchorThresholds(
        positive_effect_sesoi=float(statistics["positive_effect_sesoi"]),
        minimum_oracle_headroom=float(statistics["minimum_oracle_headroom"]),
        headroom_fraction_sesoi=float(statistics["headroom_fraction_sesoi"]),
        equivalence_margin=float(statistics["equivalence_margin"]),
        retention_noninferiority_margin=float(
            statistics["retention_noninferiority_margin"]
        ),
        oracle_absolute_ceiling=float(statistics["oracle_absolute_ceiling"]),
        alpha=float(statistics["alpha"]),
        bootstrap_samples=int(statistics["bootstrap_samples"]),
        bootstrap_confidence=float(statistics["bootstrap_confidence"]),
    )


def _e05a_artifact(run: Any, name: str) -> Path:
    artifacts = run.report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProvenanceValidationError("E05a report lacks artifacts.")
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, dict):
        raise ProvenanceValidationError(f"E05a lacks artifact {name}.")
    filename = descriptor.get("filename")
    expected_hash = descriptor.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ProvenanceValidationError(f"Unsafe E05a artifact filename for {name}.")
    if not isinstance(expected_hash, str):
        raise ProvenanceValidationError(f"E05a artifact {name} lacks a hash.")
    path = run.run_dir / filename
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
        raise ProvenanceValidationError(f"E05a artifact {name} hash mismatch.")
    expected_rows = descriptor.get("rows")
    if expected_rows is not None:
        rows = sum(1 for line in path.read_bytes().splitlines() if line.strip())
        if rows != expected_rows:
            raise ProvenanceValidationError(f"E05a artifact {name} row mismatch.")
    return path


def _registry_rows_by_seed(
    path: Path,
    *,
    split: str,
    seeds: Sequence[int],
    rows_per_seed: int,
    namespace_name: str,
    namespace_registry: SemanticNamespaceRegistry,
    memory_spec: Any,
) -> dict[int, list[Mapping[str, object]]]:
    raw = read_jsonl_strict(path)
    if any(not isinstance(row, dict) for row in raw):
        raise ProvenanceValidationError(f"{path} contains a non-object row.")
    rows = [row for row in raw if isinstance(row, dict)]
    validate_semantic_registry_rows(
        rows,
        expected_split=split,
        expected_seeds=seeds,
        expected_rows_per_seed=rows_per_seed,
        expected_namespace_name=namespace_name,
        expected_seed_slots={
            int(seed): seed_slot for seed_slot, seed in enumerate(seeds)
        },
        namespace_registry=namespace_registry,
        memory_spec=memory_spec,
        reconstruct=False,
    )
    return {
        seed: [row for row in rows if row["checkpoint_seed"] == seed]
        for seed in seeds
    }


def _examples_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    memory_spec: Any,
) -> list[SemanticExample]:
    return [
        semantic_example_from_registry_row(row, memory_spec=memory_spec)
        for row in rows
    ]


def _evaluate_baselines(
    *,
    seed: int,
    split: str,
    examples: Sequence[SemanticExample],
    factorized: Any,
    shared: Any,
    encoder: Any,
    device: torch.device,
) -> tuple[SemanticAnchorSeedMetrics, list[dict[str, object]]]:
    affected: dict[str, np.ndarray] = {}
    retention: dict[str, np.ndarray] = {}
    raw_rows: list[dict[str, object]] = []
    for name, model in (("factorized", factorized), ("shared", shared)):
        rows, local_affected, local_retention = evaluate_semantic_model(
            model,
            examples,
            encoder=encoder,
            control=SemanticControl.FULL,
            pairing_registry=None,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(
            _attach_condition(rows, seed=seed, condition=name, split=split)
        )
        affected[name] = local_affected
        retention[name] = local_retention
    rows, local_affected, local_retention = evaluate_semantic_model(
        None,
        examples,
        encoder=encoder,
        control=SemanticControl.FULL,
        pairing_registry=None,
        oracle_demand=True,
        device=device,
    )
    raw_rows.extend(
        _attach_condition(
            rows,
            seed=seed,
            condition="oracle_demand",
            split=split,
        )
    )
    affected["oracle_demand"] = local_affected
    retention["oracle_demand"] = local_retention
    return (
        seed_metrics_from_condition_arrays(
            examples,
            affected=affected,
            retention=retention,
        ),
        raw_rows,
    )


def _primary_with_controls(
    *,
    seed: int,
    examples: Sequence[SemanticExample],
    semantic_donors: Sequence[SemanticExample],
    locked_pairing_rows: Sequence[Mapping[str, object]],
    factorized: Any,
    shared: Any,
    encoder: Any,
    device: torch.device,
    norm_tolerance: float,
) -> tuple[SemanticAnchorSeedMetrics, list[dict[str, object]]]:
    baseline, raw_rows = _evaluate_baselines(
        seed=seed,
        split="primary",
        examples=examples,
        factorized=factorized,
        shared=shared,
        encoder=encoder,
        device=device,
    )
    pairings = build_control_pairing_registry(
        examples,
        semantic_donors=semantic_donors,
        norm_tolerance=norm_tolerance,
    )
    expected = [
        {"checkpoint_seed": seed, **row} for row in pairings.to_rows()
    ]
    if expected != list(locked_pairing_rows):
        raise ProvenanceValidationError(
            f"Seed {seed} regenerated control pairings differ from E05a seal."
        )
    affected = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in baseline.affected.items()
    }
    retention = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in baseline.retention.items()
    }
    for condition in CONTROL_NAMES:
        rows, values, _ = evaluate_semantic_model(
            factorized,
            examples,
            encoder=encoder,
            control=_CONDITION_TO_CONTROL[condition],
            pairing_registry=pairings,
            oracle_demand=False,
            device=device,
        )
        raw_rows.extend(
            _attach_condition(
                rows,
                seed=seed,
                condition=condition,
                split="primary",
            )
        )
        affected[condition] = values
    return (
        seed_metrics_from_condition_arrays(
            examples,
            affected=affected,
            retention=retention,
        ),
        raw_rows,
    )


def _summary_text(
    *,
    validation_status: str,
    primary_status: str,
    primary_report: Mapping[str, Any] | None,
) -> str:
    lines = [
        "# E05b Semantic Demand Inference Anchor — 결과 요약",
        "",
        "## 판정",
        "",
        "- execution_status: `PASS`",
        f"- sealed_validation_status: `{validation_status}`",
        f"- primary_semantic_anchor_status: `{primary_status}`",
        f"- full_h5_lite_claim_open: "
        f"`{bool(primary_report and primary_report.get('supported'))}`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "",
    ]
    if primary_report is None:
        lines.extend(
            [
                "Sealed validation이 통과하지 않아 primary registry를 열지 않았다.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Primary registered effects",
                "",
                "| Estimand | Estimate | 95% CI | Supported |",
                "|---|---:|---:|---|",
            ]
        )
        for key in (
            "D_shared_minus_factorized",
            "H_shared_minus_oracle",
            "Q_headroom_fraction_closed",
            "R_factorized_minus_shared_retention",
        ):
            gate = primary_report[key]
            estimate = gate.get("estimate", "NA")
            interval = gate.get("ci95", ["NA", "NA"])
            lines.append(
                f"| {key} | {estimate} | [{interval[0]}, {interval[1]}] | "
                f"{gate.get('supported', False)} |"
            )
        lines.extend(
            [
                "",
                "## 특기사항",
                "",
                "- Operation label, oracle `(e,w)`, exact mask는 learned model input에 "
                "제공하지 않았다.",
                "- Old lexical value는 숨겼고 oracle address의 visible state read만 "
                "사용했다.",
                "- Secondary paraphrase/domain/combined 결과는 primary gate에 포함하지 "
                "않았다.",
                "",
                "## 연구 흐름에서의 짧은 해석",
                "",
                "이 결과는 controlled structured-record setting의 작은 semantic "
                "anchor이며 natural-language understanding이나 hidden addressing "
                "claim으로 확장하지 않는다.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    parser.add_argument("--e05a-run-dir", required=True)
    parser.add_argument("--human-audit-run-dir", required=True)
    args = parser.parse_args()
    if args.dry_run:
        raise ValueError("E05b has no dry-run alias; E05a owns development dry-runs.")
    frozen_e05a, frozen_e05b = validate_frozen_e05_protocol()
    _validate_config_contract(
        args.config,
        frozen_e05a=frozen_e05a,
        frozen_e05b=frozen_e05b,
    )
    e05a = validate_completed_e05a_run(args.e05a_run_dir, require_go=True)
    human_audit = validate_completed_e05a_human_audit(
        args.human_audit_run_dir,
        expected_e05a=e05a,
    )
    e00 = validate_legacy_e00(args.artifact_root, require_full=True)

    seal_path = _e05a_artifact(e05a, "e05b_registry_seal")
    seal = read_json_object_strict(seal_path)
    if (
        seal.get("protocol_sha256") != PINNED_PROTOCOL_SHA256
        or seal.get("protocol_lock_sha256") != PINNED_PROTOCOL_LOCK_SHA256
        or seal.get("e05a_config")
        != {
            "canonical_sha256": PINNED_E05A_CONFIG_CANONICAL_SHA256,
            "file_sha256": PINNED_E05A_CONFIG_FILE_SHA256,
        }
        or seal.get("e05b_config")
        != {
            "canonical_sha256": PINNED_E05B_CONFIG_CANONICAL_SHA256,
            "file_sha256": PINNED_E05B_CONFIG_FILE_SHA256,
        }
        or seal.get("main_registry_sealed") is not True
        or seal.get("e05b_training_started") is not False
    ):
        raise ProvenanceValidationError("E05b registry seal contract changed.")

    dependencies = [
        e00,
        {
            **e05a.dependency_record(),
            "evidence_role": "e05a_go_and_sealed_registry",
            "e05a_design_status": "GO",
        },
        {
            **human_audit.dependency_record(),
            "evidence_role": "passed_two_human_reviewer_audit",
            "human_audit_status": "PASSED",
        },
    ]
    config, run_dir, device, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        dry_run=False,
        dependencies=dependencies,
    )
    if config != frozen_e05b:
        raise ProvenanceValidationError("Runtime E05b config differs from frozen config.")
    torch.set_num_threads(1)
    memory_spec = _memory_spec(frozen_e05a)
    namespace_registry = SemanticNamespaceRegistry.from_config(
        frozen_e05a["namespace"],
        dry_run=False,
    )
    encoder = _encoder(frozen_e05a, memory_spec)
    training_config = _training_config(frozen_e05a, dry_run=False)
    thresholds = _thresholds(config)
    bootstrap_seeds = {
        str(key): int(value)
        for key, value in config["statistics"]["bootstrap_seeds"].items()
    }
    seeds = [int(value) for value in config["seeds"]]

    train_contract = config["data"]["train"]
    validation_contract = config["data"]["sealed_validation"]
    train_rows_per_seed = (
        len(train_contract["operations"])
        * len(train_contract["domains"])
        * len(train_contract["templates"])
        * int(train_contract["count_per_cell"])
    )
    validation_rows_per_seed = (
        len(validation_contract["operations"])
        * len(validation_contract["domains"])
        * len(validation_contract["templates"])
        * int(validation_contract["count_per_cell"])
    )
    train_rows = _registry_rows_by_seed(
        _e05a_artifact(e05a, "e05b_train_registry"),
        split="train",
        seeds=seeds,
        rows_per_seed=train_rows_per_seed,
        namespace_name="e05b_train",
        namespace_registry=namespace_registry,
        memory_spec=memory_spec,
    )
    validation_rows = _registry_rows_by_seed(
        _e05a_artifact(e05a, "e05b_validation_registry"),
        split="sealed_validation",
        seeds=seeds,
        rows_per_seed=validation_rows_per_seed,
        namespace_name="e05b_validation",
        namespace_registry=namespace_registry,
        memory_spec=memory_spec,
    )

    models: dict[int, tuple[Any, Any]] = {}
    validation_metrics: dict[int, SemanticAnchorSeedMetrics] = {}
    validation_raw_rows: list[dict[str, object]] = []
    checkpoint_registry: dict[str, object] = {}
    for seed in seeds:
        train_examples = _examples_from_rows(
            train_rows[seed],
            memory_spec=memory_spec,
        )
        training = train_matched_semantic_pair(
            train_examples,
            encoder=encoder,
            hidden_dim=int(config["model"]["path_hidden_dim"]),
            config=training_config,
            seed=seed,
            device=device,
        )
        models[seed] = (training.factorized, training.shared)
        local_checkpoints: dict[str, object] = {}
        for name, model in (
            ("factorized", training.factorized),
            ("shared", training.shared),
        ):
            path = run_dir / f"seed{seed}_{name}.pt"
            torch.save(model.state_dict(), path)
            local_checkpoints[name] = _descriptor(path)
        checkpoint_registry[str(seed)] = {
            "initial_state_sha256": training.initial_state_sha256,
            "schedule_sha256": training.schedule_sha256,
            "parameter_count": training.parameter_count,
            "dense_multiply_adds_per_example": (
                training.dense_multiply_adds_per_example
            ),
            "final_loss": dict(training.final_loss),
            "checkpoints": local_checkpoints,
        }
        validation_examples = _examples_from_rows(
            validation_rows[seed],
            memory_spec=memory_spec,
        )
        local_metrics, local_rows = _evaluate_baselines(
            seed=seed,
            split="sealed_validation",
            examples=validation_examples,
            factorized=training.factorized,
            shared=training.shared,
            encoder=encoder,
            device=device,
        )
        validation_metrics[seed] = local_metrics
        validation_raw_rows.extend(local_rows)

    validation_report = evaluate_e05b_validation(
        validation_metrics,
        thresholds=thresholds,
        bootstrap_seeds=bootstrap_seeds,
    )
    validation_path = run_dir / "sealed_validation_metrics.jsonl"
    expected_validation_raw_rows = len(seeds) * validation_rows_per_seed * 3
    if len(validation_raw_rows) != expected_validation_raw_rows:
        raise AssertionError("Sealed-validation raw metric row count changed.")
    write_jsonl_strict(validation_path, validation_raw_rows)
    _assert_jsonl_rows(validation_path, expected_validation_raw_rows)

    validation_passed = bool(validation_report["passed"])
    primary_report: dict[str, object] | None = None
    primary_raw_path: Path | None = None
    secondary_raw_path: Path | None = None
    expected_primary_raw_rows: int | None = None
    expected_secondary_raw_rows: int | None = None
    primary_registry_opened = False
    secondary_summary: dict[str, object] = {}
    if validation_passed:
        primary_registry_opened = True
        primary_contract = config["data"]["primary"]
        primary_rows_per_seed = (
            len(primary_contract["operations"])
            * len(primary_contract["domains"])
            * len(primary_contract["templates"])
            * int(primary_contract["count_per_cell"])
        )
        primary_rows = _registry_rows_by_seed(
            _e05a_artifact(e05a, "e05b_primary_registry"),
            split="primary",
            seeds=seeds,
            rows_per_seed=primary_rows_per_seed,
            namespace_name="e05b_primary",
            namespace_registry=namespace_registry,
            memory_spec=memory_spec,
        )
        locked_pairing_payload = read_jsonl_strict(
            _e05a_artifact(e05a, "e05b_primary_control_pairings")
        )
        locked_pairings = {
            seed: [
                row
                for row in locked_pairing_payload
                if isinstance(row, dict) and row.get("checkpoint_seed") == seed
            ]
            for seed in seeds
        }
        primary_metrics: dict[int, SemanticAnchorSeedMetrics] = {}
        primary_raw_rows: list[dict[str, object]] = []
        for seed in seeds:
            validation_examples = _examples_from_rows(
                validation_rows[seed],
                memory_spec=memory_spec,
            )
            primary_examples = _examples_from_rows(
                primary_rows[seed],
                memory_spec=memory_spec,
            )
            factorized, shared = models[seed]
            local_metrics, local_rows = _primary_with_controls(
                seed=seed,
                examples=primary_examples,
                semantic_donors=validation_examples,
                locked_pairing_rows=locked_pairings[seed],
                factorized=factorized,
                shared=shared,
                encoder=encoder,
                device=device,
                norm_tolerance=float(
                    config["controls"]["wrong_address_norm_tolerance"]
                ),
            )
            primary_metrics[seed] = local_metrics
            primary_raw_rows.extend(local_rows)
        primary_report = evaluate_e05b_main(
            validation=validation_metrics,
            primary=primary_metrics,
            thresholds=thresholds,
            bootstrap_seeds=bootstrap_seeds,
        )
        primary_raw_path = run_dir / "primary_semantic_metrics.jsonl"
        expected_primary_raw_rows = (
            len(seeds) * primary_rows_per_seed * (3 + len(CONTROL_NAMES))
        )
        if len(primary_raw_rows) != expected_primary_raw_rows:
            raise AssertionError("Primary raw metric row count changed.")
        write_jsonl_strict(primary_raw_path, primary_raw_rows)
        _assert_jsonl_rows(primary_raw_path, expected_primary_raw_rows)

        secondary_rows: list[dict[str, object]] = []
        expected_secondary_raw_rows = 0
        for split_name, artifact_name in (
            ("heldout_paraphrase", "e05b_paraphrase_registry"),
            ("heldout_domain", "e05b_domain_registry"),
            ("combined_stress", "e05b_combined_registry"),
        ):
            contract = config["data"][split_name]
            rows_per_seed = (
                len(contract["operations"])
                * len(contract["domains"])
                * len(contract["templates"])
                * int(contract["count_per_cell"])
            )
            expected_secondary_raw_rows += len(seeds) * rows_per_seed * 3
            rows_by_seed = _registry_rows_by_seed(
                _e05a_artifact(e05a, artifact_name),
                split=split_name,
                seeds=seeds,
                rows_per_seed=rows_per_seed,
                namespace_name={
                    "heldout_paraphrase": "e05b_paraphrase",
                    "heldout_domain": "e05b_domain",
                    "combined_stress": "e05b_combined",
                }[split_name],
                namespace_registry=namespace_registry,
                memory_spec=memory_spec,
            )
            split_metrics: dict[int, SemanticAnchorSeedMetrics] = {}
            seed_model_means: dict[str, object] = {}
            for seed in seeds:
                examples = _examples_from_rows(
                    rows_by_seed[seed],
                    memory_spec=memory_spec,
                )
                factorized, shared = models[seed]
                metrics, rows = _evaluate_baselines(
                    seed=seed,
                    split=split_name,
                    examples=examples,
                    factorized=factorized,
                    shared=shared,
                    encoder=encoder,
                    device=device,
                )
                secondary_rows.extend(rows)
                split_metrics[seed] = metrics
                seed_model_means[str(seed)] = {
                    name: float(np.asarray(values).mean())
                    for name, values in metrics.affected.items()
                }
            secondary_summary[split_name] = {
                **evaluate_e05b_secondary(
                    split_metrics,
                    thresholds=thresholds,
                    bootstrap_seeds=bootstrap_seeds,
                ),
                "supplemental_seed_affected_means": seed_model_means,
            }
        secondary_raw_path = run_dir / "secondary_semantic_metrics.jsonl"
        if len(secondary_rows) != expected_secondary_raw_rows:
            raise AssertionError("Secondary raw metric row count changed.")
        write_jsonl_strict(secondary_raw_path, secondary_rows)
        _assert_jsonl_rows(secondary_raw_path, expected_secondary_raw_rows)

    validation_status = "PASSED" if validation_passed else "FAILED"
    primary_status = (
        str(primary_report["status"]) if primary_report is not None else "NOT_OPENED"
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        _summary_text(
            validation_status=validation_status,
            primary_status=primary_status,
            primary_report=primary_report,
        ),
        encoding="utf-8",
    )
    artifacts: dict[str, object] = {
        "sealed_validation_metrics": _descriptor(
            validation_path,
            rows=len(validation_raw_rows),
        ),
        "results_summary_ko": _descriptor(summary_path),
    }
    if primary_raw_path is not None:
        artifacts["primary_semantic_metrics"] = _descriptor(
            primary_raw_path,
            rows=expected_primary_raw_rows,
        )
    if secondary_raw_path is not None:
        artifacts["secondary_semantic_metrics"] = _descriptor(
            secondary_raw_path,
            rows=expected_secondary_raw_rows,
        )
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "human_audit_status": "PASSED",
        "sealed_validation_status": validation_status,
        "primary_registry_opened": primary_registry_opened,
        "primary_semantic_anchor_status": primary_status,
        "full_h5_lite_claim_open": bool(
            primary_report is not None and primary_report.get("supported")
        ),
        "validation_gate": validation_report,
        "primary_gate": primary_report,
        "secondary_descriptive": secondary_summary,
        "checkpoint_registry": checkpoint_registry,
        "artifacts": artifacts,
        "protocol_lock": {
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "protocol_lock_sha256": PINNED_PROTOCOL_LOCK_SHA256,
            "e05a_config_canonical_sha256": (
                PINNED_E05A_CONFIG_CANONICAL_SHA256
            ),
            "e05b_config_canonical_sha256": (
                PINNED_E05B_CONFIG_CANONICAL_SHA256
            ),
        },
        "claim_gate": {
            "primary_evaluated_once": primary_registry_opened,
            "secondary_not_in_primary_gate": True,
            "h5_lite_open": bool(
                primary_report is not None and primary_report.get("supported")
            ),
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
        main_eligible=True,
        full_eligible=True,
    )
    print(f"[{EXPERIMENT_ID}] PASS ({primary_status}): {run_dir}")


if __name__ == "__main__":
    main()
