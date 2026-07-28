from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.postcore_metrics import exact_sign_flip
from experiments.common import build_parser, finalize_run, initialize_run
from experiments.e19a_localization_candidate_decomposition import (
    LOCK_PATH,
    validate_e19_protocol_lock,
)

EXPERIMENT_ID = "e19b_localization_candidate_aggregate"
DEFAULT_CONFIG = "configs/e19b_localization_candidate_aggregate.yaml"
SOURCE_EXPERIMENT_ID = "e19a_localization_candidate_decomposition"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing or invalid E19a metric {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite E19a metric {key!r}")
    return value


def _validate_source_run(
    run_dir: Path,
    *,
    expected_mode: str,
    expected_status: str,
    source_config_sha256: str,
    lock_sha256: str,
    variants: list[str],
    conditions: list[str],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"Incomplete E19a source run: {run_dir}")
    manifest = _read_json_object(manifest_path)
    report = _read_json_object(report_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != expected_mode
        or manifest.get("report_sha256") != file_sha256(report_path)
    ):
        raise RuntimeError(f"E19a manifest contract failed: {run_dir}")
    if (
        report.get("status") != expected_status
        or report.get("run_mode") != expected_mode
        or report.get("rows") != len(variants) * len(conditions)
        or report.get("protocol", {}).get("source_config_sha256")
        != source_config_sha256
        or report.get("protocol", {}).get("lock_sha256") != lock_sha256
    ):
        raise RuntimeError(f"E19a report contract failed: {run_dir}")
    expected_claim_status = (
        "NOT_EVALUATED_DRY_RUN"
        if expected_mode == "DRY_RUN"
        else "PENDING_AGGREGATE"
    )
    if report.get("claim_gate", {}).get("status") != expected_claim_status:
        raise RuntimeError(f"E19a claim disposition changed: {run_dir}")
    seed = int(report["seed"])
    metrics_path = run_dir / "localization_candidate_metrics.jsonl"
    expected_metrics_hash = report.get("artifacts", {}).get("metrics_sha256")
    if (
        not metrics_path.is_file()
        or expected_metrics_hash != file_sha256(metrics_path)
    ):
        raise RuntimeError(f"E19a metric hash mismatch: {run_dir}")
    rows = _read_jsonl(metrics_path)
    expected_grid = {
        (variant, condition)
        for variant in variants
        for condition in conditions
    }
    observed_grid: set[tuple[str, str]] = set()
    initialization_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    checkpoint_hashes = report.get("artifacts", {}).get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict):
        raise RuntimeError(f"E19a checkpoint index missing: {run_dir}")
    for row in rows:
        if int(row.get("seed", -1)) != seed:
            raise RuntimeError("E19a metric row seed mismatch")
        key = (str(row.get("variant")), str(row.get("condition")))
        if key in observed_grid or key not in expected_grid:
            raise RuntimeError(f"E19a metric grid mismatch: {key}")
        observed_grid.add(key)
        initialization_hashes.add(str(row.get("initialization_sha256")))
        parameter_counts.add(int(row.get("parameter_count", -1)))
        for metric in (
            "address_accuracy",
            "candidate_recovery_mse",
            "affected_mse",
            "retention_mse",
            "old_residual",
            "architecture_extra_error",
        ):
            _finite(row, metric)
        checkpoint = Path(str(row.get("checkpoint", ""))).resolve()
        try:
            checkpoint.relative_to((run_dir / "checkpoints").resolve())
        except ValueError as error:
            raise RuntimeError("E19a checkpoint escapes its run") from error
        expected_checkpoint_hash = checkpoint_hashes.get(key[0])
        if (
            not checkpoint.is_file()
            or row.get("checkpoint_sha256") != expected_checkpoint_hash
            or file_sha256(checkpoint) != expected_checkpoint_hash
        ):
            raise RuntimeError("E19a checkpoint hash mismatch")
    if (
        observed_grid != expected_grid
        or len(initialization_hashes) != 1
        or len(parameter_counts) != 1
    ):
        raise RuntimeError(f"E19a paired grid contract failed: {run_dir}")
    full_error = {
        str(row["condition"]): _finite(row, "affected_mse")
        for row in rows
        if row["variant"] == "full"
    }
    for row in rows:
        expected_extra = (
            _finite(row, "affected_mse") - full_error[str(row["condition"])]
        )
        if abs(_finite(row, "architecture_extra_error") - expected_extra) > 1e-12:
            raise RuntimeError("E19a architecture extra error changed")
    provenance = {
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint_hashes": checkpoint_hashes,
    }
    return seed, rows, provenance


def collect_e19a_sources(
    *,
    artifact_root: str | Path,
    source_config_path: Path,
    required_seeds: list[int],
    variants: list[str],
    conditions: list[str],
    dry_run: bool,
    lock_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(artifact_root) / SOURCE_EXPERIMENT_ID
    if not source_root.is_dir():
        raise FileNotFoundError(f"No E19a sources under {source_root}")
    source_config_sha256 = file_sha256(source_config_path)
    expected_mode = "DRY_RUN" if dry_run else "MAIN"
    expected_status = "DRY_RUN" if dry_run else "PASS"
    by_seed: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json_object(manifest_path)
        if manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID:
            continue
        if manifest.get("run_mode") != expected_mode:
            continue
        seed, rows, provenance = _validate_source_run(
            run_dir,
            expected_mode=expected_mode,
            expected_status=expected_status,
            source_config_sha256=source_config_sha256,
            lock_sha256=lock_sha256,
            variants=variants,
            conditions=conditions,
        )
        if seed not in required_seeds:
            continue
        if seed in by_seed:
            raise RuntimeError(f"Duplicate eligible E19a seed: {seed}")
        by_seed[seed] = (rows, provenance)
    if set(by_seed) != set(required_seeds):
        raise RuntimeError(
            "E19a source seed grid incomplete: "
            f"expected={required_seeds}, observed={sorted(by_seed)}"
        )
    all_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for seed in required_seeds:
        rows, provenance = by_seed[seed]
        all_rows.extend(rows)
        provenance_rows.append(provenance)
    return all_rows, provenance_rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty collection")
    return sum(values) / len(values)


def compute_seed_contrasts(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
) -> list[dict[str, float | int]]:
    by_key = {
        (int(row["seed"]), str(row["variant"]), str(row["condition"])): row
        for row in rows
    }

    def metric(
        seed: int,
        variant: str,
        condition: str,
        name: str,
    ) -> float:
        return _finite(by_key[(seed, variant, condition)], name)

    result: list[dict[str, float | int]] = []
    condition_a = "A_oracle_address_oracle_candidate"
    condition_b = "B_learned_address_oracle_candidate"
    condition_c = "C_oracle_address_state_read_candidate"
    condition_d = "D_learned_address_state_read_candidate"
    all_variants = ["base", "separate_address", "state_aware", "full"]
    for seed in seeds:
        b_comparison = ["base", "state_aware"]
        b_treatment = ["separate_address", "full"]
        c_comparison = ["base", "separate_address"]
        c_treatment = ["state_aware", "full"]
        d_incomplete = ["base", "separate_address", "state_aware"]
        b_gain = _mean(
            [
                metric(seed, variant, condition_b, "affected_mse")
                for variant in b_comparison
            ]
        ) - _mean(
            [
                metric(seed, variant, condition_b, "affected_mse")
                for variant in b_treatment
            ]
        )
        c_gain = _mean(
            [
                metric(seed, variant, condition_c, "affected_mse")
                for variant in c_comparison
            ]
        ) - _mean(
            [
                metric(seed, variant, condition_c, "affected_mse")
                for variant in c_treatment
            ]
        )
        d_full_error = metric(seed, "full", condition_d, "affected_mse")
        d_best_incomplete_error = min(
            metric(seed, variant, condition_d, "affected_mse")
            for variant in d_incomplete
        )
        result.append(
            {
                "seed": seed,
                "b_separate_address_gain": b_gain,
                "c_state_read_gain": c_gain,
                "d_full_only_gain": d_best_incomplete_error - d_full_error,
                "b_retention_degradation": _mean(
                    [
                        metric(seed, variant, condition_b, "retention_mse")
                        for variant in b_treatment
                    ]
                )
                - _mean(
                    [
                        metric(seed, variant, condition_b, "retention_mse")
                        for variant in b_comparison
                    ]
                ),
                "c_retention_degradation": _mean(
                    [
                        metric(seed, variant, condition_c, "retention_mse")
                        for variant in c_treatment
                    ]
                )
                - _mean(
                    [
                        metric(seed, variant, condition_c, "retention_mse")
                        for variant in c_comparison
                    ]
                ),
                "d_retention_degradation": metric(
                    seed,
                    "full",
                    condition_d,
                    "retention_mse",
                )
                - min(
                    metric(seed, variant, condition_d, "retention_mse")
                    for variant in d_incomplete
                ),
                "b_min_capable_address_accuracy": min(
                    metric(seed, variant, condition_b, "address_accuracy")
                    for variant in b_treatment
                ),
                "d_full_address_accuracy": metric(
                    seed,
                    "full",
                    condition_d,
                    "address_accuracy",
                ),
                "c_max_capable_candidate_mse": max(
                    metric(
                        seed,
                        variant,
                        condition_c,
                        "candidate_recovery_mse",
                    )
                    for variant in c_treatment
                ),
                "d_full_candidate_mse": metric(
                    seed,
                    "full",
                    condition_d,
                    "candidate_recovery_mse",
                ),
                "maximum_capable_affected_mse": max(
                    *[
                        metric(seed, variant, condition_b, "affected_mse")
                        for variant in b_treatment
                    ],
                    *[
                        metric(seed, variant, condition_c, "affected_mse")
                        for variant in c_treatment
                    ],
                    d_full_error,
                ),
                "maximum_oracle_floor_mse": max(
                    metric(seed, variant, condition_a, "affected_mse")
                    for variant in all_variants
                ),
                "d_best_incomplete_architecture_extra_error": min(
                    metric(
                        seed,
                        variant,
                        condition_d,
                        "architecture_extra_error",
                    )
                    for variant in d_incomplete
                ),
            }
        )
    return result


def _gain_gate(
    values: list[float],
    *,
    sesoi: float,
    alpha: float,
    required_direction: float,
) -> dict[str, float | bool]:
    mean_gain = _mean(values)
    direction = sum(value > 0 for value in values) / len(values)
    p = exact_sign_flip(values, alternative="greater")
    return {
        "mean_gain": mean_gain,
        "positive_seed_fraction": direction,
        "sign_flip_p": p,
        "passed": (
            mean_gain >= sesoi
            and direction >= required_direction
            and p <= alpha
        ),
    }


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    pre_config = load_config(args.config)
    if pre_config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E19b config experiment_id mismatch")
    source_config_path = Path(
        str(pre_config["source"]["config_path"])
    ).resolve()
    lock_sha256 = validate_e19_protocol_lock(source_config_path)
    config, run_dir, _device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    seeds = [int(value) for value in config["source"]["required_seeds"]]
    if args.dry_run:
        seeds = seeds[:1]
    variants = [str(value) for value in config["source"]["required_variants"]]
    conditions = [
        str(value) for value in config["source"]["required_conditions"]
    ]
    rows, provenance_rows = collect_e19a_sources(
        artifact_root=args.artifact_root,
        source_config_path=source_config_path,
        required_seeds=seeds,
        variants=variants,
        conditions=conditions,
        dry_run=args.dry_run,
        lock_sha256=lock_sha256,
    )
    seed_rows = compute_seed_contrasts(rows, seeds=seeds)
    gate_config = config["claim_gate"]
    alpha = float(config["statistics"]["alpha"])
    sesoi = float(gate_config["selective_gain"])
    required_direction = float(
        gate_config["minimum_seed_direction_fraction"]
    )
    pattern = {
        "b_separate_address_recovery": _gain_gate(
            [float(row["b_separate_address_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            required_direction=required_direction,
        ),
        "c_state_read_recovery": _gain_gate(
            [float(row["c_state_read_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            required_direction=required_direction,
        ),
        "d_full_only_maintenance": _gain_gate(
            [float(row["d_full_only_gain"]) for row in seed_rows],
            sesoi=sesoi,
            alpha=alpha,
            required_direction=required_direction,
        ),
    }
    retention_max = max(
        float(row[key])
        for row in seed_rows
        for key in (
            "b_retention_degradation",
            "c_retention_degradation",
            "d_retention_degradation",
        )
    )
    minimum_address_accuracy = min(
        min(
            float(row["b_min_capable_address_accuracy"]),
            float(row["d_full_address_accuracy"]),
        )
        for row in seed_rows
    )
    maximum_candidate_mse = max(
        max(
            float(row["c_max_capable_candidate_mse"]),
            float(row["d_full_candidate_mse"]),
        )
        for row in seed_rows
    )
    maximum_capable_affected = max(
        float(row["maximum_capable_affected_mse"]) for row in seed_rows
    )
    maximum_oracle_floor = max(
        float(row["maximum_oracle_floor_mse"]) for row in seed_rows
    )
    conditions_passed = {
        "registered_pattern_passed": all(
            bool(gate["passed"]) for gate in pattern.values()
        ),
        "retention_noninferiority_passed": retention_max
        <= float(gate_config["retention_noninferiority"]),
        "learned_address_assay_passed": minimum_address_accuracy
        >= float(gate_config["minimum_address_accuracy"]),
        "state_read_assay_passed": maximum_candidate_mse
        <= float(gate_config["maximum_candidate_recovery_mse"]),
        "capable_floor_passed": maximum_capable_affected
        <= float(gate_config["maximum_capable_affected_mse"]),
        "oracle_floor_passed": maximum_oracle_floor
        <= float(gate_config["maximum_oracle_floor_mse"]),
    }
    supported = all(conditions_passed.values()) and not args.dry_run

    metrics_path = run_dir / "localization_candidate_paired_metrics.jsonl"
    seed_path = run_dir / "localization_candidate_seed_contrasts.jsonl"
    provenance_path = run_dir / "source_run_provenance.jsonl"
    write_jsonl(metrics_path, rows)
    write_jsonl(seed_path, seed_rows)
    write_jsonl(provenance_path, provenance_rows)
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "CONTROLLED_LEARNED_LOCALIZATION_STATE_READ_AGGREGATE",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "source_contract": {
            "experiment_id": SOURCE_EXPERIMENT_ID,
            "source_config_path": str(source_config_path),
            "source_config_sha256": file_sha256(source_config_path),
            "required_seeds": seeds,
            "required_variants": variants,
            "required_conditions": conditions,
            "source_runs": provenance_rows,
            "protocol_lock_path": str(LOCK_PATH),
            "protocol_lock_sha256": lock_sha256,
        },
        "summary": {
            "paired_seeds": len(seeds),
            "paired_rows": len(rows),
            "pattern": pattern,
            "maximum_retention_degradation": retention_max,
            "minimum_capable_address_accuracy": minimum_address_accuracy,
            "maximum_capable_candidate_mse": maximum_candidate_mse,
            "maximum_capable_affected_mse": maximum_capable_affected,
            "maximum_oracle_floor_mse": maximum_oracle_floor,
        },
        "artifacts": {
            "paired_metrics_sha256": file_sha256(metrics_path),
            "seed_contrasts_sha256": file_sha256(seed_path),
            "source_provenance_sha256": file_sha256(provenance_path),
        },
        "claim_gate": {
            "status": (
                "NOT_EVALUATED_DRY_RUN"
                if args.dry_run
                else ("SUPPORTED" if supported else "NOT_SUPPORTED")
            ),
            "supported": supported,
            "conditions": conditions_passed,
            "allowed_claim": (
                "In the registered controlled fixed-slot address-code setting, "
                "learned separate localization and current-state erase-candidate "
                "reads provide selective and complementary correction capacity."
            ),
            "forbidden_claim": (
                "Semantic or natural-language localization, novel-entity "
                "generalization, pretrained recurrent-model, agent, official-"
                "backend, or runtime-superiority transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
