from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.postcore_metrics import exact_sign_flip
from experiments import e13c_transactional_sequence_aggregate as legacy
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e13c_r1_transactional_sequence_aggregate"
DEFAULT_CONFIG = "configs/e13c_r1_transactional_sequence_aggregate.yaml"
SOURCE_EXPERIMENT_ID = "e13b_r1_transactional_sequence_memory"
CALIBRATION_EXPERIMENT_ID = "e13a_r2_sequence_floor_throughput"
RUN_START_PROVENANCE_FILES = frozenset(
    {
        "config.resolved.yaml",
        "environment.json",
        "run_manifest.json",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _finite_value(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing or invalid E13b-R1 metric {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite E13b-R1 metric {key!r}")
    return value


def _operational_incomplete_run_start_record(
    run_dir: Path,
    *,
    expected_source_config: dict,
    source_config_path: Path,
) -> dict[str, Any] | None:
    entries = tuple(run_dir.iterdir())
    names = {entry.name for entry in entries}
    if (
        names != RUN_START_PROVENANCE_FILES
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        return None

    manifest_path = run_dir / "run_manifest.json"
    resolved_config_path = run_dir / "config.resolved.yaml"
    environment_path = run_dir / "environment.json"
    manifest = _read_json(manifest_path)
    environment = _read_json(environment_path)
    resolved_config = load_config(resolved_config_path)
    source_fingerprint = manifest.get("source_fingerprint")
    source_sha = (
        source_fingerprint.get("sha256")
        if isinstance(source_fingerprint, dict)
        else None
    )
    source_files = (
        source_fingerprint.get("files")
        if isinstance(source_fingerprint, dict)
        else None
    )
    valid_source_fingerprint = bool(
        isinstance(source_sha, str)
        and len(source_sha) == 64
        and all(character in "0123456789abcdef" for character in source_sha)
        and isinstance(source_files, int)
        and source_files > 0
    )
    valid_run_start = bool(
        manifest.get("schema_version") == 2
        and manifest.get("experiment_id") == SOURCE_EXPERIMENT_ID
        and manifest.get("run_id") == run_dir.name
        and manifest.get("run_mode") == "MAIN"
        and manifest.get("source_fingerprint_phase") == "RUN_START"
        and "completed_at_utc" not in manifest
        and "report_sha256" not in manifest
        and manifest.get("config") == expected_source_config
        and resolved_config == expected_source_config
        and manifest.get("config_file_sha256")
        == file_sha256(source_config_path)
        and manifest.get("resolved_config_artifact_sha256")
        == file_sha256(resolved_config_path)
        and manifest.get("resolved_config_sha256")
        == legacy._canonical_sha256(expected_source_config)
        and bool(environment)
        and valid_source_fingerprint
    )
    if not valid_run_start:
        raise RuntimeError(
            "Manifest-only E13b-R1 directory is not a valid run-start "
            f"provenance record: {run_dir}"
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "disposition": "EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY",
        "reason": (
            "RUN_START_PROVENANCE_ONLY_NO_REPORT_METRICS_OR_CHECKPOINT"
        ),
        "files": sorted(names),
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "config_resolved_sha256": file_sha256(resolved_config_path),
        "environment_sha256": file_sha256(environment_path),
        "created_at_utc": str(manifest["created_at_utc"]),
        "source_fingerprint": source_fingerprint,
    }


def collect_e13b_r1_sources(
    *,
    artifact_root: str | Path,
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seeds, variants, updates, gaps = legacy._source_contract(
        config,
        dry_run=dry_run,
    )
    source_config_path = Path(str(config["source"]["config_path"])).resolve()
    expected_source_config = load_config(source_config_path)
    source_root = Path(artifact_root) / SOURCE_EXPERIMENT_ID
    if not source_root.is_dir():
        raise FileNotFoundError(f"No E13b-R1 runs found under {source_root}")

    by_key: dict[
        tuple[int, str],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ] = {}
    excluded: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        incomplete = _operational_incomplete_run_start_record(
            run_dir,
            expected_source_config=expected_source_config,
            source_config_path=source_config_path,
        )
        if incomplete is not None:
            excluded.append(incomplete)
            continue
        key, rows, provenance = legacy._validate_source_run(
            run_dir,
            expected_source_config=expected_source_config,
            required_updates=updates,
            required_gaps=gaps,
            dry_run=dry_run,
        )
        if key in by_key:
            raise RuntimeError(
                f"Duplicate eligible E13b-R1 runs for seed={key[0]}, "
                f"variant={key[1]}"
            )
        by_key[key] = (rows, provenance)

    expected_keys = set(product(seeds, variants))
    observed_keys = set(by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            "E13b-R1 source-run contract mismatch after excluding only "
            f"run-start-only records: missing={missing}, extra={extra}"
        )
    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        source_rows, provenance = by_key[key]
        rows.extend(source_rows)
        provenance_rows.append(provenance)
    return rows, provenance_rows, excluded


def _pin_source_manifest_integrity(
    source_runs: list[dict],
    *,
    dry_run: bool,
) -> None:
    expected_mode = "DRY_RUN" if dry_run else "MAIN"
    for source in source_runs:
        run_dir = Path(str(source["run_dir"])).resolve()
        report_path = Path(str(source["report_path"])).resolve()
        metrics_path = Path(str(source["metrics_path"])).resolve()
        checkpoint_path = Path(str(source["checkpoint_path"])).resolve()
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema_version") != 2
            or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
            or manifest.get("run_mode") != expected_mode
            or manifest.get("run_id") != run_dir.name
            or manifest.get("report_sha256") != source["report_sha256"]
        ):
            raise RuntimeError(
                f"E13b-R1 source manifest contract failed: {manifest_path}"
            )
        for path, key in (
            (report_path, "report_sha256"),
            (metrics_path, "metrics_sha256"),
            (checkpoint_path, "checkpoint_sha256"),
        ):
            if file_sha256(path) != source[key]:
                raise RuntimeError(f"Pinned E13b-R1 artifact changed: {path}")
        source["run_manifest_path"] = str(manifest_path)
        source["run_manifest_sha256"] = file_sha256(manifest_path)


def _validate_r1_metric_rows(
    rows: list[dict],
    source_runs: list[dict],
    *,
    stress_updates: int,
    stress_gap: int,
    dry_run: bool,
) -> None:
    provenance: dict[tuple[int, str], dict] = {}
    for source in source_runs:
        key = (int(source["seed"]), str(source["variant"]))
        if key in provenance:
            raise RuntimeError(f"Duplicate E13b-R1 source provenance: {key}")
        provenance[key] = source

    stress_counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        key = (int(row["seed"]), str(row["variant"]))
        if key not in provenance:
            raise RuntimeError(f"E13b-R1 row lacks source provenance: {key}")
        digest = str(row.get("base_transaction_digest", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError("E13b-R1 base transaction digest is not SHA-256")
        if row.get("checkpoint_sha256") != provenance[key]["checkpoint_sha256"]:
            raise RuntimeError("E13b-R1 metric row checkpoint hash changed")
        if row.get("activate_distractor_verified") is not False:
            raise RuntimeError("Normal E13b-R1 row unexpectedly activates distractors")
        for metric in (
            "affected_mse",
            "retention_mse",
            "old_rule_residual",
            "entity_exact_match",
            "affected_entity_exact_match",
            "verified_erase_gate_mean",
            "verified_write_gate_mean",
            "distractor_erase_gate_mean",
            "distractor_write_gate_mean",
            "distractor_joint_gate_mass_per_sequence",
        ):
            _finite_value(row, metric)

        is_stress = (
            int(row["updates"]) == stress_updates
            and int(row["gap_events"]) == stress_gap
        )
        if is_stress and not dry_run:
            stress_counts[key] += 1
            if row.get("distractor_activation_activate_distractor_verified") is not True:
                raise RuntimeError("E13b-R1 stress row lacks the active-path assay")
            for metric in (
                "distractor_activation_affected_mse",
                "distractor_activation_retention_mse",
                "distractor_activation_old_rule_residual",
                "distractor_activation_distractor_erase_gate_mean",
                "distractor_activation_distractor_write_gate_mean",
            ):
                _finite_value(row, metric)
        elif any(
            str(field).startswith("distractor_activation_")
            for field in row
        ):
            raise RuntimeError("Active-path metrics appear outside the stress cell")

    if not dry_run and any(
        stress_counts[key] != 1
        for key in provenance
    ):
        raise RuntimeError(
            "Every E13b-R1 source must contain exactly one active-path stress row"
        )


def _validate_calibration_chain(
    source_runs: list[dict],
    *,
    expected_source_config_sha256: str,
) -> dict[str, str]:
    dependencies: dict[
        tuple[str, str, str, str, str],
        dict[str, str],
    ] = {}
    for source in source_runs:
        source_report_path = Path(str(source["report_path"])).resolve()
        if file_sha256(source_report_path) != source["report_sha256"]:
            raise RuntimeError("Pinned E13b-R1 report hash changed")
        report = _read_json(source_report_path)
        dependency = report.get("calibration_dependency")
        if not isinstance(dependency, dict):
            raise RuntimeError("E13b-R1 source lacks calibration dependency")
        normalized = {str(k): str(v) for k, v in dependency.items()}
        key = (
            normalized.get("run_dir", ""),
            normalized.get("report_sha256", ""),
            normalized.get("run_manifest_sha256", ""),
            normalized.get("source_config_sha256", ""),
            normalized.get("experiment_id", ""),
        )
        dependencies[key] = normalized
        if dependency.get("experiment_id") != CALIBRATION_EXPERIMENT_ID:
            raise RuntimeError("E13b-R1 source cites the wrong calibration")
        if (
            dependency.get("source_config_sha256")
            != expected_source_config_sha256
        ):
            raise RuntimeError("E13b-R1 source config hash chain is inconsistent")
        calibration_report_path = Path(str(dependency["report_path"]))
        calibration_manifest_path = Path(str(dependency["run_manifest_path"]))
        calibration_run_dir = Path(str(dependency["run_dir"])).resolve()
        if (
            calibration_report_path.resolve().parent != calibration_run_dir
            or calibration_manifest_path.resolve().parent != calibration_run_dir
        ):
            raise RuntimeError("Pinned E13a-R2 paths do not share one run directory")
        if file_sha256(calibration_report_path) != dependency["report_sha256"]:
            raise RuntimeError("Pinned E13a-R2 report hash changed")
        if (
            file_sha256(calibration_manifest_path)
            != dependency["run_manifest_sha256"]
        ):
            raise RuntimeError("Pinned E13a-R2 manifest hash changed")
        calibration_report = _read_json(calibration_report_path)
        calibration_manifest = _read_json(calibration_manifest_path)
        if calibration_report.get("status") != "PASS" or not (
            calibration_report.get("claim_gate", {})
            .get("go_for_e13b_r1", False)
        ):
            raise RuntimeError("Pinned E13a-R2 report does not open E13b-R1")
        if not calibration_report.get("distractor_path_contract", {}).get(
            "passed",
            False,
        ):
            raise RuntimeError("Pinned E13a-R2 distractor-path contract failed")
        if (
            calibration_report.get("e13b_scale_feasibility", {}).get(
                "source_config_file_sha256"
            )
            != expected_source_config_sha256
        ):
            raise RuntimeError("Pinned E13a-R2 calibrated a different source config")
        if (
            calibration_manifest.get("schema_version") != 2
            or calibration_manifest.get("experiment_id")
            != CALIBRATION_EXPERIMENT_ID
            or calibration_manifest.get("run_mode") != "MAIN"
            or calibration_manifest.get("run_id") != calibration_run_dir.name
            or calibration_manifest.get("report_sha256")
            != dependency["report_sha256"]
        ):
            raise RuntimeError("Pinned E13a-R2 manifest is not a completed MAIN run")
    if len(dependencies) != 1:
        raise RuntimeError(
            "All E13b-R1 source runs must cite one identical E13a-R2 report"
        )
    return next(iter(dependencies.values()))


def _row_index(rows: list[dict]) -> dict[tuple[int, str, int, int], dict]:
    result: dict[tuple[int, str, int, int], dict] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["variant"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        if key in result:
            raise RuntimeError(f"Duplicate E13b-R1 metric row: {key}")
        result[key] = row
    return result


def _paired_digest_contract(
    rows: list[dict],
    *,
    required_seeds: tuple[int, ...],
    required_updates: tuple[int, ...],
    required_gaps: tuple[int, ...],
) -> bool:
    by_base: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    by_pair: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for row in rows:
        seed = int(row["seed"])
        variant = str(row["variant"])
        updates = int(row["updates"])
        gap = int(row["gap_events"])
        digest = str(row["base_transaction_digest"])
        by_base[(seed, variant, updates)].add(digest)
        by_pair[(seed, updates, gap)].add(digest)
    return bool(
        all(
            len(by_base[(seed, variant, updates)]) == 1
            for seed in required_seeds
            for variant in legacy.FIXED_VARIANTS
            for updates in required_updates
        )
        and all(
            len(by_pair[(seed, updates, gap)]) == 1
            for seed in required_seeds
            for updates in required_updates
            for gap in required_gaps
        )
    )


def _stress_statistics(
    rows: list[dict],
    *,
    seeds: tuple[int, ...],
    gaps: tuple[int, ...],
    stress_updates: int,
    stress_gap: int,
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    index = _row_index(rows)
    zero_gap = min(gaps)
    stress_rows: list[dict[str, float | int]] = []
    gains: list[float] = []
    retentions: list[float] = []
    gap_degradations: list[float] = []
    active_path_harms: list[float] = []
    for seed in seeds:
        try:
            tied = index[(seed, "tied", stress_updates, stress_gap)]
            dual = index[(seed, "dual", stress_updates, stress_gap)]
            dual_zero = index[(seed, "dual", stress_updates, zero_gap)]
        except KeyError as error:
            raise RuntimeError(
                f"Missing E13c-R1 stress or paired zero-gap cell: {error.args[0]}"
            ) from error
        gain = _finite_value(tied, "affected_mse") - _finite_value(
            dual,
            "affected_mse",
        )
        retention = _finite_value(dual, "retention_mse")
        gap_degradation = _finite_value(dual, "affected_mse") - _finite_value(
            dual_zero,
            "affected_mse",
        )
        active_path_harm = _finite_value(
            dual,
            "distractor_activation_retention_mse",
        ) - retention
        gains.append(gain)
        retentions.append(retention)
        gap_degradations.append(gap_degradation)
        active_path_harms.append(active_path_harm)
        stress_rows.append(
            {
                "seed": seed,
                "stress_affected_gain_tied_minus_dual": gain,
                "dual_stress_retention_mse": retention,
                "dual_gap2048_minus_gap0_affected_mse": gap_degradation,
                "active_path_retention_harm": active_path_harm,
            }
        )
    statistics = {
        "stress_mean_affected_gain": sum(gains) / len(gains),
        "stress_sign_flip_p": exact_sign_flip(gains, alternative="greater"),
        "stress_positive_seed_direction_fraction": (
            sum(value > 0.0 for value in gains) / len(gains)
        ),
        "maximum_dual_stress_retention_mse": max(retentions),
        "maximum_dual_gap_degradation": max(gap_degradations),
        "minimum_active_path_retention_harm": min(active_path_harms),
    }
    return stress_rows, statistics


def _stress_gate_conditions(
    statistics: dict[str, float],
    *,
    claim_gate: dict,
    alpha: float,
) -> dict[str, bool]:
    return {
        "stress_gain_passed": (
            statistics["stress_mean_affected_gain"]
            >= float(claim_gate["minimum_stress_affected_gain"])
        ),
        "stress_sign_flip_passed": (
            statistics["stress_sign_flip_p"] <= alpha
        ),
        "stress_seed_direction_passed": (
            statistics["stress_positive_seed_direction_fraction"] >= 1.0
        ),
        "dual_stress_retention_passed": (
            statistics["maximum_dual_stress_retention_mse"]
            <= float(claim_gate["maximum_dual_stress_retention_mse"])
        ),
        "dual_gap_noninferiority_passed": (
            statistics["maximum_dual_gap_degradation"]
            <= float(claim_gate["maximum_dual_gap_degradation"])
        ),
        "active_path_assay_passed": (
            statistics["minimum_active_path_retention_harm"]
            >= float(claim_gate["minimum_active_path_retention_harm"])
        ),
    }


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    legacy.SOURCE_EXPERIMENT_ID = SOURCE_EXPERIMENT_ID
    seeds, _, updates, gaps = legacy._source_contract(
        config,
        dry_run=args.dry_run,
    )
    rows, source_runs, excluded_incomplete_runs = collect_e13b_r1_sources(
        artifact_root=args.artifact_root,
        config=config,
        dry_run=args.dry_run,
    )
    stress_updates = int(config["claim_gate"]["stress_updates"])
    stress_gap = int(config["claim_gate"]["stress_gap_events"])
    _pin_source_manifest_integrity(source_runs, dry_run=args.dry_run)
    _validate_r1_metric_rows(
        rows,
        source_runs,
        stress_updates=stress_updates,
        stress_gap=stress_gap,
        dry_run=args.dry_run,
    )
    paired, seed_mean_gain, seed_mean_retention = legacy.aggregate_paired_rows(
        rows,
        required_seeds=seeds,
        required_updates=updates,
        required_gaps=gaps,
    )

    source_config_sha256 = file_sha256(config["source"]["config_path"])
    calibration_dependency = None
    if not args.dry_run:
        calibration_dependency = _validate_calibration_chain(
            source_runs,
            expected_source_config_sha256=source_config_sha256,
        )

    alpha = float(config["statistics"]["alpha"])
    sesoi = float(config["claim_gate"]["minimum_asymmetric_gain"])
    retention_margin = float(config["claim_gate"]["retention_noninferiority"])
    direction_threshold = float(
        config["claim_gate"]["minimum_seed_direction_fraction"]
    )
    mean_gain = sum(seed_mean_gain) / len(seed_mean_gain)
    overall_p = exact_sign_flip(seed_mean_gain, alternative="greater")
    positive_fraction = sum(value > 0.0 for value in seed_mean_gain) / len(
        seed_mean_gain
    )
    max_mean_retention = max(seed_mean_retention)
    digest_contract = _paired_digest_contract(
        rows,
        required_seeds=seeds,
        required_updates=updates,
        required_gaps=gaps,
    )

    stress_seed_rows: list[dict[str, float | int]] = []
    stress_statistics: dict[str, float] | None = None
    stress_conditions = {
        "stress_gain_passed": False,
        "stress_sign_flip_passed": False,
        "stress_seed_direction_passed": False,
        "dual_stress_retention_passed": False,
        "dual_gap_noninferiority_passed": False,
        "active_path_assay_passed": False,
    }
    if not args.dry_run:
        stress_seed_rows, stress_statistics = _stress_statistics(
            rows,
            seeds=seeds,
            gaps=gaps,
            stress_updates=stress_updates,
            stress_gap=stress_gap,
        )
        stress_conditions = _stress_gate_conditions(
            stress_statistics,
            claim_gate=config["claim_gate"],
            alpha=alpha,
        )

    conditions = {
        "overall_mean_gain_passed": mean_gain >= sesoi,
        "overall_sign_flip_passed": overall_p <= alpha,
        "overall_seed_direction_passed": (
            positive_fraction >= direction_threshold
        ),
        "overall_retention_noninferiority_passed": (
            max_mean_retention <= retention_margin
        ),
        "base_transaction_digest_contract_passed": digest_contract,
        **stress_conditions,
        "calibration_hash_chain_passed": calibration_dependency is not None,
    }
    supported = bool(not args.dry_run and all(conditions.values()))
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "LEARNED_DISTRACTOR_TRANSACTIONAL_SEQUENCE_AGGREGATE",
        "source_runs": source_runs,
        "excluded_operational_incomplete_runs": excluded_incomplete_runs,
        "calibration_dependency": calibration_dependency,
        "source_contract": {
            "fixed_seeds": list(seeds),
            "variants": list(legacy.FIXED_VARIANTS),
            "updates": list(updates),
            "gap_events": list(gaps),
            "source_config_path": str(config["source"]["config_path"]),
            "source_config_sha256": source_config_sha256,
            "complete_grid_required": True,
            "unique_run_per_seed_variant_required": True,
            "same_base_transaction_across_gaps_required": True,
        },
        "summary": {
            "paired_cells": len(paired),
            "paired_seeds": len(seed_mean_gain),
            "mean_affected_gain": mean_gain,
            "sign_flip_p": overall_p,
            "positive_seed_direction_fraction": positive_fraction,
            "max_seed_mean_retention_degradation": max_mean_retention,
            "stress_mean_affected_gain": (
                None
                if stress_statistics is None
                else stress_statistics["stress_mean_affected_gain"]
            ),
            "stress_sign_flip_p": (
                None
                if stress_statistics is None
                else stress_statistics["stress_sign_flip_p"]
            ),
            "stress_positive_seed_direction_fraction": (
                None
                if stress_statistics is None
                else stress_statistics[
                    "stress_positive_seed_direction_fraction"
                ]
            ),
            "maximum_dual_stress_retention_mse": (
                None
                if stress_statistics is None
                else stress_statistics["maximum_dual_stress_retention_mse"]
            ),
            "maximum_dual_gap_degradation": (
                None
                if stress_statistics is None
                else stress_statistics["maximum_dual_gap_degradation"]
            ),
            "minimum_active_path_retention_harm": (
                None
                if stress_statistics is None
                else stress_statistics["minimum_active_path_retention_harm"]
            ),
        },
        "claim_gate": {
            "supported": supported,
            "conditions": conditions,
            "allowed_claim": (
                "In the structured learned-distractor sequence bridge, "
                "independent erase/write control improves repeated-update "
                "correction and remains effective through the registered "
                "2,048-event distractor stress."
            ),
            "forbidden_claim": (
                "Natural-language, learned-addressing, recurrent-LM, agent, "
                "planning, or official-backend transfer."
            ),
        },
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    write_jsonl(run_dir / "sequence_paired_metrics.jsonl", paired)
    write_jsonl(run_dir / "sequence_stress_seed_metrics.jsonl", stress_seed_rows)
    write_jsonl(run_dir / "source_run_provenance.jsonl", source_runs)
    write_jsonl(
        run_dir / "excluded_operational_incomplete_runs.jsonl",
        excluded_incomplete_runs,
    )
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
