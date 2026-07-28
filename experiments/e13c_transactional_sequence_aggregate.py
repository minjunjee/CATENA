from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.postcore_metrics import exact_sign_flip
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e13c_transactional_sequence_aggregate"
DEFAULT_CONFIG = "configs/e13c_transactional_sequence_aggregate.yaml"
SOURCE_EXPERIMENT_ID = "e13b_transactional_sequence_memory"
FIXED_SEEDS = (101, 211, 307, 401, 503)
FIXED_VARIANTS = ("tied", "dual")
FIXED_UPDATES = (1, 4, 8)
FIXED_GAPS = (0, 128, 512, 2048)


def _read_json(path: Path) -> dict[str, Any]:
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
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_contract(
    config: dict[str, Any], *, dry_run: bool
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("E13c config must define a source mapping")
    if source.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError(
            f"source.experiment_id must be {SOURCE_EXPERIMENT_ID!r}"
        )
    seeds = tuple(int(value) for value in source.get("required_seeds", ()))
    variants = tuple(str(value) for value in source.get("required_variants", ()))
    updates = tuple(int(value) for value in source.get("required_updates", ()))
    gaps = tuple(int(value) for value in source.get("required_gap_events", ()))
    if seeds != FIXED_SEEDS:
        raise ValueError(f"source.required_seeds must equal {list(FIXED_SEEDS)}")
    if variants != FIXED_VARIANTS:
        raise ValueError(
            f"source.required_variants must equal {list(FIXED_VARIANTS)}"
        )
    if updates != FIXED_UPDATES:
        raise ValueError(
            f"source.required_updates must equal {list(FIXED_UPDATES)}"
        )
    if gaps != FIXED_GAPS:
        raise ValueError(
            f"source.required_gap_events must equal {list(FIXED_GAPS)}"
        )
    if dry_run:
        return seeds[:1], variants, updates[:1], gaps[:1]
    return seeds, variants, updates, gaps


def _finite_float(row: dict[str, Any], key: str, *, path: Path) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in {path}")
    return value


def _validate_source_run(
    run_dir: Path,
    *,
    expected_source_config: dict[str, Any],
    required_updates: tuple[int, ...],
    required_gaps: tuple[int, ...],
    dry_run: bool,
) -> tuple[tuple[int, str], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "report.json"
    metrics_path = run_dir / "sequence_main_metrics.jsonl"
    missing = [
        str(path.name)
        for path in (manifest_path, report_path, metrics_path)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"Ineligible E13b run {run_dir}: missing {', '.join(missing)}"
        )

    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    expected_status = "DRY_RUN" if dry_run else "PASS"
    expected_gate_status = "DRY_RUN" if dry_run else "PENDING_AGGREGATE"
    if report.get("status") != expected_status:
        raise RuntimeError(
            f"Ineligible E13b run {run_dir}: status={report.get('status')!r}, "
            f"expected {expected_status!r}"
        )
    gate = report.get("claim_gate")
    if not isinstance(gate, dict) or gate.get("status") != expected_gate_status:
        raise RuntimeError(
            f"Ineligible E13b run {run_dir}: claim gate is not "
            f"{expected_gate_status!r}"
        )
    if manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise RuntimeError(
            f"Ineligible E13b run {run_dir}: wrong experiment_id"
        )
    if manifest.get("config") != expected_source_config:
        raise RuntimeError(
            f"Ineligible E13b run {run_dir}: resolved source config mismatch"
        )

    rows = _read_jsonl(metrics_path)
    if not rows:
        raise RuntimeError(f"Ineligible E13b run {run_dir}: empty metrics")
    variants = {str(row["variant"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    if len(variants) != 1 or len(seeds) != 1:
        raise RuntimeError(
            f"E13b run {run_dir} must contain exactly one variant and one seed"
        )
    variant = next(iter(variants))
    seed = next(iter(seeds))
    expected_cells = set(product(required_updates, required_gaps))
    observed_cells: set[tuple[int, int]] = set()
    checkpoints: set[Path] = set()
    for row in rows:
        cell = (int(row["updates"]), int(row["gap_events"]))
        if cell in observed_cells:
            raise RuntimeError(f"Duplicate E13b cell {cell} in {run_dir}")
        observed_cells.add(cell)
        for metric in (
            "affected_mse",
            "retention_mse",
            "old_rule_residual",
            "entity_exact_match",
        ):
            _finite_float(row, metric, path=metrics_path)
        checkpoint = Path(str(row["checkpoint"])).resolve()
        checkpoints.add(checkpoint)
    if observed_cells != expected_cells:
        missing_cells = sorted(expected_cells - observed_cells)
        extra_cells = sorted(observed_cells - expected_cells)
        raise RuntimeError(
            f"Incomplete E13b grid in {run_dir}: "
            f"missing={missing_cells}, extra={extra_cells}"
        )
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"E13b run {run_dir} must reference exactly one checkpoint"
        )
    checkpoint = next(iter(checkpoints))
    checkpoint_root = (run_dir / "checkpoints").resolve()
    if checkpoint.parent != checkpoint_root or not checkpoint.is_file():
        raise RuntimeError(
            f"E13b checkpoint is missing or outside its run directory: {checkpoint}"
        )
    if int(report.get("rows", -1)) != len(rows):
        raise RuntimeError(f"E13b report row count mismatch in {run_dir}")

    provenance = {
        "run_dir": str(run_dir.resolve()),
        "seed": seed,
        "variant": variant,
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_config_canonical_sha256": _canonical_sha256(
            expected_source_config
        ),
    }
    return (seed, variant), rows, provenance


def collect_e13b_sources(
    *,
    artifact_root: str | Path,
    config: dict[str, Any],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds, variants, updates, gaps = _source_contract(config, dry_run=dry_run)
    source_config_path = Path(str(config["source"]["config_path"]))
    expected_source_config = load_config(source_config_path)
    source_root = Path(artifact_root) / SOURCE_EXPERIMENT_ID
    if not source_root.is_dir():
        raise FileNotFoundError(f"No E13b runs found under {source_root}")

    by_key: dict[tuple[int, str], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        key, rows, provenance = _validate_source_run(
            run_dir,
            expected_source_config=expected_source_config,
            required_updates=updates,
            required_gaps=gaps,
            dry_run=dry_run,
        )
        if key in by_key:
            raise RuntimeError(
                f"Duplicate eligible E13b runs for seed={key[0]}, "
                f"variant={key[1]}"
            )
        by_key[key] = (rows, provenance)

    expected_keys = set(product(seeds, variants))
    observed_keys = set(by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            f"E13b source-run contract mismatch: missing={missing}, extra={extra}"
        )
    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        source_rows, provenance = by_key[key]
        rows.extend(source_rows)
        provenance_rows.append(provenance)
    return rows, provenance_rows


def aggregate_paired_rows(
    rows: list[dict[str, Any]],
    *,
    required_seeds: tuple[int, ...],
    required_updates: tuple[int, ...],
    required_gaps: tuple[int, ...],
) -> tuple[list[dict[str, float | int]], list[float], list[float]]:
    key_to_rows: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (int(row["seed"]), int(row["updates"]), int(row["gap_events"]))
        variant = str(row["variant"])
        if variant in key_to_rows[key]:
            raise RuntimeError(f"Duplicate paired variant {variant!r} for {key}")
        key_to_rows[key][variant] = row

    expected_cells = set(product(required_seeds, required_updates, required_gaps))
    if set(key_to_rows) != expected_cells:
        missing = sorted(expected_cells - set(key_to_rows))
        extra = sorted(set(key_to_rows) - expected_cells)
        raise RuntimeError(
            f"Paired E13b cell contract mismatch: missing={missing}, extra={extra}"
        )

    paired: list[dict[str, float | int]] = []
    seed_gains: dict[int, list[float]] = defaultdict(list)
    retention_diffs: dict[int, list[float]] = defaultdict(list)
    for (seed, updates, gap), variants in sorted(key_to_rows.items()):
        if set(variants) != set(FIXED_VARIANTS):
            raise RuntimeError(
                f"E13b paired cell {(seed, updates, gap)} lacks tied/dual"
            )
        tied = variants["tied"]
        dual = variants["dual"]
        gain = float(tied["affected_mse"]) - float(dual["affected_mse"])
        retention = float(dual["retention_mse"]) - float(tied["retention_mse"])
        seed_gains[seed].append(gain)
        retention_diffs[seed].append(retention)
        paired.append(
            {
                "seed": seed,
                "updates": updates,
                "gap_events": gap,
                "affected_gain_tied_minus_dual": gain,
                "retention_dual_minus_tied": retention,
                "old_rule_residual_gain": float(tied["old_rule_residual"])
                - float(dual["old_rule_residual"]),
                "exact_match_gain": float(dual["entity_exact_match"])
                - float(tied["entity_exact_match"]),
            }
        )
    seed_mean_gain = [
        sum(seed_gains[seed]) / len(seed_gains[seed])
        for seed in required_seeds
    ]
    seed_mean_retention = [
        sum(retention_diffs[seed]) / len(retention_diffs[seed])
        for seed in required_seeds
    ]
    return paired, seed_mean_gain, seed_mean_retention


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
    seeds, _, updates, gaps = _source_contract(
        config, dry_run=args.dry_run
    )
    rows, source_runs = collect_e13b_sources(
        artifact_root=args.artifact_root,
        config=config,
        dry_run=args.dry_run,
    )
    paired, seed_mean_gain, seed_mean_retention = aggregate_paired_rows(
        rows,
        required_seeds=seeds,
        required_updates=updates,
        required_gaps=gaps,
    )

    alpha = float(config["statistics"]["alpha"])
    sesoi = float(config["claim_gate"]["minimum_asymmetric_gain"])
    retention_margin = float(config["claim_gate"]["retention_noninferiority"])
    direction_fraction_threshold = float(
        config["claim_gate"]["minimum_seed_direction_fraction"]
    )
    mean_gain = sum(seed_mean_gain) / len(seed_mean_gain)
    max_retention = max(seed_mean_retention)
    p = exact_sign_flip(seed_mean_gain, alternative="greater")
    direction_fraction = sum(value > 0.0 for value in seed_mean_gain) / len(
        seed_mean_gain
    )
    supported = bool(
        not args.dry_run
        and mean_gain >= sesoi
        and p <= alpha
        and direction_fraction >= direction_fraction_threshold
        and max_retention <= retention_margin
    )
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "source_runs": source_runs,
        "source_contract": {
            "fixed_seeds": list(seeds),
            "variants": list(FIXED_VARIANTS),
            "updates": list(updates),
            "gap_events": list(gaps),
            "complete_grid_required": True,
            "unique_run_per_seed_variant_required": True,
        },
        "summary": {
            "paired_cells": len(paired),
            "paired_seeds": len(seed_mean_gain),
            "mean_affected_gain": mean_gain,
            "sign_flip_p": p,
            "minimum_exact_sign_flip_p": 1.0 / (2 ** len(seed_mean_gain)),
            "positive_seed_direction_fraction": direction_fraction,
            "max_seed_mean_retention_degradation": max_retention,
        },
        "claim_gate": {
            "supported": supported,
            "allowed_claim": (
                "In the structured event-sequence bridge, independent erase/write "
                "control improves repeated-update correction without material "
                "retention loss."
            ),
            "forbidden_claim": (
                "Natural-language, agent, or official-backbone transfer."
            ),
        },
    }
    write_jsonl(run_dir / "sequence_paired_metrics.jsonl", paired)
    write_jsonl(run_dir / "source_run_provenance.jsonl", source_runs)
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
