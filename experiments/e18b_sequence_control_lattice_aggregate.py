from __future__ import annotations

import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.postcore_metrics import exact_sign_flip
from catena.models.sequence_control_lattice import SequenceControlFreedom
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e18b_sequence_control_lattice_aggregate"
DEFAULT_CONFIG = "configs/e18b_sequence_control_lattice_aggregate.yaml"
SOURCE_EXPERIMENT_ID = "e18a_sequence_control_lattice"
PROTOCOL_LOCK = Path("docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line, parse_constant=reject_nonfinite)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def _finite(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing/invalid E18a metric {key!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite E18a metric {key!r}")
    return value


def _source_contract(config: dict) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    source = config["source"]
    seeds = tuple(int(value) for value in source["required_seeds"])
    variants = tuple(str(value) for value in source["required_variants"])
    demands = tuple(str(value) for value in source["required_demands"])
    updates = tuple(int(value) for value in source["required_updates"])
    gaps = tuple(int(value) for value in source["required_gap_events"])
    if len(seeds) != 5 or len(set(seeds)) != len(seeds):
        raise ValueError("E18b requires five unique paired seeds")
    if variants != tuple(value.value for value in SequenceControlFreedom):
        raise ValueError("E18b controller lattice order is not canonical")
    if len(demands) != 4 or len(set(demands)) != 4:
        raise ValueError("E18b requires four ordered demand families")
    if updates != (1, 4, 8) or gaps != (0, 128, 512, 2048):
        raise ValueError("E18b test grid differs from the registered grid")
    return seeds, variants, demands, updates, gaps


def validate_protocol_lock(
    *,
    aggregate_config_path: str | Path,
    source_config_path: str | Path,
) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = PROTOCOL_LOCK.resolve()
    lock = _read_json_object(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or lock.get("aggregate_experiment_id") != EXPERIMENT_ID
        or lock.get("evaluation_started") is not False
        or lock.get("protocol_frozen_before_evaluation") is not True
    ):
        raise RuntimeError("E18 protocol lock is invalid or not prospective")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("E18 protocol lock lacks its file map")
    for relative_path, expected_hash in files.items():
        candidate = (repo_root / str(relative_path)).resolve()
        if not candidate.is_file() or file_sha256(candidate) != expected_hash:
            raise RuntimeError(f"E18 locked file changed: {relative_path}")
    for path in (aggregate_config_path, source_config_path):
        resolved = Path(path).resolve()
        relative = resolved.relative_to(repo_root).as_posix()
        if files.get(relative) != file_sha256(resolved):
            raise RuntimeError(f"E18 locked file changed: {relative}")
    return {
        "path": str(lock_path),
        "sha256": file_sha256(lock_path),
        "aggregate_config_sha256": file_sha256(aggregate_config_path),
        "source_config_sha256": file_sha256(source_config_path),
    }


def _validate_source_run(
    *,
    run_dir: Path,
    expected_config: dict,
    source_config_path: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    demands: tuple[str, ...],
    updates: tuple[int, ...],
    gaps: tuple[int, ...],
    protocol_lock_sha256: str,
) -> tuple[tuple[int, str], list[dict[str, Any]], dict[str, Any]] | None:
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    if not report_path.exists():
        if manifest_path.exists():
            manifest = _read_json_object(manifest_path)
            if manifest.get("run_mode") == "MAIN":
                raise RuntimeError(f"Incomplete E18a MAIN run: {run_dir}")
        return None
    report = _read_json_object(report_path)
    manifest = _read_json_object(manifest_path)
    if report.get("run_mode") == "DRY_RUN":
        if manifest.get("run_mode") != "DRY_RUN":
            raise RuntimeError(f"E18a dry-run mode mismatch: {run_dir}")
        return None
    report_sha256 = file_sha256(report_path)
    if (
        report.get("status") != "PASS"
        or report.get("run_mode") != "MAIN"
        or report.get("claim_gate", {}).get("status") != "PENDING_AGGREGATE"
        or report.get("distractor_path_contract", {}).get("passed") is not True
        or manifest.get("schema_version") != 2
        or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or manifest.get("run_id") != run_dir.name
        or manifest.get("run_mode") != "MAIN"
        or manifest.get("report_sha256") != report_sha256
        or manifest.get("config") != expected_config
        or manifest.get("config_file_sha256") != file_sha256(source_config_path)
    ):
        raise RuntimeError(f"Invalid completed E18a MAIN provenance: {run_dir}")
    seed = int(report["seed"])
    variant = str(report["variant"])
    if seed not in seeds or variant not in variants:
        raise RuntimeError(f"E18a run is outside registered seed/variant grid: {run_dir}")
    expected_rows = len(demands) * len(updates) * len(gaps)
    if report.get("rows") != expected_rows or report.get("expected_rows") != expected_rows:
        raise RuntimeError(f"E18a report row count mismatch: {run_dir}")
    report_lock = report.get("protocol_lock", {})
    if report_lock.get("sha256") != protocol_lock_sha256:
        raise RuntimeError(f"E18a protocol lock hash mismatch: {run_dir}")

    metrics_path = run_dir / "sequence_control_lattice_metrics.jsonl"
    rows = _read_jsonl(metrics_path)
    expected_grid = set(product(demands, updates, gaps))
    observed_grid: set[tuple[str, int, int]] = set()
    checkpoint_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    initialization_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    optimizers: set[str] = set()
    assay_rows = 0
    for row in rows:
        if int(row.get("seed", -1)) != seed or str(row.get("variant")) != variant:
            raise RuntimeError(f"E18a row identity mismatch: {run_dir}")
        key = (
            str(row.get("demand_family")),
            int(row.get("updates", -1)),
            int(row.get("gap_events", -1)),
        )
        if key in observed_grid:
            raise RuntimeError(f"Duplicate E18a metric row {key}: {run_dir}")
        observed_grid.add(key)
        checkpoint_paths.add(str(row.get("checkpoint")))
        checkpoint_hashes.add(str(row.get("checkpoint_sha256")))
        initialization_hashes.add(str(row.get("initialization_sha256")))
        parameter_counts.add(int(row.get("parameter_count", -1)))
        optimizers.add(str(row.get("optimizer")))
        if str(row.get("protocol_lock_sha256")) != protocol_lock_sha256:
            raise RuntimeError(f"E18a row lock mismatch: {run_dir}")
        for metric in ("affected_mse", "retention_mse", "state_mse"):
            _finite(row, metric)
        is_assay = (
            key[1] == 8
            and key[2] == 2048
            and "distractor_activation_retention_harm" in row
        )
        assay_rows += int(is_assay)
        if is_assay:
            _finite(row, "distractor_activation_retention_harm")
    if observed_grid != expected_grid or len(rows) != expected_rows:
        raise RuntimeError(f"Incomplete E18a metric grid: {run_dir}")
    if (
        len(checkpoint_paths) != 1
        or len(checkpoint_hashes) != 1
        or len(initialization_hashes) != 1
        or len(parameter_counts) != 1
        or optimizers != {"AdamW"}
        or assay_rows != len(demands)
    ):
        raise RuntimeError(f"E18a paired/artifact contract mismatch: {run_dir}")
    checkpoint_path = Path(next(iter(checkpoint_paths)))
    checkpoint_sha256 = next(iter(checkpoint_hashes))
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError(f"E18a checkpoint hash mismatch: {run_dir}")
    provenance = {
        "seed": seed,
        "variant": variant,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": report_sha256,
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "initialization_sha256": next(iter(initialization_hashes)),
        "parameter_count": next(iter(parameter_counts)),
        "optimizer": next(iter(optimizers)),
        "distractor_path_contract_passed": True,
    }
    return (seed, variant), rows, provenance


def collect_main_sources(
    *,
    artifact_root: str | Path,
    config: dict,
    protocol_lock_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds, variants, demands, updates, gaps = _source_contract(config)
    source_config_path = Path(config["source"]["config_path"]).resolve()
    expected_config = load_config(source_config_path)
    source_root = Path(artifact_root) / SOURCE_EXPERIMENT_ID
    if not source_root.is_dir():
        raise FileNotFoundError(f"No E18a source namespace: {source_root}")
    by_key: dict[
        tuple[int, str],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ] = {}
    for run_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        result = _validate_source_run(
            run_dir=run_dir,
            expected_config=expected_config,
            source_config_path=source_config_path,
            variants=variants,
            seeds=seeds,
            demands=demands,
            updates=updates,
            gaps=gaps,
            protocol_lock_sha256=protocol_lock_sha256,
        )
        if result is None:
            continue
        key, rows, provenance = result
        if key in by_key:
            raise RuntimeError(f"Duplicate eligible E18a source run: {key}")
        by_key[key] = (rows, provenance)
    expected_keys = set(product(seeds, variants))
    if set(by_key) != expected_keys:
        raise RuntimeError(
            "E18a complete paired source grid mismatch: "
            f"missing={sorted(expected_keys - set(by_key))}, "
            f"extra={sorted(set(by_key) - expected_keys)}"
        )
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        source_rows, source_provenance = by_key[key]
        rows.extend(source_rows)
        provenance.append(source_provenance)
    return rows, provenance


def _synthetic_dry_rows(config: dict) -> list[dict[str, Any]]:
    seeds, variants, demands, updates, gaps = _source_contract(config)
    rows: list[dict[str, Any]] = []
    for seed, variant, demand, update_count, gap in product(
        seeds,
        variants,
        demands,
        updates,
        gaps,
    ):
        freedom_rank = variants.index(variant)
        requirement_rank = demands.index(demand) + 1
        affected = 0.0001 if freedom_rank >= requirement_rank else 0.004
        digest = hashlib.sha256(
            f"{seed}:{demand}:{update_count}".encode()
        ).hexdigest()
        row: dict[str, Any] = {
            "seed": seed,
            "variant": variant,
            "demand_family": demand,
            "updates": update_count,
            "gap_events": gap,
            "evaluation_seed": 100_000 + seed + update_count,
            "affected_mse": affected,
            "retention_mse": 0.00001,
            "state_mse": affected / 2.0,
            "base_transaction_digest": digest,
            "initialization_sha256": hashlib.sha256(
                f"init:{seed}".encode()
            ).hexdigest(),
            "parameter_count": 1000,
            "optimizer": "AdamW",
        }
        if update_count == 8 and gap == 2048:
            row["distractor_activation_retention_harm"] = 0.01
        rows.append(row)
    return rows


def paired_grid_contract(
    *,
    rows: list[dict[str, Any]],
    config: dict,
) -> bool:
    seeds, variants, demands, updates, gaps = _source_contract(config)
    expected_count = (
        len(seeds) * len(variants) * len(demands) * len(updates) * len(gaps)
    )
    if len(rows) != expected_count:
        return False
    initialization: dict[int, set[str]] = {seed: set() for seed in seeds}
    parameters: dict[int, set[int]] = {seed: set() for seed in seeds}
    paired: dict[tuple[int, str, int, int], set[tuple[int, str]]] = {}
    across_gaps: dict[tuple[int, str, str, int], set[str]] = {}
    for row in rows:
        seed = int(row["seed"])
        initialization[seed].add(str(row["initialization_sha256"]))
        parameters[seed].add(int(row["parameter_count"]))
        pair_key = (
            seed,
            str(row["demand_family"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        paired.setdefault(pair_key, set()).add(
            (
                int(row["evaluation_seed"]),
                str(row["base_transaction_digest"]),
            )
        )
        gap_key = (
            seed,
            str(row["variant"]),
            str(row["demand_family"]),
            int(row["updates"]),
        )
        across_gaps.setdefault(gap_key, set()).add(
            str(row["base_transaction_digest"])
        )
    return bool(
        all(len(values) == 1 for values in initialization.values())
        and all(len(values) == 1 for values in parameters.values())
        and all(len(values) == 1 for values in paired.values())
        and all(len(values) == 1 for values in across_gaps.values())
        and len(paired) == len(seeds) * len(demands) * len(updates) * len(gaps)
    )


def aggregate_contrasts(
    *,
    rows: list[dict[str, Any]],
    config: dict,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    seeds, variants, demands, updates, gaps = _source_contract(config)
    index: dict[tuple[int, str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["variant"]),
            str(row["demand_family"]),
            int(row["updates"]),
            int(row["gap_events"]),
        )
        if key in index:
            raise RuntimeError(f"Duplicate E18 aggregate metric row: {key}")
        index[key] = row
    adjacent_names = (
        "magnitude_factorization",
        "value_granularity",
        "address_decoupling",
        "state_conditioning",
    )
    gate = config["claim_gate"]
    paired_rows: list[dict[str, Any]] = []
    contrasts: dict[str, dict[str, Any]] = {}
    for demand_index, name in enumerate(adjacent_names):
        baseline = variants[demand_index]
        treatment = variants[demand_index + 1]
        target_demand = demands[demand_index]
        seed_gains: list[float] = []
        stress_gains: list[float] = []
        simpler_degradations: list[float] = []
        retention_degradations: list[float] = []
        for seed in seeds:
            cell_gains = [
                _finite(
                    index[(seed, baseline, target_demand, update_count, gap)],
                    "affected_mse",
                )
                - _finite(
                    index[(seed, treatment, target_demand, update_count, gap)],
                    "affected_mse",
                )
                for update_count, gap in product(updates, gaps)
            ]
            seed_gain = sum(cell_gains) / len(cell_gains)
            stress_gain = _finite(
                index[(seed, baseline, target_demand, 8, 2048)],
                "affected_mse",
            ) - _finite(
                index[(seed, treatment, target_demand, 8, 2048)],
                "affected_mse",
            )
            seed_gains.append(seed_gain)
            stress_gains.append(stress_gain)
            paired_rows.append(
                {
                    "contrast": name,
                    "seed": seed,
                    "baseline": baseline,
                    "treatment": treatment,
                    "target_demand": target_demand,
                    "mean_corresponding_demand_gain": seed_gain,
                    "stress_gain": stress_gain,
                }
            )
            for simpler in demands[:demand_index]:
                for update_count, gap in product(updates, gaps):
                    simpler_degradations.append(
                        _finite(
                            index[(seed, treatment, simpler, update_count, gap)],
                            "affected_mse",
                        )
                        - _finite(
                            index[(seed, baseline, simpler, update_count, gap)],
                            "affected_mse",
                        )
                    )
            for demand, update_count, gap in product(demands, updates, gaps):
                retention_degradations.append(
                    _finite(
                        index[(seed, treatment, demand, update_count, gap)],
                        "retention_mse",
                    )
                    - _finite(
                        index[(seed, baseline, demand, update_count, gap)],
                        "retention_mse",
                    )
                )
        mean_gain = sum(seed_gains) / len(seed_gains)
        max_simpler = max(simpler_degradations, default=0.0)
        max_retention = max(retention_degradations, default=0.0)
        stress_fraction = sum(value > 0.0 for value in stress_gains) / len(
            stress_gains
        )
        stress_p = exact_sign_flip(stress_gains, alternative="greater")
        passed = bool(
            mean_gain >= float(gate["minimum_corresponding_demand_gain"])
            and max_simpler <= float(gate["maximum_simpler_demand_degradation"])
            and max_retention <= float(gate["maximum_retention_degradation"])
            and stress_fraction >= float(gate["stress_positive_seed_fraction"])
            and stress_p <= float(config["statistics"]["alpha"])
        )
        contrasts[name] = {
            "baseline": baseline,
            "treatment": treatment,
            "target_demand": target_demand,
            "mean_corresponding_demand_gain": mean_gain,
            "maximum_simpler_demand_degradation": max_simpler,
            "maximum_retention_degradation": max_retention,
            "stress_positive_seed_fraction": stress_fraction,
            "stress_sign_flip_p": stress_p,
            "passed": passed,
        }

    active_rows: list[dict[str, Any]] = []
    for seed, variant, demand in product(seeds, variants, demands):
        row = index[(seed, variant, demand, 8, 2048)]
        harm = _finite(row, "distractor_activation_retention_harm")
        active_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "demand_family": demand,
                "active_path_retention_harm": harm,
            }
        )
    return contrasts, paired_rows, active_rows


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    source_config_path = str(config["source"]["config_path"])
    lock = validate_protocol_lock(
        aggregate_config_path=args.config,
        source_config_path=source_config_path,
    )
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if args.dry_run:
        rows = _synthetic_dry_rows(config)
        source_runs: list[dict[str, Any]] = []
    else:
        rows, source_runs = collect_main_sources(
            artifact_root=args.artifact_root,
            config=config,
            protocol_lock_sha256=lock["sha256"],
        )
    grid_passed = paired_grid_contract(rows=rows, config=config)
    contrasts, paired_rows, active_rows = aggregate_contrasts(
        rows=rows,
        config=config,
    )
    minimum_active_harm = min(
        float(row["active_path_retention_harm"])
        for row in active_rows
    )
    active_path_passed = minimum_active_harm >= float(
        config["claim_gate"]["minimum_active_path_retention_harm"]
    )
    source_provenance_passed = bool(
        args.dry_run
        or (
            len(source_runs)
            == len(config["source"]["required_seeds"])
            * len(config["source"]["required_variants"])
            and all(
                row["distractor_path_contract_passed"]
                for row in source_runs
            )
        )
    )
    all_adjacent_passed = all(
        bool(contrast["passed"]) for contrast in contrasts.values()
    )
    conditions = {
        "all_adjacent_contrasts_passed": all_adjacent_passed,
        "full_paired_grid_passed": grid_passed,
        "source_provenance_passed": source_provenance_passed,
        "model_visible_active_path_assay_passed": active_path_passed,
    }
    supported = bool(not args.dry_run and all(conditions.values()))
    report = {
        "status": "DRY_RUN" if args.dry_run else "PASS",
        "run_scope": "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE_AGGREGATE",
        "protocol_lock": lock,
        "source_runs": source_runs,
        "source_contract": {
            **config["source"],
            "source_config_sha256": file_sha256(source_config_path),
            "complete_unique_run_per_seed_variant_required": True,
            "paired_initialization_parameter_data_and_digest_required": True,
        },
        "contrasts": contrasts,
        "summary": {
            "source_runs": len(source_runs),
            "metric_rows": len(rows),
            "paired_contrast_seed_rows": len(paired_rows),
            "active_path_rows": len(active_rows),
            "minimum_active_path_retention_harm": minimum_active_harm,
        },
        "claim_gate": {
            "supported": supported,
            "conditions": conditions,
            "allowed_claim": (
                "In controlled structured transaction sequences with oracle "
                "addresses and candidates, each added memory-control freedom "
                "provides a selective benefit on the registered demand family "
                "that requires it, including the 2,048-event stress."
            ),
            "forbidden_claim": (
                "Natural-language, learned-candidate/address, recurrent-LM, "
                "agent, planning, or official-backend transfer."
            ),
        },
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    write_jsonl(
        run_dir / "sequence_control_lattice_paired_metrics.jsonl",
        paired_rows,
    )
    write_jsonl(
        run_dir / "sequence_control_lattice_active_path_metrics.jsonl",
        active_rows,
    )
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
