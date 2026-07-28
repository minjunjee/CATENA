from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json, write_jsonl
from catena.post_e21.contracts import (
    PostE21ContractError,
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.locality_data import parse_locality_methods
from catena.post_e21.locality_eval import (
    build_active_cell_rows,
    compute_locality_seed_summaries,
    select_locality_method,
    selection_summary_ko,
    validate_paired_metric_grid,
)
from catena.post_e21.locality_protocol import (
    load_parent_threshold_contract,
    require_temp_dry_root,
    threshold_float,
    validate_e21_freeze_dependency,
    validate_parent_binding,
)
from catena.post_e21.locality_runner import (
    run_locality_method_grid,
    runtime_locality_config,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e22a_locality_method_selection"
DEFAULT_CONFIG = "configs/e22a_locality_method_selection.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E22A_LOCALITY_METHOD_SELECTION_PROTOCOL_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E22a prospective active-path locality method selection"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv("CATENA_ARTIFACT_ROOT", "artifacts"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parent-e21-freeze",
        help="Explicit immutable E21 freeze; mandatory for E22a MAIN.",
    )
    return parser


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        require_temp_dry_root(args.artifact_root)
    config = load_config(args.config)
    if (
        config.get("experiment_id") != EXPERIMENT_ID
        or config.get("protocol", {}).get("phase") != "development_selection"
        or "thresholds" in config
        or "claim_gate" in config
    ):
        raise ValueError("E22a config identity or threshold-inheritance contract failed")
    snapshot = validate_protocol_lock(
        lock_path=LOCK_PATH,
        config_path=args.config,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    validate_parent_binding(snapshot=snapshot, parent=parent)
    dependency: dict[str, Any] | None = None
    if args.dry_run:
        if args.parent_e21_freeze:
            dependency = validate_e21_freeze_dependency(
                freeze_path=args.parent_e21_freeze,
                parent=parent,
            )
    else:
        if not args.parent_e21_freeze:
            raise PostE21ContractError(
                "BLOCKED_DEPENDENCY: E22a MAIN requires explicit --parent-e21-freeze"
            )
        try:
            dependency = validate_e21_freeze_dependency(
                freeze_path=args.parent_e21_freeze,
                parent=parent,
            )
        except PostE21ContractError as error:
            raise PostE21ContractError(f"BLOCKED_DEPENDENCY: {error}") from error

    methods = parse_locality_methods(config["methods"])
    seeds = [int(value) for value in config["development_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("E22a requires exactly three unique development seeds")
    initialized, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if initialized != config:
        raise RuntimeError("E22a config changed at run start")
    protocol_copy = copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    runtime = runtime_locality_config(config, dry_run=args.dry_run)
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload={
            "phase": "E22a",
            "dry_run": bool(args.dry_run),
            "development_seeds": seeds,
            "methods": [method.as_dict() for method in methods],
            "variants": list(runtime["model"]["variants"]),
            "conditions": list(runtime["conditions"]),
            "demand_families": list(runtime["demand_families"]),
            "training_grid": dict(runtime["training"]),
            "evaluation_grid": dict(runtime["evaluation"]),
            "namespaces": dict(runtime["namespaces"]),
            "parent_e21_lock_sha256": parent.sha256,
        },
    )
    rows, checkpoint_hashes, runtime_metadata = run_locality_method_grid(
        runtime=runtime,
        methods=methods,
        seeds=seeds,
        run_dir=run_dir,
        device=device,
        parent_lock_sha256=parent.sha256,
        protocol_lock_sha256=snapshot.sha256,
        risk_scale=threshold_float(
            parent.thresholds,
            "maximum_nontarget_degradation",
        ),
    )
    method_ids = [method.method_id for method in methods]
    validate_paired_metric_grid(
        rows,
        seeds=seeds,
        methods=method_ids,
        variants=[str(value) for value in runtime["model"]["variants"]],
        conditions=[str(value) for value in runtime["conditions"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
    )
    baseline = next(method for method in methods if method.baseline)
    seed_rows = compute_locality_seed_summaries(
        rows,
        seeds=seeds,
        method_ids=method_ids,
        updates_grid=[int(value) for value in runtime["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in runtime["evaluation"]["gap_events"]],
        demand_families=[str(value) for value in runtime["demand_families"]],
        stress_updates=int(runtime["evaluation"]["stress"]["updates"]),
        stress_gap_events=int(runtime["evaluation"]["stress"]["gap_events"]),
    )
    selection = select_locality_method(
        seed_rows,
        methods=methods,
        thresholds=parent.thresholds,
        dry_run=args.dry_run,
    )
    row_artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=rows,
        seed_rows=seed_rows,
    )
    active_cell_rows = build_active_cell_rows(rows)
    active_cells_path = run_dir / "active_cell_metrics.jsonl"
    write_jsonl(active_cells_path, active_cell_rows)
    scores_path = run_dir / "selection_scores.jsonl"
    write_jsonl(scores_path, list(selection["method_summaries"]))
    selected_id = selection["selected_method_id"]
    selected_method = (
        next(method.as_dict() for method in methods if method.method_id == selected_id)
        if selected_id is not None
        else None
    )
    selection_lock_path = run_dir / "selection_lock.json"
    selection_lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_dir.name,
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "selection_status": selection["status"],
        "selected_method": selected_method,
        "baseline_method": baseline.as_dict(),
        "development_seeds": seeds,
        "development_seed_claim_eligible": False,
        "selection_rule": selection["selection_rule"],
        "parent_e21_lock_sha256": parent.sha256,
        "e22a_protocol_lock_sha256": snapshot.sha256,
        "config_sha256": snapshot.config_sha256,
        "data_sha256": data_sha256,
        "raw_metrics_sha256": row_artifacts["raw"]["sha256"],
        "seed_metrics_sha256": row_artifacts["seed"]["sha256"],
        "selection_scores_sha256": file_sha256(scores_path),
        "active_cell_metrics_sha256": file_sha256(active_cells_path),
        "main_confirmatory_unlock": bool(not args.dry_run and selection["status"] == "SELECTED"),
    }
    write_json(selection_lock_path, selection_lock)
    summary_path = run_dir / str(config["results_summary"]["filename"])
    summary_path.write_text(
        selection_summary_ko(
            selection=selection,
            seeds=seeds,
            dry_run=args.dry_run,
        ),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > int(
        config["results_summary"]["maximum_lines"]
    ):
        raise RuntimeError("E22a results summary exceeds one-page contract")
    common = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=False,
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PASS",
        "status": "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "E22A_DEVELOPMENT_METHOD_SELECTION",
        **common,
        "parent_e21": {
            "lock_path": str(parent.path),
            "lock_sha256": parent.sha256,
            "inherited_thresholds": parent.thresholds,
            "explicit_freeze_dependency": dependency,
        },
        "runtime_metadata": runtime_metadata,
        "selection": selection,
        "claim_gate": {
            "status": selection["status"],
            "claim_eligible": False,
            "confirmatory_unlocked": bool(selection_lock["main_confirmatory_unlock"]),
            "allowed_claim": "Development-only method selection diagnostics.",
            "forbidden_claim": (
                "E21 reinterpretation, confirmatory locality, H5, natural "
                "language, LM, agent, official backend, or runtime transfer."
            ),
        },
        "artifacts": {
            "protocol_lock": {
                "path": str(protocol_copy.resolve()),
                "sha256": file_sha256(protocol_copy),
            },
            "data_manifest": {
                "path": str(data_manifest_path.resolve()),
                "sha256": file_sha256(data_manifest_path),
            },
            "rows": row_artifacts,
            "active_cell_metrics": {
                "path": str(active_cells_path.resolve()),
                "sha256": file_sha256(active_cells_path),
                "rows": len(active_cell_rows),
            },
            "selection_scores": {
                "path": str(scores_path.resolve()),
                "sha256": file_sha256(scores_path),
                "rows": len(selection["method_summaries"]),
            },
            "selection_lock": {
                "path": str(selection_lock_path.resolve()),
                "sha256": file_sha256(selection_lock_path),
            },
            "results_summary_ko": _descriptor(summary_path),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS/{selection['status']}: {run_dir}")


if __name__ == "__main__":
    main()
