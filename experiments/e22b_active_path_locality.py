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
from catena.post_e21.locality_data import method_by_id, parse_locality_methods
from catena.post_e21.locality_eval import (
    assess_locality_confirmatory,
    build_active_cell_rows,
    compute_locality_seed_summaries,
    confirmatory_summary_ko,
    validate_paired_metric_grid,
)
from catena.post_e21.locality_protocol import (
    load_parent_threshold_contract,
    require_temp_dry_root,
    threshold_float,
    validate_parent_binding,
    validate_selection_run_dependency,
)
from catena.post_e21.locality_runner import (
    run_locality_method_grid,
    runtime_locality_config,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e22b_active_path_locality"
DEFAULT_CONFIG = "configs/e22b_active_path_locality.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E22B_ACTIVE_PATH_LOCALITY_PROTOCOL_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E22b selected-vs-mean active-path locality confirmation"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv("CATENA_ARTIFACT_ROOT", "artifacts"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selection-run",
        required=True,
        help="Explicit completed E22a run directory.",
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
        or config.get("protocol", {}).get("phase") != "confirmatory"
        or "thresholds" in config
        or "claim_gate" in config
    ):
        raise ValueError("E22b config identity or threshold-inheritance contract failed")
    snapshot = validate_protocol_lock(
        lock_path=LOCK_PATH,
        config_path=args.config,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    parent = load_parent_threshold_contract(repo_root=REPO_ROOT)
    validate_parent_binding(snapshot=snapshot, parent=parent)

    selection_contract = config["selection_contract"]
    e22a_config_path = REPO_ROOT / str(selection_contract["config_path"])
    e22a_lock_path = REPO_ROOT / str(selection_contract["protocol_lock_path"])
    e22a_snapshot = validate_protocol_lock(
        lock_path=e22a_lock_path,
        config_path=e22a_config_path,
        experiment_id=str(selection_contract["experiment_id"]),
        repo_root=REPO_ROOT,
    )
    if snapshot.payload.get("parent_e22a_static_lock_sha256") != e22a_snapshot.sha256:
        raise PostE21ContractError("E22b static lock does not bind E22a")
    try:
        dependency = validate_selection_run_dependency(
            selection_run=args.selection_run,
            parent=parent,
            expected_protocol_lock_sha256=e22a_snapshot.sha256,
            dry_run=args.dry_run,
        )
    except PostE21ContractError as error:
        raise PostE21ContractError(f"BLOCKED_DEPENDENCY: {error}") from error
    e22a_config = load_config(e22a_config_path)
    e22a_methods = parse_locality_methods(e22a_config["methods"])
    selected_payload = dependency["selected_method"]
    baseline_payload = dependency["baseline_method"]
    selected = method_by_id(
        e22a_methods,
        str(selected_payload["method_id"]),
    )
    baseline = method_by_id(
        e22a_methods,
        str(baseline_payload["method_id"]),
    )
    if selected.as_dict() != selected_payload or baseline.as_dict() != baseline_payload:
        raise PostE21ContractError("E22a selected method differs from frozen grid")
    methods = [baseline, selected]
    seeds = [int(value) for value in config["confirmatory_seeds"]]
    development_seeds = {int(value) for value in e22a_config["development_seeds"]}
    if len(seeds) != 8 or len(set(seeds)) != 8 or set(seeds) & development_seeds:
        raise ValueError("E22b requires eight unique fresh paired seeds")

    initialized, run_dir, device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    if initialized != config:
        raise RuntimeError("E22b config changed at run start")
    protocol_copy = copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)
    runtime = runtime_locality_config(config, dry_run=args.dry_run)
    data_manifest_path, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload={
            "phase": "E22b",
            "dry_run": bool(args.dry_run),
            "confirmatory_seeds": seeds,
            "methods": [method.as_dict() for method in methods],
            "variants": list(runtime["model"]["variants"]),
            "conditions": list(runtime["conditions"]),
            "demand_families": list(runtime["demand_families"]),
            "training_grid": dict(runtime["training"]),
            "evaluation_grid": dict(runtime["evaluation"]),
            "namespaces": dict(runtime["namespaces"]),
            "parent_e21_lock_sha256": parent.sha256,
            "selection_lock_sha256": dependency["selection_lock_sha256"],
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
    assessment = assess_locality_confirmatory(
        seed_rows,
        selected_method_id=selected.method_id,
        baseline_method_id=baseline.method_id,
        required_seeds=seeds,
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
    selection_provenance_path = run_dir / "selection_provenance.json"
    write_json(selection_provenance_path, dependency)
    summary_path = run_dir / str(config["results_summary"]["filename"])
    summary_path.write_text(
        confirmatory_summary_ko(
            assessment=assessment,
            dry_run=args.dry_run,
        ),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > int(
        config["results_summary"]["maximum_lines"]
    ):
        raise RuntimeError("E22b results summary exceeds one-page contract")
    common = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes=checkpoint_hashes,
        evidence_tier="CONTROLLED_REFERENCE",
        claim_eligible=bool(
            assessment["status"] == "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED"
            and assessment["supported"]
        ),
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PASS",
        "status": "PASS",
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "run_scope": "E22B_ACTIVE_PATH_LOCALITY_CONFIRMATORY",
        **common,
        "parent_e21": {
            "lock_path": str(parent.path),
            "lock_sha256": parent.sha256,
            "inherited_thresholds": parent.thresholds,
        },
        "phase_dependency": dependency,
        "runtime_metadata": runtime_metadata,
        "summary": assessment,
        "claim_gate": {
            "status": assessment["status"],
            "supported": bool(assessment["supported"]),
            "allowed_claim": (
                "Selected locality objective versus paired mean retention in "
                "controlled structured-event sequences, if every gate passes."
            ),
            "forbidden_claim": (
                "E21 retrospective repair, H5, natural-language, novel-ID, "
                "LM, agent/planning, official backend, or runtime transfer."
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
            "selection_provenance": {
                "path": str(selection_provenance_path.resolve()),
                "sha256": file_sha256(selection_provenance_path),
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
    print(f"[{EXPERIMENT_ID}] PASS/{assessment['status']}: {run_dir}")


if __name__ == "__main__":
    main()
