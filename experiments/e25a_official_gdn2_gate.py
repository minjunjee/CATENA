from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_json
from catena.eval.official_operator_gate import run_official_operator_gate
from catena.post_e21.contracts import (
    copy_protocol_snapshot,
    report_contract_metadata,
    validate_protocol_lock,
    write_data_manifest,
    write_required_rows,
)
from catena.post_e21.official_gdn2 import (
    E25aNotConfigured,
    official_source_manifest,
    validate_gate_dependency,
)
from experiments.common import finalize_run, initialize_run

EXPERIMENT_ID = "e25a_official_gdn2_gate"
DEFAULT_CONFIG = "configs/e25a_official_gdn2_gate.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E25A_OFFICIAL_GDN2_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=("gate", "replication"), default="gate")
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--allow-scientific-replication", action="store_true")
    return parser


def _replication(
    *,
    config: dict[str, Any],
    gate_report: Path | None,
    authorized: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not authorized:
        raise PermissionError(
            "Official scientific replication requires "
            "--allow-scientific-replication after explicit user approval"
        )
    if gate_report is None:
        raise ValueError("replication requires an explicit --gate-report path")
    dependency = validate_gate_dependency(
        gate_report,
        expected_experiment_id=str(config["replication"]["required_gate_experiment_id"]),
    )
    module_name = str(config["replication"]["plugin_module"])
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise E25aNotConfigured(
            f"official replication plugin is unavailable: {module_name}"
        ) from error
    runner = getattr(plugin, "run_minimal_replications", None)
    if not callable(runner):
        raise E25aNotConfigured(f"{module_name} must expose run_minimal_replications(config)")
    result = runner(dict(config))
    if not isinstance(result, dict):
        raise TypeError("official replication plugin must return a dictionary")
    passed = _validate_replication_result(result, config=config)
    blocked = result.get("blocked_dependency") is True
    return {
        **result,
        "status": ("PASS" if passed else ("BLOCKED_DEPENDENCY" if blocked else "FAIL")),
        "scientific_evidence": bool(passed and result["scientific_evidence"]),
    }, dependency


def _validate_replication_result(
    result: dict[str, Any],
    *,
    config: dict[str, Any],
) -> bool:
    rows = result.get("rows")
    seed_rows = result.get("seed_rows")
    checks = result.get("checks")
    decisions = result.get("subset_decisions")
    passed = result.get("passed")
    declared_scientific = result.get("scientific_evidence")
    blocked = result.get("blocked_dependency") is True
    if not isinstance(rows, list) or (not rows and not blocked):
        raise TypeError(
            "official replication result.rows must be nonempty unless dependency-blocked"
        )
    if not isinstance(seed_rows, list) or (not seed_rows and not blocked):
        raise TypeError(
            "official replication result.seed_rows must be nonempty unless dependency-blocked"
        )
    if not isinstance(checks, dict) or not checks:
        raise TypeError("official replication result.checks must be a nonempty mapping")
    if not isinstance(decisions, dict):
        raise TypeError("official replication result.subset_decisions must be a mapping")
    if not isinstance(passed, bool) or not isinstance(declared_scientific, bool):
        raise TypeError("official replication pass/evidence declarations must be boolean")
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"official replication row {index} must be a mapping")
        if not isinstance(row.get("subset"), str):
            raise TypeError(f"official replication row {index} lacks subset")
        if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
            raise TypeError(f"official replication row {index} lacks integer seed")
        normalized_rows.append(row)
    required = ("e02b_magnitude_factorization", "e18_magnitude_sequence")
    for subset in required:
        subset_rows = [row for row in normalized_rows if row["subset"] == subset]
        decision = decisions.get(subset)
        if not isinstance(decision, dict):
            raise TypeError(f"official replication lacks decision for {subset}")
        if blocked:
            if subset_rows or decision.get("include") is not False:
                raise ValueError(f"blocked replication must not emit {subset} outcomes")
        else:
            if not subset_rows:
                raise ValueError(f"official replication omitted required subset {subset}")
            check = checks.get(subset)
            if not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
                raise TypeError(f"official replication lacks authoritative check for {subset}")
            if decision.get("include") is not True:
                raise ValueError(f"official replication did not include required subset {subset}")
    e22_decision = decisions.get("e22_locality_if_supported")
    if not isinstance(e22_decision, dict) or not isinstance(e22_decision.get("include"), bool):
        raise TypeError("official replication lacks an explicit E22 include/skip decision")
    e22_rows = [row for row in normalized_rows if row["subset"] == "e22_locality_if_supported"]
    if e22_decision["include"] is True and e22_decision.get("implemented") is False:
        if not blocked or e22_rows:
            raise ValueError("unimplemented official E22 route must block without rows")
    elif bool(e22_decision["include"]) != bool(e22_rows):
        raise ValueError("official E22 locality rows disagree with the include/skip decision")
    authoritative_pass = all(
        isinstance(item, dict) and item.get("passed") is True for item in checks.values()
    )
    if passed is not authoritative_pass:
        raise ValueError("plugin result.passed disagrees with authoritative subset checks")
    if blocked and (passed or authoritative_pass):
        raise ValueError("dependency-blocked replication cannot pass")
    if declared_scientific is not passed:
        raise ValueError("plugin scientific_evidence must exactly match authoritative PASS")
    registered = set(str(value) for value in config["replication"]["subsets"])
    observed = {str(row["subset"]) for row in normalized_rows}
    if not observed.issubset(registered):
        raise ValueError("official replication returned an unregistered subset")
    return authoritative_pass


def _artifact_rows(
    *,
    stage: str,
    status: str,
    source: dict[str, Any],
    result: dict[str, Any],
    dependency: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if stage == "replication" and status in {"PASS", "FAIL"}:
        dependency_sha = dependency["sha256"] if dependency is not None else None
        common = {
            "stage": "REPLICATION",
            "status": status,
            "official_source_status": source["status"],
            "gate_dependency_sha256": dependency_sha,
            "replication_plugin_sha256": source["replication_plugin"]["sha256"],
        }
        raw_rows = [{**common, **dict(row)} for row in result["rows"]]
        seed_rows = [{**common, **dict(row)} for row in result["seed_rows"]]
        return raw_rows, seed_rows
    return [
        {
            "stage": stage.upper(),
            "status": status,
            "official_source_status": source["status"],
            "configured": bool(result.get("configured", source.get("configured", False))),
            "scientific_evidence": bool(result.get("scientific_evidence", False)),
            "checks": result.get("checks", {}),
            "metrics": result.get("metrics", {}),
        }
    ], [
        {
            "stage": stage.upper(),
            "status": status,
            "passed_checks": sum(
                int(bool(item.get("passed")))
                for item in dict(result.get("checks", {})).values()
                if isinstance(item, dict)
            ),
            "required_checks": (
                len(dict(config["backend"]["checks"]))
                if stage == "gate"
                else len(dict(result.get("checks", {})))
            ),
        }
    ]


def _summary(*, stage: str, status: str, source_status: str) -> str:
    return "\n".join(
        [
            "# E25a Official GDN2/KDA Operator Gate 결과 요약",
            "",
            f"- Stage: `{stage.upper()}`",
            f"- Execution status: `{status}`",
            f"- Official source status: `{source_status}`",
            "- Reference/mock fallback: `false`",
            "",
            "Gate PASS 전에는 official magnitude replication과 architecture claim이 열리지 않는다.",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E25a config experiment_id mismatch")
    if args.dry_run and args.stage != "gate":
        raise ValueError("E25a dry-run supports the gate contract only")
    snapshot = validate_protocol_lock(
        lock_path=LOCK_PATH,
        config_path=args.config,
        experiment_id=EXPERIMENT_ID,
        repo_root=REPO_ROOT,
    )
    config_for_run, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="DRY_RUN" if args.dry_run else "MAIN",
    )
    copy_protocol_snapshot(snapshot=snapshot, run_dir=run_dir)

    source: dict[str, Any]
    dependency: dict[str, Any] | None = None
    try:
        source = official_source_manifest(
            config_for_run,
            dry_run=args.dry_run,
            stage=args.stage,
        )
        if args.stage == "gate":
            result = run_official_operator_gate(
                backend=dict(config_for_run["backend"]),
                dry_run=bool(args.dry_run),
            )
        else:
            result, dependency = _replication(
                config=config_for_run,
                gate_report=args.gate_report,
                authorized=bool(args.allow_scientific_replication),
            )
    except E25aNotConfigured as error:
        source = {
            "status": "NOT_CONFIGURED",
            "configured": False,
            "reference_fallback": False,
            "error": f"{type(error).__name__}: {error}",
        }
        result = {
            "status": "NOT_CONFIGURED",
            "configured": False,
            "scientific_evidence": False,
            "checks": {},
            "error": source["error"],
        }
    if args.dry_run:
        status = "DRY_RUN"
    else:
        status = str(result.get("status", "FAIL"))
        if status not in {
            "PASS",
            "FAIL",
            "NOT_CONFIGURED",
            "BLOCKED_DEPENDENCY",
        }:
            status = "FAIL"

    rows, seed_rows = _artifact_rows(
        stage=args.stage,
        status=status,
        source=source,
        result=result,
        dependency=dependency,
        config=config_for_run,
    )
    _, data_sha256 = write_data_manifest(
        run_dir=run_dir,
        payload={
            "stage": args.stage.upper(),
            "official_sources": source,
            "gate_dependency": dependency,
            "registered_subsets": list(config_for_run["replication"]["subsets"]),
        },
    )
    artifacts = write_required_rows(
        run_dir=run_dir,
        raw_rows=rows,
        seed_rows=seed_rows,
        raw_filename="official_operator_metrics.jsonl",
        seed_filename="official_operator_gate_summary.jsonl",
    )
    write_json(run_dir / "official_source_manifest.json", source)
    write_json(run_dir / "official_gate_or_replication.json", result)
    gate_passed = bool(
        args.stage == "gate" and status == "PASS" and result.get("scientific_evidence") is True
    )
    replication_passed = bool(
        args.stage == "replication"
        and status == "PASS"
        and result.get("scientific_evidence") is True
    )
    evidence_tier = (
        "OFFICIAL_OPERATOR"
        if gate_passed or replication_passed
        else "NOT_CONFIGURED_OR_FAILED_OFFICIAL"
    )
    claim_eligible = bool(gate_passed or replication_passed)
    metadata = report_contract_metadata(
        run_dir=run_dir,
        snapshot=snapshot,
        data_sha256=data_sha256,
        checkpoint_hashes={},
        evidence_tier=evidence_tier,
        claim_eligible=claim_eligible,
    )
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        _summary(
            stage=args.stage,
            status=status,
            source_status=str(source["status"]),
        ),
        encoding="utf-8",
    )
    summary_line_count = len(summary_path.read_text(encoding="utf-8").splitlines())
    if summary_line_count > 45:
        raise RuntimeError("E25a results summary exceeds one-page contract")
    report = {
        "experiment_id": EXPERIMENT_ID,
        "execution_status": status,
        "run_mode": "DRY_RUN" if args.dry_run else "MAIN",
        "stage": args.stage.upper(),
        **metadata,
        "scientific_evidence": bool(claim_eligible and result.get("scientific_evidence") is True),
        "official_operator_gate_passed": gate_passed,
        "official_minimal_replication_passed": replication_passed,
        "official_sources": source,
        "gate_dependency": dependency,
        "backend": result,
        "artifacts": artifacts,
        "results_summary": {
            "path": str(summary_path.resolve()),
            "sha256": file_sha256(summary_path),
            "line_count": summary_line_count,
        },
        "claim_gate": {
            "official_operator_parity_claim_eligible": gate_passed,
            "official_magnitude_replication_claim_eligible": replication_passed,
            "language_model_claim_eligible": False,
            "agent_claim_eligible": False,
            "transformer_or_kveraser_claim_eligible": False,
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {args.stage.upper()}/{status}: {run_dir}")


if __name__ == "__main__":
    main()
