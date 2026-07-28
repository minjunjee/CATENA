from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catena.core.config import load_config
from catena.core.io import file_sha256, write_jsonl
from catena.eval.structured_sequence_localization_r1 import (
    assess_e21b_r1,
    compute_e21b_r1_seed_contrasts,
)
from experiments.common import finalize_run, initialize_run
from experiments.e21_structured_sequence_localization_transfer import (
    _read_json_object,
    _validate_source_run,
    validate_e21_protocol_lock,
)

EXPERIMENT_ID = "e21b_r1_structured_sequence_localization_aggregate"
DEFAULT_CONFIG = "configs/e21b_r1_structured_sequence_localization_aggregate.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "docs/E21B_R1_STRUCTURED_SEQUENCE_AGGREGATE_LOCK.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E21b-R1 aggregate repair")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv("CATENA_ARTIFACT_ROOT", "artifacts"),
    )
    parser.add_argument("--source-run", action="append", default=[])
    return parser


def _validate_repair_lock(config_path: str | Path) -> dict[str, str]:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise RuntimeError("E21b-R1 prospective lock is missing or unsafe")
    lock = _read_json_object(LOCK_PATH)
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("frozen_before_any_e21a_main_report") is not True
        or lock.get("e21a_main_report_count_at_freeze") != 0
        or lock.get("main_execution_started") is not False
    ):
        raise RuntimeError("E21b-R1 lock is not prospectively valid")
    parent_path = REPO_ROOT / "docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json"
    if file_sha256(parent_path) != lock.get("parent_e21_lock_sha256"):
        raise RuntimeError("Parent E21 protocol lock changed")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("E21b-R1 lock file map is missing")
    for relative, expected_hash in files.items():
        candidate = (REPO_ROOT / str(relative)).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError as error:
            raise RuntimeError("E21b-R1 locked path escapes repo") from error
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or file_sha256(candidate) != expected_hash
        ):
            raise RuntimeError(f"E21b-R1 locked file changed: {relative}")
    config = Path(config_path).resolve()
    if file_sha256(config) != files.get(
        "configs/e21b_r1_structured_sequence_localization_aggregate.yaml"
    ):
        raise RuntimeError("E21b-R1 config does not match lock")
    return {
        "path": str(LOCK_PATH.resolve()),
        "sha256": file_sha256(LOCK_PATH),
        "config_sha256": file_sha256(config),
    }


def _summary(
    *,
    assessment: dict[str, Any],
    seeds: list[int],
) -> str:
    status = "SUPPORTED" if assessment["supported"] else "NOT_SUPPORTED"
    pattern = assessment["pattern"]
    observed = assessment["observed"]
    lines = [
        "# E21b-R1 Active-Guardrail Repair 결과",
        "",
        f"- 판정: **{status}**",
        f"- paired seeds: `{len(seeds)}`",
        "- original E21b: `INCONCLUSIVE_GATE_IMPLEMENTATION`",
        "- evidence tier: `CONTROLLED_REFERENCE`",
        "",
        "## Primary contrast (원본 estimand 유지)",
        "",
        "| Contrast | Mean gain | +seed | p | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in pattern.items():
        lines.append(
            f"| `{name}` | {float(result['mean_gain']):.6g} | "
            f"{float(result['positive_seed_fraction']):.3f} | "
            f"{float(result['sign_flip_p']):.6g} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "## Repaired guardrail",
            "",
            f"- max active non-target cell degradation: "
            f"`{float(observed['maximum_nontarget_degradation']):.6g}`",
            f"- max primary-context retention cell degradation: "
            f"`{float(observed['maximum_retention_degradation']):.6g}`",
            "- State-read는 실제 route가 활성화되는 C/D에서만 비교했다.",
            "- 각 update×gap cell maximum을 사용해 평균 상쇄를 금지했다.",
            "",
            "## 경계",
            "",
            "- Fixed identifier schema와 explicit demand/provenance field의",
            "  controlled repeated-sequence evidence로만 해석한다.",
            "- H5/자연어/novel-ID/recurrent-LM/agent/official claim은 닫혀 있다.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if (
        config.get("experiment_id") != EXPERIMENT_ID
        or config.get("protocol", {}).get("aggregate_only") is not True
    ):
        raise ValueError("E21b-R1 config identity mismatch")
    repair_lock = _validate_repair_lock(args.config)
    source_config_path = REPO_ROOT / str(config["source_contract"]["config_path"])
    source_config = load_config(source_config_path)
    source_lock = validate_e21_protocol_lock(source_config_path)
    required_seeds = [int(value) for value in config["seeds"]]
    if len(args.source_run) != len(required_seeds):
        raise ValueError("E21b-R1 requires exactly five explicit source runs")

    initialized, run_dir, _device = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
        run_mode="MAIN",
    )
    if initialized != config:
        raise RuntimeError("E21b-R1 config changed at run start")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for source_value in args.source_run:
        source = Path(source_value).resolve()
        report = _read_json_object(source / "report.json")
        seed = int(report.get("seed", -1))
        if seed not in required_seeds or seed in observed_seeds:
            raise RuntimeError("E21b-R1 source seed duplicate/unregistered")
        source_rows, source_provenance = _validate_source_run(
            source,
            expected_seed=seed,
            expected_mode="MAIN",
            config=source_config,
            lock=source_lock,
            dry_run=False,
        )
        rows.extend(source_rows)
        provenance.append(source_provenance)
        observed_seeds.add(seed)
    if observed_seeds != set(required_seeds):
        raise RuntimeError("E21b-R1 exact five-seed source grid is incomplete")

    seed_rows = compute_e21b_r1_seed_contrasts(
        rows,
        seeds=required_seeds,
        updates_grid=[int(value) for value in config["evaluation"]["updates"]],
        gaps_grid=[int(value) for value in config["evaluation"]["gap_events"]],
        demand_families=[str(value) for value in config["demand_families"]],
        stress_updates=int(config["evaluation"]["stress"]["updates"]),
        stress_gap_events=int(config["evaluation"]["stress"]["gap_events"]),
    )
    assessment = assess_e21b_r1(
        seed_rows,
        thresholds=config["claim_gate"],
        alpha=float(config["statistics"]["alpha"]),
    )
    metrics_path = run_dir / "structured_sequence_paired_metrics.jsonl"
    contrasts_path = run_dir / "structured_sequence_seed_contrasts_r1.jsonl"
    provenance_path = run_dir / "source_run_provenance.jsonl"
    write_jsonl(metrics_path, rows)
    write_jsonl(contrasts_path, seed_rows)
    write_jsonl(provenance_path, provenance)
    summary_path = run_dir / "RESULTS_SUMMARY_KO.md"
    summary_path.write_text(
        _summary(assessment=assessment, seeds=required_seeds),
        encoding="utf-8",
    )
    if len(summary_path.read_text(encoding="utf-8").splitlines()) > 55:
        raise RuntimeError("E21b-R1 summary exceeds one-page contract")

    status = "SUPPORTED" if assessment["supported"] else "NOT_SUPPORTED"
    report = {
        "status": "PASS",
        "run_mode": "MAIN",
        "run_scope": "E21_AGGREGATE_ACTIVE_GUARDRAIL_REPAIR",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
        "original_e21b_disposition": "INCONCLUSIVE_GATE_IMPLEMENTATION",
        "source_contract": {
            "required_seeds": required_seeds,
            "explicit_source_runs_only": True,
            "source_runs": provenance,
            "source_protocol_lock_sha256": source_lock["sha256"],
            "repair_protocol_lock_sha256": repair_lock["sha256"],
            "repair_config_sha256": repair_lock["config_sha256"],
        },
        "summary": assessment,
        "artifacts": {
            "paired_metrics_sha256": file_sha256(metrics_path),
            "seed_contrasts_sha256": file_sha256(contrasts_path),
            "source_provenance_sha256": file_sha256(provenance_path),
            "results_summary_ko": {
                "path": str(summary_path.resolve()),
                "sha256": file_sha256(summary_path),
                "line_count": len(summary_path.read_text(encoding="utf-8").splitlines()),
            },
        },
        "claim_gate": {
            "status": status,
            "supported": bool(assessment["supported"]),
            "allowed_claim": (
                "Controlled repeated structured-event sequence transfer with "
                "fixed identifiers and explicit demand/provenance fields."
            ),
            "forbidden_claim": (
                "H5, semantic/natural-language, novel identifier, recurrent LM, "
                "agent/planning, official backend, or runtime transfer."
            ),
        },
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS/{status}: {run_dir}")


if __name__ == "__main__":
    main()
