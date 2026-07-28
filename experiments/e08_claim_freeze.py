from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catena.core.io import read_latest_pointer, write_json
from catena.eval.claims import evaluate_claims
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e08_claim_freeze"
DEFAULT_CONFIG = "configs/e08_claim_freeze.yaml"


def _load(root: str, experiment_id: str):
    try:
        run = read_latest_pointer(root, experiment_id)
    except FileNotFoundError:
        return None, None
    with (run / "report.json").open("r", encoding="utf-8") as handle:
        return json.load(handle), run


def _path(report: dict[str, Any] | None, *keys: str) -> bool:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return bool(value)


def _parse_bool(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "pass", "preserved", "leak"}:
        return True
    if normalized in {"0", "false", "no", "n", "fail", "not_preserved", "no_leak"}:
        return False
    return None


def _cohen_kappa(first: list[bool], second: list[bool]) -> float | None:
    if len(first) != len(second) or not first:
        return None
    agreement = sum(a == b for a, b in zip(first, second, strict=True)) / len(first)
    p_first = sum(first) / len(first)
    p_second = sum(second) / len(second)
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    if abs(1.0 - expected) <= 1e-12:
        return None
    return (agreement - expected) / (1.0 - expected)


def _audit(run: Path | None, thresholds: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "passed": False,
        "reason": "missing audit",
        "item_count": 0,
    }
    if run is None or not (run / "naturalization_audit.csv").exists():
        return empty
    with (run / "naturalization_audit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return empty

    reviewer_a_meaning: list[bool] = []
    reviewer_b_meaning: list[bool] = []
    reviewer_a_leakage: list[bool] = []
    reviewer_b_leakage: list[bool] = []
    adjudicated_meaning: list[bool] = []
    adjudicated_leakage: list[bool] = []
    incomplete = 0
    for row in rows:
        parsed = [
            _parse_bool(row.get("reviewer_a_meaning_preserved", "")),
            _parse_bool(row.get("reviewer_a_answer_leakage", "")),
            _parse_bool(row.get("reviewer_b_meaning_preserved", "")),
            _parse_bool(row.get("reviewer_b_answer_leakage", "")),
            _parse_bool(row.get("adjudication_meaning_preserved", "")),
            _parse_bool(row.get("adjudication_answer_leakage", "")),
        ]
        if any(value is None for value in parsed):
            incomplete += 1
            continue
        a_meaning, a_leak, b_meaning, b_leak, final_meaning, final_leak = parsed
        reviewer_a_meaning.append(bool(a_meaning))
        reviewer_a_leakage.append(bool(a_leak))
        reviewer_b_meaning.append(bool(b_meaning))
        reviewer_b_leakage.append(bool(b_leak))
        adjudicated_meaning.append(bool(final_meaning))
        adjudicated_leakage.append(bool(final_leak))

    complete = len(adjudicated_meaning)
    if complete == 0:
        return {
            **empty,
            "item_count": len(rows),
            "incomplete_count": incomplete,
            "reason": "audit rows are not completed",
        }

    meaning_agreement = sum(
        first == second
        for first, second in zip(
            reviewer_a_meaning, reviewer_b_meaning, strict=True
        )
    ) / complete
    leakage_agreement = sum(
        first == second
        for first, second in zip(
            reviewer_a_leakage, reviewer_b_leakage, strict=True
        )
    ) / complete
    meaning_rate = sum(adjudicated_meaning) / complete
    leakage_rate = sum(adjudicated_leakage) / complete

    passed = (
        complete >= int(thresholds["minimum_completed_items"])
        and incomplete == 0
        and meaning_rate >= float(thresholds["minimum_meaning_preserved_rate"])
        and leakage_rate <= float(thresholds["maximum_answer_leakage_rate"])
        and meaning_agreement >= float(thresholds["minimum_reviewer_agreement"])
        and leakage_agreement >= float(thresholds["minimum_reviewer_agreement"])
    )
    return {
        "passed": passed,
        "item_count": len(rows),
        "completed_count": complete,
        "incomplete_count": incomplete,
        "adjudicated_meaning_preserved_rate": meaning_rate,
        "adjudicated_answer_leakage_rate": leakage_rate,
        "reviewer_meaning_agreement": meaning_agreement,
        "reviewer_leakage_agreement": leakage_agreement,
        "reviewer_meaning_kappa": _cohen_kappa(
            reviewer_a_meaning, reviewer_b_meaning
        ),
        "reviewer_leakage_kappa": _cohen_kappa(
            reviewer_a_leakage, reviewer_b_leakage
        ),
        "thresholds": thresholds,
    }


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    args = parser.parse_args()
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
    )
    ids = [
        "e01b_constrained_behavioral_reachability",
        "e02_magnitude_factorization",
        "e03_granularity_orientation",
        "e04_functional_mediation",
        "e05_semantic_demand_inference",
        "e06_reusable_state_assimilation",
        "e07_transformer_boundary",
    ]
    loaded = {experiment_id: _load(args.artifact_root, experiment_id) for experiment_id in ids}
    reports = {experiment_id: result[0] for experiment_id, result in loaded.items()}
    runs = {experiment_id: result[1] for experiment_id, result in loaded.items()}

    h5_report = reports["e05_semantic_demand_inference"]
    h5_direction = bool(
        h5_report and float(h5_report.get("primary", {}).get("mean", 0.0)) > 0
    )
    audit_report = _audit(
        runs["e05_semantic_demand_inference"], config["audit"]
    )
    evidence = {
        "h1": _path(
            reports["e01b_constrained_behavioral_reachability"],
            "claim_gate",
            "constrained_behavioral_reachability_predicts_error",
        ),
        "h2": _path(
            reports["e02_magnitude_factorization"], "claim_gate", "supported"
        ),
        "h3": _path(
            reports["e03_granularity_orientation"], "claim_gate", "supported"
        ),
        "h4": _path(
            reports["e04_functional_mediation"], "claim_gate", "supported"
        ),
        "h5_direction": h5_direction,
        "h5_audit": bool(audit_report["passed"]),
        "h6": _path(
            reports["e06_reusable_state_assimilation"],
            "claim_gate",
            "multi_update",
        )
        and _path(
            reports["e06_reusable_state_assimilation"],
            "claim_gate",
            "external_read_break_even",
        ),
        "rqt": _path(reports["e07_transformer_boundary"], "scientific_evidence"),
    }
    decisions = evaluate_claims(evidence)
    frozen = {
        "evidence": evidence,
        "claims": [asdict(decision) for decision in decisions],
        "h5_audit": audit_report,
        "source_reports": {
            experiment_id: {
                "present": reports[experiment_id] is not None,
                "run_dir": str(runs[experiment_id]) if runs[experiment_id] else None,
            }
            for experiment_id in ids
        },
        "scope_note": (
            "REALM critical path: H1-H4 plus a small H5 anchor. "
            "H6/RQ-T are post-workshop."
        ),
    }
    write_json(run_dir / "claim_freeze.json", frozen)
    report = {
        "status": "PASS",
        "allowed_claims": [
            decision.claim_id for decision in decisions if decision.allowed
        ],
        "blocked_claims": [
            decision.claim_id for decision in decisions if not decision.allowed
        ],
        "claim_tiers": {
            decision.claim_id: decision.tier for decision in decisions
        },
        "h5_audit": audit_report,
    }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
