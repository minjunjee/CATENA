from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENTS = [
    "e10_learned_rank_scaling",
    "e10b_floor_aware_rank_scaling",
    "e11_representation_control_coadaptation",
    "e11b_scale_normalized_coadaptation",
    "e12_control_algebra_lattice",
    "e13a_sequence_floor_throughput",
    "e13a_r1_sequence_floor_throughput",
    "e13a_r2_sequence_floor_throughput",
    "e13b_transactional_sequence_memory",
    "e13b_r1_transactional_sequence_memory",
    "e13c_transactional_sequence_aggregate",
    "e13c_r1_transactional_sequence_aggregate",
    "e14_plan_continuation",
    "e15_official_backend_gate",
    "e15a_official_gdn2_kda_gate",
    "e15a_r1_official_gdn2_kda_gate",
    "e15b_official_kveraser_gate",
    "e16_core_evidence_freeze",
    "e17_postcore_evidence_freeze",
    "e18a_sequence_control_lattice",
    "e18b_sequence_control_lattice_aggregate",
    "e19a_localization_candidate_decomposition",
    "e19b_localization_candidate_aggregate",
    "e20_quality_constrained_break_even",
    "e21a_structured_sequence_localization_transfer",
    "e21b_structured_sequence_localization_aggregate",
    "e21b_r1_structured_sequence_localization_aggregate",
]

STATIC_CLAIM_STATUS = {
    "e13a_sequence_floor_throughput": "CALIBRATION_PILOT_ONLY",
    "e13b_transactional_sequence_memory": "NOT_RUN",
    "e13c_transactional_sequence_aggregate": "NOT_RUN",
    "e16_core_evidence_freeze": "CORE_EVIDENCE_FROZEN",
    "e17_postcore_evidence_freeze": "POSTCORE_EVIDENCE_FROZEN",
    # The original aggregate implementation is structurally invalid regardless
    # of the value it writes in report.json.  Only E21b-R1 is claim eligible.
    "e21b_structured_sequence_localization_aggregate": (
        "INCONCLUSIVE_GATE_IMPLEMENTATION"
    ),
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _claim_status(
    experiment_id: str,
    report: dict[str, Any],
    artifact_root: Path,
) -> tuple[str, bool | None]:
    if experiment_id in STATIC_CLAIM_STATUS:
        return STATIC_CLAIM_STATUS[experiment_id], None

    if experiment_id == "e13a_r1_sequence_floor_throughput":
        amendment_path = (
            artifact_root
            / "E13A_R1_RESULT_STATUS_AMENDMENT_FREEZE_V1.json"
        )
        if amendment_path.is_file():
            amendment = _load(amendment_path)
            status = str(
                amendment.get("e13a_r1", {}).get(
                    "calibration_status",
                    "LEGACY_PIPELINE_ONLY",
                )
            )
            return status, False
        return "HISTORICAL_GO_UNAMENDED", None

    gate = report.get("claim_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    if experiment_id == "e21b_r1_structured_sequence_localization_aggregate":
        gate_status = gate.get("status")
        supported = gate.get("supported")
        if isinstance(gate_status, str) and isinstance(supported, bool):
            return gate_status, supported
    supported = gate.get("supported")
    if supported is not None:
        value = bool(supported)
        return ("SUPPORTED" if value else "NOT_OPENED"), value
    if gate.get("go_for_e13b_r1") is not None:
        value = bool(gate["go_for_e13b_r1"])
        return ("GO_FOR_E13B_R1" if value else "NO_GO_FOR_E13B_R1"), value
    if gate.get("status") is not None:
        return str(gate["status"]), None
    if gate.get("official_backend_ready") is not None:
        value = bool(gate["official_backend_ready"])
        return ("OFFICIAL_READY" if value else "NOT_CONFIGURED"), value
    if gate.get("official_operator_claim_eligible") is not None:
        value = bool(gate["official_operator_claim_eligible"])
        return (
            "OFFICIAL_CLAIM_ELIGIBLE" if value else "OFFICIAL_CLAIM_CLOSED",
            value,
        )
    return "NO_CLAIM_GATE", None


def collect(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENTS:
        pointer = artifact_root / experiment_id / "latest.json"
        if not pointer.exists():
            rows.append({"experiment_id": experiment_id, "status": "NOT_RUN"})
            continue
        try:
            run_dir = Path(_load(pointer)["run_dir"])
            report_path = run_dir / "report.json"
            report = _load(report_path)
            claim_status, supported = _claim_status(
                experiment_id,
                report,
                artifact_root,
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "status": report.get("status", report.get("execution_status", "UNKNOWN")),
                    "supported_or_go": supported,
                    "claim_status": claim_status,
                    "run_dir": str(run_dir),
                    "report_path": str(report_path),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "status": "MALFORMED",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CATENA post-core experiment state")
    parser.add_argument("--artifact-root", default="/data/minjun_dev/CATENA/artifacts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.artifact_root)
    rows = collect(root)
    if args.json:
        print(
            json.dumps(
                {
                    "artifact_root": str(root.resolve()),
                    "experiments": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    print(f"Artifact root: {root}")
    experiment_width = max(45, max(len(item) for item in EXPERIMENTS) + 1)
    print(f"{'Experiment':{experiment_width}} {'Status':12} Claim")
    print("-" * (experiment_width + 32))
    for row in rows:
        claim_status = row.get("claim_status", "-")
        print(
            f"{row['experiment_id']:{experiment_width}} "
            f"{row['status']:12} "
            f"{claim_status}"
        )
        if row.get("error"):
            print(f"  error: {row['error']}")


if __name__ == "__main__":
    main()
