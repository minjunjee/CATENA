from __future__ import annotations

import json
from pathlib import Path

from tools.postcore_status import EXPERIMENTS, _claim_status


def test_postcore_inventory_covers_frozen_extension_and_e21() -> None:
    required = {
        "e17_postcore_evidence_freeze",
        "e15b_official_kveraser_gate",
        "e18b_sequence_control_lattice_aggregate",
        "e19b_localization_candidate_aggregate",
        "e20_quality_constrained_break_even",
        "e21a_structured_sequence_localization_transfer",
        "e21b_structured_sequence_localization_aggregate",
        "e21b_r1_structured_sequence_localization_aggregate",
    }

    assert required.issubset(EXPERIMENTS)
    assert len(EXPERIMENTS) == len(set(EXPERIMENTS))


def test_r1_status_uses_additive_amendment(tmp_path: Path) -> None:
    amendment = {
        "e13a_r1": {
            "calibration_status": (
                "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY"
            )
        }
    }
    (tmp_path / "E13A_R1_RESULT_STATUS_AMENDMENT_FREEZE_V1.json").write_text(
        json.dumps(amendment),
        encoding="utf-8",
    )

    status, supported = _claim_status(
        "e13a_r1_sequence_floor_throughput",
        {"claim_gate": {"go_for_e13b": True}},
        tmp_path,
    )

    assert status == "GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY"
    assert supported is False


def test_r2_and_aggregate_statuses_are_distinct(tmp_path: Path) -> None:
    r2_status, r2_supported = _claim_status(
        "e13a_r2_sequence_floor_throughput",
        {"claim_gate": {"go_for_e13b_r1": True}},
        tmp_path,
    )
    aggregate_status, aggregate_supported = _claim_status(
        "e13c_r1_transactional_sequence_aggregate",
        {"claim_gate": {"supported": True}},
        tmp_path,
    )

    assert (r2_status, r2_supported) == ("GO_FOR_E13B_R1", True)
    assert (aggregate_status, aggregate_supported) == ("SUPPORTED", True)


def test_original_e13a_is_never_shown_as_repaired_go(
    tmp_path: Path,
) -> None:
    status, supported = _claim_status(
        "e13a_sequence_floor_throughput",
        {"claim_gate": {"go_for_e13b": True}},
        tmp_path,
    )

    assert status == "CALIBRATION_PILOT_ONLY"
    assert supported is None


def test_original_e21b_is_never_shown_as_claim_eligible(
    tmp_path: Path,
) -> None:
    status, supported = _claim_status(
        "e21b_structured_sequence_localization_aggregate",
        {"claim_gate": {"supported": True, "status": "SUPPORTED"}},
        tmp_path,
    )

    assert status == "INCONCLUSIVE_GATE_IMPLEMENTATION"
    assert supported is None


def test_official_operator_failure_is_explicitly_closed(
    tmp_path: Path,
) -> None:
    status, supported = _claim_status(
        "e15a_r1_official_gdn2_kda_gate",
        {"claim_gate": {"official_operator_claim_eligible": False}},
        tmp_path,
    )

    assert status == "OFFICIAL_CLAIM_CLOSED"
    assert supported is False


def test_e21b_r1_preserves_not_supported_wording(
    tmp_path: Path,
) -> None:
    status, supported = _claim_status(
        "e21b_r1_structured_sequence_localization_aggregate",
        {"claim_gate": {"status": "NOT_SUPPORTED", "supported": False}},
        tmp_path,
    )

    assert status == "NOT_SUPPORTED"
    assert supported is False
