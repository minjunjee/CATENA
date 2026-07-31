from __future__ import annotations

import pytest

from catena.lm.audit_contract import E26_AUDIT_LOCKED_HASH_KEYS
from catena.lm.checkpointing import (
    restart_audit_receipt,
    validate_restart_audit_coverage,
)


def _locked_hashes() -> dict[str, str]:
    return {
        key: f"{index:064x}"
        for index, key in enumerate(sorted(E26_AUDIT_LOCKED_HASH_KEYS), start=1)
    }


def _cases(candidate_ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        f"{candidate_id}__{variant}__{transition}": {
            "candidate_id": candidate_id,
            "variant": variant,
            "transition": transition,
            "passed": True,
        }
        for candidate_id in candidate_ids
        for variant in ("dual_delta_lm", "projected_tied_delta_lm")
        for transition in ("general_to_transaction", "transaction_to_general")
    }


def test_restart_receipt_requires_every_locked_candidate() -> None:
    candidate_ids = ("candidate_a", "candidate_b")
    locked = _locked_hashes()
    receipt = restart_audit_receipt(
        resume_cases=_cases(candidate_ids),
        cursor_replays={candidate_id: {"passed": True} for candidate_id in candidate_ids},
        expected_candidate_ids=candidate_ids,
        locked_hashes=locked,
        source_inventory={"source_tree_sha256": locked["source_tree_sha256"]},
    )
    assert validate_restart_audit_coverage(
        receipt,
        expected_candidate_ids=candidate_ids,
    ) == {"candidate_a": True, "candidate_b": True}

    incomplete = _cases(candidate_ids)
    incomplete.pop("candidate_b__projected_tied_delta_lm__transaction_to_general")
    with pytest.raises(ValueError, match="exact candidate/variant/transition grid"):
        restart_audit_receipt(
            resume_cases=incomplete,
            cursor_replays={candidate_id: {"passed": True} for candidate_id in candidate_ids},
            expected_candidate_ids=candidate_ids,
            locked_hashes=locked,
            source_inventory={"source_tree_sha256": locked["source_tree_sha256"]},
        )
