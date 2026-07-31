import json
from pathlib import Path

import pytest

from catena.lm.transaction_data import (
    TransactionDataError,
    TransactionReplaySpec,
    write_transaction_replay_manifest,
)


def test_transaction_replay_manifest_never_opens_main_test(tmp_path: Path) -> None:
    spec = TransactionReplaySpec(
        seed=260_026,
        splits=("train", "validation", "calibration"),
        domains=("access_control",),
        operations=("PRESERVE", "ADD", "INVALIDATE", "SUPERSEDE"),
        items_per_cell=2,
        distractor_units=1,
    )
    path = write_transaction_replay_manifest(tmp_path / "transactions.json", spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["replay_identical"] is True
    assert payload["main_test_opened"] is False
    assert payload["visible_operation_gate_address_future_query_leakage"] == 0
    assert payload["split_audit"]["disjoint"] is True


def test_transaction_stage2_rejects_main_test_access() -> None:
    with pytest.raises(TransactionDataError, match="must not open main_test"):
        TransactionReplaySpec(
            seed=1,
            splits=("main_test",),
            domains=("access_control",),
            operations=("ADD",),
            items_per_cell=1,
            distractor_units=0,
        )
