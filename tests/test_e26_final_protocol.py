from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from catena.lm.e26_final_protocol import (
    E26FinalProtocolError,
    load_protocol,
    validate_protocol,
)

CONFIG = Path("configs/e26_final_gdn2_1p3b_transactional_transfer.yaml")


def test_protocol_validates_registered_contract() -> None:
    payload = validate_protocol(load_protocol(CONFIG))
    assert payload["experiment_id"] == "E26_FINAL_GDN2_1P3B_TRANSACTIONAL_TRANSFER"
    assert payload["runtime"]["automatic_execution_after_gates"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("statistics", "asymmetric_absolute_sesoi", 0.019),
        ("speed", "minimum_train_tokens_per_second_per_gpu", 11_999),
        ("training", "seeds", [1, 2, 3, 4, 5]),
        ("source", "checkpoint", {}),
    ],
)
def test_protocol_rejects_registered_field_drift(
    section: str,
    field: str,
    replacement: object,
) -> None:
    payload = load_protocol(CONFIG)
    changed = deepcopy(payload)
    if section == "source" and field == "checkpoint":
        changed[section][field] = replacement
    else:
        changed[section][field] = replacement
    with pytest.raises(E26FinalProtocolError):
        validate_protocol(changed)


def test_protocol_requires_autonomous_continuation() -> None:
    payload = load_protocol(CONFIG)
    payload["runtime"]["automatic_execution_after_gates"] = False
    with pytest.raises(E26FinalProtocolError, match="autonomously"):
        validate_protocol(payload)
