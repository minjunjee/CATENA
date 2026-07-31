from __future__ import annotations

from copy import deepcopy

import pytest

from catena.lm.e26a_population_lock import (
    E26AValidationPopulationError,
    e26a_validation_population_payload,
    validate_e26a_validation_population_lock,
)


def _config() -> dict[str, object]:
    return {
        "data": {
            "transaction_generator_version": "v8.1",
            "operations": [
                "PRESERVE",
                "ADD",
                "INVALIDATE",
                "SUPERSEDE",
                "ADD_EXCEPTION",
            ],
        },
        "gate_population": {
            "generation_seed": 260001,
            "namespace": "e26a_gate_population_v1",
            "splits": ["train", "validation", "main_test", "heldout_domain"],
            "domains": [
                "access_control",
                "api_configuration",
                "workflow",
                "versioned_preference",
            ],
            "items_per_operation_per_split": 4,
            "distractor_units": 1,
            "population_hash_required": True,
        },
    }


def test_validation_lock_materializes_only_validation_bytes() -> None:
    config = _config()
    first = e26a_validation_population_payload(config)
    second = e26a_validation_population_payload(config)
    assert first == second
    assert first["episode_count"] == 20
    assert first["main_test_opened"] is False
    assert first["main_test_access_count"] == 0
    assert first["heldout_domain_opened"] is False
    assert {row["split"] for row in first["records"]} == {"validation"}
    assert {row["operation"] for row in first["records"]} == {
        "PRESERVE",
        "ADD",
        "INVALIDATE",
        "SUPERSEDE",
        "ADD_EXCEPTION",
    }
    episodes = validate_e26a_validation_population_lock(first, config=config)
    assert len(episodes) == 20


def test_validation_lock_tamper_fails_closed() -> None:
    config = _config()
    payload = deepcopy(e26a_validation_population_payload(config))
    payload["records"][0]["split"] = "main_test"
    with pytest.raises(
        E26AValidationPopulationError,
        match="differs from deterministic config replay",
    ):
        validate_e26a_validation_population_lock(payload, config=config)
