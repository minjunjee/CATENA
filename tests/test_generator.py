from __future__ import annotations

from catena.data.generator import generate_episode
from catena.data.validate import validate_episode


def _episode(index: int = 0):
    return generate_episode(
        split="test",
        index=index,
        seed=13,
        history_token_target=512,
        domain="api",
        operation="SUPERSEDE",
        dependency_depth=2,
        query_gap_tokens=32,
        schema_family="payment-client",
    )


def test_episode_is_deterministic_and_valid():
    left = _episode()
    right = _episode()
    assert left.to_dict() == right.to_dict()
    assert not validate_episode(left)
    assert left.current_state["version"] == left.initial_state["version"] + 1
    assert {q.kind for q in left.queries} >= {
        "affected_direct",
        "affected_derived",
        "unaffected",
        "old_rule_probe",
    }


def test_tx_only_cannot_contain_full_action_in_closure():
    episode = _episode(1)
    closure = " ".join(item.text for item in episode.closure)
    assert episode.metadata["current_action"] not in closure
