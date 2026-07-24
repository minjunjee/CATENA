from __future__ import annotations

from catena.data.generator import generate_episode
from catena.methods.policies import apply_text_policy, build_base_state
from catena.models.mock import MockStatefulModel


def test_text_policy_state_paths_are_distinct():
    episode = generate_episode(
        split="test",
        index=0,
        seed=13,
        history_token_target=128,
        domain="api",
        operation="SUPERSEDE",
        dependency_depth=1,
        query_gap_tokens=0,
        schema_family="payment-client",
    )
    model = MockStatefulModel()
    base = build_base_state(model, episode)
    stale = apply_text_policy(model, episode, base, "stale")
    closure = apply_text_policy(model, episode, base, "typed_closure")
    reset = apply_text_policy(model, episode, base, "reset_snapshot")
    assert stale.text == base.text
    assert closure.text != base.text
    assert reset.text != base.text
    assert len(reset.text) < len(closure.text)
