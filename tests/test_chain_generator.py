from __future__ import annotations

from catena.data.chain_generator import generate_chain_episode


def test_long_chain_keeps_valid_queries():
    episode = generate_chain_episode(
        split="test",
        index=0,
        seed=13,
        history_token_target=256,
        chain_length=16,
        domain="workflow",
        operations=["SUPERSEDE", "AMEND", "INVALIDATE", "ADD_EXCEPTION"],
        dependency_depth=2,
    )
    assert len(episode.transactions) == 16
    assert episode.final_state["version"] == episode.initial_state["version"] + 16
    assert any(q.kind == "unaffected" for q in episode.queries)
    assert all(0 <= q.gold_index < len(q.candidates) for q in episode.queries)
