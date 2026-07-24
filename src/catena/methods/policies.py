from __future__ import annotations

import random
from typing import Any

from catena.data.render import (
    render_compact_snapshot,
    render_history_prompt,
    render_plain_correction,
    render_refresh_prompt,
    render_typed_closure,
    render_typed_transaction,
)
from catena.data.schema import ClosureItem, Episode, Query


def build_base_state(model, episode: Episode):
    return model.prefill_text(render_history_prompt(episode), None)


def apply_text_policy(
    model,
    episode: Episode,
    base_state: Any,
    policy: str,
    *,
    query: Query | None = None,
    shuffle_seed: int = 0,
):
    """Apply a non-learned update policy and return the resulting model state."""

    if policy in {"stale", "stale_kv"}:
        return model.clone_state(base_state)
    if policy in {"plain_correction", "plain_append"}:
        return model.prefill_text(render_plain_correction(episode.transaction), model.clone_state(base_state))
    if policy in {"typed_transaction", "typed_append"}:
        return model.prefill_text(render_typed_transaction(episode.transaction), model.clone_state(base_state))
    if policy in {"typed_closure", "closure_append"}:
        text = render_typed_closure(episode.transaction, episode.closure)
        return model.prefill_text(text, model.clone_state(base_state))
    if policy == "shuffled_closure":
        shuffled = list(episode.closure)
        random.Random(shuffle_seed + episode.seed).shuffle(shuffled)
        corrupted: list[ClosureItem] = []
        for index, item in enumerate(shuffled):
            # Swap relation labels while retaining similar length and vocabulary.
            relation = shuffled[(index + 1) % len(shuffled)].relation if shuffled else item.relation
            corrupted.append(ClosureItem(item.node_id, relation, item.text))
        text = render_typed_closure(episode.transaction, corrupted)
        return model.prefill_text(text, model.clone_state(base_state))
    if policy == "tx_only_zero_state":
        return model.prefill_text(render_typed_closure(episode.transaction, episode.closure), None)
    if policy in {"reset_snapshot", "compact_capsule"}:
        return model.prefill_text(render_compact_snapshot(episode), None)
    if policy == "query_time_retrieval":
        if query is None:
            raise ValueError("query_time_retrieval requires the current query")
        relevant = {
            key: episode.current_state.get(key)
            for key in query.affected_keys
            if key in episode.current_state
        }
        if not relevant:
            relevant = episode.current_state
        text = "[LATEST RETRIEVED MEMORY]\n" + str(relevant)
        return model.prefill_text(text, model.clone_state(base_state))
    if policy in {"exact_refresh", "full_reprefill"}:
        return model.prefill_text(render_refresh_prompt(episode), None)
    raise ValueError(f"Unsupported policy: {policy}")
