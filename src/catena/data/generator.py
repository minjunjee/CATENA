from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from catena.config import load_yaml

from .schema import ClosureItem, Episode, HistorySegment, Query, Transaction
from .templates import DOMAIN_FIELDS, FILLER_SENTENCES, SCHEMA_FAMILIES


def _stable_id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _choice_other(rng: random.Random, values: list[str], old: str) -> str:
    choices = [value for value in values if value != old]
    return rng.choice(choices)


def _target_key(domain: str, rng: random.Random) -> str:
    fields = list(DOMAIN_FIELDS[domain])
    # Prefer fields whose changes naturally affect an action.
    preferred = {
        "api": ["auth_method", "endpoint", "retry_policy"],
        "access": ["role", "approval", "session_ttl"],
        "workflow": ["owner", "channel", "approval"],
    }[domain]
    return rng.choice(preferred if rng.random() < 0.8 else fields)


def _make_initial_state(domain: str, schema_family: str, rng: random.Random) -> dict[str, Any]:
    state = {key: rng.choice(values) for key, values in DOMAIN_FIELDS[domain].items()}
    state["schema_family"] = schema_family
    state["enabled"] = True
    state["version"] = rng.randint(2, 7)
    state["exception_scope"] = None
    return state


def _apply_transaction(
    state: dict[str, Any], operation: str, target: str, rng: random.Random
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    new_state = dict(state)
    old_value = state.get(target)
    scope: dict[str, Any] = {}

    if operation in {"SUPERSEDE", "AMEND"}:
        new_value = _choice_other(rng, DOMAIN_FIELDS[_domain_from_state(state)][target], str(old_value))
        new_state[target] = new_value
    elif operation == "INVALIDATE":
        target = "enabled"
        old_value = bool(state.get("enabled", True))
        new_value = False
        new_state["enabled"] = False
    elif operation == "ADD_EXCEPTION":
        target = "exception_scope"
        old_value = state.get("exception_scope")
        new_value = rng.choice(["EU-only", "emergency-only", "auditor-only", "weekend-only"])
        new_state["exception_scope"] = new_value
        scope = {"scope": new_value}
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    new_state["version"] = int(state["version"]) + 1
    return new_state, old_value, new_value, {"target": target, **scope}


def _domain_from_state(state: dict[str, Any]) -> str:
    keys = set(state)
    if "auth_method" in keys:
        return "api"
    if "resource" in keys:
        return "access"
    return "workflow"


def _action_for_state(domain: str, state: dict[str, Any]) -> str:
    if not state.get("enabled", True):
        return json.dumps({"action": "ABSTAIN", "reason": "rule_invalidated"}, sort_keys=True)
    if domain == "api":
        return json.dumps(
            {
                "tool": "call_api",
                "endpoint": state["endpoint"],
                "auth": state["auth_method"],
                "region": state["region"],
                "retry": state["retry_policy"],
            },
            sort_keys=True,
        )
    if domain == "access":
        return json.dumps(
            {
                "tool": "request_access",
                "role": state["role"],
                "resource": state["resource"],
                "approval": state["approval"],
                "ttl": state["session_ttl"],
            },
            sort_keys=True,
        )
    return json.dumps(
        {
            "tool": "run_workflow",
            "owner": state["owner"],
            "channel": state["channel"],
            "cadence": state["cadence"],
            "approval": state["approval"],
        },
        sort_keys=True,
    )


def _field_sentence(domain: str, schema_family: str, key: str, value: Any, version: int) -> str:
    return (
        f"In {schema_family} version {version}, the canonical field {key} is {value}. "
        f"This statement belongs to the {domain} configuration."
    )


def _make_history(
    domain: str,
    schema_family: str,
    initial: dict[str, Any],
    target: str,
    history_token_target: int,
    rng: random.Random,
) -> list[HistorySegment]:
    segments: list[HistorySegment] = []
    version = int(initial["version"])
    for key, value in initial.items():
        if key in {"schema_family", "version"}:
            continue
        segments.append(
            HistorySegment(
                segment_id=_stable_id(schema_family, version, key),
                kind="canonical_old",
                text=_field_sentence(domain, schema_family, key, value, version),
                entities=[schema_family, key],
                affected=(key == target or (target == "exception_scope" and key == "exception_scope")),
            )
        )

    old_action = _action_for_state(domain, initial)
    segments.append(
        HistorySegment(
            segment_id=_stable_id(schema_family, "old-plan", version),
            kind="derived_old_plan",
            text=f"The agent derived and recorded this plan under version {version}: {old_action}",
            entities=[schema_family, "plan"],
            affected=True,
        )
    )

    # Approximate one token as 0.75 words. Exact token lengths are materialized later by the adapter.
    target_words = max(64, int(history_token_target * 0.75))
    current_words = sum(len(segment.text.split()) for segment in segments)
    filler_idx = 0
    while current_words < target_words:
        text = f"Log {filler_idx}: {rng.choice(FILLER_SENTENCES)}"
        segments.insert(
            max(0, len(segments) - 1),
            HistorySegment(
                segment_id=_stable_id(schema_family, "filler", filler_idx, rng.random()),
                kind="distractor",
                text=text,
                entities=["unrelated"],
                affected=False,
            ),
        )
        current_words += len(text.split())
        filler_idx += 1
    return segments


def _make_closure(
    schema_family: str,
    target: str,
    old_version: int,
    dependency_depth: int,
) -> list[ClosureItem]:
    items = [
        ClosureItem(
            node_id=f"{schema_family}:{target}:v{old_version}",
            relation="SUPERSEDED_OR_CHANGED",
            text=f"The previous value of {target} at version {old_version} is no longer current.",
        )
    ]
    if dependency_depth >= 1:
        items.append(
            ClosureItem(
                node_id=f"{schema_family}:plan:v{old_version}",
                relation="INVALIDATES",
                text=f"The stored plan derived from {target} at version {old_version} must be reconsidered.",
            )
        )
    if dependency_depth >= 2:
        items.append(
            ClosureItem(
                node_id=f"{schema_family}:tool-args:v{old_version}",
                relation="INVALIDATES",
                text="Any tool arguments copied from the invalidated plan are stale.",
            )
        )
    if dependency_depth >= 3:
        items.append(
            ClosureItem(
                node_id=f"{schema_family}:retry:v{old_version}",
                relation="RECHECKS",
                text="A downstream retry or approval decision must be recomputed from the current state.",
            )
        )
    return items


def _candidate_values(current: dict[str, Any], initial: dict[str, Any], target: str) -> list[str]:
    values = [str(current.get(target)), str(initial.get(target))]
    domain = _domain_from_state(current)
    if target in DOMAIN_FIELDS[domain]:
        values.extend(DOMAIN_FIELDS[domain][target])
    values.extend(["UNKNOWN", "NOT_APPLICABLE"])
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped[:6]


def _make_queries(
    domain: str,
    schema_family: str,
    initial: dict[str, Any],
    current: dict[str, Any],
    target: str,
    rng: random.Random,
) -> list[Query]:
    candidates = _candidate_values(current, initial, target)
    gold = str(current.get(target))
    direct = Query(
        query_id="q_affected_direct",
        kind="affected_direct",
        prompt=f"What is the current value of {target} for {schema_family}? Return only the value.",
        candidates=candidates,
        gold_index=candidates.index(gold),
        affected_keys=[target],
    )

    current_action = _action_for_state(domain, current)
    old_action = _action_for_state(domain, initial)
    action_candidates = [current_action]
    if old_action not in action_candidates:
        action_candidates.append(old_action)
    attempts = 0
    while len(action_candidates) < 4 and attempts < 64:
        attempts += 1
        perturbed = dict(current)
        # When the current rule is invalidated, changing ordinary fields still
        # produces the same ABSTAIN action.  Reactivate only the distractor copy so
        # the candidate set remains finite and behaviorally distinct.
        if not perturbed.get("enabled", True):
            perturbed["enabled"] = True
        key = rng.choice([k for k in DOMAIN_FIELDS[domain] if k in perturbed])
        perturbed[key] = _choice_other(rng, DOMAIN_FIELDS[domain][key], str(perturbed[key]))
        action = _action_for_state(domain, perturbed)
        if action not in action_candidates:
            action_candidates.append(action)
    # Defensive fallback for degenerate schemas.  These candidates are valid JSON
    # but cannot equal the canonical action.
    while len(action_candidates) < 4:
        action_candidates.append(
            json.dumps({"action": "REVIEW", "variant": len(action_candidates)}, sort_keys=True)
        )
    derived = Query(
        query_id="q_affected_derived",
        kind="affected_derived",
        prompt=(
            f"Produce the correct JSON action for {schema_family} using the current canonical state. "
            "Return one candidate exactly."
        ),
        candidates=action_candidates,
        gold_index=action_candidates.index(current_action),
        affected_keys=[target, "plan"],
        tool_schema={"type": "object"},
    )

    unaffected_keys = [
        key
        for key in DOMAIN_FIELDS[domain]
        if key != target and str(initial.get(key)) == str(current.get(key))
    ]
    # Long transaction chains may eventually touch every domain field.  In that
    # case retain a field that is unaffected by the *final* transaction; the query
    # still tests whether the current value survives the last state update.
    if not unaffected_keys:
        unaffected_keys = [key for key in DOMAIN_FIELDS[domain] if key != target]
    keep_key = rng.choice(unaffected_keys)
    keep_candidates = _candidate_values(current, initial, keep_key)
    keep_gold = str(current[keep_key])
    unaffected = Query(
        query_id="q_unaffected",
        kind="unaffected",
        prompt=f"What is the retained current value of {keep_key} for {schema_family}? Return only the value.",
        candidates=keep_candidates,
        gold_index=keep_candidates.index(keep_gold),
        affected_keys=[],
    )

    probe_candidates = ["CURRENT", "STALE", "UNKNOWN"]
    old_probe = Query(
        query_id="q_old_rule_probe",
        kind="old_rule_probe",
        prompt=(
            f"Is the old {target} value {initial.get(target)} still the current rule for {schema_family}? "
            "Answer CURRENT, STALE, or UNKNOWN."
        ),
        candidates=probe_candidates,
        gold_index=probe_candidates.index("STALE"),
        affected_keys=[target],
    )
    return [direct, derived, unaffected, old_probe]


def _make_refresh_segments(
    history: list[HistorySegment],
    domain: str,
    schema_family: str,
    current: dict[str, Any],
    tx: Transaction,
    closure: list[ClosureItem],
    query_gap_tokens: int,
) -> list[HistorySegment]:
    result = list(history)
    result.append(
        HistorySegment(
            segment_id=_stable_id(schema_family, "transaction", tx.new_version),
            kind="verified_transaction",
            text=(
                f"Verified transaction {tx.operation}: {tx.target} changed from {tx.old_value} "
                f"to {tx.new_value}; canonical version is now {tx.new_version}."
            ),
            entities=[schema_family, tx.target],
            affected=True,
        )
    )
    for item in closure:
        result.append(
            HistorySegment(
                segment_id=item.node_id,
                kind="invalidation",
                text=item.text,
                entities=[schema_family],
                affected=True,
            )
        )
    result.append(
        HistorySegment(
            segment_id=_stable_id(schema_family, "current-snapshot", tx.new_version),
            kind="canonical_current",
            text=(
                f"Current canonical state for {schema_family} in the {domain} domain: "
                + json.dumps(current, sort_keys=True)
            ),
            entities=[schema_family],
            affected=True,
        )
    )
    gap_words = max(0, int(query_gap_tokens * 0.75))
    idx = 0
    while gap_words > 0:
        text = f"Post-update unrelated note {idx}: {FILLER_SENTENCES[idx % len(FILLER_SENTENCES)]}"
        result.append(
            HistorySegment(
                segment_id=_stable_id(schema_family, "post-gap", idx),
                kind="post_update_distractor",
                text=text,
                entities=["unrelated"],
                affected=False,
            )
        )
        gap_words -= len(text.split())
        idx += 1
    return result


def generate_episode(
    *,
    split: str,
    index: int,
    seed: int,
    history_token_target: int,
    domain: str,
    operation: str,
    dependency_depth: int,
    query_gap_tokens: int,
    schema_family: str | None = None,
) -> Episode:
    rng = random.Random(_stable_id(seed, split, index, domain, operation))
    schema_family = schema_family or rng.choice(SCHEMA_FAMILIES[domain])
    initial = _make_initial_state(domain, schema_family, rng)
    requested_target = _target_key(domain, rng)
    current, old_value, new_value, tx_meta = _apply_transaction(initial, operation, requested_target, rng)
    target = str(tx_meta["target"])
    old_version = int(initial["version"])
    new_version = int(current["version"])
    valid_from = (date(2026, 1, 1) + timedelta(days=rng.randint(0, 365))).isoformat()
    closure = _make_closure(schema_family, target, old_version, dependency_depth)
    tx = Transaction(
        operation=operation,  # type: ignore[arg-type]
        target=target,
        old_value=old_value,
        new_value=new_value,
        old_version=old_version,
        new_version=new_version,
        valid_from=valid_from,
        invalidates=[item.node_id for item in closure if item.relation == "INVALIDATES"],
        affects=[item.node_id for item in closure],
        scope={k: v for k, v in tx_meta.items() if k != "target"},
    )
    history = _make_history(domain, schema_family, initial, target, history_token_target, rng)
    queries = _make_queries(domain, schema_family, initial, current, target, rng)
    refresh = _make_refresh_segments(
        history, domain, schema_family, current, tx, closure, query_gap_tokens
    )
    episode_id = f"{split}-{_stable_id(seed, index, domain, schema_family, operation)}"
    affected_segment_index = min(
        (i for i, segment in enumerate(history) if segment.affected),
        default=0,
    )
    return Episode(
        episode_id=episode_id,
        split=split,
        domain=domain,
        schema_family=schema_family,
        seed=seed,
        history_token_target=history_token_target,
        dependency_depth=dependency_depth,
        query_gap_tokens=query_gap_tokens,
        initial_state=initial,
        current_state=current,
        history_segments=history,
        transaction=tx,
        closure=closure,
        queries=queries,
        refresh_segments=refresh,
        metadata={
            "affected_segment_index": affected_segment_index,
            "old_action": _action_for_state(domain, initial),
            "current_action": _action_for_state(domain, current),
        },
    )


def _split_schema_choices(domain: str, split: str, holdout: bool) -> list[str]:
    families = SCHEMA_FAMILIES[domain]
    if not holdout:
        return families
    if split == "train":
        return families[:2]
    if split == "val":
        return families[1:3]
    return families[2:]


def _iter_episodes(config: dict[str, Any], split: str, count: int) -> Iterable[Episode]:
    seed = int(config["seed"])
    domains = list(config["domains"])
    operations = list(config["operations"])
    lengths = list(
        config.get(f"{split}_history_token_targets", config["history_token_targets"])
    )
    depths = list(config["dependency_depths"])
    gaps = list(config["query_gap_tokens"])
    holdout = bool(config.get("split_holdout_schema_families", False))
    for index in range(count):
        rng = random.Random(_stable_id(seed, split, index))
        domain = rng.choice(domains)
        schema_family = rng.choice(_split_schema_choices(domain, split, holdout))
        yield generate_episode(
            split=split,
            index=index,
            seed=seed,
            history_token_target=rng.choice(lengths),
            domain=domain,
            operation=rng.choice(operations),
            dependency_depth=rng.choice(depths),
            query_gap_tokens=rng.choice(gaps),
            schema_family=schema_family,
        )


def write_jsonl(path: Path, episodes: Iterable[Episode]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def generate_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": int(config.get("num_train", 0)),
        "val": int(config.get("num_val", 0)),
        "test": int(config.get("num_test", 0)),
    }
    manifest: dict[str, Any] = {"config": config, "files": {}}
    for split, count in counts.items():
        path = output_dir / f"{split}.jsonl"
        written = write_jsonl(path, _iter_episodes(config, split, count))
        manifest["files"][split] = {"path": str(path), "episodes": written}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
