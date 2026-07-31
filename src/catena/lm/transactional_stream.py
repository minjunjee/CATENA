from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

from .hashing import hash_mapping


class Operation(StrEnum):
    PRESERVE = "PRESERVE"
    ADD = "ADD"
    INVALIDATE = "INVALIDATE"
    SUPERSEDE = "SUPERSEDE"
    ADD_EXCEPTION = "ADD_EXCEPTION"


class QueryType(StrEnum):
    CURRENT_STATE = "current_state"
    DERIVED_ACTION = "derived_action"
    STALE_PROBE = "stale_probe"
    UNAFFECTED_RETENTION = "unaffected_retention"


@dataclass(frozen=True)
class QueryRecord:
    query_type: str
    prompt: str
    candidate_answers: tuple[str, ...]
    gold_index: int
    structured_gold: dict[str, Any] | None
    affected_entities: tuple[str, ...]
    retained_entities: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.candidate_answers) < 2:
            raise ValueError("Each query needs at least two candidate answers")
        if not 0 <= self.gold_index < len(self.candidate_answers):
            raise ValueError("gold_index is outside candidate_answers")

    @property
    def gold_answer(self) -> str:
        return self.candidate_answers[self.gold_index]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidate_answers"] = list(self.candidate_answers)
        value["affected_entities"] = list(self.affected_entities)
        value["retained_entities"] = list(self.retained_entities)
        return value


@dataclass(frozen=True)
class TransactionEpisode:
    episode_id: str
    split: str
    domain: str
    operation: str
    template_family: str
    entity_namespace: str
    seed: int
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    materialization_text: str
    transaction_text: str
    dependency_closure: tuple[str, ...]
    distractor_text: str
    branch_prefix_text: str
    queries: tuple[QueryRecord, ...]
    exact_refresh_text: str
    protected_fields: dict[str, Any]
    visible_input_audit: dict[str, bool]

    def __post_init__(self) -> None:
        expected = {item.value for item in QueryType}
        observed = {query.query_type for query in self.queries}
        if observed != expected or len(self.queries) != len(expected):
            raise ValueError(f"Expected one query of each type; got {sorted(observed)}")
        if self.operation not in {item.value for item in Operation}:
            raise ValueError(f"Unknown operation: {self.operation}")

    @property
    def signature(self) -> str:
        return hash_mapping(
            {
                "split": self.split,
                "domain": self.domain,
                "operation": self.operation,
                "template_family": self.template_family,
                "entity_namespace": self.entity_namespace,
                "protected_fields": self.protected_fields,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dependency_closure"] = list(self.dependency_closure)
        result["queries"] = [query.to_dict() for query in self.queries]
        return result

    def render_training_example(self, query_type: str) -> str:
        query = next(item for item in self.queries if item.query_type == query_type)
        return f"{self.branch_prefix_text}\n\n{query.prompt}\nANSWER: {query.gold_answer}"


_SPLIT_NAMESPACES: dict[str, dict[str, str]] = {
    "train": {"entity": "tr", "template": "tmpl-tr"},
    "validation": {"entity": "va", "template": "tmpl-va"},
    "calibration": {"entity": "ca", "template": "tmpl-ca"},
    "main_test": {"entity": "te", "template": "tmpl-te"},
    "heldout_domain": {"entity": "hd", "template": "tmpl-hd"},
}

_TOKENS = {
    "resources": ["ledger", "archive", "deploy", "billing", "audit"],
    "values": ["amber", "cobalt", "jade", "silver", "violet", "ivory"],
    "routes": ["/v1/report", "/v2/search", "/ops/export", "/team/sync"],
    "states": ["queued", "approved", "paused", "released", "archived"],
}


def _opaque(prefix: str, kind: str, index: int) -> str:
    # It is intentionally not a transparent numeric address such as slot-3.
    return f"{prefix}-{kind}-{index:04x}-qz"


def _choice(rng: random.Random, values: Sequence[str], exclude: str | None = None) -> str:
    candidates = [value for value in values if value != exclude]
    if not candidates:
        raise ValueError("No candidate values remain")
    return rng.choice(candidates)


def _domain_record(
    domain: str,
    *,
    target: str,
    retained: str,
    old_value: str,
    new_value: str,
    resource: str,
    route: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if domain == "access_control":
        before = {
            target: {"resource": resource, "decision": old_value, "version": 1},
            retained: {"resource": "audit", "decision": "allow", "version": 4},
        }
        metadata = {"tool": "authorize", "resource": resource}
    elif domain == "api_configuration":
        before = {
            target: {"route": route, "profile": old_value, "version": 2},
            retained: {"route": "/health", "profile": "silver", "version": 3},
        }
        metadata = {"tool": "configure_api", "route": route}
    elif domain == "workflow":
        before = {
            target: {"job": resource, "state": old_value, "version": 7},
            retained: {"job": "audit", "state": "approved", "version": 2},
        }
        metadata = {"tool": "transition_workflow", "job": resource}
    elif domain == "versioned_preference":
        before = {
            target: {"scope": resource, "preference": old_value, "version": 5},
            retained: {"scope": "notifications", "preference": "jade", "version": 9},
        }
        metadata = {"tool": "apply_preference", "scope": resource}
    else:
        raise ValueError(f"Unknown domain: {domain}")
    metadata["new_value"] = new_value
    return before, metadata


def _current_field(domain: str) -> str:
    return {
        "access_control": "decision",
        "api_configuration": "profile",
        "workflow": "state",
        "versioned_preference": "preference",
    }[domain]


def _materialize(domain: str, target: str, retained: str, state: dict[str, Any]) -> str:
    target_record = state[target]
    retained_record = state[retained]
    field = _current_field(domain)
    if domain == "access_control":
        action = (
            f"The planner used policy version {target_record['version']} and prepared an "
            f"authorization decision of {target_record[field]} for {target} on "
            f"{target_record['resource']}."
        )
    elif domain == "api_configuration":
        action = (
            f"The deployment plan selected profile {target_record[field]} for route "
            f"{target_record['route']} under record {target}."
        )
    elif domain == "workflow":
        action = (
            f"The execution plan treated job {target_record['job']} as "
            f"{target_record[field]} using record {target}."
        )
    else:
        action = (
            f"The assistant prepared a response using preference {target_record[field]} "
            f"for scope {target_record['scope']} from record {target}."
        )
    return (
        f"CURRENT RECORDS\n- {target}: {json.dumps(target_record, sort_keys=True)}\n"
        f"- {retained}: {json.dumps(retained_record, sort_keys=True)}\n\n"
        f"MATERIALIZED CONSEQUENCE\n{action}"
    )


def _apply_operation(
    before: dict[str, Any],
    *,
    operation: Operation,
    target: str,
    new_value: str,
) -> dict[str, Any]:
    after = json.loads(json.dumps(before))
    field = next(
        key for key in ("decision", "profile", "state", "preference") if key in after[target]
    )
    if operation is Operation.PRESERVE:
        after[target]["version"] += 1
    elif operation is Operation.ADD:
        additions = list(after[target].get("additions", []))
        additions.append(new_value)
        after[target]["additions"] = additions
        after[target]["version"] += 1
    elif operation is Operation.INVALIDATE:
        after[target]["active"] = False
        after[target]["version"] += 1
    elif operation is Operation.SUPERSEDE:
        after[target][field] = new_value
        after[target]["version"] += 1
    elif operation is Operation.ADD_EXCEPTION:
        exceptions = dict(after[target].get("exceptions", {}))
        exceptions["emergency"] = new_value
        after[target]["exceptions"] = exceptions
        after[target]["version"] += 1
    else:  # pragma: no cover - exhaustive enum
        raise AssertionError(operation)
    return cast(dict[str, Any], after)


def _transaction_text(
    domain: str,
    operation: Operation,
    target: str,
    before: dict[str, Any],
    after: dict[str, Any],
    new_value: str,
) -> str:
    old_version = before[target]["version"]
    new_version = after[target]["version"]
    subject = {
        "access_control": "policy record",
        "api_configuration": "configuration record",
        "workflow": "workflow record",
        "versioned_preference": "preference record",
    }[domain]
    if operation is Operation.PRESERVE:
        body = "The review confirmed that its effective setting remains unchanged."
    elif operation is Operation.ADD:
        body = (
            f"A new compatible setting {new_value} is now also active; "
            "the prior setting remains valid."
        )
    elif operation is Operation.INVALIDATE:
        body = "The prior setting is revoked immediately and has no replacement."
    elif operation is Operation.SUPERSEDE:
        body = f"The prior setting is retired and replaced by {new_value}."
    else:
        body = f"The base setting remains, but emergency cases now use {new_value}."
    return (
        f"VERIFIED UPDATE\nFor {subject} {target}, version {old_version} is followed by "
        f"version {new_version}. {body}"
    )


def _distractor(rng: random.Random, retained: str, gap_units: int) -> str:
    if gap_units <= 0:
        return ""
    sentences = []
    for index in range(gap_units):
        decoy = _opaque("dx", "entity", rng.randrange(1 << 16))
        color = rng.choice(_TOKENS["values"])
        sentences.append(
            f"Unrelated note {index + 1}: {decoy} keeps profile {color}; "
            f"{retained} is not modified."
        )
    return "DISTRACTOR NOTES\n" + " ".join(sentences)


def _tool_gold(
    domain: str,
    metadata: dict[str, Any],
    target: str,
    operation: Operation,
    after: dict[str, Any],
) -> dict[str, Any]:
    record = after[target]
    field = _current_field(domain)
    if operation is Operation.INVALIDATE:
        effective = "inactive"
    elif operation is Operation.ADD:
        effective = "+".join([str(record[field]), *record.get("additions", [])])
    elif operation is Operation.ADD_EXCEPTION:
        effective = str(record.get("exceptions", {}).get("emergency"))
    else:
        effective = str(record[field])
    gold = {
        "tool": metadata["tool"],
        "record": target,
        "effective": effective,
        "version": record["version"],
    }
    for key in ("resource", "route", "job", "scope"):
        if key in metadata:
            gold[key] = metadata[key]
    return gold


def _queries(
    domain: str,
    operation: Operation,
    target: str,
    retained: str,
    before: dict[str, Any],
    after: dict[str, Any],
    metadata: dict[str, Any],
    new_value: str,
) -> tuple[QueryRecord, ...]:
    field = _current_field(domain)
    old_value = str(before[target][field])
    if operation is Operation.INVALIDATE:
        current_value = "inactive"
    elif operation is Operation.ADD:
        current_value = "+".join([old_value, *after[target].get("additions", [])])
    elif operation is Operation.ADD_EXCEPTION:
        current_value = old_value
    else:
        current_value = str(after[target][field])

    current_candidates = (current_value, old_value if current_value != old_value else new_value)
    tool = _tool_gold(domain, metadata, target, operation, after)
    wrong_tool = dict(tool)
    wrong_tool["version"] = before[target]["version"]
    tool_gold = json.dumps(tool, sort_keys=True, separators=(",", ":"))
    tool_wrong = json.dumps(wrong_tool, sort_keys=True, separators=(",", ":"))

    if operation is Operation.PRESERVE:
        stale_gold = f"PREVIOUS_CONTENT_REMAINS_CURRENT:{old_value}"
        stale_wrong = "PREVIOUS_CONTENT_WAS_REVOKED"
    elif operation is Operation.ADD:
        stale_gold = f"PREVIOUS_CONTENT_REMAINS_BUT_IS_NOT_COMPLETE:{old_value}"
        stale_wrong = f"USE_PREVIOUS_CONTENT_ONLY:{old_value}"
    elif operation in (Operation.INVALIDATE, Operation.SUPERSEDE):
        stale_gold = f"DO_NOT_USE_PREVIOUS_VERSION:{old_value}"
        stale_wrong = f"USE_PREVIOUS_VERSION:{old_value}"
    else:
        stale_gold = "BASE_RULE_REMAINS_BUT_EXCEPTION_MUST_BE_CHECKED"
        stale_wrong = f"USE_BASE_WITHOUT_CHECKING_EXCEPTION:{old_value}"
    retained_value = str(before[retained][field])
    retained_wrong = _choice(
        random.Random(before[retained]["version"]), _TOKENS["values"], retained_value
    )

    return (
        QueryRecord(
            query_type=QueryType.CURRENT_STATE.value,
            prompt=f"What is the current effective setting for {target}?",
            candidate_answers=current_candidates,
            gold_index=0,
            structured_gold=None,
            affected_entities=(target,),
            retained_entities=(retained,),
        ),
        QueryRecord(
            query_type=QueryType.DERIVED_ACTION.value,
            prompt=(
                f"Return the exact compact JSON action that should be executed for {target} now."
            ),
            candidate_answers=(tool_gold, tool_wrong),
            gold_index=0,
            structured_gold=tool,
            affected_entities=(target,),
            retained_entities=(retained,),
        ),
        QueryRecord(
            query_type=QueryType.STALE_PROBE.value,
            prompt=f"How should the previously materialized setting for {target} be treated now?",
            candidate_answers=(stale_gold, stale_wrong),
            gold_index=0,
            structured_gold=None,
            affected_entities=(target,),
            retained_entities=(retained,),
        ),
        QueryRecord(
            query_type=QueryType.UNAFFECTED_RETENTION.value,
            prompt=f"What setting remains active for unrelated record {retained}?",
            candidate_answers=(retained_value, retained_wrong),
            gold_index=0,
            structured_gold=None,
            affected_entities=(),
            retained_entities=(retained,),
        ),
    )


def generate_episode(
    *,
    seed: int,
    split: str,
    domain: str,
    operation: Operation | str,
    index: int,
    distractor_units: int = 0,
) -> TransactionEpisode:
    if split not in _SPLIT_NAMESPACES:
        raise ValueError(f"Unknown split: {split}")
    operation = Operation(operation)
    stable = int.from_bytes(
        hashlib.sha256(f"{split}|{domain}|{operation.value}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random((seed << 20) ^ index ^ stable)
    namespace = _SPLIT_NAMESPACES[split]
    target = _opaque(namespace["entity"], "target", index)
    retained = _opaque(namespace["entity"], "retain", index + 10_000)
    old_value = rng.choice(_TOKENS["values"])
    new_value = _choice(rng, _TOKENS["values"], old_value)
    resource = rng.choice(_TOKENS["resources"])
    route = rng.choice(_TOKENS["routes"])
    before, metadata = _domain_record(
        domain,
        target=target,
        retained=retained,
        old_value=old_value,
        new_value=new_value,
        resource=resource,
        route=route,
    )
    after = _apply_operation(before, operation=operation, target=target, new_value=new_value)
    materialization = _materialize(domain, target, retained, before)
    transaction = _transaction_text(domain, operation, target, before, after, new_value)
    distractor = _distractor(rng, retained, distractor_units)
    prefix = "\n\n".join(part for part in (materialization, transaction, distractor) if part)
    queries = _queries(domain, operation, target, retained, before, after, metadata, new_value)
    exact_refresh = (
        "CURRENT CANONICAL VIEW\n"
        f"- {target}: {json.dumps(after[target], sort_keys=True)}\n"
        f"- {retained}: {json.dumps(after[retained], sort_keys=True)}\n"
        f"The previous version of {target} is historical and must not govern current action."
    )
    template_family = f"{namespace['template']}-{domain}-{index % 7}"
    protected = {
        "target_entity": target,
        "retained_entity": retained,
        "old_value": old_value,
        "new_value": new_value,
        "old_version": before[target]["version"],
        "new_version": after[target]["version"],
    }
    audit = visible_input_audit(prefix, queries)
    return TransactionEpisode(
        episode_id=f"{split}-{domain}-{operation.value.lower()}-{seed}-{index}",
        split=split,
        domain=domain,
        operation=operation.value,
        template_family=template_family,
        entity_namespace=namespace["entity"],
        seed=seed,
        state_before=before,
        state_after=after,
        materialization_text=materialization,
        transaction_text=transaction,
        dependency_closure=(target,),
        distractor_text=distractor,
        branch_prefix_text=prefix,
        queries=queries,
        exact_refresh_text=exact_refresh,
        protected_fields=protected,
        visible_input_audit=audit,
    )


def visible_input_audit(prefix: str, queries: Sequence[QueryRecord]) -> dict[str, bool]:
    lower = prefix.lower()
    operation_tokens = [f"operation={item.value.lower()}" for item in Operation]
    gate_tokens = ["erase_bit", "write_bit", "gate_label", "e=0", "w=1"]
    exact_tool_calls = [
        query.gold_answer for query in queries if query.query_type == QueryType.DERIVED_ACTION.value
    ]
    future_prompts = [query.prompt for query in queries]
    return {
        "contains_operation_label": any(token in lower for token in operation_tokens),
        "contains_gate_label": any(token in lower for token in gate_tokens),
        "contains_exact_address_mask": "address_mask" in lower or "target_slot" in lower,
        "contains_future_query": any(prompt in prefix for prompt in future_prompts),
        "contains_exact_final_tool_call": any(answer in prefix for answer in exact_tool_calls),
    }


def validate_episode(episode: TransactionEpisode) -> list[str]:
    errors: list[str] = []
    for name, value in episode.visible_input_audit.items():
        if value:
            errors.append(f"visible_input_audit failed: {name}")
    if episode.branch_prefix_text != "\n\n".join(
        part
        for part in (
            episode.materialization_text,
            episode.transaction_text,
            episode.distractor_text,
        )
        if part
    ):
        errors.append("branch_prefix_text does not match materialization+transaction+distractor")
    if episode.protected_fields["target_entity"] not in episode.exact_refresh_text:
        errors.append("exact refresh is missing target entity")
    if (
        episode.state_after == episode.state_before
        and episode.operation != Operation.PRESERVE.value
    ):
        errors.append("non-preserve operation did not change canonical state")
    if episode.operation == Operation.PRESERVE.value:
        before = episode.state_before[episode.protected_fields["target_entity"]]
        after = episode.state_after[episode.protected_fields["target_entity"]]
        comparable_before = dict(before)
        comparable_after = dict(after)
        comparable_before.pop("version", None)
        comparable_after.pop("version", None)
        if comparable_before != comparable_after:
            errors.append("preserve operation changed effective content")
    return errors


def audit_split_disjointness(episodes: Iterable[TransactionEpisode]) -> dict[str, Any]:
    by_split: dict[str, dict[str, set[str]]] = {}
    signatures: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    validation_errors: dict[str, list[str]] = {}
    for episode in episodes:
        bucket = by_split.setdefault(
            episode.split,
            {"entities": set(), "templates": set(), "episode_ids": set()},
        )
        target = str(episode.protected_fields["target_entity"])
        retained = str(episode.protected_fields["retained_entity"])
        bucket["entities"].update((target, retained))
        bucket["templates"].add(episode.template_family)
        bucket["episode_ids"].add(episode.episode_id)
        normalized = re.sub(r"\s+", " ", episode.branch_prefix_text.strip().lower())
        signature = hash_mapping(normalized)
        if signature in signatures:
            duplicates.append((signatures[signature], episode.episode_id))
        else:
            signatures[signature] = episode.episode_id
        errors = validate_episode(episode)
        if errors:
            validation_errors[episode.episode_id] = errors

    overlaps: list[dict[str, Any]] = []
    split_names = sorted(by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            for category in ("entities", "templates", "episode_ids"):
                common = by_split[left][category] & by_split[right][category]
                if common:
                    overlaps.append(
                        {
                            "left": left,
                            "right": right,
                            "category": category,
                            "count": len(common),
                            "examples": sorted(common)[:5],
                        }
                    )
    return {
        "disjoint": not overlaps,
        "overlaps": overlaps,
        "duplicates": duplicates,
        "validation_errors": validation_errors,
        "split_counts": {split: len(values["episode_ids"]) for split, values in by_split.items()},
    }


def generate_grid(
    *,
    seed: int,
    splits: Sequence[str],
    domains: Sequence[str],
    operations: Sequence[Operation | str],
    items_per_cell: int,
    distractor_units: int = 1,
) -> Iterator[TransactionEpisode]:
    for split in splits:
        for domain in domains:
            for operation in operations:
                for index in range(items_per_cell):
                    yield generate_episode(
                        seed=seed,
                        split=split,
                        domain=domain,
                        operation=operation,
                        index=index,
                        distractor_units=distractor_units,
                    )
