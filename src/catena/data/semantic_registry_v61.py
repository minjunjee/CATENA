from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from catena.core.schema import Operation
from catena.data.semantic_transactions_v61 import (
    ALLOWED_SAFE_FIELDS,
    BANNED_SURFACE_CUES,
    SafeSemanticRecord,
    SemanticExample,
    SemanticMemorySpec,
    SemanticNamespaceRegistry,
    build_safe_record_for_operation,
    build_semantic_example,
    derive_operation,
    find_banned_surface_cues,
    semantic_private_identifiers,
)

REGISTRY_SCHEMA_VERSION = 1


def semantic_registry_row_from_design(
    *,
    namespace_name: str,
    split: str,
    numeric_seed: int,
    checkpoint_seed: int,
    seed_slot: int,
    operation: Operation,
    domain: str,
    template_surface: str,
) -> dict[str, object]:
    record = build_safe_record_for_operation(
        operation=operation,
        numeric_seed=numeric_seed,
        domain=domain,
        template_surface=template_surface,
    )
    identifiers = semantic_private_identifiers(
        numeric_seed=numeric_seed,
        checkpoint_seed=checkpoint_seed,
    )
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "split": split,
        "seed_slot": int(seed_slot),
        "checkpoint_seed": int(checkpoint_seed),
        "namespace_name": namespace_name,
        "numeric_seed": int(numeric_seed),
        "example_id": identifiers["example_id"],
        "transaction_id": identifiers["transaction_id"],
        "episode_id": identifiers["episode_id"],
        "operation_private": operation.value,
        "old_value_token_private": identifiers["old_value_token"],
        "safe_record": asdict(record),
    }


def semantic_example_registry_row(
    example: SemanticExample,
    *,
    split: str,
    seed_slot: int,
) -> dict[str, object]:
    private = example.private_episode
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "split": split,
        "seed_slot": int(seed_slot),
        "checkpoint_seed": int(private.checkpoint_seed),
        "namespace_name": private.namespace_name,
        "numeric_seed": int(private.numeric_seed),
        "example_id": private.example_id,
        "transaction_id": private.transaction_id,
        "episode_id": private.episode_id,
        "operation_private": private.operation.value,
        "old_value_token_private": private.old_value_token,
        "safe_record": asdict(example.safe_record),
    }


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Semantic registry field {field} must be a nonempty string.")
    return value


def _required_integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Semantic registry field {field} must be an integer.")
    return value


def semantic_example_from_registry_row(
    row: Mapping[str, object],
    *,
    memory_spec: SemanticMemorySpec,
) -> SemanticExample:
    if row.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Semantic registry schema version changed.")
    safe_payload = row.get("safe_record")
    if not isinstance(safe_payload, dict):
        raise ValueError("Semantic registry row lacks a safe record.")
    record = SafeSemanticRecord(**safe_payload)
    example = build_semantic_example(
        safe_record=record,
        namespace_name=_required_string(row, "namespace_name"),
        numeric_seed=_required_integer(row, "numeric_seed"),
        checkpoint_seed=_required_integer(row, "checkpoint_seed"),
        memory_spec=memory_spec,
    )
    private = example.private_episode
    expected = {
        "example_id": private.example_id,
        "transaction_id": private.transaction_id,
        "episode_id": private.episode_id,
        "operation_private": private.operation.value,
        "old_value_token_private": private.old_value_token,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"Semantic registry reconstruction differs at {field}.")
    return example


def validate_semantic_registry_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_split: str,
    expected_seeds: Sequence[int],
    expected_rows_per_seed: int,
    expected_namespace_name: str,
    expected_seed_slots: Mapping[int, int],
    namespace_registry: SemanticNamespaceRegistry,
    memory_spec: SemanticMemorySpec,
    reconstruct: bool,
) -> None:
    if not rows:
        raise ValueError("Semantic registry must not be empty.")
    identifiers: set[str] = set()
    transaction_ids: set[str] = set()
    episode_ids: set[str] = set()
    counts = {int(seed): 0 for seed in expected_seeds}
    numeric_seeds = {int(seed): set() for seed in expected_seeds}
    if set(expected_seed_slots) != set(counts):
        raise ValueError("Semantic registry seed-slot mapping differs from fixed seeds.")
    for row in rows:
        if row.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError("Semantic registry schema version changed.")
        if _required_string(row, "split") != expected_split:
            raise ValueError("Semantic registry contains the wrong split.")
        seed = _required_integer(row, "checkpoint_seed")
        if seed not in counts:
            raise ValueError(f"Semantic registry contains unexpected seed {seed}.")
        if _required_string(row, "namespace_name") != expected_namespace_name:
            raise ValueError("Semantic registry contains the wrong namespace.")
        seed_slot = _required_integer(row, "seed_slot")
        if seed_slot != int(expected_seed_slots[seed]):
            raise ValueError("Semantic registry seed-slot mapping changed.")
        numeric_seed = _required_integer(row, "numeric_seed")
        example_id = _required_string(row, "example_id")
        transaction_id = _required_string(row, "transaction_id")
        episode_id = _required_string(row, "episode_id")
        if (
            example_id in identifiers
            or transaction_id in transaction_ids
            or episode_id in episode_ids
        ):
            raise ValueError("Semantic registry contains duplicate identifiers.")
        identifiers.add(example_id)
        transaction_ids.add(transaction_id)
        episode_ids.add(episode_id)
        numeric_seeds[seed].add(numeric_seed)
        counts[seed] += 1
        safe_payload = row.get("safe_record")
        if not isinstance(safe_payload, dict) or set(safe_payload) != set(
            ALLOWED_SAFE_FIELDS
        ):
            raise ValueError("Semantic registry safe-record schema changed.")
        record = SafeSemanticRecord(**safe_payload)
        if row.get("operation_private") != derive_operation(record).value:
            raise ValueError("Semantic registry operation differs from raw predicates.")
        expected_identifiers = semantic_private_identifiers(
            numeric_seed=numeric_seed,
            checkpoint_seed=seed,
        )
        for field, expected_value in (
            ("example_id", expected_identifiers["example_id"]),
            ("transaction_id", expected_identifiers["transaction_id"]),
            ("episode_id", expected_identifiers["episode_id"]),
            ("old_value_token_private", expected_identifiers["old_value_token"]),
        ):
            if row.get(field) != expected_value:
                raise ValueError(
                    f"Semantic registry private identifier differs at {field}."
                )
        if reconstruct:
            semantic_example_from_registry_row(row, memory_spec=memory_spec)
    if any(count != expected_rows_per_seed for count in counts.values()):
        raise ValueError(
            f"Semantic registry seed counts differ: {counts}, "
            f"expected {expected_rows_per_seed}."
        )
    for seed, seed_slot in expected_seed_slots.items():
        expected_numeric_seeds = {
            namespace_registry.numeric_seed(
                expected_namespace_name,
                seed_slot=int(seed_slot),
                index=index,
            )
            for index in range(expected_rows_per_seed)
        }
        if numeric_seeds[int(seed)] != expected_numeric_seeds:
            raise ValueError(
                f"Semantic registry numeric namespace is not contiguous for seed {seed}."
            )


def render_naturalized_record(record: SafeSemanticRecord) -> str:
    """Render a neutral human-audit view without an operation word."""

    if find_banned_surface_cues(record):
        raise ValueError("Cannot render a record that contains a banned cue.")
    if record.template_surface == "ledger":
        text = (
            f"Domain {record.domain}; subject {record.entity_description}; "
            f"relation {record.current_relation}. On day {record.observation_day}, "
            f"the prior version {record.prior_version} has interval "
            f"{record.prior_valid_from_day}–{record.prior_valid_to_day}. "
            f"Evidence {record.incoming_evidence} for item "
            f"{record.incoming_value_token}, version {record.evidence_version}, "
            f"was recorded on day {record.evidence_timestamp_day} with interval "
            f"{record.evidence_valid_from_day}–{record.evidence_valid_to_day}, "
            f"scope {record.scope}, source {record.source}, trace "
            f"{record.provenance}."
        )
    elif record.template_surface == "paraphrase":
        text = (
            f"For {record.entity_description} in {record.domain}, inspect day "
            f"{record.observation_day}. The earlier entry is version "
            f"{record.prior_version} and spans days "
            f"{record.prior_valid_from_day} through {record.prior_valid_to_day}. "
            f"A filing dated {record.evidence_timestamp_day} describes "
            f"{record.incoming_value_token} as version {record.evidence_version}, "
            f"covering days {record.evidence_valid_from_day} through "
            f"{record.evidence_valid_to_day}; its scope is {record.scope}, its "
            f"source is {record.source}, and its trace is {record.provenance}. "
            f"The relation field is {record.current_relation}."
        )
    else:
        text = (
            f"At day {record.observation_day}, domain {record.domain} lists subject "
            f"{record.entity_description} under {record.current_relation}. Prior "
            f"version {record.prior_version} spans day "
            f"{record.prior_valid_from_day} to {record.prior_valid_to_day}. "
            f"Statement {record.incoming_evidence}, filed day "
            f"{record.evidence_timestamp_day}, associates item "
            f"{record.incoming_value_token} with version "
            f"{record.evidence_version} for day "
            f"{record.evidence_valid_from_day} to "
            f"{record.evidence_valid_to_day}, scope {record.scope}, source "
            f"{record.source}, trace {record.provenance}."
        )
    tokens = {token.strip(".,;:()[]{}").lower() for token in text.split()}
    leaked = sorted(set(BANNED_SURFACE_CUES) & tokens)
    if leaked:
        raise AssertionError(f"Naturalized record leaks banned cues: {leaked}.")
    return text


def audit_id(split: str, example_id: str) -> str:
    return "audit_" + hashlib.sha256(
        f"{split}\0{example_id}".encode()
    ).hexdigest()[:24]
