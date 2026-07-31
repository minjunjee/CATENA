"""Frozen v8.1 transaction replay and leakage receipts for E26 Stage-2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

from .transactional_stream import (
    Operation,
    TransactionEpisode,
    audit_split_disjointness,
    generate_grid,
    validate_episode,
)


class TransactionDataError(RuntimeError):
    """Raised when transaction replay or leakage checks fail closed."""


@dataclass(frozen=True, slots=True)
class TransactionReplaySpec:
    seed: int
    splits: tuple[str, ...]
    domains: tuple[str, ...]
    operations: tuple[str, ...]
    items_per_cell: int
    distractor_units: int

    def __post_init__(self) -> None:
        if "main_test" in self.splits:
            raise TransactionDataError("Stage-2 transaction preparation must not open main_test")
        if self.items_per_cell < 1:
            raise ValueError("items_per_cell must be positive")


def _episode_lock_record(episode: TransactionEpisode) -> dict[str, Any]:
    visible = {
        "episode_id": episode.episode_id,
        "split": episode.split,
        "domain": episode.domain,
        "operation": episode.operation,
        "template_family": episode.template_family,
        "entity_namespace": episode.entity_namespace,
        "branch_prefix_sha256": sha256_canonical_json(episode.branch_prefix_text),
        "query_bundle_sha256": sha256_canonical_json(
            [query.to_dict() for query in episode.queries]
        ),
        "protected_signature_sha256": sha256_canonical_json(episode.protected_fields),
        "visible_input_audit": episode.visible_input_audit,
    }
    visible["record_sha256"] = sha256_canonical_json(visible)
    return visible


def generate_locked_transaction_records(
    spec: TransactionReplaySpec,
) -> tuple[dict[str, Any], ...]:
    episodes = list(
        generate_grid(
            seed=spec.seed,
            splits=spec.splits,
            domains=spec.domains,
            operations=spec.operations,
            items_per_cell=spec.items_per_cell,
            distractor_units=spec.distractor_units,
        )
    )
    errors = {
        episode.episode_id: validate_episode(episode)
        for episode in episodes
        if validate_episode(episode)
    }
    if errors:
        raise TransactionDataError(f"Transaction visible-input leakage audit failed: {errors}")
    return tuple(_episode_lock_record(episode) for episode in episodes)


def write_transaction_replay_manifest(
    path: str | Path,
    spec: TransactionReplaySpec,
) -> Path:
    first = generate_locked_transaction_records(spec)
    second = generate_locked_transaction_records(spec)
    if first != second:
        raise TransactionDataError("Transaction generator replay is not byte deterministic")
    episodes = list(
        generate_grid(
            seed=spec.seed,
            splits=spec.splits,
            domains=spec.domains,
            operations=spec.operations,
            items_per_cell=spec.items_per_cell,
            distractor_units=spec.distractor_units,
        )
    )
    split_audit = audit_split_disjointness(episodes)
    if (
        not split_audit["disjoint"]
        or split_audit["duplicates"]
        or split_audit["validation_errors"]
    ):
        raise TransactionDataError(f"Transaction split audit failed: {split_audit}")
    record_digest = sha256_canonical_json(list(first))
    payload = {
        "schema_version": "catena-e26-transaction-replay-v1",
        "manifest_type": "E26_TRANSACTION_DETERMINISTIC_REPLAY",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "scientific_main_input_eligible": True,
        "generator_version": "v8.1",
        "seed": spec.seed,
        "splits": list(spec.splits),
        "domains": list(spec.domains),
        "operations": list(spec.operations),
        "items_per_cell": spec.items_per_cell,
        "distractor_units": spec.distractor_units,
        "episode_count": len(first),
        "records_sha256": record_digest,
        "replay_count": 2,
        "replay_identical": True,
        "split_audit": split_audit,
        "visible_operation_gate_address_future_query_leakage": 0,
        "main_test_opened": False,
        "main_test_access_forbidden": True,
        "records": list(first),
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite transaction manifest: {destination}")
    write_json_strict(destination, payload)
    if sha256_file(destination) == "":
        raise AssertionError("unreachable")
    return destination


def default_stage2_transaction_spec() -> TransactionReplaySpec:
    return TransactionReplaySpec(
        seed=260_026,
        splits=("train", "validation", "calibration"),
        domains=(
            "access_control",
            "api_configuration",
            "workflow",
            "versioned_preference",
        ),
        operations=tuple(item.value for item in Operation),
        items_per_cell=100,
        distractor_units=1,
    )
