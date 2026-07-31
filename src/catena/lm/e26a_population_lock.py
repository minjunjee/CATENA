"""Prospective, validation-only population lock for the E26a scientific gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    read_json_object_strict,
    sha256_canonical_json,
    write_json_strict,
)

from .transactional_stream import (
    Operation,
    QueryType,
    TransactionEpisode,
    generate_grid,
    validate_episode,
)


class E26AValidationPopulationError(RuntimeError):
    """Raised when the validation population is not prospectively identifiable."""


def _population_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    population = config.get("gate_population")
    data = config.get("data")
    if not isinstance(population, Mapping) or not isinstance(data, Mapping):
        raise E26AValidationPopulationError("E26a config lacks gate_population or data")
    if population.get("population_hash_required") is not True:
        raise E26AValidationPopulationError(
            "gate_population.population_hash_required must remain true"
        )
    return population


def generate_e26a_validation_episodes(
    config: Mapping[str, Any],
) -> tuple[TransactionEpisode, ...]:
    """Generate only the locked E26a validation split.

    The broader split list in the prospective config remains a derivation
    specification. This function deliberately never iterates over that list,
    so main-test and held-out-domain bytes cannot be materialized here.
    """

    population = _population_config(config)
    configured_splits = population.get("splits")
    if not isinstance(configured_splits, list) or "validation" not in configured_splits:
        raise E26AValidationPopulationError("gate_population must prospectively include validation")
    domains_raw = population.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise E26AValidationPopulationError("gate_population.domains is invalid")
    domains = tuple(str(value) for value in domains_raw)
    data = config.get("data")
    assert isinstance(data, Mapping)
    operations_raw = data.get("operations")
    if not isinstance(operations_raw, list) or not operations_raw:
        raise E26AValidationPopulationError("data.operations is invalid")
    try:
        operations = tuple(Operation(str(value)) for value in operations_raw)
    except ValueError as error:
        raise E26AValidationPopulationError(
            "data.operations contains an unknown operation"
        ) from error
    items_per_operation = int(population.get("items_per_operation_per_split", 0))
    if items_per_operation <= 0 or items_per_operation % len(domains):
        raise E26AValidationPopulationError(
            "items_per_operation_per_split must be positive and divisible by domains"
        )
    episodes = tuple(
        generate_grid(
            seed=int(population["generation_seed"]),
            splits=("validation",),
            domains=domains,
            operations=operations,
            items_per_cell=items_per_operation // len(domains),
            distractor_units=int(population["distractor_units"]),
        )
    )
    errors = {
        episode.episode_id: validate_episode(episode)
        for episode in episodes
        if validate_episode(episode)
    }
    if errors:
        raise E26AValidationPopulationError(f"E26a validation visible-input audit failed: {errors}")
    expected_count = items_per_operation * len(operations)
    if len(episodes) != expected_count:
        raise E26AValidationPopulationError(
            f"E26a validation population size changed: {len(episodes)} != {expected_count}"
        )
    if any(episode.split != "validation" for episode in episodes):
        raise E26AValidationPopulationError(
            "E26a validation generator materialized a forbidden split"
        )
    return episodes


def e26a_validation_population_payload(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    first = generate_e26a_validation_episodes(config)
    second = generate_e26a_validation_episodes(config)
    first_records = [episode.to_dict() for episode in first]
    second_records = [episode.to_dict() for episode in second]
    if first_records != second_records:
        raise E26AValidationPopulationError(
            "E26a validation population replay is not byte deterministic"
        )
    population = _population_config(config)
    data = config["data"]
    assert isinstance(data, Mapping)
    records_sha256 = sha256_canonical_json(first_records)
    payload: dict[str, Any] = {
        "schema_version": "catena-e26a-validation-population-v1",
        "manifest_type": "E26A_VALIDATION_POPULATION_LOCK",
        "scientific_evidence": False,
        "evidence_tier": "SCIENTIFIC_INPUT_PROVENANCE",
        "claim_ceiling": "PROTOCOL_IDENTIFIABILITY_ONLY",
        "scientific_main_input_eligible": True,
        "namespace": str(population["namespace"]),
        "generator_version": str(data["transaction_generator_version"]),
        "generation_seed": int(population["generation_seed"]),
        "split": "validation",
        "domains": [str(value) for value in population["domains"]],
        "operations": [str(value) for value in data["operations"]],
        "query_types": [item.value for item in QueryType],
        "items_per_operation": int(population["items_per_operation_per_split"]),
        "distractor_units": int(population["distractor_units"]),
        "episode_count": len(first_records),
        "records_sha256": records_sha256,
        "replay_count": 2,
        "replay_identical": True,
        "visible_input_validation_error_count": 0,
        "main_test_opened": False,
        "main_test_access_count": 0,
        "heldout_domain_opened": False,
        "heldout_domain_access_count": 0,
        "forbidden_materialized_splits": ["main_test", "heldout_domain"],
        "records": first_records,
    }
    payload["manifest_sha256"] = sha256_canonical_json(payload)
    return payload


def write_e26a_validation_population_lock(
    path: str | Path,
    config: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite validation population lock: {destination}")
    write_json_strict(destination, e26a_validation_population_payload(config))
    return destination


def validate_e26a_validation_population_lock(
    payload: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> tuple[TransactionEpisode, ...]:
    if payload.get("manifest_type") != "E26A_VALIDATION_POPULATION_LOCK":
        raise E26AValidationPopulationError(
            "Validation population lock has the wrong manifest_type"
        )
    if payload.get("scientific_evidence") is not False:
        raise E26AValidationPopulationError(
            "Validation population lock must remain scientific_evidence=false"
        )
    if payload.get("scientific_main_input_eligible") is not True:
        raise E26AValidationPopulationError(
            "Validation population lock is not an eligible scientific input"
        )
    if payload.get("main_test_opened") is not False or payload.get("main_test_access_count") != 0:
        raise E26AValidationPopulationError("Validation population lock records main-test access")
    if (
        payload.get("heldout_domain_opened") is not False
        or payload.get("heldout_domain_access_count") != 0
    ):
        raise E26AValidationPopulationError(
            "Validation population lock records held-out-domain access"
        )
    expected = e26a_validation_population_payload(config)
    if dict(payload) != expected:
        raise E26AValidationPopulationError(
            "Validation population lock differs from deterministic config replay"
        )
    return generate_e26a_validation_episodes(config)


def load_e26a_validation_population_lock(
    path: str | Path,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[TransactionEpisode, ...]]:
    payload = read_json_object_strict(path)
    episodes = validate_e26a_validation_population_lock(payload, config=config)
    return payload, episodes
