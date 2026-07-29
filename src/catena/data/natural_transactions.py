from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from catena.core.schema import Operation


@dataclass(slots=True)
class NaturalTransaction:
    transaction_id: str
    domain: str
    operation: Operation
    text: str
    entity: str
    old_value: str
    new_value: str
    template_id: str


_TEMPLATES: dict[str, dict[Operation, list[str]]] = {
    "api": {
        Operation.PRESERVE: [
            "The authentication policy for {entity} remains unchanged.",
            "No authentication change applies to {entity}.",
        ],
        Operation.ADD: [
            "{entity} now also accepts {new} in addition to its current method.",
            "A new accepted method, {new}, has been introduced for {entity}.",
        ],
        Operation.INVALIDATE: [
            "The legacy method {old} is no longer accepted by {entity}.",
            "{entity} has withdrawn support for {old}.",
        ],
        Operation.SUPERSEDE: [
            "{new} replaces {old} as the authentication method for {entity}.",
            "{entity} has migrated from {old} to {new}.",
        ],
    },
    "access": {
        Operation.PRESERVE: [
            "The permission assigned to {entity} is unchanged.",
            "There is no role update for {entity}.",
        ],
        Operation.ADD: [
            "{entity} receives the additional permission {new}.",
            "The permission {new} is added to {entity}'s existing access.",
        ],
        Operation.INVALIDATE: [
            "The permission {old} is revoked from {entity}.",
            "{entity} can no longer use {old}.",
        ],
        Operation.SUPERSEDE: [
            "The role {new} replaces {old} for {entity}.",
            "{entity}'s access changes from {old} to {new}.",
        ],
    },
    "workflow": {
        Operation.PRESERVE: [
            "The workflow setting for {entity} remains as before.",
            "No workflow setting changes for {entity}.",
        ],
        Operation.ADD: [
            "{entity} gains an additional workflow option: {new}.",
            "The option {new} is added without removing the current setting for {entity}.",
        ],
        Operation.INVALIDATE: [
            "The workflow option {old} is retired for {entity}.",
            "{entity} must no longer use {old}.",
        ],
        Operation.SUPERSEDE: [
            "{entity} now uses {new} instead of {old}.",
            "The workflow for {entity} switches from {old} to {new}.",
        ],
    },
}


def generate_natural_transactions(
    *,
    count: int,
    seed: int,
    domains: list[str] | None = None,
    operations: list[Operation] | None = None,
) -> list[NaturalTransaction]:
    rng = random.Random(seed)
    selected_domains = domains or list(_TEMPLATES)
    selected_operations = operations or list(Operation)
    rows: list[NaturalTransaction] = []
    for index in range(count):
        domain = rng.choice(selected_domains)
        operation = rng.choice(selected_operations)
        entity = f"entity_{rng.randrange(128):03d}"
        old_value = f"old_{rng.randrange(64):02d}"
        new_value = f"new_{rng.randrange(64):02d}"
        template_pool = _TEMPLATES[domain][operation]
        template_idx = rng.randrange(len(template_pool))
        template = template_pool[template_idx]
        text = template.format(entity=entity, old=old_value, new=new_value)
        digest = hashlib.sha256(f"{seed}-{index}-{text}".encode()).hexdigest()[:16]
        rows.append(
            NaturalTransaction(
                transaction_id=digest,
                domain=domain,
                operation=operation,
                text=text,
                entity=entity,
                old_value=old_value,
                new_value=new_value,
                template_id=f"{domain}-{operation.value}-{template_idx}",
            )
        )
    return rows
