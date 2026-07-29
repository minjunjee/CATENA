from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from catena.core.schema import Operation


@dataclass(slots=True)
class SemanticTransaction:
    transaction_id: str
    domain: str
    operation: Operation
    text: str
    entity: str
    old_value: str
    new_value: str
    template_family: str
    surface_style: str


def _text(domain: str, entity: str, old: str, new: str, operation: Operation, style: str) -> str:
    if style == "structured":
        current = {
            Operation.PRESERVE: f"The current record for {entity} has the same active entry as version one.",
            Operation.ADD: f"The current record for {entity} retains the prior entry and contains an additional active entry {new}.",
            Operation.INVALIDATE: f"The prior entry for {entity} remains only as historical evidence and no active replacement is listed.",
            Operation.SUPERSEDE: f"The prior entry for {entity} is historical while {new} is the current active entry.",
        }[operation]
        return f"Domain {domain}. Version one records {old} for {entity}. {current}"
    if style == "indirect":
        return {
            Operation.PRESERVE: f"A later audit of {entity} agrees with its earlier {domain} record.",
            Operation.ADD: f"A later audit of {entity} accepts the earlier entry and a second entry {new} together.",
            Operation.INVALIDATE: f"A later audit of {entity} cites the earlier entry only for history, with nothing active in its place.",
            Operation.SUPERSEDE: f"A later audit of {entity} cites the earlier entry only for history and treats {new} as current.",
        }[operation]
    if style == "paraphrase":
        return {
            Operation.PRESERVE: f"No currently usable {domain} setting for {entity} differs from the earlier record.",
            Operation.ADD: f"The usable {domain} settings for {entity} include the former entry and {new} as well.",
            Operation.INVALIDATE: f"The former {domain} setting for {entity} is documented but cannot be used now.",
            Operation.SUPERSEDE: f"The former {domain} setting for {entity} is no longer usable; {new} is usable now.",
        }[operation]
    raise ValueError(style)


def generate_semantic_transactions(
    *, count: int, seed: int, domains: list[str], operations: list[Operation], styles: list[str]
) -> list[SemanticTransaction]:
    rng = random.Random(seed)
    result = []
    for index in range(count):
        domain = rng.choice(domains); operation = rng.choice(operations); style = rng.choice(styles)
        entity = f"entity_{rng.randrange(512):03d}"
        old = f"prior_{rng.randrange(256):03d}"; new = f"current_{rng.randrange(256):03d}"
        text = _text(domain, entity, old, new, operation, style)
        digest = hashlib.sha256(f"{seed}-{index}-{text}".encode()).hexdigest()[:16]
        result.append(SemanticTransaction(digest, domain, operation, text, entity, old, new, f"{domain}-{style}", style))
    return result
