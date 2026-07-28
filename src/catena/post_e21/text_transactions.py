"""Leakage-controlled text-form transactions for E25b."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import torch

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_POLICY_TOKEN = re.compile(r"\bpolicy-[0-9]+\b", re.IGNORECASE)


class TextDemand(StrEnum):
    MAGNITUDE = "magnitude"
    VALUE = "value"
    ADDRESS = "address"
    STATE_CONDITIONING = "state_conditioning"


class MagnitudeOperation(StrEnum):
    ADD = "add_anchor"
    INVALIDATE = "invalidate_anchor"
    SUPERSEDE = "supersede_composition"


class OldRuleStatus(StrEnum):
    """Categorical status of the previously stored rule after assimilation."""

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class TextSplit(StrEnum):
    DEVELOPMENT = "development"
    TRAIN = "train"
    VALIDATION = "validation"
    PRIMARY = "primary"
    PARAPHRASE = "paraphrase"
    IDENTIFIER = "identifier"
    DOMAIN = "domain"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class TextTransaction:
    """One model-visible transaction plus private evaluation targets."""

    example_id: str
    minimal_pair_id: str
    state_counterpair_id: str
    split: TextSplit
    demand: TextDemand
    text: str
    query_direct: str
    query_derived: str
    query_old_rule: str
    query_unaffected: str
    entity: str
    other_entity: str
    domain: str
    day: int
    active: bool
    magnitude_operation: str
    old_value_label: str
    new_value_label: str
    candidate_decoder_seed: int
    template_index: int
    evidence_index: int
    memory_entities: tuple[str, ...]
    state: torch.Tensor
    target_state: torch.Tensor
    erase_index: int
    write_index: int
    old_value: torch.Tensor
    new_value: torch.Tensor
    coordinate_mask: torch.Tensor
    derived_action: int
    derived_action_label: str
    derived_action_rule: str
    direct_fact_entity: str
    direct_fact_answer: str
    old_rule_status: OldRuleStatus

    def __post_init__(self) -> None:
        if self.state.ndim != 2 or self.target_state.shape != self.state.shape:
            raise ValueError("state and target_state must be same-shaped matrices")
        slots, value_dim = self.state.shape
        if len(self.memory_entities) != slots or len(set(self.memory_entities)) != slots:
            raise ValueError("memory entity keys must be unique and aligned with state slots")
        if not 0 <= self.erase_index < slots or not 0 <= self.write_index < slots:
            raise ValueError("private address lies outside the memory")
        if self.old_value.shape != (value_dim,) or self.new_value.shape != (value_dim,):
            raise ValueError("candidate vectors have the wrong shape")
        if self.coordinate_mask.shape != (value_dim,):
            raise ValueError("coordinate mask has the wrong shape")
        if self.entity != self.memory_entities[self.erase_index]:
            raise ValueError("erase entity is not aligned with its private address")
        if self.other_entity != self.memory_entities[self.write_index]:
            raise ValueError("write entity is not aligned with its private address")
        if self.day < 0 or not self.old_value_label or not self.new_value_label:
            raise ValueError("transaction semantic metadata is incomplete")
        if self.demand is TextDemand.MAGNITUDE:
            MagnitudeOperation(self.magnitude_operation)
        elif self.magnitude_operation != "not_applicable":
            raise ValueError("non-magnitude transaction carries a magnitude operation")
        decoded_candidate = decode_visible_policy_candidate(
            self.text,
            dimension=value_dim,
            semantic_value_seed=self.candidate_decoder_seed,
        )
        if not bool(torch.equal(decoded_candidate, self.new_value)):
            raise ValueError("visible policy token does not decode to the private candidate")
        if self.derived_action_label != _ACTION_TAXONOMY[self.derived_action]:
            raise ValueError("derived action label does not follow the registered taxonomy")
        if self.derived_action_rule != _ACTION_RULE:
            raise ValueError("derived action rule drifted")
        expected_old_rule_status = old_rule_status_for(
            demand=self.demand,
            magnitude_operation=self.magnitude_operation,
            active=self.active,
        )
        if self.old_rule_status is not expected_old_rule_status:
            raise ValueError("old-rule status is inconsistent with the private demand")
        for tensor in (
            self.state,
            self.target_state,
            self.old_value,
            self.new_value,
            self.coordinate_mask,
        ):
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError("text transaction contains non-finite data")


_TRAIN_DOMAINS = ("access", "billing", "routing")
_OOD_DOMAINS = ("clinical", "aviation")
_TRAIN_TEMPLATES = (
    "{entity} has a standing record. {evidence}",
    "For {entity}, a prior rule remains stored. {evidence}",
)
_PARAPHRASE_TEMPLATES = (
    "A standing instruction is stored for {entity}. {evidence}",
    "{entity} has an existing rule on record. {evidence}",
)
_EVIDENCE = {
    TextDemand.VALUE: (
        "At day {day}, {new} governs the security half while all other fields stay as recorded.",
        "The security coordinates adopt {new} on day {day}; the remaining coordinates persist.",
    ),
    TextDemand.ADDRESS: (
        "The entitlement leaves {entity} and becomes active for {other} as {new} on day {day}.",
        "After day {day}, {entity} no longer holds the entitlement; {other} carries {new}.",
    ),
    TextDemand.STATE_CONDITIONING: (
        "At day {day}, apply {new} only when the standing record is active; "
        "otherwise retain the record.",
        "The day-{day} notice takes effect as {new} conditional on an active standing record.",
    ),
}

_ACTION_TAXONOMY = ("HOLD", "AUTHORIZE", "ESCALATE", "MONITOR")
_ACTION_RULE = (
    "Take the largest coordinate index of the gold direct-fact vector; "
    "index modulo 4 maps 0=HOLD, 1=AUTHORIZE, 2=ESCALATE, 3=MONITOR."
)


def old_rule_status_for(
    *,
    demand: TextDemand,
    magnitude_operation: str,
    active: bool,
) -> OldRuleStatus:
    """Return the prospectively registered demand-aware old-rule gold."""

    if demand is TextDemand.MAGNITUDE:
        operation = MagnitudeOperation(magnitude_operation)
        return OldRuleStatus.FULL if operation is MagnitudeOperation.ADD else OldRuleStatus.NONE
    if demand is TextDemand.VALUE:
        return OldRuleStatus.PARTIAL
    if demand is TextDemand.ADDRESS:
        return OldRuleStatus.NONE
    return OldRuleStatus.NONE if active else OldRuleStatus.FULL


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _render_text(
    *,
    demand: TextDemand,
    magnitude_operation: str,
    entity: str,
    other: str,
    new_value_label: str,
    day: int,
    domain: str,
    template_index: int,
    evidence_index: int,
    paraphrase: bool,
) -> str:
    templates = _PARAPHRASE_TEMPLATES if paraphrase else _TRAIN_TEMPLATES
    if demand is TextDemand.MAGNITUDE:
        operation = MagnitudeOperation(magnitude_operation)
        if operation is MagnitudeOperation.ADD:
            evidence = (
                f"Alongside the standing rule, {new_value_label} also takes effect from day {day}."
            )
        elif operation is MagnitudeOperation.INVALIDATE:
            evidence = (
                f"A notice cites {new_value_label} for review. From day {day}, "
                "the standing rule has no continuing force and the cited policy "
                "remains pending."
            )
        else:
            evidence = (
                f"From day {day}, {new_value_label} governs and the former term "
                "has no continuing force."
            )
    else:
        evidence = _EVIDENCE[demand][evidence_index % len(_EVIDENCE[demand])].format(
            entity=entity,
            other=other,
            new=new_value_label,
            day=day,
        )
    return templates[template_index % len(templates)].format(
        entity=entity,
        evidence=f"In {domain}, {evidence}",
    )


def _direct_fact_answer(
    *,
    demand: TextDemand,
    target_row: torch.Tensor,
    old_value: torch.Tensor,
    new_value: torch.Tensor,
    old_value_label: str,
    new_value_label: str,
) -> str:
    if bool(torch.allclose(target_row, torch.zeros_like(target_row), atol=1.0e-7, rtol=0.0)):
        return "NO_ACTIVE_RULE"
    if bool(torch.allclose(target_row, old_value, atol=1.0e-7, rtol=0.0)):
        return old_value_label
    if bool(torch.allclose(target_row, new_value, atol=1.0e-7, rtol=0.0)):
        return new_value_label
    if demand is TextDemand.VALUE:
        return f"SECURITY_HALF={new_value_label};REMAINDER={old_value_label}"
    return f"COEXISTING_RULES={old_value_label}+{new_value_label}"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(text))


def lexical_leakage(text: str, blacklist: Iterable[str]) -> tuple[str, ...]:
    tokens = set(tokenize(text))
    return tuple(sorted({word.lower() for word in blacklist} & tokens))


def _seed(namespace_seed: int, *parts: object) -> int:
    payload = "::".join((str(namespace_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _vector(token: str, *, dimension: int, namespace_seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(namespace_seed, "value", token))
    vector = torch.randn(dimension, generator=generator)
    return cast(torch.Tensor, vector / vector.norm().clamp_min(1.0e-8))


def decode_visible_policy_candidate(
    text: str,
    *,
    dimension: int,
    semantic_value_seed: int,
) -> torch.Tensor:
    """Decode the one visible policy token through the registered fixed map."""

    labels = {match.group(0).lower() for match in _POLICY_TOKEN.finditer(text)}
    if not labels:
        return torch.zeros(dimension)
    if len(labels) != 1:
        raise ValueError("transaction text must expose exactly one policy candidate")
    return _vector(
        next(iter(labels)),
        dimension=dimension,
        namespace_seed=semantic_value_seed,
    )


def _entity_name(index: int, *, heldout: bool) -> str:
    first = (
        ("Cedar", "Juniper", "Maple", "Willow")
        if not heldout
        else (
            "Aurora",
            "Nimbus",
            "Solstice",
            "Zephyr",
        )
    )
    second = ("Kestrel", "Lynx", "Orca", "Raven")
    return f"{first[index % len(first)]}-{second[(index // len(first)) % len(second)]}"


def _target_for_demand(
    *,
    demand: TextDemand,
    state: torch.Tensor,
    erase_index: int,
    write_index: int,
    old_value: torch.Tensor,
    new_value: torch.Tensor,
    coordinate_mask: torch.Tensor,
    active: bool,
    magnitude_operation: str,
) -> torch.Tensor:
    target = state.clone()
    if demand is TextDemand.MAGNITUDE:
        operation = MagnitudeOperation(magnitude_operation)
        if operation is MagnitudeOperation.ADD:
            target[write_index] = old_value + new_value
        elif operation is MagnitudeOperation.INVALIDATE:
            target[erase_index].zero_()
        elif operation is MagnitudeOperation.SUPERSEDE:
            target[erase_index] = new_value
    elif demand is TextDemand.VALUE:
        target[erase_index] = coordinate_mask * new_value + (1.0 - coordinate_mask) * old_value
    elif demand is TextDemand.ADDRESS:
        target[erase_index].zero_()
        target[write_index] = new_value
    elif active:
        target[erase_index] = new_value
    return target


def build_text_transactions(
    *,
    split: TextSplit,
    demand_families: Sequence[TextDemand],
    count_per_demand: int,
    slots: int,
    value_dim: int,
    namespace_seed: int,
    semantic_value_seed: int,
    blacklist: Sequence[str],
) -> list[TextTransaction]:
    """Generate an outcome-independent, deterministic namespace."""

    if count_per_demand < 1 or slots < 4 or value_dim < 4:
        raise ValueError("text transaction dimensions are too small")
    heldout_identifier = split in {TextSplit.IDENTIFIER, TextSplit.COMBINED}
    heldout_domain = split in {TextSplit.DOMAIN, TextSplit.COMBINED}
    paraphrase = split in {TextSplit.PARAPHRASE, TextSplit.COMBINED}
    domains = _OOD_DOMAINS if heldout_domain else _TRAIN_DOMAINS
    rows: list[TextTransaction] = []
    for demand_index, demand in enumerate(demand_families):
        for item in range(count_per_demand):
            # A minimal-pair member may change the demand relation and its
            # required address, but never the base state, old/new values, or
            # effective day.  In particular, the private demand label cannot
            # change the RNG stream.
            semantic_item = item // 2
            local_seed = _seed(namespace_seed, split.value, "minimal-pair", item)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(local_seed)
            state = 0.20 * torch.randn(slots, value_dim, generator=generator)
            erase_index = int(semantic_item % slots)
            write_index = (
                int((erase_index + 1 + demand_index) % slots)
                if demand is TextDemand.ADDRESS
                else erase_index
            )
            old_token = f"standing-{(semantic_item * 3) % 19}"
            new_token = f"policy-{(semantic_item * 5 + 1) % 23}"
            new_value = _vector(
                new_token,
                dimension=value_dim,
                namespace_seed=semantic_value_seed,
            )
            old_basis = _vector(
                old_token,
                dimension=value_dim,
                namespace_seed=semantic_value_seed,
            )
            old_basis = old_basis - torch.dot(old_basis, new_value) * new_value
            old_basis = old_basis / old_basis.norm().clamp_min(1.0e-8)
            # Each surface form occurs once with an active state and once with
            # an inactive state.  The state branch is represented only by the
            # sign of A; template/entity/policy/day are identical.
            active = bool(item % 2 == 0)
            state[erase_index] = old_basis if active else -old_basis
            if demand is TextDemand.MAGNITUDE:
                if split in {
                    TextSplit.DEVELOPMENT,
                    TextSplit.TRAIN,
                    TextSplit.VALIDATION,
                }:
                    magnitude_operation = (
                        MagnitudeOperation.ADD.value
                        if semantic_item % 2 == 0
                        else MagnitudeOperation.INVALIDATE.value
                    )
                else:
                    magnitude_operation = tuple(MagnitudeOperation)[
                        semantic_item % len(MagnitudeOperation)
                    ].value
            else:
                magnitude_operation = "not_applicable"
            old_value = state[erase_index].clone()
            coordinate_mask = torch.zeros(value_dim)
            coordinate_mask[: value_dim // 2] = 1.0
            target = _target_for_demand(
                demand=demand,
                state=state,
                erase_index=erase_index,
                write_index=write_index,
                old_value=old_value,
                new_value=new_value,
                coordinate_mask=coordinate_mask,
                active=active,
                magnitude_operation=magnitude_operation,
            )
            memory_entities = tuple(
                _entity_name(slot, heldout=heldout_identifier) for slot in range(slots)
            )
            entity = memory_entities[erase_index]
            other = memory_entities[write_index]
            day = 30 + semantic_item % 300
            domain = domains[semantic_item % len(domains)]
            template_index = semantic_item % 2
            evidence_index = semantic_item % 2
            text = _render_text(
                demand=demand,
                magnitude_operation=magnitude_operation,
                entity=entity,
                other=other,
                new_value_label=new_token,
                day=day,
                domain=domain,
                template_index=template_index,
                evidence_index=evidence_index,
                paraphrase=paraphrase,
            )
            leaked = lexical_leakage(text, blacklist)
            if leaked:
                raise ValueError(f"Generated model text contains forbidden cues: {leaked}")
            example_id = hashlib.sha256(
                f"{split.value}:{demand.value}:{item}:{namespace_seed}".encode()
            ).hexdigest()[:20]
            minimal_pair_id = hashlib.sha256(
                f"{split.value}:minimal-pair:{item}:{namespace_seed}".encode()
            ).hexdigest()[:20]
            state_counterpair_id = hashlib.sha256(
                (
                    f"{split.value}:{demand.value}:state-counterpair:"
                    f"{semantic_item}:{namespace_seed}"
                ).encode()
            ).hexdigest()[:20]
            direct_fact_index = write_index if demand is TextDemand.ADDRESS else erase_index
            old_rule_status = old_rule_status_for(
                demand=demand,
                magnitude_operation=magnitude_operation,
                active=active,
            )
            derived_action = int(target[direct_fact_index].argmax().item() % 4)
            rows.append(
                TextTransaction(
                    example_id=example_id,
                    minimal_pair_id=minimal_pair_id,
                    state_counterpair_id=state_counterpair_id,
                    split=split,
                    demand=demand,
                    text=text,
                    query_direct=(
                        "What rule now applies to "
                        f"{other if demand is TextDemand.ADDRESS else entity}?"
                    ),
                    query_derived="Which downstream action follows from the current rule?",
                    query_old_rule=(
                        "What is the previously stored rule's post-transaction status: "
                        "FULL, PARTIAL, or NONE?"
                    ),
                    query_unaffected="Did unrelated active records remain unchanged?",
                    entity=entity,
                    other_entity=other,
                    domain=domain,
                    day=day,
                    active=active,
                    magnitude_operation=magnitude_operation,
                    old_value_label=old_token,
                    new_value_label=new_token,
                    candidate_decoder_seed=semantic_value_seed,
                    template_index=template_index,
                    evidence_index=evidence_index,
                    memory_entities=memory_entities,
                    state=state,
                    target_state=target,
                    erase_index=erase_index,
                    write_index=write_index,
                    old_value=old_value,
                    new_value=new_value,
                    coordinate_mask=coordinate_mask,
                    # The derived-action label follows the materialized
                    # post-transaction fact.  In particular, an inactive
                    # state-conditioned transaction is a genuine no-op and
                    # must not inherit the incoming proposal's action label.
                    derived_action=derived_action,
                    derived_action_label=_ACTION_TAXONOMY[derived_action],
                    derived_action_rule=_ACTION_RULE,
                    direct_fact_entity=memory_entities[direct_fact_index],
                    direct_fact_answer=_direct_fact_answer(
                        demand=demand,
                        target_row=target[direct_fact_index],
                        old_value=old_value,
                        new_value=new_value,
                        old_value_label=old_token,
                        new_value_label=new_token,
                    ),
                    old_rule_status=old_rule_status,
                )
            )
    return rows


def visible_registry_rows(examples: Sequence[TextTransaction]) -> list[dict[str, Any]]:
    """Return only model-visible/audit fields; private targets are represented by hashes."""

    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "example_id": example.example_id,
                "minimal_pair_id": example.minimal_pair_id,
                "state_counterpair_id": example.state_counterpair_id,
                "split": example.split.value,
                "demand_family_private_audit_only": example.demand.value,
                "magnitude_operation_private_audit_only": example.magnitude_operation,
                "text": example.text,
                "queries": {
                    "direct_fact": example.query_direct,
                    "derived_action": example.query_derived,
                    "old_rule_probe": example.query_old_rule,
                    "unaffected_retention": example.query_unaffected,
                },
                "memory_entities": list(example.memory_entities),
                "current_state_sha256": tensor_sha256(example.state),
                "private_target_sha256": tensor_sha256(example.target_state),
            }
        )
    return rows


def shuffled_texts(examples: Sequence[TextTransaction]) -> Mapping[str, str]:
    """Shuffle policy/day content while holding entity and address text fixed."""

    grouped: dict[tuple[TextSplit, TextDemand], list[TextTransaction]] = {}
    for example in examples:
        grouped.setdefault((example.split, example.demand), []).append(example)
    overrides: dict[str, str] = {}
    for group in grouped.values():
        for example in group:
            donor = next(
                (
                    candidate
                    for candidate in group
                    if (
                        candidate.new_value_label,
                        candidate.day,
                    )
                    != (
                        example.new_value_label,
                        example.day,
                    )
                ),
                None,
            )
            if donor is None:
                raise ValueError(
                    "shuffled-text control requires at least two distinct policy/day pairs"
                )
            overrides[example.example_id] = _render_text(
                demand=example.demand,
                magnitude_operation=example.magnitude_operation,
                entity=example.entity,
                other=example.other_entity,
                new_value_label=donor.new_value_label,
                day=donor.day,
                domain=example.domain,
                template_index=example.template_index,
                evidence_index=example.evidence_index,
                paraphrase=example.split in {TextSplit.PARAPHRASE, TextSplit.COMBINED},
            )
    return overrides


def wrong_entity_texts(examples: Sequence[TextTransaction]) -> Mapping[str, str]:
    """Change only entity/address strings, preserving policy semantics exactly."""

    overrides: dict[str, str] = {}
    for example in examples:
        wrong_entity_index = (example.erase_index + 1) % len(example.memory_entities)
        offset = (example.write_index - example.erase_index) % len(example.memory_entities)
        wrong_write_index = (wrong_entity_index + offset) % len(example.memory_entities)
        overrides[example.example_id] = _render_text(
            demand=example.demand,
            magnitude_operation=example.magnitude_operation,
            entity=example.memory_entities[wrong_entity_index],
            other=example.memory_entities[wrong_write_index],
            new_value_label=example.new_value_label,
            day=example.day,
            domain=example.domain,
            template_index=example.template_index,
            evidence_index=example.evidence_index,
            paraphrase=example.split in {TextSplit.PARAPHRASE, TextSplit.COMBINED},
        )
    return overrides
