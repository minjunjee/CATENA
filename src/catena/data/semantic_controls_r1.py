from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from catena.data.semantic_controls_v61 import (
    ControlPairing,
    ControlPairingRegistry,
)
from catena.data.semantic_transactions_v61 import (
    SafeSemanticRecord,
    SemanticExample,
    derive_operation,
)

_COHERENT_PREDICATE_FIELDS = (
    "current_relation",
    "incoming_evidence",
    "prior_version",
    "evidence_version",
    "observation_day",
    "evidence_timestamp_day",
    "prior_valid_from_day",
    "prior_valid_to_day",
    "evidence_valid_from_day",
    "evidence_valid_to_day",
    "scope",
)
_NUISANCE_FIELDS = ("source", "provenance")
_RELATION_IRRELEVANT_PREDICATE_FIELDS = (
    "incoming_evidence",
    "evidence_timestamp_day",
)


def _stable_index(identifier: str, label: str, size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot select from an empty R1 donor pool.")
    digest = hashlib.sha256(f"{identifier}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % size


def _different_donor(
    donors: Sequence[SemanticExample],
    *,
    excluded_id: str,
    identifier: str,
    label: str,
) -> SemanticExample:
    if len(donors) < 2:
        raise ValueError("R1 multi-donor derangement requires at least two donors.")
    start = _stable_index(identifier, label, len(donors))
    for offset in range(len(donors)):
        candidate = donors[(start + offset) % len(donors)]
        if candidate.example_id != excluded_id:
            return candidate
    raise ValueError("Could not select a distinct R1 semantic donor.")


def _replace_from_donors(
    recipient: SafeSemanticRecord,
    assignments: Mapping[str, SafeSemanticRecord],
) -> SafeSemanticRecord:
    return replace(
        recipient,
        **{
            field_name: getattr(donor, field_name)
            for field_name, donor in assignments.items()
        },
    )


def _visible_candidates(
    state: torch.Tensor,
    address: torch.Tensor,
    incoming_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_read = address @ state
    return torch.outer(address, state_read), torch.outer(address, incoming_value)


def _norm_match_scale(
    target: torch.Tensor,
    candidate: torch.Tensor,
    *,
    tolerance: float,
) -> tuple[float, float]:
    target_norm = float(torch.linalg.vector_norm(target).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate).item())
    if candidate_norm <= tolerance:
        if target_norm <= tolerance:
            return 0.0, 0.0
        raise ValueError("A near-zero R1 candidate cannot be norm matched.")
    scale = target_norm / candidate_norm
    mismatch = abs(
        float(torch.linalg.vector_norm(candidate * scale).item()) - target_norm
    )
    return scale, mismatch


def _group_key(example: SemanticExample) -> tuple[int, str, str]:
    return example.seed, example.domain, example.template


def build_control_pairing_registry_r1(
    examples: Sequence[SemanticExample],
    *,
    norm_tolerance: float = 1e-6,
) -> ControlPairingRegistry:
    """Build deterministic R1 controls without outcome-dependent search.

    The original fieldwise random search can be unidentifiable for a fully
    balanced factorial. R1 instead selects a coherent operation-changing donor
    first, then replaces two demand-irrelevant predicate fields and both
    nuisance fields from distinct donors. This is a deterministic multi-donor
    derangement; the operation-changing relation is fixed before evaluation.
    """

    if norm_tolerance <= 0.0:
        raise ValueError("norm_tolerance must be positive.")
    recipients = list(examples)
    if not recipients:
        raise ValueError("R1 control pairing requires nonempty examples.")

    groups: dict[tuple[int, str, str], list[SemanticExample]] = defaultdict(list)
    for example in recipients:
        groups[_group_key(example)].append(example)
    for pool in groups.values():
        pool.sort(key=lambda example: example.example_id)

    pairings: list[ControlPairing] = []
    for recipient in sorted(recipients, key=lambda example: example.example_id):
        eligible = [
            donor
            for donor in groups[_group_key(recipient)]
            if donor.example_id != recipient.example_id
            and donor.operation is not recipient.operation
        ]
        if len(eligible) < 2:
            raise ValueError(
                "R1 controls require at least two same-seed/domain/template "
                "operation-changing donors."
            )
        coherent = eligible[
            _stable_index(
                recipient.example_id,
                "r1-wrong-semantics",
                len(eligible),
            )
        ]
        secondary = _different_donor(
            eligible,
            excluded_id=coherent.example_id,
            identifier=recipient.example_id,
            label="r1-shuffled-secondary",
        )
        nuisance = _different_donor(
            eligible,
            excluded_id=coherent.example_id,
            identifier=recipient.example_id,
            label="r1-shuffled-nuisance",
        )

        coherent_assignments = {
            field_name: coherent.safe_record
            for field_name in _COHERENT_PREDICATE_FIELDS
        }
        wrong_semantics_record = _replace_from_donors(
            recipient.safe_record,
            coherent_assignments,
        )
        target_operation = derive_operation(wrong_semantics_record)
        if target_operation is recipient.operation:
            raise AssertionError("R1 coherent donor did not change operation.")

        shuffled_assignments = dict(coherent_assignments)
        for field_name in _RELATION_IRRELEVANT_PREDICATE_FIELDS:
            shuffled_assignments[field_name] = secondary.safe_record
        shuffled_assignments["source"] = nuisance.safe_record
        shuffled_assignments["provenance"] = secondary.safe_record
        shuffled_record = _replace_from_donors(
            recipient.safe_record,
            shuffled_assignments,
        )
        if derive_operation(shuffled_record) is not target_operation:
            raise AssertionError("R1 multi-donor derangement changed target operation.")

        donor_ids = tuple(
            (
                field_name,
                (
                    secondary.example_id
                    if field_name in _RELATION_IRRELEVANT_PREDICATE_FIELDS
                    else coherent.example_id
                ),
            )
            for field_name in _COHERENT_PREDICATE_FIELDS
        ) + (
            ("source", nuisance.example_id),
            ("provenance", secondary.example_id),
        )

        private = recipient.private_episode
        wrong_address_index = (private.affected_index + 1) % private.keys.shape[0]
        correct_address = private.keys[private.affected_index]
        wrong_address = private.keys[wrong_address_index]
        correct_erase, correct_write = _visible_candidates(
            private.state,
            correct_address,
            private.new_value,
        )
        wrong_erase, wrong_write = _visible_candidates(
            private.state,
            wrong_address,
            private.new_value,
        )
        erase_scale, erase_mismatch = _norm_match_scale(
            correct_erase,
            wrong_erase,
            tolerance=norm_tolerance,
        )
        write_scale, write_mismatch = _norm_match_scale(
            correct_write,
            wrong_write,
            tolerance=norm_tolerance,
        )
        if max(erase_mismatch, write_mismatch) > norm_tolerance:
            raise ValueError("R1 wrong-address norm mismatch exceeds tolerance.")

        pairings.append(
            ControlPairing(
                example_id=recipient.example_id,
                shuffled_record=shuffled_record,
                shuffled_field_donor_ids=donor_ids,
                wrong_semantics_record=wrong_semantics_record,
                wrong_semantics_donor_id=coherent.example_id,
                wrong_address_index=wrong_address_index,
                wrong_address_erase_scale=float(erase_scale),
                wrong_address_write_scale=float(write_scale),
                wrong_address_erase_norm_mismatch=float(erase_mismatch),
                wrong_address_write_norm_mismatch=float(write_mismatch),
                mappings_use_outcomes=False,
            )
        )

    return ControlPairingRegistry(
        pairings=tuple(pairings),
        norm_tolerance=float(norm_tolerance),
        mappings_use_outcomes=False,
    )


__all__ = ["build_control_pairing_registry_r1"]
