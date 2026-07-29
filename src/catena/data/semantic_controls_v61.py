from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

import torch

from catena.data.semantic_transactions_v61 import (
    ALLOWED_SAFE_FIELDS,
    SafeSemanticRecord,
    SemanticExample,
    derive_operation,
)


class SemanticControl(StrEnum):
    FULL = "full"
    TRANSACTION_ONLY = "transaction_only"
    STATE_ONLY = "state_only"
    SHUFFLED_FIELDS = "shuffled_fields"
    WRONG_ADDRESS = "wrong_address"
    WRONG_SEMANTICS = "wrong_semantics"


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
_DERANGED_NUISANCE_FIELDS = ("source", "provenance")
_PRESERVED_CONTROL_FIELDS = (
    "entity_description",
    "domain",
    "incoming_value_token",
    "template_surface",
)

if set(_COHERENT_PREDICATE_FIELDS + _DERANGED_NUISANCE_FIELDS) | set(
    _PRESERVED_CONTROL_FIELDS
) != set(ALLOWED_SAFE_FIELDS):
    raise AssertionError("Semantic control field partition differs from the frozen schema.")


@dataclass(frozen=True, slots=True)
class VisibleUpdateContext:
    """Only the public state, address, and incoming candidate used by an update."""

    visible_state: torch.Tensor
    visible_address: torch.Tensor
    incoming_value: torch.Tensor
    erase_candidate_scale: float = 1.0
    write_candidate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.visible_state.ndim != 2:
            raise ValueError("visible_state must be a matrix.")
        if self.visible_address.shape != (self.visible_state.shape[0],):
            raise ValueError("visible_address has an incompatible shape.")
        if self.incoming_value.shape != (self.visible_state.shape[1],):
            raise ValueError("incoming_value has an incompatible shape.")
        if (
            self.visible_state.device != self.visible_address.device
            or self.visible_state.device != self.incoming_value.device
        ):
            raise ValueError("Visible update tensors must share a device.")
        if (
            self.visible_state.dtype != self.visible_address.dtype
            or self.visible_state.dtype != self.incoming_value.dtype
        ):
            raise ValueError("Visible update tensors must share a dtype.")
        for name in ("visible_state", "visible_address", "incoming_value"):
            if not bool(torch.isfinite(getattr(self, name)).all().item()):
                raise FloatingPointError(f"{name} contains a non-finite value.")
        for name in ("erase_candidate_scale", "write_candidate_scale"):
            value = getattr(self, name)
            if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

    @property
    def address_resolved_state_read(self) -> torch.Tensor:
        return self.visible_address @ self.visible_state


@dataclass(frozen=True, slots=True)
class SemanticControlView:
    semantic_record: SafeSemanticRecord | None
    update_context: VisibleUpdateContext

    @property
    def semantic_fields_visible(self) -> bool:
        return self.semantic_record is not None


@dataclass(frozen=True, slots=True)
class ControlPairing:
    example_id: str
    shuffled_record: SafeSemanticRecord
    shuffled_field_donor_ids: tuple[tuple[str, str], ...]
    wrong_semantics_record: SafeSemanticRecord
    wrong_semantics_donor_id: str
    wrong_address_index: int
    wrong_address_erase_scale: float
    wrong_address_write_scale: float
    wrong_address_erase_norm_mismatch: float
    wrong_address_write_norm_mismatch: float
    mappings_use_outcomes: bool = False

    def __post_init__(self) -> None:
        if self.mappings_use_outcomes:
            raise ValueError("Semantic control mappings may not use outcomes.")
        if not self.example_id or not self.wrong_semantics_donor_id:
            raise ValueError("Control pairing identifiers must be nonempty.")
        if len(self.shuffled_field_donor_ids) != len(
            _COHERENT_PREDICATE_FIELDS + _DERANGED_NUISANCE_FIELDS
        ):
            raise ValueError("Shuffled field donor registry is incomplete.")
        donor_fields = tuple(field_name for field_name, _ in self.shuffled_field_donor_ids)
        if donor_fields != _COHERENT_PREDICATE_FIELDS + _DERANGED_NUISANCE_FIELDS:
            raise ValueError("Shuffled field donor order differs from the frozen schema.")
        if any(donor_id == self.example_id for _, donor_id in self.shuffled_field_donor_ids):
            raise ValueError("Shuffled field mapping contains a self-map.")
        predicate_donor_ids = {
            donor_id
            for field_name, donor_id in self.shuffled_field_donor_ids
            if field_name in _COHERENT_PREDICATE_FIELDS
        }
        if len(predicate_donor_ids) < 2:
            raise ValueError(
                "Shuffled predicate fields must come from at least two donors."
            )
        if self.wrong_semantics_donor_id == self.example_id:
            raise ValueError("Wrong-semantics mapping contains a self-map.")
        if derive_operation(self.shuffled_record) is not derive_operation(
            self.wrong_semantics_record
        ):
            raise ValueError(
                "Shuffled and wrong-semantics controls must induce the same demand."
            )
        for name in (
            "wrong_address_erase_scale",
            "wrong_address_write_scale",
            "wrong_address_erase_norm_mismatch",
            "wrong_address_write_norm_mismatch",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")

    @property
    def maximum_wrong_address_norm_mismatch(self) -> float:
        return max(
            self.wrong_address_erase_norm_mismatch,
            self.wrong_address_write_norm_mismatch,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "shuffled_record": asdict(self.shuffled_record),
            "shuffled_field_donor_ids": [
                {"field": field_name, "donor_example_id": donor_id}
                for field_name, donor_id in self.shuffled_field_donor_ids
            ],
            "wrong_semantics_record": asdict(self.wrong_semantics_record),
            "wrong_semantics_donor_id": self.wrong_semantics_donor_id,
            "wrong_address_index": self.wrong_address_index,
            "wrong_address_erase_scale": self.wrong_address_erase_scale,
            "wrong_address_write_scale": self.wrong_address_write_scale,
            "wrong_address_erase_norm_mismatch": (
                self.wrong_address_erase_norm_mismatch
            ),
            "wrong_address_write_norm_mismatch": (
                self.wrong_address_write_norm_mismatch
            ),
            "mappings_use_outcomes": self.mappings_use_outcomes,
        }


@dataclass(frozen=True, slots=True)
class ControlPairingRegistry:
    pairings: tuple[ControlPairing, ...]
    norm_tolerance: float
    mappings_use_outcomes: bool = False
    _pairing_by_id: Mapping[str, ControlPairing] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.mappings_use_outcomes:
            raise ValueError("Semantic control registry may not use outcomes.")
        if self.norm_tolerance <= 0.0:
            raise ValueError("norm_tolerance must be positive.")
        identifiers = [pairing.example_id for pairing in self.pairings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Control pairing registry contains duplicate recipients.")
        object.__setattr__(
            self,
            "_pairing_by_id",
            MappingProxyType(
                {pairing.example_id: pairing for pairing in self.pairings}
            ),
        )
        if any(
            pairing.maximum_wrong_address_norm_mismatch > self.norm_tolerance
            for pairing in self.pairings
        ):
            raise ValueError("Wrong-address norm mismatch exceeds the registered tolerance.")

    def for_example(self, example: SemanticExample | str) -> ControlPairing:
        example_id = example if isinstance(example, str) else example.example_id
        try:
            return self._pairing_by_id[example_id]
        except KeyError:
            raise KeyError(
                f"No control pairing is registered for {example_id!r}."
            ) from None

    def to_rows(self) -> list[dict[str, object]]:
        return [pairing.to_dict() for pairing in self.pairings]

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_rows(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _stable_index(identifier: str, label: str, size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot select from an empty donor pool.")
    digest = hashlib.sha256(f"{identifier}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % size


def _record_with_donor_fields(
    recipient: SafeSemanticRecord,
    assignments: Mapping[str, SafeSemanticRecord],
) -> SafeSemanticRecord:
    updates = {
        field_name: getattr(donor, field_name)
        for field_name, donor in assignments.items()
    }
    return replace(recipient, **updates)


def _fieldwise_shuffled_record(
    recipient: SemanticExample,
    eligible: Sequence[SemanticExample],
    *,
    target_operation: object,
) -> tuple[SafeSemanticRecord, tuple[tuple[str, str], ...]]:
    """Build a deterministic multi-donor derangement with a matched demand.

    The search depends only on frozen identifiers and semantic fields. It never
    reads model outputs, losses, or target-state errors.
    """

    fields_to_derange = _COHERENT_PREDICATE_FIELDS + _DERANGED_NUISANCE_FIELDS
    for attempt in range(4096):
        assignments: dict[str, SafeSemanticRecord] = {}
        donor_ids: list[tuple[str, str]] = []
        for field_name in fields_to_derange:
            donor = eligible[
                _stable_index(
                    recipient.example_id,
                    f"shuffled-{attempt}-{field_name}",
                    len(eligible),
                )
            ]
            assignments[field_name] = donor.safe_record
            donor_ids.append((field_name, donor.example_id))
        predicate_donor_ids = {
            donor_id
            for field_name, donor_id in donor_ids
            if field_name in _COHERENT_PREDICATE_FIELDS
        }
        if len(predicate_donor_ids) < 2:
            continue
        try:
            shuffled_record = _record_with_donor_fields(
                recipient.safe_record,
                assignments,
            )
        except ValueError:
            continue
        if derive_operation(shuffled_record) is target_operation:
            return shuffled_record, tuple(donor_ids)
    raise ValueError(
        "Could not construct the frozen operation-matched fieldwise derangement "
        f"for {recipient.example_id}."
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
        raise ValueError("A near-zero wrong-address candidate cannot be norm matched.")
    scale = target_norm / candidate_norm
    matched_norm = float(torch.linalg.vector_norm(candidate * scale).item())
    return scale, abs(matched_norm - target_norm)


def _group_key(example: SemanticExample) -> tuple[int, str, str]:
    return example.seed, example.domain, example.template


def build_control_pairing_registry(
    examples: Sequence[SemanticExample],
    *,
    semantic_donors: Sequence[SemanticExample] | None = None,
    norm_tolerance: float = 1e-6,
) -> ControlPairingRegistry:
    """Freeze outcome-independent semantic and wrong-address mappings."""

    if norm_tolerance <= 0.0:
        raise ValueError("norm_tolerance must be positive.")
    recipients = list(examples)
    donors = list(semantic_donors if semantic_donors is not None else examples)
    if not recipients or not donors:
        raise ValueError("Control pairing requires recipients and semantic donors.")
    donor_groups: dict[tuple[int, str, str], list[SemanticExample]] = defaultdict(list)
    for donor in donors:
        donor_groups[_group_key(donor)].append(donor)
    for pool in donor_groups.values():
        pool.sort(key=lambda example: example.example_id)

    pairings: list[ControlPairing] = []
    for recipient in sorted(recipients, key=lambda example: example.example_id):
        eligible = [
            donor
            for donor in donor_groups.get(_group_key(recipient), [])
            if donor.example_id != recipient.example_id
            and donor.operation is not recipient.operation
        ]
        if not eligible:
            raise ValueError(
                "No same-seed/domain/template operation-changing semantic donor exists "
                f"for {recipient.example_id}."
            )

        wrong_donor = eligible[
            _stable_index(recipient.example_id, "wrong-semantics", len(eligible))
        ]
        coherent_assignments = {
            field_name: wrong_donor.safe_record
            for field_name in _COHERENT_PREDICATE_FIELDS
        }
        wrong_semantics_record = _record_with_donor_fields(
            recipient.safe_record,
            coherent_assignments,
        )
        if derive_operation(wrong_semantics_record) is recipient.operation:
            raise AssertionError("Wrong-semantics donor did not change the derived demand.")

        shuffled_record, shuffled_donor_ids = _fieldwise_shuffled_record(
            recipient,
            eligible,
            target_operation=wrong_donor.operation,
        )
        if derive_operation(shuffled_record) is recipient.operation:
            raise AssertionError("Shuffled mapping did not change the derived demand.")
        if derive_operation(shuffled_record) is not derive_operation(
            wrong_semantics_record
        ):
            raise AssertionError(
                "Shuffled and wrong-semantics mappings induce different demands."
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
            raise ValueError("Wrong-address norm match exceeds the frozen tolerance.")

        pairings.append(
            ControlPairing(
                example_id=recipient.example_id,
                shuffled_record=shuffled_record,
                shuffled_field_donor_ids=shuffled_donor_ids,
                wrong_semantics_record=wrong_semantics_record,
                wrong_semantics_donor_id=wrong_donor.example_id,
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


def _normalize_control(control: SemanticControl | str) -> SemanticControl:
    return control if isinstance(control, SemanticControl) else SemanticControl(control)


def semantic_record_for_control(
    example: SemanticExample,
    control: SemanticControl | str,
    pairing: ControlPairing | None = None,
) -> SafeSemanticRecord | None:
    selected = _normalize_control(control)
    if pairing is not None and pairing.example_id != example.example_id:
        raise ValueError("Control pairing belongs to a different recipient.")
    if selected is SemanticControl.STATE_ONLY:
        return None
    if selected is SemanticControl.SHUFFLED_FIELDS:
        if pairing is None:
            raise ValueError("Shuffled-fields control requires a frozen pairing.")
        return pairing.shuffled_record
    if selected is SemanticControl.WRONG_SEMANTICS:
        if pairing is None:
            raise ValueError("Wrong-semantics control requires a frozen pairing.")
        return pairing.wrong_semantics_record
    return example.safe_record


def build_visible_update_context(
    example: SemanticExample,
    control: SemanticControl | str,
    pairing: ControlPairing | None = None,
) -> VisibleUpdateContext:
    """Build a public update context without exposing private targets or demand."""

    selected = _normalize_control(control)
    if pairing is not None and pairing.example_id != example.example_id:
        raise ValueError("Control pairing belongs to a different recipient.")
    private = example.private_episode
    address_index = private.affected_index
    state = private.state.clone()
    incoming_value = private.new_value.clone()
    erase_scale = 1.0
    write_scale = 1.0

    if selected is SemanticControl.TRANSACTION_ONLY:
        state = torch.zeros_like(state)
    elif selected is SemanticControl.STATE_ONLY:
        incoming_value = torch.zeros_like(incoming_value)
    elif selected is SemanticControl.WRONG_ADDRESS:
        if pairing is None:
            raise ValueError("Wrong-address control requires a frozen pairing.")
        address_index = pairing.wrong_address_index
        erase_scale = pairing.wrong_address_erase_scale
        write_scale = pairing.wrong_address_write_scale

    return VisibleUpdateContext(
        visible_state=state,
        visible_address=private.keys[address_index].clone(),
        incoming_value=incoming_value,
        erase_candidate_scale=float(erase_scale),
        write_candidate_scale=float(write_scale),
    )


def build_control_view(
    example: SemanticExample,
    control: SemanticControl | str,
    pairing: ControlPairing | None = None,
) -> SemanticControlView:
    return SemanticControlView(
        semantic_record=semantic_record_for_control(example, control, pairing),
        update_context=build_visible_update_context(example, control, pairing),
    )


def build_visible_candidates(
    context: VisibleUpdateContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    erase_candidate, write_candidate = _visible_candidates(
        context.visible_state,
        context.visible_address,
        context.incoming_value,
    )
    return (
        erase_candidate * context.erase_candidate_scale,
        write_candidate * context.write_candidate_scale,
    )


def _scalar_gate(
    value: torch.Tensor | float,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.numel() != 1:
        raise ValueError(f"{name} gate must be scalar.")
    result = result.reshape(())
    if not bool(torch.isfinite(result).item()):
        raise FloatingPointError(f"{name} gate is non-finite.")
    if not 0.0 <= float(result.item()) <= 1.0:
        raise ValueError(f"{name} gate lies outside [0,1].")
    return result


def apply_visible_update(
    context: VisibleUpdateContext,
    erase: torch.Tensor | float,
    write: torch.Tensor | float,
) -> torch.Tensor:
    """Apply gates using only the model-visible context and derived candidates."""

    erase_gate = _scalar_gate(erase, reference=context.visible_state, name="erase")
    write_gate = _scalar_gate(write, reference=context.visible_state, name="write")
    erase_candidate, write_candidate = build_visible_candidates(context)
    return (
        context.visible_state
        - erase_gate * erase_candidate
        + write_gate * write_candidate
    )
