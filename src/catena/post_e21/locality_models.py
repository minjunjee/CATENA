from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from catena.data.structured_sequence_localization import (
    StructuredSequenceTransferBatch,
    StructuredTransferCondition,
    concatenate_structured_event_features,
)
from catena.models.structured_sequence_localization import (
    MatchedStructuredSequenceController,
    StructuredSequenceFreedom,
    StructuredSequenceTransferOutput,
)


@dataclass(slots=True)
class LocalitySequenceTransferOutput(StructuredSequenceTransferOutput):
    """E21-compatible output plus the route actually used by E22."""

    applied_route_mask: torch.Tensor
    applied_update_deltas: torch.Tensor


class LocalityStructuredSequenceController(MatchedStructuredSequenceController):
    """Instrumented E21 controller with optional hard-forward top-k routing."""

    def __init__(
        self,
        *,
        freedom: StructuredSequenceFreedom,
        slots: int,
        identifier_dim: int,
        value_dim: int,
        hidden_dim: int,
        address_temperature: float,
        active_fraction: float | None = None,
    ) -> None:
        super().__init__(
            freedom=freedom,
            slots=slots,
            identifier_dim=identifier_dim,
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            address_temperature=address_temperature,
        )
        if active_fraction is not None and not 0.0 < active_fraction <= 1.0:
            raise ValueError("active_fraction must lie in (0, 1]")
        self.active_fraction = active_fraction

    def _apply_route(
        self,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the frozen hard-forward/soft-backward top-k estimator."""

        if self.active_fraction is None:
            return weights, weights.detach() > 0.0
        support = max(1, math.ceil(float(self.active_fraction) * self.slots))
        indices = (
            weights.detach()
            .topk(
                support,
                dim=-1,
                largest=True,
                sorted=False,
            )
            .indices
        )
        hard_mask = torch.zeros_like(weights).scatter_(-1, indices, 1.0)
        # Forward is an exact binary top-k mask. Backward follows the soft
        # address weights, which keeps the registered sparse route trainable.
        straight_through_mask = hard_mask + weights - weights.detach()
        routed = weights * straight_through_mask
        routed = routed / routed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return routed, routed.detach() > 0.0

    def _project_event_update(
        self,
        *,
        batch: StructuredSequenceTransferBatch,
        time_index: int,
        update_delta: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the event update and applied route for ordinary controllers."""

        del batch, time_index
        return update_delta, route_mask

    def forward(
        self,
        batch: StructuredSequenceTransferBatch,
        condition: StructuredTransferCondition,
    ) -> LocalitySequenceTransferOutput:
        event_features = concatenate_structured_event_features(batch.inputs)
        state = batch.inputs.initial_state.clone()
        batch_size, sequence_length = batch.erase_addresses.shape
        erase_weights_rows: list[torch.Tensor] = []
        write_weights_rows: list[torch.Tensor] = []
        candidate_rows: list[torch.Tensor] = []
        activity_rows: list[torch.Tensor] = []
        raw_address_rows: list[torch.Tensor] = []
        raw_candidate_rows: list[torch.Tensor] = []
        raw_activity_rows: list[torch.Tensor] = []
        route_mask_rows: list[torch.Tensor] = []
        update_delta_rows: list[torch.Tensor] = []

        for time_index in range(sequence_length):
            hidden = self.event_encoder(event_features[:, time_index])
            raw_address = self.address_head(hidden).view(
                batch_size,
                2,
                self.slots,
            )
            raw_candidate = self.candidate_head(hidden)
            raw_activity = self.activity_head(hidden).squeeze(-1)

            if self.freedom.has_separate_address:
                erase_logits = raw_address[:, 0]
                write_logits = raw_address[:, 1]
            else:
                shared_logits = raw_address.mean(dim=1)
                erase_logits = shared_logits
                write_logits = shared_logits

            if condition.uses_oracle_address:
                erase_weights = nn.functional.one_hot(
                    batch.erase_addresses[:, time_index],
                    num_classes=self.slots,
                ).to(state.dtype)
                write_weights = nn.functional.one_hot(
                    batch.write_addresses[:, time_index],
                    num_classes=self.slots,
                ).to(state.dtype)
            else:
                erase_weights = torch.softmax(
                    erase_logits / self.address_temperature,
                    dim=-1,
                )
                write_weights = torch.softmax(
                    write_logits / self.address_temperature,
                    dim=-1,
                )

            erase_weights, erase_route_mask = self._apply_route(erase_weights)
            write_weights, write_route_mask = self._apply_route(write_weights)
            state_read = torch.einsum("bs,bsv->bv", erase_weights, state)
            if condition.uses_oracle_candidate:
                erase_candidate = batch.old_candidates[:, time_index]
            elif self.freedom.has_state_read:
                erase_candidate = state_read
            else:
                erase_candidate = raw_candidate

            erase_gate, write_gate = self._visible_demand_gates(
                family_one_hot=batch.inputs.family_one_hot[:, time_index],
                operation_one_hot=batch.inputs.operation_one_hot[:, time_index],
                channel_mask=batch.inputs.channel_masks[:, time_index],
                erase_candidate=erase_candidate,
            )
            activity = torch.sigmoid(raw_activity)
            erase_delta = activity[:, None] * erase_gate * erase_candidate
            write_delta = (
                activity[:, None] * write_gate * batch.inputs.new_candidates[:, time_index]
            )
            update_delta = (
                -erase_weights[:, :, None] * erase_delta[:, None, :]
                + write_weights[:, :, None] * write_delta[:, None, :]
            )
            route_mask = torch.stack((erase_route_mask, write_route_mask), dim=1)
            update_delta, route_mask = self._project_event_update(
                batch=batch,
                time_index=time_index,
                update_delta=update_delta,
                route_mask=route_mask,
            )
            state = state + update_delta

            erase_weights_rows.append(erase_weights)
            write_weights_rows.append(write_weights)
            candidate_rows.append(erase_candidate)
            activity_rows.append(activity)
            raw_address_rows.append(raw_address)
            raw_candidate_rows.append(raw_candidate)
            raw_activity_rows.append(raw_activity)
            route_mask_rows.append(route_mask)
            update_delta_rows.append(update_delta)

        return LocalitySequenceTransferOutput(
            state=state,
            erase_address_weights=torch.stack(erase_weights_rows, dim=1),
            write_address_weights=torch.stack(write_weights_rows, dim=1),
            erase_candidates=torch.stack(candidate_rows, dim=1),
            activity_gates=torch.stack(activity_rows, dim=1),
            raw_address_logits=torch.stack(raw_address_rows, dim=1),
            raw_candidates=torch.stack(raw_candidate_rows, dim=1),
            raw_activity_logits=torch.stack(raw_activity_rows, dim=1),
            applied_route_mask=torch.stack(route_mask_rows, dim=1),
            applied_update_deltas=torch.stack(update_delta_rows, dim=1),
        )


class ProtectedLocalityDiagnosticController(LocalityStructuredSequenceController):
    """Oracle projection upper bound; never eligible for selection.

    The associative readout basis is the state-slot basis.  Removing every
    active non-target readout direction therefore means retaining an event
    update only on its registered erase/write target slots.  Distractor events
    have no registered target and are projected to zero.  Projection occurs
    before recurrent state carry, not as a post-hoc evaluation mask.
    """

    def _project_event_update(
        self,
        *,
        batch: StructuredSequenceTransferBatch,
        time_index: int,
        update_delta: torch.Tensor,
        route_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        verified = batch.update_mask[:, time_index]
        erase = nn.functional.one_hot(
            batch.erase_addresses[:, time_index],
            num_classes=self.slots,
        ).to(torch.bool)
        write = nn.functional.one_hot(
            batch.write_addresses[:, time_index],
            num_classes=self.slots,
        ).to(torch.bool)
        target_subspace = (erase | write) & verified[:, None]
        projected_update = update_delta * target_subspace[:, :, None].to(update_delta.dtype)
        projected_route = route_mask & target_subspace[:, None, :]
        return projected_update, projected_route
