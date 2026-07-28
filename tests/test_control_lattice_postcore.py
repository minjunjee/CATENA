from __future__ import annotations

import torch

from catena.data.control_lattice import (
    GRANULARITY_START_DESCRIPTOR_INDEX,
    GRANULARITY_WIDTH_DESCRIPTOR_INDEX,
    DemandAxis,
    generate_control_lattice_batch,
)
from catena.models.lattice_controllers import (
    ControlFreedom,
    LatticeOutput,
    MatchedControlLatticeController,
)
from catena.training.lattice_training import evaluate_lattice_controller


def test_control_lattice_batch_and_controller() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    batch = generate_control_lattice_batch(
        family=DemandAxis.ADDRESS,
        batch_size=6,
        slots=5,
        value_dim=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    assert batch.state.shape == (6, 5, 8)
    model = MatchedControlLatticeController(
        freedom=ControlFreedom.SEPARATE_ADDRESS,
        descriptor_dim=12,
        value_dim=8,
        hidden_dim=16,
    )
    output = model(batch)
    assert output.state.shape == batch.state.shape
    assert torch.isfinite(output.state).all()


def test_granularity_descriptor_identifies_the_complete_contiguous_mask() -> None:
    value_dim = 16
    generator = torch.Generator(device="cpu")
    generator.manual_seed(19)
    batch = generate_control_lattice_batch(
        family=DemandAxis.GRANULARITY,
        batch_size=64,
        slots=5,
        value_dim=value_dim,
        generator=generator,
        device=torch.device("cpu"),
    )

    widths = torch.round(batch.descriptor[:, GRANULARITY_WIDTH_DESCRIPTOR_INDEX] * value_dim).to(
        torch.long
    )
    starts = torch.round(
        batch.descriptor[:, GRANULARITY_START_DESCRIPTOR_INDEX] * (value_dim - 1)
    ).to(torch.long)
    reconstructed = torch.zeros_like(batch.erase_mask)
    for row, (start, width) in enumerate(zip(starts.tolist(), widths.tolist(), strict=True)):
        reconstructed[row, start : start + width] = 1.0

    assert torch.equal(reconstructed, batch.erase_mask)
    assert torch.equal(reconstructed, batch.write_mask)
    assert torch.unique(starts).numel() > 1


def test_state_conditioned_target_uses_the_value_read_by_state_aware_control() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    batch = generate_control_lattice_batch(
        family=DemandAxis.STATE_CONDITIONED,
        batch_size=128,
        slots=7,
        value_dim=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    batch_index = torch.arange(batch.state.shape[0])
    marker = (batch.old_value[:, 0] > 0).to(batch.state.dtype)
    expected = batch.state.clone()
    expected[batch_index, batch.erase_address] = (
        batch.old_value
        - marker[:, None] * batch.old_value
        + (1.0 - marker[:, None]) * batch.new_value
    )

    assert torch.equal(
        batch.state[batch_index, batch.erase_address],
        batch.old_value,
    )
    assert torch.allclose(batch.target, expected)

    # This guards against regressing to the unrelated global slot-zero marker.
    unrelated_marker = batch.state[:, 0, 0] > 0
    assert torch.any(unrelated_marker != marker.bool())


def test_only_state_aware_controller_conditions_gates_on_the_affected_old_value() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(39)
    batch = generate_control_lattice_batch(
        family=DemandAxis.STATE_CONDITIONED,
        batch_size=16,
        slots=5,
        value_dim=8,
        generator=generator,
        device=torch.device("cpu"),
    )

    captured: dict[ControlFreedom, torch.Tensor] = {}
    for freedom in (ControlFreedom.SEPARATE_ADDRESS, ControlFreedom.STATE_AWARE):
        model = MatchedControlLatticeController(
            freedom=freedom,
            descriptor_dim=12,
            value_dim=8,
            hidden_dim=16,
        )

        def capture_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            key: ControlFreedom = freedom,
        ) -> None:
            captured[key] = inputs[0].detach().clone()

        handle = model.encoder[0].register_forward_pre_hook(capture_input)
        model(batch)
        handle.remove()

    assert torch.equal(
        captured[ControlFreedom.SEPARATE_ADDRESS][:, 12:],
        torch.zeros_like(batch.old_value),
    )
    assert torch.equal(captured[ControlFreedom.STATE_AWARE][:, 12:], batch.old_value)


def test_retention_metric_counts_the_actual_unaffected_slots() -> None:
    class UnitRetentionError(torch.nn.Module):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            state = batch.target.clone()
            row = torch.arange(state.shape[0])
            unaffected = torch.ones(
                state.shape[:2],
                dtype=torch.bool,
                device=state.device,
            )
            unaffected[row, batch.erase_address] = False
            unaffected[row, batch.write_address] = False
            state[unaffected] += 1.0
            gate = torch.zeros(state.shape[0], 1, device=state.device)
            return LatticeOutput(state=state, erase_gate=gate, write_gate=gate)

    metrics = evaluate_lattice_controller(
        model=UnitRetentionError(),  # type: ignore[arg-type]
        family=DemandAxis.MAGNITUDE,
        episodes=32,
        batch_size=8,
        slots=5,
        value_dim=8,
        device=torch.device("cpu"),
        seed=49,
    )

    assert metrics["affected_mse"] == 0.0
    assert abs(metrics["retention_mse"] - 1.0) < 1e-7


def test_all_control_freedoms_can_start_from_identical_maximal_parameters() -> None:
    states: list[dict[str, torch.Tensor]] = []
    for freedom in ControlFreedom:
        torch.manual_seed(1234)
        model = MatchedControlLatticeController(
            freedom=freedom,
            descriptor_dim=12,
            value_dim=8,
            hidden_dim=16,
        )
        states.append(
            {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        )

    reference = states[0]
    for state in states[1:]:
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[name], reference[name]) for name in reference)
