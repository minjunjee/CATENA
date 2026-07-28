from __future__ import annotations

import torch

from catena.data.localization_candidate import (
    LocalizationCandidateCondition,
    generate_localization_candidate_batch,
    make_address_codebook,
)
from catena.models.localization_candidate import (
    LocalizationCandidateFreedom,
    MatchedLocalizationCandidateController,
)
from catena.training.localization_candidate import (
    evaluate_localization_candidate_controller,
)
from experiments.e19b_localization_candidate_aggregate import (
    _gain_gate,
    compute_seed_contrasts,
)


def _batch(*, batch_size: int = 32):
    codebook = make_address_codebook(slots=6, code_dim=8, seed=19)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    batch = generate_localization_candidate_batch(
        batch_size=batch_size,
        slots=6,
        value_dim=10,
        state_scale=0.5,
        address_codebook=codebook,
        generator=generator,
        device=torch.device("cpu"),
    )
    return batch, codebook


def _model(
    freedom: LocalizationCandidateFreedom,
) -> MatchedLocalizationCandidateController:
    return MatchedLocalizationCandidateController(
        freedom=freedom,
        descriptor_dim=2 * 8 + 10,
        slots=6,
        value_dim=10,
        hidden_dim=16,
        address_temperature=0.5,
    )


def test_localization_candidate_data_contract() -> None:
    batch, _codebook = _batch()
    row = torch.arange(batch.state.shape[0])
    assert torch.all(batch.erase_address != batch.write_address)
    assert torch.equal(
        batch.old_candidate,
        batch.state[row, batch.erase_address],
    )
    assert torch.equal(batch.descriptor[:, -10:], batch.new_candidate)
    expected = batch.state.clone()
    expected[row, batch.erase_address] -= batch.old_candidate
    expected[row, batch.write_address] += batch.new_candidate
    assert torch.equal(batch.target, expected)


def test_oracle_address_and_candidate_are_an_exact_positive_control() -> None:
    batch, _codebook = _batch()
    for freedom in LocalizationCandidateFreedom:
        model = _model(freedom)
        output = model(
            batch,
            LocalizationCandidateCondition.A_ORACLE_ADDRESS_ORACLE_CANDIDATE,
        )
        assert torch.allclose(output.state, batch.target)
        assert torch.equal(output.erase_candidate, batch.old_candidate)


def test_projection_freedoms_share_one_maximal_parameter_surface() -> None:
    states: list[dict[str, torch.Tensor]] = []
    for freedom in LocalizationCandidateFreedom:
        torch.manual_seed(1901)
        model = _model(freedom)
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


def test_shared_address_and_state_read_projections_are_explicit() -> None:
    batch, _codebook = _batch()
    base = _model(LocalizationCandidateFreedom.BASE)
    state_aware = _model(LocalizationCandidateFreedom.STATE_AWARE)
    state_aware.load_state_dict(base.state_dict())
    condition = (
        LocalizationCandidateCondition.D_LEARNED_ADDRESS_STATE_READ_CANDIDATE
    )
    base_output = base(batch, condition)
    state_output = state_aware(batch, condition)
    assert torch.equal(
        base_output.erase_address_weights,
        base_output.write_address_weights,
    )
    assert torch.equal(
        state_output.erase_address_weights,
        state_output.write_address_weights,
    )
    expected_read = torch.einsum(
        "bs,bsv->bv",
        state_output.erase_address_weights,
        batch.state,
    )
    assert torch.allclose(state_output.erase_candidate, expected_read)
    assert torch.equal(base_output.erase_candidate, base_output.raw_candidate)


def test_evaluation_exposes_all_registered_metrics() -> None:
    _batch_value, codebook = _batch()
    model = _model(LocalizationCandidateFreedom.FULL)
    metrics = evaluate_localization_candidate_controller(
        model=model,
        condition=(
            LocalizationCandidateCondition.A_ORACLE_ADDRESS_ORACLE_CANDIDATE
        ),
        episodes=32,
        batch_size=8,
        slots=6,
        value_dim=10,
        state_scale=0.5,
        address_codebook=codebook,
        device=torch.device("cpu"),
        seed=39,
    )
    assert set(metrics) == {
        "state_mse",
        "address_accuracy",
        "candidate_recovery_mse",
        "affected_mse",
        "retention_mse",
        "old_residual",
    }
    assert metrics["address_accuracy"] == 1.0
    assert metrics["candidate_recovery_mse"] == 0.0
    assert metrics["affected_mse"] < 1e-12
    assert metrics["retention_mse"] == 0.0


def _synthetic_aggregate_rows() -> list[dict]:
    conditions = {
        "A": "A_oracle_address_oracle_candidate",
        "B": "B_learned_address_oracle_candidate",
        "C": "C_oracle_address_state_read_candidate",
        "D": "D_learned_address_state_read_candidate",
    }
    variants = ["base", "separate_address", "state_aware", "full"]
    rows: list[dict] = []
    for seed in [101, 211, 307, 401, 503]:
        for condition_key, condition in conditions.items():
            for variant in variants:
                if condition_key == "A":
                    error = 0.0
                elif condition_key == "B":
                    error = 0.0 if variant in {"separate_address", "full"} else 0.01
                elif condition_key == "C":
                    error = 0.0 if variant in {"state_aware", "full"} else 0.01
                else:
                    error = 0.0 if variant == "full" else 0.01
                full_error = 0.0
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "condition": condition,
                        "address_accuracy": (
                            1.0
                            if condition_key in {"A", "C"}
                            or variant in {"separate_address", "full"}
                            else 0.5
                        ),
                        "candidate_recovery_mse": (
                            0.0
                            if condition_key in {"A", "B"}
                            or variant in {"state_aware", "full"}
                            else 0.25
                        ),
                        "affected_mse": error,
                        "retention_mse": 0.0,
                        "old_residual": error,
                        "architecture_extra_error": error - full_error,
                    }
                )
    return rows


def test_registered_aggregate_pattern_uses_seed_as_the_unit() -> None:
    seed_rows = compute_seed_contrasts(
        _synthetic_aggregate_rows(),
        seeds=[101, 211, 307, 401, 503],
    )
    assert len(seed_rows) == 5
    assert all(row["b_separate_address_gain"] == 0.01 for row in seed_rows)
    assert all(row["c_state_read_gain"] == 0.01 for row in seed_rows)
    assert all(row["d_full_only_gain"] == 0.01 for row in seed_rows)
    gate = _gain_gate(
        [float(row["d_full_only_gain"]) for row in seed_rows],
        sesoi=0.001,
        alpha=0.05,
        required_direction=1.0,
    )
    assert gate["passed"] is True
    assert gate["sign_flip_p"] == 0.03125
