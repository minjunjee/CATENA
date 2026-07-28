import torch

from catena.core.schema import CandidateMode, Operation
from catena.data.tamp import TAMPConfig, build_episode
from catena.training.losses import affected_read_mse, unaffected_retention_mse


def test_exact_target_has_zero_affected_and_retention_error() -> None:
    episode = build_episode(
        seed=3,
        operation=Operation.SUPERSEDE,
        candidate_mode=CandidateMode.ORACLE,
        config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8),
    )
    assert torch.isclose(affected_read_mse(episode.target_state, episode), torch.tensor(0.0))
    assert torch.isclose(
        unaffected_retention_mse(episode.target_state, episode), torch.tensor(0.0)
    )


def test_retention_ignores_affected_key() -> None:
    episode = build_episode(
        seed=4,
        operation=Operation.ADD,
        candidate_mode=CandidateMode.ORACLE,
        config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8),
    )
    output = episode.target_state.clone()
    key = episode.keys[episode.affected_index]
    output = output + torch.outer(key, torch.ones(output.shape[1]))
    assert affected_read_mse(output, episode) > 0
    # Correlated keys can create small spillover; the metric must still use only unaffected indices.
    manual_keys = episode.keys[episode.unaffected_indices]
    manual = torch.mean((manual_keys @ output - manual_keys @ episode.target_state) ** 2)
    assert torch.allclose(unaffected_retention_mse(output, episode), manual)
