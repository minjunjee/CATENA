import torch

from catena.core.schema import CandidateMode, ControllerKind, Operation
from catena.data.tamp import TAMPConfig, build_episode
from catena.theory.reachability import ReadoutMode, constrained_reachability


def test_dual_feasible_regret_closes_asymmetric_oracle_corner():
    episode = build_episode(seed=1, operation=Operation.ADD, candidate_mode=CandidateMode.ORACLE, config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8))
    tied = constrained_reachability(episode, ControllerKind.TIED_SCALAR, mode=ReadoutMode.STATE)
    dual = constrained_reachability(episode, ControllerKind.DUAL_SCALAR, mode=ReadoutMode.STATE)
    assert dual.feasible_mse < 1e-10
    assert tied.feasible_mse > dual.feasible_mse


def test_behavioral_readout_is_finite():
    episode = build_episode(seed=2, operation=Operation.SUPERSEDE, candidate_mode=CandidateMode.ORACLE, config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8))
    report = constrained_reachability(episode, ControllerKind.DUAL_SCALAR, mode=ReadoutMode.BEHAVIORAL)
    assert torch.isfinite(torch.tensor(report.feasible_mse))


def test_behavioral_mse_equal_weights_correction_and_retention():
    from catena.theory.reachability import behavioral_mse
    from catena.training.losses import affected_read_mse, unaffected_retention_mse

    episode = build_episode(
        seed=3,
        operation=Operation.ADD,
        candidate_mode=CandidateMode.ORACLE,
        config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8),
    )
    output = episode.state
    expected = 0.5 * affected_read_mse(output, episode) + 0.5 * unaffected_retention_mse(
        output, episode
    )
    assert torch.allclose(behavioral_mse(output, episode), expected, atol=1e-7)
