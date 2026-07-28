import pytest
import torch

from catena.core.schema import CandidateMode, Operation
from catena.data.geometry_sweep import build_geometry_episode
from catena.models.matched_controllers import MatchedScalarController, ScalarConstraint
from catena.training.matched_probe import MatchedTrainConfig, train_matched_controller


def test_tied_and_dual_have_identical_parameter_count():
    tied = MatchedScalarController(10, 16, ScalarConstraint.TIED)
    dual = MatchedScalarController(10, 16, ScalarConstraint.DUAL)
    assert sum(p.numel() for p in tied.parameters()) == sum(p.numel() for p in dual.parameters())


def test_tied_outputs_equal_gates():
    model = MatchedScalarController(10, 16, ScalarConstraint.TIED)
    output = model(torch.randn(4, 10))
    assert torch.allclose(output.erase, output.write)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression path")
def test_matched_training_builds_features_before_episode_device_transfer():
    episode = build_geometry_episode(
        seed=7,
        operation=Operation.ADD,
        candidate_mode=CandidateMode.ORACLE,
        key_dim=8,
        value_dim=8,
        num_associations=6,
        key_correlation=0.2,
        old_scale=1.0,
        new_scale=1.0,
        old_new_cosine=0.0,
        episode_index=0,
    )
    model = MatchedScalarController(10, 8, ScalarConstraint.TIED)
    losses = train_matched_controller(
        model=model,
        episodes=[episode],
        config=MatchedTrainConfig(steps=1, learning_rate=0.003),
        device=torch.device("cuda:0"),
    )
    assert len(losses) == 1
