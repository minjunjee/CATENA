from catena.core.schema import CandidateMode, ControllerKind, Operation
from catena.data.tamp import TAMPConfig, build_episode
from catena.theory.control_geometry import local_control_geometry


def test_dual_has_no_more_regret_than_tied() -> None:
    config = TAMPConfig(num_associations=8, key_dim=8, value_dim=8)
    for operation in Operation:
        episode = build_episode(
            seed=15,
            operation=operation,
            candidate_mode=CandidateMode.ORACLE,
            config=config,
        )
        tied = local_control_geometry(episode, ControllerKind.TIED_SCALAR)
        dual = local_control_geometry(episode, ControllerKind.DUAL_SCALAR)
        assert dual.projection_regret <= tied.projection_regret + 1e-6


def test_asymmetric_operation_has_positive_tied_regret() -> None:
    episode = build_episode(
        seed=16,
        operation=Operation.ADD,
        candidate_mode=CandidateMode.ORACLE,
        config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8),
    )
    tied = local_control_geometry(episode, ControllerKind.TIED_SCALAR)
    assert tied.projection_regret > 0
