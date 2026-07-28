import torch

from catena.core.schema import CandidateMode, Operation
from catena.data.tamp import TAMPConfig, build_episode, validate_episode


def test_target_state_invariants() -> None:
    config = TAMPConfig(num_associations=8, key_dim=8, value_dim=8)
    for mode in CandidateMode:
        for operation in Operation:
            episode = build_episode(
                seed=10,
                operation=operation,
                candidate_mode=mode,
                config=config,
            )
            validate_episode(episode)


def test_unaffected_excludes_affected() -> None:
    episode = build_episode(
        seed=2,
        operation=Operation.SUPERSEDE,
        candidate_mode=CandidateMode.ORACLE,
        config=TAMPConfig(num_associations=8, key_dim=8, value_dim=8),
    )
    assert episode.affected_index not in episode.unaffected_indices.tolist()


def test_recurrent_read_differs_under_correlation() -> None:
    config = TAMPConfig(
        num_associations=8,
        key_dim=8,
        value_dim=8,
        key_correlation=0.8,
    )
    oracle = build_episode(
        seed=99,
        operation=Operation.INVALIDATE,
        candidate_mode=CandidateMode.ORACLE,
        config=config,
    )
    recurrent = build_episode(
        seed=99,
        operation=Operation.INVALIDATE,
        candidate_mode=CandidateMode.RECURRENT_READ,
        config=config,
    )
    assert not torch.allclose(oracle.erase_candidate, recurrent.erase_candidate)
