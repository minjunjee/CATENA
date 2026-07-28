from catena.core.schema import CandidateMode, Operation
from catena.data.geometry_sweep import build_geometry_episode
from experiments.e01b_constrained_behavioral_reachability import (
    _claim_eligibility,
    _paired_episode_digest,
)


def _episode(mode: CandidateMode):
    return build_geometry_episode(
        seed=101,
        operation=Operation.SUPERSEDE,
        candidate_mode=mode,
        key_dim=8,
        value_dim=8,
        num_associations=6,
        key_correlation=0.2,
        old_scale=1.0,
        new_scale=1.2,
        old_new_cosine=0.1,
        episode_index=0,
    )


def test_candidate_modes_share_base_episode_and_only_change_erase_candidate():
    report = _paired_episode_digest(
        [_episode(CandidateMode.ORACLE)],
        [_episode(CandidateMode.RECURRENT_READ)],
    )
    assert report["only_erase_candidate_varies"] is True
    assert report["mean_erase_candidate_mse"] > 0.0


def test_e01b_claim_requires_main_and_all_eight_preregistered_seeds():
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    eligible, reasons = _claim_eligibility(
        dry_run=False,
        seeds=seeds,
        configured_seeds=seeds,
        condition_count=4,
        row_count=239_616,
        expected_row_count=239_616,
        checkpoint_count=32,
    )
    assert eligible
    assert reasons == []

    dry_eligible, dry_reasons = _claim_eligibility(
        dry_run=True,
        seeds=seeds[:7],
        configured_seeds=seeds,
        condition_count=4,
        row_count=1,
        expected_row_count=1,
        checkpoint_count=28,
    )
    assert not dry_eligible
    assert "dry_run" in dry_reasons
    assert "requires_exactly_8_unique_training_seeds" in dry_reasons
