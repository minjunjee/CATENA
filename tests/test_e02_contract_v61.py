import numpy as np
import pytest

from catena.core.schema import Operation
from experiments.e02_magnitude_factorization import (
    _assert_finite_payload,
    _confirmatory_eligibility,
    _episodes,
    _fixed_seed_episode_bootstrap,
    _h2_claim_supported,
)

DATA = {
    "num_associations": 6,
    "key_dim": 8,
    "value_dim": 8,
    "key_correlation": 0.1,
    "old_scale": 1.0,
    "new_scale": 1.0,
    "old_new_cosine": 0.0,
}


def test_e02_episode_prefix_is_operation_balanced():
    episodes = _episodes(1000, 3, DATA)
    assert [episode.operation for episode in episodes[:4]] == list(Operation)
    assert {
        operation: sum(episode.operation is operation for episode in episodes)
        for operation in Operation
    } == {operation: 3 for operation in Operation}


def test_e02_fixed_seed_bootstrap_keeps_seed_and_operation_strata():
    effects = {
        11: np.array([1.0, 1.0, 3.0, 3.0]),
        22: np.array([5.0, 5.0, 7.0, 7.0]),
    }
    operations = {
        11: np.array(["add", "add", "invalidate", "invalidate"]),
        22: np.array(["add", "add", "invalidate", "invalidate"]),
    }
    interval = _fixed_seed_episode_bootstrap(
        effects,
        operations,
        samples=100,
        seed=9,
    )
    assert interval.estimate == pytest.approx(4.0)
    assert interval.low == pytest.approx(4.0)
    assert interval.high == pytest.approx(4.0)


def test_e02_strict_json_contract_allows_null_but_rejects_nan():
    _assert_finite_payload({"normalized_gain": None})
    with pytest.raises(FloatingPointError):
        _assert_finite_payload({"normalized_gain": float("nan")})


def test_e02_confirmatory_gate_requires_all_six_guards_and_main_design():
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    assert _confirmatory_eligibility(
        dry_run=False,
        seeds=seeds,
        configured_seeds=seeds,
        run_tuning=True,
        tuning_record_count=8,
        asymmetric_evaluable=True,
        supersede_evaluable=True,
    )
    assert not _confirmatory_eligibility(
        dry_run=True,
        seeds=seeds,
        configured_seeds=seeds,
        run_tuning=True,
        tuning_record_count=8,
        asymmetric_evaluable=True,
        supersede_evaluable=True,
    )

    guards = {
        "confirmatory_eligible": True,
        "asymmetric_gain": True,
        "preserve_equivalence": True,
        "supersede_equivalence": True,
        "positive_interaction": True,
        "retention_noninferiority": True,
        "tuned_direction_consistency": True,
    }
    assert _h2_claim_supported(**guards)
    for name in guards:
        one_failed = dict(guards)
        one_failed[name] = False
        assert not _h2_claim_supported(**one_failed)
