from catena.core.config import load_config
from experiments.e01b_constrained_behavioral_reachability import _claim_eligibility
from experiments.v61_common import (
    _expected_e01b_contract,
    _expected_e02_contract,
)


def test_main_full_eligibility_is_execution_not_hypothesis_support():
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    eligible, failures = _claim_eligibility(
        dry_run=False,
        seeds=seeds,
        configured_seeds=seeds,
        condition_count=4,
        row_count=239_616,
        expected_row_count=239_616,
        checkpoint_count=32,
    )
    assert eligible
    assert failures == []


def test_expected_row_contracts_are_derived_from_locked_configs():
    assert _expected_e01b_contract(
        load_config("configs/e01b_constrained_behavioral_reachability.yaml")
    ) == (239_616, 32)
    assert _expected_e02_contract(
        load_config("configs/e02_magnitude_factorization.yaml")
    ) == (16_384, 16)
