import inspect

from catena.core.config import load_config
from experiments.e05a_semantic_protocol_lock import (
    _AUDIT_SPLITS,
    _access_manifest,
    _memory_spec,
)
from experiments.e05b_semantic_anchor import _validate_config_contract, main


def test_e05a_access_manifest_has_no_private_update_fields():
    config = load_config("configs/e05a_semantic_protocol_lock.yaml")
    manifest = _access_manifest(config)
    assert manifest["forbidden_access_test_passed"] is True
    assert manifest["forbidden_field_overlap"] == []
    assert manifest["private_target_or_demand_in_public_update_context"] is False
    assert config["training"]["target_state_weight"] == 0.0


def test_e05b_contract_matches_e05a_and_keeps_four_audit_splits():
    e05a = load_config("configs/e05a_semantic_protocol_lock.yaml")
    e05b = load_config("configs/e05b_semantic_anchor.yaml")
    _validate_config_contract(
        "configs/e05b_semantic_anchor.yaml",
        frozen_e05a=e05a,
        frozen_e05b=e05b,
    )
    assert _AUDIT_SPLITS == (
        "primary",
        "heldout_paraphrase",
        "heldout_domain",
        "combined_stress",
    )
    assert _memory_spec(e05a).num_associations == 16


def test_e05b_unseal_consumes_validation_passed_not_claim_supported():
    source = inspect.getsource(main)
    assert 'validation_report["passed"]' in source
    assert 'validation_report["supported"]' not in source
    assert "evaluate_e05b_secondary" in source
