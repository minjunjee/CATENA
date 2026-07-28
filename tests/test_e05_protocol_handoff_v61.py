from pathlib import Path

from experiments.e05_common_v61 import (
    PINNED_E04_FREEZE_SHA256,
    validate_frozen_e04_dependency,
    validate_frozen_e05_protocol,
)


def test_e05_protocol_and_configs_match_the_prospective_lock():
    e05a, e05b = validate_frozen_e05_protocol()
    assert e05a["protocol"]["e05_runs_before_lock"] == 0
    assert e05b["human_audit"]["required_before_training"] is True
    assert e05b["statistics"]["positive_effect_sesoi"] == 0.001


def test_e04_additive_freeze_is_the_exact_h5_dependency():
    root = Path("/data/minjun_dev/CATENA/artifacts")
    frozen = validate_frozen_e04_dependency(root)
    assert frozen.freeze["claim_status"]["full_h4_claim_open"] is True
    assert frozen.dependency_record()["freeze_sha256"] == PINNED_E04_FREEZE_SHA256
    assert (
        frozen.dependency_record()["original_e02_confirmatory_status"]
        == "INCONCLUSIVE"
    )
