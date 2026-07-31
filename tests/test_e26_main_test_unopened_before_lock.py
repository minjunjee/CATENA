from __future__ import annotations

from pathlib import Path

import pytest

from catena.core.provenance_v61 import sha256_canonical_json
from catena.lm.artifacts import ArtifactContractError, ArtifactRun
from catena.lm.e26a_gate import assert_main_test_unopened, zero_main_test_access_ledger

_POPULATION_FILE_SHA256 = "a" * 64


def _population_lock() -> dict[str, object]:
    return {
        "manifest_type": "E26A_VALIDATION_POPULATION_LOCK",
        "split": "validation",
        "records_sha256": "b" * 64,
        "episode_count": 500,
    }


def _run(root: Path) -> ArtifactRun:
    return ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=root,
        canonical_artifact_root=root,
        run_mode="MAIN",
        dry_run=False,
        scientific_evidence=False,
        evidence_tier="SCIENTIFIC_PROTOCOL_GATE",
        claim_ceiling="PROTOCOL_IDENTIFIABILITY_ONLY",
    )


def test_zero_main_test_ledger_is_explicit_and_hash_bound(tmp_path: Path) -> None:
    population = _population_lock()
    ledger = zero_main_test_access_ledger(
        validation_population_lock=population,
        validation_population_lock_sha256=_POPULATION_FILE_SHA256,
    )
    assert ledger["main_test_opened"] is False
    assert ledger["main_test_access_count"] == 0
    assert ledger["permitted_materialized_splits"] == ["validation"]
    assert ledger["forbidden_materialized_splits"] == [
        "main_test",
        "heldout_domain",
    ]
    assert len(ledger["ledger_sha256"]) == 64

    run = _run(tmp_path)
    run.write("main_test_access_ledger.json", ledger)
    assert_main_test_unopened(
        run,
        validation_population_lock=population,
        validation_population_lock_sha256=_POPULATION_FILE_SHA256,
    )


def test_nonzero_main_test_access_blocks_finalization_path(tmp_path: Path) -> None:
    run = _run(tmp_path)
    population = _population_lock()
    ledger = zero_main_test_access_ledger(
        validation_population_lock=population,
        validation_population_lock_sha256=_POPULATION_FILE_SHA256,
    )
    ledger["main_test_opened"] = True
    ledger["main_test_access_count"] = 1
    ledger.pop("ledger_sha256")
    ledger["ledger_sha256"] = sha256_canonical_json(ledger)
    run.write("main_test_access_ledger.json", ledger)
    with pytest.raises(ArtifactContractError, match="exact validation-only"):
        assert_main_test_unopened(
            run,
            validation_population_lock=population,
            validation_population_lock_sha256=_POPULATION_FILE_SHA256,
        )


def test_main_test_ledger_tampering_is_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    population = _population_lock()
    ledger = zero_main_test_access_ledger(
        validation_population_lock=population,
        validation_population_lock_sha256=_POPULATION_FILE_SHA256,
    )
    ledger["permitted_materialized_splits"].append("main_test")
    run.write("main_test_access_ledger.json", ledger)
    with pytest.raises(ArtifactContractError, match="exact validation-only"):
        assert_main_test_unopened(
            run,
            validation_population_lock=population,
            validation_population_lock_sha256=_POPULATION_FILE_SHA256,
        )


def test_rehashed_main_test_permission_is_still_rejected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    population = _population_lock()
    ledger = zero_main_test_access_ledger(
        validation_population_lock=population,
        validation_population_lock_sha256=_POPULATION_FILE_SHA256,
    )
    ledger["permitted_materialized_splits"].append("main_test")
    ledger.pop("ledger_sha256")
    ledger["ledger_sha256"] = sha256_canonical_json(ledger)
    run.write("main_test_access_ledger.json", ledger)
    with pytest.raises(ArtifactContractError, match="exact validation-only"):
        assert_main_test_unopened(
            run,
            validation_population_lock=population,
            validation_population_lock_sha256=_POPULATION_FILE_SHA256,
        )
