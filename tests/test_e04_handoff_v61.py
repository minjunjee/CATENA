import pytest

from experiments.e04_functional_mediation import (
    _heldout_base_seed,
    _is_baseline_dose,
    _row_key,
    _validate_row_identities,
)


def _row(*, intervention: str, dose: float) -> dict[str, object]:
    return {
        "seed": 11,
        "base_index": 3,
        "operation": "add",
        "intervention": intervention,
        "dose": dose,
    }


def test_e04_heldout_seed_uses_configured_reserved_region():
    assert (
        _heldout_base_seed(seed=11, base_index=0, heldout_seed_offset=75_000)
        == 1_175_000
    )
    assert (
        _heldout_base_seed(seed=11, base_index=127, heldout_seed_offset=75_000)
        == 1_175_127
    )

    with pytest.raises(ValueError, match="inside one seed block"):
        _heldout_base_seed(
            seed=11,
            base_index=25_000,
            heldout_seed_offset=75_000,
        )


def test_e04_row_key_is_unique_and_dose_one_is_canonical_baseline():
    rows = [
        _row(intervention="baseline", dose=1.0),
        *[
            _row(intervention="relevant_dose", dose=dose)
            for dose in (0.0, 0.25, 0.5, 0.75)
        ],
    ]

    _validate_row_identities(rows)
    assert len({_row_key(row) for row in rows}) == len(rows)
    assert _is_baseline_dose(1.0)


def test_e04_rejects_exact_duplicate_row_key():
    duplicate = _row(intervention="baseline", dose=1.0)
    with pytest.raises(ValueError, match="Duplicate intervention row key"):
        _validate_row_identities([duplicate, dict(duplicate)])


def test_e04_rejects_redundant_relevant_dose_one_alias():
    with pytest.raises(ValueError, match="duplicates the canonical baseline"):
        _validate_row_identities(
            [
                _row(intervention="baseline", dose=1.0),
                _row(intervention="relevant_dose", dose=1.0),
            ]
        )


def test_e04_full_schema_validation_rejects_minimal_legacy_rows():
    with pytest.raises(ValueError, match="incomplete schemas"):
        _validate_row_identities(
            [_row(intervention="baseline", dose=1.0)],
            require_full_schema=True,
        )
