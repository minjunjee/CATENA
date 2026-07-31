from __future__ import annotations

import hashlib

import pytest

from catena.core.provenance_v61 import sha256_canonical_json
from catena.lm.audit_contract import (
    E26_AUDIT_LOCKED_HASH_KEYS,
    e26_execution_source_inventory,
    validate_e26_audit_locked_hashes,
)
from catena.lm.numerical_audit import candidate_matrix_numerical_audit_receipt


def _locked(source_sha256: str) -> dict[str, str]:
    values = {key: hashlib.sha256(key.encode()).hexdigest() for key in E26_AUDIT_LOCKED_HASH_KEYS}
    values["source_tree_sha256"] = source_sha256
    return values


def _variant_row() -> dict:
    report = {"passed": True}
    return {
        "arbitrary_partitions": {
            "zero_state": {"fp32": report, "bf16": report},
            "prefilled_state": {"fp32": report, "bf16": report},
        },
        "gradient_accumulation": {
            "fp32": [{"passed": True}],
            "bf16": [{"passed": True}],
        },
        "passed": True,
    }


def test_execution_inventory_excludes_report_markdown_and_is_explicit(tmp_path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("value: 1\n", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
    (tmp_path / "RESULTS.md").write_text("not executable\n", encoding="utf-8")
    inventory = e26_execution_source_inventory(tmp_path)
    assert [row["path"] for row in inventory["rows"]] == [
        "config.yaml",
        "module.py",
        "pytest.ini",
    ]
    original = inventory["source_tree_sha256"]
    (tmp_path / "RESULTS.md").write_text("report-only edit\n", encoding="utf-8")
    assert e26_execution_source_inventory(tmp_path)["source_tree_sha256"] == original


def test_candidate_receipt_requires_exact_grid_and_locked_hash_contract(tmp_path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    inventory = e26_execution_source_inventory(tmp_path)
    candidate_ids = ("candidate_a", "candidate_b")
    audits = {
        candidate_id: {
            "model_config_sha256": sha256_canonical_json({"id": candidate_id}),
            "variants": {
                "dual_delta_lm": _variant_row(),
                "projected_tied_delta_lm": _variant_row(),
            },
            "passed": True,
        }
        for candidate_id in candidate_ids
    }
    receipt = candidate_matrix_numerical_audit_receipt(
        candidate_audits=audits,
        expected_candidate_ids=candidate_ids,
        locked_hashes=_locked(inventory["source_tree_sha256"]),
        source_inventory=inventory,
    )
    assert receipt["passed"] is True
    assert receipt["scientific_evidence"] is False
    assert receipt["main_test_opened"] is False
    assert len(receipt["receipt_sha256"]) == 64

    invalid = _locked(inventory["source_tree_sha256"])
    invalid.pop("protocol_lock_sha256")
    with pytest.raises(ValueError, match="missing"):
        validate_e26_audit_locked_hashes(invalid)
