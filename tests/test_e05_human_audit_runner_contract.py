from pathlib import Path

import pytest

from experiments.e05a_semantic_audit_adjudication import (
    _regular_external_file,
    _validate_config,
)


def test_human_audit_config_is_locked_to_parent_protocol():
    config = _validate_config("configs/e05a_semantic_audit_adjudication.yaml")
    assert config["thresholds"]["total_items"] == 300
    assert config["protocol"]["modifies_e05a_artifact"] is False
    assert config["claim"]["pass_is_dependency_not_h5_support"] is True
    assert Path("docs/E05A_HUMAN_AUDIT_ADJUDICATION_LOCK_KO.md").is_file()


def test_human_review_inputs_must_be_external_copies(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "e05a" / "run"
    run_dir.mkdir(parents=True)
    inside = run_dir / "reviewer.csv"
    inside.write_text("audit_id,meaning_preserved,answer_leakage\n")
    external = tmp_path / "human_reviews" / "reviewer.csv"
    external.parent.mkdir()
    external.write_text("audit_id,meaning_preserved,answer_leakage\n")

    with pytest.raises(ValueError, match="external copy"):
        _regular_external_file(
            str(inside),
            "reviewer A",
            forbidden_roots=(artifact_root, run_dir),
        )
    assert (
        _regular_external_file(
            str(external),
            "reviewer A",
            forbidden_roots=(artifact_root, run_dir),
        )
        == external.resolve()
    )
