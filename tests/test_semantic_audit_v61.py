import csv

import pytest

from catena.eval.semantic_audit_v61 import evaluate_semantic_human_audit


def _write(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_two_reviewer_audit_passes_only_with_complete_adjudication(tmp_path):
    ids = [f"audit-{index:03d}" for index in range(300)]
    items = tmp_path / "items.csv"
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    adjudication = tmp_path / "adjudication.csv"
    _write(items, ["audit_id", "split", "text"], [[value, "primary", "record"] for value in ids])
    _write(a, ["audit_id", "meaning_preserved", "answer_leakage"], [[value, 1, 0] for value in ids])
    _write(b, ["audit_id", "meaning_preserved", "answer_leakage"], [[value, 1, 0] for value in ids])
    _write(
        adjudication,
        ["audit_id", "adjudicated_meaning_preserved", "adjudicated_answer_leakage"],
        [[value, 1, 0] for value in ids],
    )
    report = evaluate_semantic_human_audit(
        audit_items_path=items,
        reviewer_a_path=a,
        reviewer_b_path=b,
        adjudication_path=adjudication,
    )
    assert report.passed
    assert report.items == 300
    assert report.adjudicated_meaning_preservation == 1.0
    assert report.adjudicated_answer_leakage == 0.0


def test_audit_rejects_model_outcomes_and_incomplete_review(tmp_path):
    ids = [f"audit-{index:03d}" for index in range(300)]
    items = tmp_path / "items.csv"
    review = tmp_path / "review.csv"
    adjudication = tmp_path / "adjudication.csv"
    _write(
        items,
        ["audit_id", "model_error"],
        [[value, 0.0] for value in ids],
    )
    _write(
        review,
        ["audit_id", "meaning_preserved", "answer_leakage"],
        [[value, 1, 0] for value in ids],
    )
    _write(
        adjudication,
        ["audit_id", "adjudicated_meaning_preserved", "adjudicated_answer_leakage"],
        [[value, 1, 0] for value in ids[:-1]],
    )
    with pytest.raises(ValueError, match="outcome columns"):
        evaluate_semantic_human_audit(
            audit_items_path=items,
            reviewer_a_path=review,
            reviewer_b_path=review,
            adjudication_path=adjudication,
        )
