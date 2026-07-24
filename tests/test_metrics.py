from __future__ import annotations

from catena.eval.metrics import PredictionRecord, aggregate


def test_correction_retention_harmonic_mean():
    rows = [
        PredictionRecord(
            episode_id="e1",
            query_id="a",
            query_kind="affected_direct",
            policy="p",
            prediction_index=1,
            gold_index=1,
            exact_prediction_index=1,
            teacher_correct=True,
        ),
        PredictionRecord(
            episode_id="e1",
            query_id="r",
            query_kind="unaffected",
            policy="p",
            prediction_index=0,
            gold_index=0,
            exact_prediction_index=0,
            teacher_correct=True,
        ),
    ]
    summary = aggregate(rows)
    assert summary["c_update"] == 1.0
    assert summary["c_retain"] == 1.0
    assert summary["c_joint"] == 1.0
