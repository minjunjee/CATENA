from __future__ import annotations

from catena.cli import _run_mock_smoke


def test_mock_smoke_checks_pipeline_invariants() -> None:
    report = _run_mock_smoke()
    assert report["passed"] is True
    assert report["base_bytes"] > 0
    assert report["exact_scores_finite"] is True
    assert report["typed_scores_finite"] is True
    assert report["states_distinct"] is True
