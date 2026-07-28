from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from catena.eval.official_operator_gate import evaluate_metric_checks
from experiments.e15a_official_gdn2_kda_gate import main as e15a_main


def test_metric_contract_is_authoritative() -> None:
    checks, passed = evaluate_metric_checks(
        metrics={"relative_l2": 1.0e-6, "finite": True},
        check_contract={
            "parity": {
                "metric_key": "relative_l2",
                "comparison": "le",
                "threshold": 1.0e-5,
            },
            "backward": {
                "metric_key": "finite",
                "comparison": "eq",
                "expected": True,
            },
        },
    )
    assert passed is True
    assert checks["parity"]["passed"] is True
    assert checks["backward"]["passed"] is True


def test_metric_contract_rejects_missing_required_metric() -> None:
    with pytest.raises(KeyError, match="required metric"):
        evaluate_metric_checks(
            metrics={},
            check_contract={
                "parity": {
                    "metric_key": "relative_l2",
                    "comparison": "le",
                    "threshold": 1.0e-5,
                }
            },
        )


def test_e15a_dry_run_cannot_open_official_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e15a_official_gdn2_kda_gate.py",
            "--config",
            "configs/e15a_official_gdn2_kda_gate.yaml",
            "--device",
            "cpu",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
    )
    e15a_main()
    pointer = json.loads(
        (
            artifact_root / "e15a_official_gdn2_kda_gate" / "latest.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (Path(pointer["run_dir"]) / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "DRY_RUN"
    assert report["backend"]["status"] == "DRY_RUN"
    assert report["claim_gate"]["official_operator_claim_eligible"] is False
    assert report["claim_gate"]["scientific_evidence"] is False

