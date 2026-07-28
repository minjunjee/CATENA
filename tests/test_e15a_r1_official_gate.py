from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.e15a_r1_official_gdn2_kda_gate import main


def test_e15a_r1_dry_run_cannot_open_official_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e15a_r1_official_gdn2_kda_gate.py",
            "--config",
            "configs/e15a_r1_official_gdn2_kda_gate.yaml",
            "--device",
            "cpu",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
    )
    main()
    pointer = json.loads(
        (
            artifact_root
            / "e15a_r1_official_gdn2_kda_gate"
            / "latest.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (Path(pointer["run_dir"]) / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "DRY_RUN"
    assert report["backend"]["status"] == "DRY_RUN"
    assert report["repair"]["scientific_gate_changed"] is False
    assert report["claim_gate"]["official_operator_claim_eligible"] is False
    assert report["claim_gate"]["scientific_evidence"] is False
