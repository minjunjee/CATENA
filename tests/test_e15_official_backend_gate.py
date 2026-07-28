from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.e15_official_backend_gate import (
    _authoritative_plugin_row,
    main,
)


def test_plugin_cannot_override_authoritative_status_or_commit() -> None:
    row, passed = _authoritative_plugin_row(
        base={"name": "official", "repo_path": "/repo"},
        result={
            "passed": True,
            "scientific_evidence": True,
            "status": "NOT_CONFIGURED",
            "commit": "spoofed",
            "metrics": {"relative_l2": 0.0},
        },
        commit="a" * 40,
    )

    assert passed is True
    assert row["status"] == "PASS"
    assert row["commit"] == "a" * 40
    assert row["scientific_evidence"] is True


def test_passing_plugin_must_explicitly_mark_scientific_evidence() -> None:
    with pytest.raises(ValueError, match="scientific_evidence=true"):
        _authoritative_plugin_row(
            base={"name": "official", "repo_path": "/repo"},
            result={"passed": True, "scientific_evidence": False},
            commit="b" * 40,
        )


def test_dry_run_cannot_open_official_backend_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e15_official_backend_gate.py",
            "--config",
            "configs/e15_official_backend_gate.yaml",
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
            artifact_root / "e15_official_backend_gate" / "latest.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (Path(pointer["run_dir"]) / "report.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "DRY_RUN"
    assert report["claim_gate"]["official_backend_ready"] is False
    assert all(row["status"] == "DRY_RUN" for row in report["backends"])
    assert all(
        row["scientific_evidence"] is False for row in report["backends"]
    )
