from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import write_postcore_results_summary_index as indexer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_summary(
    artifact_root: Path,
    *,
    number: int,
    experiment_id: str | None = None,
    serial: int = 0,
    content: str | None = None,
) -> Path:
    experiment = experiment_id or f"e{number:02d}_fixture"
    run_id = f"20260728T12{number:02d}{serial:02d}.000000Z"
    run_dir = artifact_root / experiment / run_id
    report_path = run_dir / "report.json"
    _write_json(
        report_path,
        {
            "status": "PASS",
            "evidence_tier": "CONTROLLED_REFERENCE",
            "scientific_evidence": False,
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": experiment,
            "run_id": run_id,
            "run_mode": "MAIN",
            "report_sha256": hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest(),
        },
    )
    summary_path = run_dir / indexer.SUMMARY_FILENAME
    summary_path.write_text(
        content
        or (
            f"# E{number} result\n\n"
            f"Run `{run_id}` completed under controlled-reference scope.\n"
        ),
        encoding="utf-8",
    )
    return summary_path


def _make_required_grid(artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    for number in indexer.REQUIRED_NUMBERS:
        experiment_id = (
            indexer.REQUIRED_E18_AGGREGATE
            if number == 18
            else f"e{number:02d}_fixture"
        )
        _make_summary(
            artifact_root,
            number=number,
            experiment_id=experiment_id,
        )


def test_build_plan_audits_required_grid_and_optional_e21(
    tmp_path: Path,
) -> None:
    _make_required_grid(tmp_path)
    plan = indexer.build_index_plan(tmp_path)
    assert len(plan.records) == 11
    assert plan.e21_present is False
    assert plan.output_path.name == indexer.OUTPUT_FILENAME
    assert "## E21" in plan.content
    assert "검증 가능한 completed summary가 없다" in plan.content
    for record in plan.records:
        assert record.lines <= indexer.MAX_SUMMARY_LINES
        assert record.utf8_bytes <= indexer.MAX_SUMMARY_BYTES
        assert record.sha256 in plan.content

    _make_summary(tmp_path, number=21)
    with_e21 = indexer.build_index_plan(tmp_path)
    assert len(with_e21.records) == 12
    assert with_e21.e21_present is True
    assert "`e21_fixture`" in with_e21.content


def test_exclusive_create_never_overwrites_new_or_legacy_index(
    tmp_path: Path,
) -> None:
    _make_required_grid(tmp_path)
    legacy = tmp_path / indexer.LEGACY_INDEX_FILENAME
    legacy.write_text("legacy-sentinel\n", encoding="utf-8")
    plan = indexer.build_index_plan(tmp_path)
    indexer._write_exclusive(plan.output_path, plan.content)
    assert plan.output_path.read_text(encoding="utf-8") == plan.content
    assert legacy.read_text(encoding="utf-8") == "legacy-sentinel\n"
    with pytest.raises(FileExistsError):
        indexer._write_exclusive(plan.output_path, "replacement")
    assert plan.output_path.read_text(encoding="utf-8") == plan.content


def test_missing_e18_aggregate_blocks_index_creation(tmp_path: Path) -> None:
    _make_required_grid(tmp_path)
    aggregate = next(
        path
        for path in tmp_path.glob(
            f"{indexer.REQUIRED_E18_AGGREGATE}/*/{indexer.SUMMARY_FILENAME}"
        )
    )
    aggregate.unlink()
    with pytest.raises(RuntimeError, match="coverage is incomplete: E18"):
        indexer.build_index_plan(tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("\n".join(f"line {index}" for index in range(61)), "lines=61/60"),
        ("x" * 8_001, "bytes=8001/8000"),
    ],
)
def test_summary_size_contract_is_enforced(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    _make_required_grid(tmp_path)
    summary = next(tmp_path.glob("e10_fixture/*/RESULTS_SUMMARY_KO.md"))
    summary.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        indexer.build_index_plan(tmp_path)


def test_manifest_report_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    _make_required_grid(tmp_path)
    report = next(tmp_path.glob("e12_fixture/*/report.json"))
    report.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="report hash mismatch"):
        indexer.build_index_plan(tmp_path)


def test_validate_existing_detects_tampering(tmp_path: Path) -> None:
    _make_required_grid(tmp_path)
    plan = indexer.build_index_plan(tmp_path)
    indexer._write_exclusive(plan.output_path, plan.content)
    indexer.validate_existing(plan)
    plan.output_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        indexer.validate_existing(plan)
