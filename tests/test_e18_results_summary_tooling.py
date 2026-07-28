from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import pytest
import yaml

from experiments import e18b_sequence_control_lattice_aggregate as e18b
from scripts import check_e18_status
from scripts import write_e18_result_summaries as summaries

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_PATH = (
    REPO_ROOT / check_e18_status.SOURCE_CONFIG_RELATIVE_PATH
)
AGGREGATE_CONFIG_PATH = (
    REPO_ROOT / check_e18_status.AGGREGATE_CONFIG_RELATIVE_PATH
)
LOCK_PATH = REPO_ROOT / check_e18_status.LOCK_RELATIVE_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _run_id(serial: int) -> str:
    return f"20260728T1200{serial:02d}.000000Z"


def _make_valid_source(
    artifact_root: Path,
    *,
    seed: int,
    variant: str,
    serial: int,
) -> Path:
    config = _read_yaml(SOURCE_CONFIG_PATH)
    lock_sha256 = _sha256(LOCK_PATH)
    run_dir = (
        artifact_root
        / summaries.SOURCE_EXPERIMENT_ID
        / _run_id(serial)
    )
    checkpoint = run_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{seed}:{variant}".encode())
    checkpoint_sha256 = _sha256(checkpoint)
    initialization_sha256 = hashlib.sha256(f"init:{seed}".encode()).hexdigest()

    families = [str(value) for value in config["data"]["families"]]
    variants = [str(value) for value in config["model"]["variants"]]
    freedom_rank = variants.index(variant)
    rows: list[dict] = []
    for family, update_count, gap in product(
        families,
        (1, 4, 8),
        (0, 128, 512, 2048),
    ):
        requirement_rank = families.index(family) + 1
        affected = 0.0001 if freedom_rank >= requirement_rank else 0.004
        row = {
            "seed": seed,
            "variant": variant,
            "demand_family": family,
            "updates": update_count,
            "gap_events": gap,
            "evaluation_seed": (
                100_000
                + 10_000 * seed
                + 100 * families.index(family)
                + update_count
            ),
            "affected_mse": affected,
            "retention_mse": 0.00001,
            "state_mse": affected / 2.0,
            "base_transaction_digest": hashlib.sha256(
                f"{seed}:{family}:{update_count}".encode()
            ).hexdigest(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "initialization_sha256": initialization_sha256,
            "protocol_lock_sha256": lock_sha256,
            "parameter_count": 12345,
            "optimizer": "AdamW",
            "train_final_loss": 0.0012,
            "train_best_loss": 0.0010,
            "examples_per_second": 987.6,
            "peak_memory_bytes": 123456,
        }
        if update_count == 8 and gap == 2048:
            row["distractor_activation_retention_harm"] = 0.01
        rows.append(row)
    _write_jsonl(run_dir / "sequence_control_lattice_metrics.jsonl", rows)

    report = {
        "status": "PASS",
        "run_scope": "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE",
        "run_mode": "MAIN",
        "variant": variant,
        "seed": seed,
        "rows": 48,
        "expected_rows": 48,
        "protocol_lock": {"sha256": lock_sha256},
        "claim_gate": {"status": "PENDING_AGGREGATE"},
        "distractor_path_contract": {"passed": True},
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": summaries.SOURCE_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
            "completed_at_utc": "2026-07-28T12:30:00+00:00",
            "config": config,
            "config_file_sha256": _sha256(SOURCE_CONFIG_PATH),
            "report_sha256": _sha256(report_path),
        },
    )
    return run_dir


def _make_complete_source_grid(artifact_root: Path) -> None:
    for serial, cell in enumerate(check_e18_status.canonical_cells()):
        _make_valid_source(
            artifact_root,
            seed=cell.seed,
            variant=cell.variant,
            serial=serial,
        )


def _make_valid_aggregate(artifact_root: Path) -> Path:
    config = _read_yaml(AGGREGATE_CONFIG_PATH)
    runtime = json.loads(json.dumps(config))
    runtime["source"]["config_path"] = str(SOURCE_CONFIG_PATH.resolve())
    lock_sha256 = _sha256(LOCK_PATH)
    rows, provenance = e18b.collect_main_sources(
        artifact_root=artifact_root,
        config=runtime,
        protocol_lock_sha256=lock_sha256,
    )
    contrasts, paired_rows, active_rows = e18b.aggregate_contrasts(
        rows=rows,
        config=runtime,
    )
    minimum_active_harm = min(
        float(row["active_path_retention_harm"]) for row in active_rows
    )
    conditions = {
        "all_adjacent_contrasts_passed": all(
            bool(contrast["passed"]) for contrast in contrasts.values()
        ),
        "full_paired_grid_passed": e18b.paired_grid_contract(
            rows=rows,
            config=runtime,
        ),
        "source_provenance_passed": True,
        "model_visible_active_path_assay_passed": (
            minimum_active_harm
            >= float(config["claim_gate"]["minimum_active_path_retention_harm"])
        ),
    }
    run_dir = (
        artifact_root
        / summaries.AGGREGATE_EXPERIMENT_ID
        / "20260728T130000.000000Z"
    )
    report = {
        "status": "PASS",
        "run_scope": (
            "SEQUENCE_CONTROL_ARCHITECTURE_DEMAND_LATTICE_AGGREGATE"
        ),
        "protocol_lock": {"sha256": lock_sha256},
        "source_runs": provenance,
        "contrasts": contrasts,
        "summary": {
            "source_runs": len(provenance),
            "metric_rows": len(rows),
            "paired_contrast_seed_rows": len(paired_rows),
            "active_path_rows": len(active_rows),
            "minimum_active_path_retention_harm": minimum_active_harm,
        },
        "claim_gate": {
            "supported": all(conditions.values()),
            "conditions": conditions,
        },
        "evidence_tier": "CONTROLLED_REFERENCE",
        "scientific_evidence": False,
    }
    _write_jsonl(
        run_dir / "sequence_control_lattice_paired_metrics.jsonl",
        paired_rows,
    )
    _write_jsonl(
        run_dir / "sequence_control_lattice_active_path_metrics.jsonl",
        active_rows,
    )
    _write_jsonl(run_dir / "source_run_provenance.jsonl", provenance)
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": summaries.AGGREGATE_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
            "completed_at_utc": "2026-07-28T13:00:01+00:00",
            "config": config,
            "config_file_sha256": _sha256(AGGREGATE_CONFIG_PATH),
            "report_sha256": _sha256(report_path),
        },
    )
    return run_dir


def test_source_summary_dry_run_then_exclusive_create_and_skip(
    tmp_path: Path,
) -> None:
    run_dir = _make_valid_source(
        tmp_path,
        seed=101,
        variant="tied_scalar",
        serial=0,
    )
    plan = summaries.build_summary_plan(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        scope="source",
        include_live=False,
    )
    assert len(plan.actions) == 1
    dry_results = summaries.execute_summary_plan(plan, dry_run=True)
    assert dry_results[0]["disposition"] == "WOULD_CREATE"
    assert not (run_dir / summaries.SUMMARY_FILENAME).exists()

    results = summaries.execute_summary_plan(plan, dry_run=False)
    assert results[0]["disposition"] == "CREATED"
    summary_path = run_dir / summaries.SUMMARY_FILENAME
    original = summary_path.read_bytes()
    assert b"PENDING_AGGREGATE" in original
    source_text = original.decode()
    assert "explicit demand descriptor" in source_text
    assert "model-visible verified bit" in source_text
    assert "stress SESOI" in source_text
    assert len(original.decode().splitlines()) <= summaries.MAX_SUMMARY_LINES

    second = summaries.execute_summary_plan(plan, dry_run=False)
    assert second[0]["disposition"] == "SKIP_EXISTING"
    assert summary_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        summaries._write_exclusive(summary_path, "replacement")
    assert summary_path.read_bytes() == original


def test_incomplete_duplicate_and_live_source_each_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = (
        tmp_path
        / summaries.SOURCE_EXPERIMENT_ID
        / "20260728T120000.000000Z"
    )
    _write_json(
        incomplete / "run_manifest.json",
        {"schema_version": 2, "run_mode": "MAIN"},
    )
    with pytest.raises(RuntimeError, match="incomplete_main_runs=1"):
        summaries.build_summary_plan(
            repo_root=REPO_ROOT,
            artifact_root=tmp_path,
            include_live=False,
        )

    clean = tmp_path / "duplicates"
    _make_valid_source(
        clean,
        seed=101,
        variant="tied_scalar",
        serial=0,
    )
    second = _make_valid_source(
        clean,
        seed=101,
        variant="tied_scalar",
        serial=1,
    )
    assert second.exists()
    with pytest.raises(RuntimeError, match="duplicate_completed_cells=1"):
        summaries.build_summary_plan(
            repo_root=REPO_ROOT,
            artifact_root=clean,
            include_live=False,
        )

    live_root = tmp_path / "live"
    monkeypatch.setattr(
        check_e18_status,
        "_live_e18_processes",
        lambda: [
            check_e18_status.LiveRun(
                pid=123,
                command="e18a",
                seed=101,
                variant="tied_scalar",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="live_main_processes=1"):
        summaries.build_summary_plan(
            repo_root=REPO_ROOT,
            artifact_root=live_root,
            include_live=True,
        )


def test_aggregate_absence_is_nonmutating_and_explicit(
    tmp_path: Path,
) -> None:
    plan = summaries.build_summary_plan(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        scope="aggregate",
        include_live=False,
    )
    assert plan.aggregate_available is False
    assert plan.actions == ()


def test_aggregate_summary_requires_exact_reproducible_sources(
    tmp_path: Path,
) -> None:
    _make_complete_source_grid(tmp_path)
    run_dir = _make_valid_aggregate(tmp_path)
    plan = summaries.build_summary_plan(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        scope="aggregate",
        include_live=False,
    )
    assert plan.aggregate_available is True
    assert len(plan.actions) == 1
    assert plan.actions[0].kind == "AGGREGATE"
    results = summaries.execute_summary_plan(plan, dry_run=False)
    assert results[0]["disposition"] == "CREATED"
    content = (run_dir / summaries.SUMMARY_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "`SUPPORTED`" in content
    assert content.count("| PASS |") == 4
    assert "Grid-mean affected gain" in content
    assert "every-cell" in content
    assert "개선을 뜻하지" in content
    assert "Stress는 5/5 방향성" in content
    assert "explicit demand" in content
    assert "model-visible verified bit" in content
    assert "absolute" in content
    assert "accuracy가 아니다" in content
    assert len(content.splitlines()) <= summaries.MAX_SUMMARY_LINES

    paired_path = run_dir / "sequence_control_lattice_paired_metrics.jsonl"
    paired_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="derived artifact does not reproduce"):
        summaries.build_summary_plan(
            repo_root=REPO_ROOT,
            artifact_root=tmp_path,
            scope="aggregate",
            include_live=False,
        )
