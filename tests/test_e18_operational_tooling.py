from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

from scripts import check_e18_status as status

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/launch_e18_sequence_lattice_wave.py"
SOURCE_CONFIG = REPO_ROOT / status.SOURCE_CONFIG_RELATIVE_PATH
LOCK_PATH = REPO_ROOT / status.LOCK_RELATIVE_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_config() -> dict:
    payload = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _make_valid_main(
    artifact_root: Path,
    *,
    seed: int,
    variant: str,
    suffix: str,
) -> Path:
    config = _source_config()
    lock_sha256 = _sha256(LOCK_PATH)
    run_id = f"20260728T120000.{seed:03d}{suffix}Z"
    run_dir = artifact_root / status.SOURCE_EXPERIMENT_ID / run_id
    checkpoint = run_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint:{seed}:{variant}:{suffix}".encode())
    checkpoint_sha256 = _sha256(checkpoint)
    initialization_sha256 = hashlib.sha256(f"init:{seed}".encode()).hexdigest()

    families = [str(value) for value in config["data"]["families"]]
    updates = [1, 4, 8]
    gaps = [0, 128, 512, 2048]
    rows: list[dict] = []
    for family_index, (family, update_count, gap) in enumerate(
        product(families, updates, gaps)
    ):
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
            "affected_mse": 0.002 + family_index * 1e-8,
            "retention_mse": 0.00001,
            "state_mse": 0.001,
            "base_transaction_digest": hashlib.sha256(
                f"{seed}:{family}:{update_count}".encode()
            ).hexdigest(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "initialization_sha256": initialization_sha256,
            "protocol_lock_sha256": lock_sha256,
            "parameter_count": 12345,
            "optimizer": "AdamW",
        }
        if update_count == 8 and gap == 2048:
            row["distractor_activation_retention_harm"] = 0.01
        rows.append(row)
    metrics = run_dir / "sequence_control_lattice_metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "run_mode": "MAIN",
        "variant": variant,
        "seed": seed,
        "rows": 48,
        "expected_rows": 48,
        "protocol_lock": {"sha256": lock_sha256},
        "claim_gate": {"status": "PENDING_AGGREGATE"},
        "distractor_path_contract": {"passed": True},
    }
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": status.SOURCE_EXPERIMENT_ID,
            "run_id": run_id,
            "run_mode": "MAIN",
            "config": config,
            "config_file_sha256": _sha256(SOURCE_CONFIG),
            "report_sha256": _sha256(report_path),
        },
    )
    return run_dir


def _launcher_command(artifact_root: Path) -> list[str]:
    return [
        sys.executable,
        str(LAUNCHER),
        "--repo-root",
        str(REPO_ROOT),
        "--artifact-root",
        str(artifact_root),
        "--python-bin",
        sys.executable,
        "--expected-python-prefix",
        sys.prefix,
        "--dry-run",
    ]


def test_dry_run_prints_fixed_25_cell_three_gpu_schedule_without_artifacts(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "not-created"
    result = subprocess.run(
        _launcher_command(artifact_root),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    schedule_lines = [
        line
        for line in result.stdout.splitlines()
        if line[:2].isdigit() and "action=" in line
    ]
    assert len(schedule_lines) == 25
    assert "00 gpu=0 seed=101 variant=tied_scalar" in schedule_lines[0]
    assert "01 gpu=1 seed=101 variant=dual_scalar" in schedule_lines[1]
    assert "02 gpu=2 seed=101 variant=diagonal_value" in schedule_lines[2]
    assert "24 gpu=0 seed=503 variant=state_aware" in schedule_lines[-1]
    assert all("action=RUN" in line for line in schedule_lines)
    assert "latest_pointer_used=false aggregate_autorun=false" in result.stdout
    assert not artifact_root.exists()


def test_incomplete_main_explicitly_blocks_launch(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / status.SOURCE_EXPERIMENT_ID
        / "20260728T130000.000000Z"
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": status.SOURCE_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
        },
    )
    snapshot = status.scan_e18_status(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        include_live=False,
    )
    assert len(snapshot.incomplete_runs) == 1
    assert snapshot.launch_safe is False

    result = subprocess.run(
        _launcher_command(tmp_path),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "incomplete_main_runs=1" in result.stderr


def test_completed_cell_is_skipped_and_latest_pointer_is_ignored(
    tmp_path: Path,
) -> None:
    _make_valid_main(
        tmp_path,
        seed=101,
        variant="tied_scalar",
        suffix="001",
    )
    _write_json(
        tmp_path / status.SOURCE_EXPERIMENT_ID / "latest.json",
        {"run_dir": "/outside/ambiguous/latest"},
    )
    snapshot = status.scan_e18_status(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        include_live=False,
    )
    assert snapshot.cell_state(status.Cell(seed=101, variant="tied_scalar")) == (
        "COMPLETED"
    )
    assert len(snapshot.completed_cells) == 1
    assert len(snapshot.missing_cells) == 24

    result = subprocess.run(
        _launcher_command(tmp_path),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "00 gpu=0 seed=101 variant=tied_scalar        "
        "action=SKIP_COMPLETED"
    ) in result.stdout
    assert result.stdout.count("action=RUN") == 24


def test_duplicate_completed_cell_explicitly_blocks_launch(
    tmp_path: Path,
) -> None:
    for suffix in ("001", "002"):
        _make_valid_main(
            tmp_path,
            seed=101,
            variant="tied_scalar",
            suffix=suffix,
        )
    snapshot = status.scan_e18_status(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        include_live=False,
    )
    assert len(snapshot.duplicates) == 1
    assert snapshot.launch_safe is False

    result = subprocess.run(
        _launcher_command(tmp_path),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "duplicate_completed_cells=1" in result.stderr


def test_aggregate_ready_requires_exact_25_run_provenance(
    tmp_path: Path,
) -> None:
    for index, cell in enumerate(status.canonical_cells()):
        _make_valid_main(
            tmp_path,
            seed=cell.seed,
            variant=cell.variant,
            suffix=f"{index:03d}",
        )
    snapshot = status.scan_e18_status(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        include_live=False,
    )
    assert len(snapshot.completed_cells) == 25
    assert snapshot.aggregate_source_runs == 25
    assert snapshot.aggregate_source_rows == 1200
    assert snapshot.aggregate_provenance_error is None
    assert snapshot.aggregate_ready is True
