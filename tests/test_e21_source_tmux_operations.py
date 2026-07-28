from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from scripts import manage_e21_source_tmux as operations

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/manage_e21_source_tmux.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_completed(artifact_root: Path, *, seed: int, suffix: str) -> Path:
    run_id = f"20260728T120000.{suffix}Z"
    run_dir = artifact_root / operations.SOURCE_EXPERIMENT_ID / run_id
    metrics = run_dir / "structured_sequence_transfer_metrics.jsonl"
    summary = run_dir / "RESULTS_SUMMARY_KO.md"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, Path] = {}
    checkpoint_hashes: dict[str, str] = {}
    for variant in operations.REGISTERED_VARIANTS:
        checkpoint = run_dir / "checkpoints" / f"seed{seed}_{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{seed}:{variant}:checkpoint".encode())
        checkpoints[variant] = checkpoint.resolve()
        checkpoint_hashes[variant] = _sha256(checkpoint)
    rows = []
    for variant, condition, family, updates, gap in product(
        operations.REGISTERED_VARIANTS,
        operations.REGISTERED_CONDITIONS,
        operations.REGISTERED_FAMILIES,
        operations.REGISTERED_UPDATES,
        operations.REGISTERED_GAPS,
    ):
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "condition": condition,
                "demand_family": family,
                "updates": updates,
                "gap_events": gap,
                "checkpoint": str(checkpoints[variant]),
                "checkpoint_sha256": checkpoint_hashes[variant],
                "initialization_sha256": "a" * 64,
                "parameter_count": 1000,
                "base_transaction_digest": (
                    f"digest:{family}:{updates}:{gap}"
                ),
            }
        )
    metrics.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary.write_text("# E21a\n\nPASS\n", encoding="utf-8")
    report = {
        "status": "PASS",
        "run_mode": "MAIN",
        "seed": seed,
        "protocol": {
            "lock_sha256": operations.EXPECTED_PROTOCOL_LOCK_SHA256,
            "source_config_sha256": _sha256(
                REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE
            ),
        },
        "artifacts": {
            "metrics_sha256": _sha256(metrics),
            "checkpoint_hashes": checkpoint_hashes,
            "results_summary_ko": {
                "sha256": _sha256(summary),
                "line_count": 3,
            },
        },
    }
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": operations.SOURCE_EXPERIMENT_ID,
            "run_id": run_id,
            "run_mode": "MAIN",
            "completed_at_utc": "2026-07-28T12:30:00+00:00",
            "report_sha256": _sha256(report_path),
            "config_file_sha256": _sha256(
                REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE
            ),
        },
    )
    return run_dir


def _cli(artifact_root: Path, *tail: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(REPO_ROOT),
        "--artifact-root",
        str(artifact_root),
        "--config",
        str(REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE),
        "--python-bin",
        sys.executable,
        *tail,
    ]


def test_registered_contract_and_exact_source_command() -> None:
    contract = operations.validate_registered_contract(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE,
    )
    assert contract["lock_sha256"] == operations.EXPECTED_PROTOCOL_LOCK_SHA256
    command = operations.build_source_command(
        python_bin=Path(sys.executable),
        config_path=REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE,
        seed=113,
        artifact_root=Path("/data/example"),
    )
    assert command[1:3] == ("-m", operations.SOURCE_MODULE)
    assert command[-6:] == (
        "--seed",
        "113",
        "--device",
        "cuda:0",
        "--artifact-root",
        "/data/example",
    )
    assert "--aggregate" not in command
    assert "--dry-run" not in command


def test_completed_seed_blocks_launch_without_creating_launcher_files(
    tmp_path: Path,
) -> None:
    _make_completed(tmp_path, seed=113, suffix="000001")
    result = subprocess.run(
        _cli(tmp_path, "launch", "--seed", "113", "--gpu", "0"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "already has 1 completed MAIN run" in result.stderr
    assert not (tmp_path / "_launcher_logs").exists()


def test_report_manifest_metrics_summary_shell_is_not_completed(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / operations.SOURCE_EXPERIMENT_ID
        / "20260728T120000.000009Z"
    )
    metrics = run_dir / "structured_sequence_transfer_metrics.jsonl"
    summary = run_dir / "RESULTS_SUMMARY_KO.md"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text('{"seed": 113}\n', encoding="utf-8")
    summary.write_text("# shell\n", encoding="utf-8")
    report_path = run_dir / "report.json"
    _write_json(
        report_path,
        {
            "status": "PASS",
            "run_mode": "MAIN",
            "seed": 113,
            "protocol": {
                "lock_sha256": operations.EXPECTED_PROTOCOL_LOCK_SHA256,
                "source_config_sha256": _sha256(
                    REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE
                ),
            },
            "artifacts": {
                "metrics_sha256": _sha256(metrics),
                "checkpoint_hashes": {
                    variant: "0" * 64
                    for variant in operations.REGISTERED_VARIANTS
                },
                "results_summary_ko": {
                    "sha256": _sha256(summary),
                    "line_count": 1,
                },
            },
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": operations.SOURCE_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
            "completed_at_utc": "2026-07-28T12:01:00+00:00",
            "report_sha256": _sha256(report_path),
            "config_file_sha256": _sha256(
                REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE
            ),
        },
    )
    completed, incomplete = operations.scan_artifact_runs(tmp_path)
    assert completed == ()
    assert len(incomplete) == 1
    assert incomplete[0].seed == 113

    planned = subprocess.run(
        _cli(
            tmp_path,
            "launch",
            "--seed",
            "113",
            "--gpu",
            "0",
            "--allow-incomplete-retry",
        ),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, planned.stderr
    assert "execute=False" in planned.stdout


def test_checkpoint_tamper_removes_completed_classification(
    tmp_path: Path,
) -> None:
    run_dir = _make_completed(tmp_path, seed=113, suffix="000010")
    completed, incomplete = operations.scan_artifact_runs(tmp_path)
    assert len(completed) == 1
    assert incomplete == ()

    checkpoint = run_dir / "checkpoints/seed113_full.pt"
    checkpoint.write_bytes(b"tampered")
    completed, incomplete = operations.scan_artifact_runs(tmp_path)
    assert completed == ()
    assert len(incomplete) == 1
    assert incomplete[0].seed == 113


def test_overlong_summary_is_not_a_completed_source(tmp_path: Path) -> None:
    run_dir = _make_completed(tmp_path, seed=113, suffix="000011")
    summary = run_dir / "RESULTS_SUMMARY_KO.md"
    summary.write_text("\n".join(f"line {index}" for index in range(56)) + "\n")
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["results_summary_ko"] = {
        "sha256": _sha256(summary),
        "line_count": 56,
    }
    _write_json(report_path, report)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report_sha256"] = _sha256(report_path)
    _write_json(manifest_path, manifest)

    completed, incomplete = operations.scan_artifact_runs(tmp_path)
    assert completed == ()
    assert len(incomplete) == 1


def test_incomplete_run_is_preserved_and_retry_requires_explicit_flag(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / operations.SOURCE_EXPERIMENT_ID
        / "20260728T120000.000002Z"
    )
    checkpoint = run_dir / "checkpoints/seed223_base.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial")
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": operations.SOURCE_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
        },
    )
    before = checkpoint.read_bytes()
    blocked = subprocess.run(
        _cli(tmp_path, "launch", "--seed", "223", "--gpu", "1"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "--allow-incomplete-retry" in blocked.stderr

    planned = subprocess.run(
        _cli(
            tmp_path,
            "launch",
            "--seed",
            "223",
            "--gpu",
            "1",
            "--allow-incomplete-retry",
        ),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert planned.returncode == 0, planned.stderr
    assert "detached_tmux=true execute=False" in planned.stdout
    assert "no files or processes created" in planned.stdout
    assert checkpoint.read_bytes() == before
    assert not (tmp_path / "_launcher_logs").exists()


def test_process_parser_recovers_physical_gpu_and_filters_artifact_root(
    tmp_path: Path,
) -> None:
    command = (
        "/env/bin/python",
        "-m",
        operations.SOURCE_MODULE,
        "--config",
        str(REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE),
        "--seed",
        "331",
        "--device",
        "cuda:0",
        "--artifact-root",
        str(tmp_path),
    )
    record = operations.parse_source_process(
        pid=4321,
        command=command,
        environ={"CUDA_VISIBLE_DEVICES": "2"},
        cwd=REPO_ROOT,
        artifact_root=tmp_path,
    )
    assert record is not None
    assert (record.seed, record.gpu, record.pid) == (331, 2, 4321)
    assert (
        operations.parse_source_process(
            pid=4321,
            command=command,
            environ={"CUDA_VISIBLE_DEVICES": "2"},
            cwd=REPO_ROOT,
            artifact_root=tmp_path / "other",
        )
        is None
    )


def test_active_seed_and_busy_gpu_are_both_rejected(tmp_path: Path) -> None:
    snapshot = operations.StatusSnapshot(
        artifact_root=str(tmp_path),
        completed_runs=(),
        incomplete_runs=(),
        active_processes=(
            operations.ActiveProcess(
                pid=99,
                seed=449,
                gpu=3,
                command=("python",),
            ),
        ),
        active_reservations=(),
        invalid_launcher_records=(),
    )
    with pytest.raises(RuntimeError, match="seed 449 already has an active"):
        operations._assert_launchable(
            snapshot,
            seed=449,
            gpu=3,
            allow_incomplete_retry=False,
        )
    with pytest.raises(RuntimeError, match="physical GPU 3 already"):
        operations._assert_launchable(
            snapshot,
            seed=557,
            gpu=3,
            allow_incomplete_retry=False,
        )


@pytest.mark.parametrize(
    ("seed", "gpu", "expected"),
    [
        ("999", "0", "unregistered E21 seed"),
        ("113", "4", "unregistered physical GPU"),
    ],
)
def test_unregistered_seed_or_gpu_is_rejected_before_launch(
    tmp_path: Path,
    seed: str,
    gpu: str,
    expected: str,
) -> None:
    result = subprocess.run(
        _cli(tmp_path, "launch", "--seed", seed, "--gpu", gpu),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert expected in result.stderr
    assert not (tmp_path / "_launcher_logs").exists()


def test_active_named_tmux_reservation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_dir = (
        tmp_path
        / "_launcher_logs"
        / "e21_source_tmux_20260728T120000.000003Z_seed557"
    )
    session = "catena_e21_seed557_20260728T120000_000003Z"
    _write_json(
        launch_dir / "launch_record.json",
        {
            "schema_version": 1,
            "experiment_id": operations.SOURCE_EXPERIMENT_ID,
            "artifact_root": str(tmp_path.resolve()),
            "seed": 557,
            "gpu": 3,
            "session_name": session,
        },
    )
    monkeypatch.setattr(operations, "_tmux_sessions", lambda _binary: {session})
    reservations, invalid = operations.scan_active_reservations(tmp_path)
    assert invalid == ()
    assert len(reservations) == 1
    assert reservations[0].session_name == session


def test_launch_record_is_additive_and_names_detached_session(
    tmp_path: Path,
) -> None:
    contract = operations.validate_registered_contract(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE,
    )
    record_path, record = operations._create_launch_record(
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        config_path=REPO_ROOT / operations.SOURCE_CONFIG_RELATIVE,
        python_bin=Path(sys.executable),
        seed=557,
        gpu=3,
        allow_incomplete_retry=True,
        contract=contract,
    )
    assert record_path.is_file()
    assert record["session_name"].startswith("catena_e21_seed557_")
    assert record["resume_from_partial_checkpoint"] is False
    assert record["aggregate_autorun"] is False
    assert record["source_command"][1:3] == ["-m", operations.SOURCE_MODULE]
    with pytest.raises(FileExistsError):
        operations._write_new_json(record_path, record)
