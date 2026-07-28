from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from catena.core.config import load_config
from catena.core.io import file_sha256
from experiments import e13b_r1_transactional_sequence_memory as e13b
from experiments import e13c_r1_transactional_sequence_aggregate as e13c

REPO_ROOT = Path(__file__).resolve().parents[1]
E13B_CONFIG = REPO_ROOT / "configs/e13b_r1_transactional_sequence_memory.yaml"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _strict_json(path: Path) -> object:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )


def _fake_r2_dependency(artifact_root: Path) -> dict[str, str]:
    run_dir = (
        artifact_root
        / e13b.CALIBRATION_EXPERIMENT_ID
        / "20260727T200000.000000Z"
    )
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    source_hash = file_sha256(E13B_CONFIG)
    report = {
        "status": "PASS",
        "claim_gate": {"go_for_e13b_r1": True},
        "distractor_path_contract": {"passed": True},
        "e13b_scale_feasibility": {
            "passed": True,
            "source_config_file_sha256": source_hash,
        },
    }
    _write_json(report_path, report)
    report_hash = file_sha256(report_path)
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "experiment_id": e13b.CALIBRATION_EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
            "report_sha256": report_hash,
        },
    )
    _write_json(
        artifact_root / e13b.CALIBRATION_EXPERIMENT_ID / "latest.json",
        {"run_dir": str(run_dir.resolve())},
    )
    return {
        "experiment_id": e13b.CALIBRATION_EXPERIMENT_ID,
        "run_dir": str(run_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "report_sha256": report_hash,
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": file_sha256(manifest_path),
        "source_config_sha256": source_hash,
    }


def _metric_row(
    *,
    seed: int = 101,
    variant: str = "dual",
    updates: int = 8,
    gap: int = 2048,
    checkpoint_hash: str = "c" * 64,
) -> dict:
    row = {
        "seed": seed,
        "variant": variant,
        "updates": updates,
        "gap_events": gap,
        "checkpoint_sha256": checkpoint_hash,
        "base_transaction_digest": "d" * 64,
        "activate_distractor_verified": False,
        "affected_mse": 0.001,
        "retention_mse": 0.0001,
        "old_rule_residual": 0.0001,
        "entity_exact_match": 0.99,
        "affected_entity_exact_match": 0.98,
        "verified_erase_gate_mean": 0.5,
        "verified_write_gate_mean": 0.5,
        "distractor_erase_gate_mean": 0.01,
        "distractor_write_gate_mean": 0.01,
        "distractor_joint_gate_mass_per_sequence": 2.0,
    }
    if updates == 8 and gap == 2048:
        row.update(
            {
                "distractor_activation_activate_distractor_verified": True,
                "distractor_activation_affected_mse": 0.003,
                "distractor_activation_retention_mse": 0.0012,
                "distractor_activation_old_rule_residual": 0.002,
                "distractor_activation_distractor_erase_gate_mean": 0.5,
                "distractor_activation_distractor_write_gate_mean": 0.5,
            }
        )
    return row


def _write_run_start_only(run_dir: Path) -> None:
    config = load_config(E13B_CONFIG)
    resolved_path = run_dir / "config.resolved.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_json(run_dir / "environment.json", {"python": "test"})
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 2,
            "experiment_id": e13b.EXPERIMENT_ID,
            "run_id": run_dir.name,
            "run_mode": "MAIN",
            "created_at_utc": "2026-07-27T19:00:00+00:00",
            "config": config,
            "config_file_sha256": file_sha256(E13B_CONFIG),
            "resolved_config_artifact_sha256": file_sha256(resolved_path),
            "resolved_config_sha256": (
                e13c.legacy._canonical_sha256(config)
            ),
            "source_fingerprint_phase": "RUN_START",
            "source_fingerprint": {"files": 100, "sha256": "a" * 64},
        },
    )


def test_b_r1_dependency_pins_main_manifest_and_report_hash(
    tmp_path: Path,
) -> None:
    expected = _fake_r2_dependency(tmp_path)

    observed = e13b._load_calibration_dependency(
        artifact_root=str(tmp_path),
        source_config_path=str(E13B_CONFIG),
    )

    assert observed == expected
    report_path = Path(expected["report_path"])
    report = _strict_json(report_path)
    assert isinstance(report, dict)
    report["tampered"] = True
    _write_json(report_path, report)
    with pytest.raises(RuntimeError, match="manifest"):
        e13b._load_calibration_dependency(
            artifact_root=str(tmp_path),
            source_config_path=str(E13B_CONFIG),
        )


def test_b_r1_main_rows_pair_digests_and_confine_active_assay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_train(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            final_loss=0.1,
            best_loss=0.05,
            examples_per_second=1000.0,
            peak_memory_bytes=0,
        )

    def fake_evaluate(
        *,
        updates: int,
        seed: int,
        activate_distractor_verified: bool = False,
        **_kwargs: object,
    ) -> dict:
        digest = hashlib.sha256(f"{seed}:{updates}".encode()).hexdigest()
        retention = 0.002 if activate_distractor_verified else 0.0001
        return {
            "affected_mse": 0.003 if activate_distractor_verified else 0.001,
            "retention_mse": retention,
            "old_rule_residual": 0.0001,
            "entity_exact_match": 0.99,
            "affected_entity_exact_match": 0.98,
            "affected_entity_count": 8,
            "unaffected_entity_count": 24,
            "verified_event_count": 8,
            "distractor_event_count": 16,
            "verified_erase_gate_mean": 0.5,
            "verified_write_gate_mean": 0.5,
            "distractor_erase_gate_mean": 0.01,
            "distractor_write_gate_mean": 0.01,
            "distractor_joint_gate_mass_per_sequence": 2.0,
            "activate_distractor_verified": activate_distractor_verified,
            "base_transaction_digest": digest,
        }

    monkeypatch.setattr(e13b, "train_sequence_memory_v2", fake_train)
    monkeypatch.setattr(e13b, "evaluate_sequence_memory_v2", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e13b-r1",
            "--config",
            str(E13B_CONFIG),
            "--device",
            "cpu",
            "--artifact-root",
            str(tmp_path),
            "--variant",
            "dual",
            "--seed",
            "101",
            "--ignore-calibration",
        ],
    )
    e13b.main()

    latest = _strict_json(
        tmp_path / e13b.EXPERIMENT_ID / "latest.json"
    )
    assert isinstance(latest, dict)
    run_dir = Path(str(latest["run_dir"]))
    rows = [
        json.loads(line)
        for line in (run_dir / "sequence_main_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 12
    for updates in (1, 4, 8):
        update_rows = [row for row in rows if row["updates"] == updates]
        assert len({row["base_transaction_digest"] for row in update_rows}) == 1
    assay_rows = [
        row
        for row in rows
        if "distractor_activation_retention_mse" in row
    ]
    assert len(assay_rows) == 1
    assert (assay_rows[0]["updates"], assay_rows[0]["gap_events"]) == (8, 2048)
    assert assay_rows[0][
        "distractor_activation_activate_distractor_verified"
    ] is True
    _strict_json(run_dir / "report.json")
    for line in (run_dir / "sequence_main_metrics.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda value: pytest.fail(value))


def test_c_r1_stress_statistics_apply_all_registered_stress_gates() -> None:
    seeds = (101, 211, 307, 401, 503)
    rows: list[dict] = []
    for seed in seeds:
        rows.extend(
            [
                {
                    "seed": seed,
                    "variant": "tied",
                    "updates": 8,
                    "gap_events": 2048,
                    "affected_mse": 0.003,
                },
                {
                    "seed": seed,
                    "variant": "dual",
                    "updates": 8,
                    "gap_events": 0,
                    "affected_mse": 0.0008,
                },
                {
                    "seed": seed,
                    "variant": "dual",
                    "updates": 8,
                    "gap_events": 2048,
                    "affected_mse": 0.001,
                    "retention_mse": 0.0001,
                    "distractor_activation_retention_mse": 0.0012,
                },
            ]
        )
    stress_rows, statistics = e13c._stress_statistics(
        rows,
        seeds=seeds,
        gaps=(0, 128, 512, 2048),
        stress_updates=8,
        stress_gap=2048,
    )
    config = load_config(
        "configs/e13c_r1_transactional_sequence_aggregate.yaml"
    )
    conditions = e13c._stress_gate_conditions(
        statistics,
        claim_gate=config["claim_gate"],
        alpha=float(config["statistics"]["alpha"]),
    )

    assert len(stress_rows) == 5
    assert statistics["stress_sign_flip_p"] == 0.03125
    assert all(conditions.values())


def test_c_r1_rejects_checkpoint_tamper_and_duplicate_rows() -> None:
    provenance = [
        {
            "seed": 101,
            "variant": "dual",
            "checkpoint_sha256": "c" * 64,
        }
    ]
    row = _metric_row()
    e13c._validate_r1_metric_rows(
        [row],
        provenance,
        stress_updates=8,
        stress_gap=2048,
        dry_run=False,
    )

    tampered = deepcopy(row)
    tampered["checkpoint_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        e13c._validate_r1_metric_rows(
            [tampered],
            provenance,
            stress_updates=8,
            stress_gap=2048,
            dry_run=False,
        )
    with pytest.raises(RuntimeError, match="Duplicate E13b-R1 metric row"):
        e13c._row_index([row, deepcopy(row)])


def test_c_r1_excludes_only_exact_run_start_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "20260727T190000.000000Z"
    _write_run_start_only(run_dir)
    config = load_config(E13B_CONFIG)

    record = e13c._operational_incomplete_run_start_record(
        run_dir,
        expected_source_config=config,
        source_config_path=E13B_CONFIG,
    )

    assert record is not None
    assert record["run_dir"] == str(run_dir.resolve())
    assert record["reason"] == (
        "RUN_START_PROVENANCE_ONLY_NO_REPORT_METRICS_OR_CHECKPOINT"
    )
    assert record["run_manifest_sha256"] == file_sha256(
        run_dir / "run_manifest.json"
    )

    manifest_path = run_dir / "run_manifest.json"
    manifest = _strict_json(manifest_path)
    assert isinstance(manifest, dict)
    manifest["experiment_id"] = "tampered"
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="not a valid run-start"):
        e13c._operational_incomplete_run_start_record(
            run_dir,
            expected_source_config=config,
            source_config_path=E13B_CONFIG,
        )

    _write_run_start_only(run_dir)
    _write_json(run_dir / "report.json", {"status": "PASS"})
    assert (
        e13c._operational_incomplete_run_start_record(
            run_dir,
            expected_source_config=config,
            source_config_path=E13B_CONFIG,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="missing"):
        e13c.legacy._validate_source_run(
            run_dir,
            expected_source_config=config,
            required_updates=(1, 4, 8),
            required_gaps=(0, 128, 512, 2048),
            dry_run=False,
        )


def test_c_r1_collection_reports_exclusion_and_still_rejects_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / e13b.EXPERIMENT_ID
    _write_run_start_only(
        source_root / "20260727T190000.000000Z"
    )
    seeds = (101, 211, 307, 401, 503)
    variants = ("tied", "dual")
    for seed in seeds:
        for variant in variants:
            run_dir = source_root / f"complete_{seed}_{variant}"
            run_dir.mkdir(parents=True)
            (run_dir / "complete.marker").write_text("complete")

    def fake_validate(
        run_dir: Path,
        **_kwargs: object,
    ) -> tuple[tuple[int, str], list[dict], dict]:
        _, seed_text, variant = run_dir.name.split("_")
        seed = int(seed_text)
        return (
            (seed, variant),
            [{"seed": seed, "variant": variant}],
            {"seed": seed, "variant": variant},
        )

    monkeypatch.setattr(
        e13c.legacy,
        "SOURCE_EXPERIMENT_ID",
        e13c.SOURCE_EXPERIMENT_ID,
    )
    monkeypatch.setattr(
        e13c.legacy,
        "_validate_source_run",
        fake_validate,
    )
    config = load_config(
        "configs/e13c_r1_transactional_sequence_aggregate.yaml"
    )
    rows, provenance, excluded = e13c.collect_e13b_r1_sources(
        artifact_root=tmp_path,
        config=config,
        dry_run=False,
    )

    assert len(rows) == 10
    assert len(provenance) == 10
    assert len(excluded) == 1
    assert excluded[0]["disposition"] == (
        "EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY"
    )

    duplicate = source_root / "duplicate_101_tied"
    duplicate.mkdir()
    (duplicate / "complete.marker").write_text("complete")
    with pytest.raises(RuntimeError, match="Duplicate eligible E13b-R1"):
        e13c.collect_e13b_r1_sources(
            artifact_root=tmp_path,
            config=config,
            dry_run=False,
        )


def test_r2_pipeline_cpu_dry_runs_are_non_supporting_and_strict_json(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    def run(module: str, config: str, *extra: str) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                "--config",
                config,
                "--device",
                "cpu",
                "--artifact-root",
                str(tmp_path),
                "--dry-run",
                *extra,
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    run(
        "experiments.e13a_r2_sequence_floor_throughput",
        "configs/e13a_r2_sequence_floor_throughput.yaml",
    )
    for variant in ("tied", "dual"):
        run(
            "experiments.e13b_r1_transactional_sequence_memory",
            "configs/e13b_r1_transactional_sequence_memory.yaml",
            "--variant",
            variant,
            "--seed",
            "101",
            "--ignore-calibration",
        )
    run(
        "experiments.e13c_r1_transactional_sequence_aggregate",
        "configs/e13c_r1_transactional_sequence_aggregate.yaml",
    )

    for path in sorted(tmp_path.rglob("*.json")):
        _strict_json(path)
    for path in sorted(tmp_path.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                json.loads(
                    line,
                    parse_constant=lambda value: pytest.fail(value),
                )
    r2_latest = _strict_json(
        tmp_path / e13b.CALIBRATION_EXPERIMENT_ID / "latest.json"
    )
    assert isinstance(r2_latest, dict)
    r2_report = _strict_json(Path(str(r2_latest["run_dir"])) / "report.json")
    assert isinstance(r2_report, dict)
    assert r2_report["claim_gate"]["go_for_e13b_r1"] is False
    c_latest = _strict_json(
        tmp_path / e13c.EXPERIMENT_ID / "latest.json"
    )
    assert isinstance(c_latest, dict)
    c_report = _strict_json(Path(str(c_latest["run_dir"])) / "report.json")
    assert isinstance(c_report, dict)
    assert c_report["claim_gate"]["supported"] is False
    assert all(
        c_report["summary"][key] is None
        for key in (
            "stress_mean_affected_gain",
            "stress_sign_flip_p",
            "maximum_dual_gap_degradation",
            "minimum_active_path_retention_harm",
        )
    )


def test_r1_launcher_preflight_uses_exact_environment_and_rejects_duplicate(
    tmp_path: Path,
) -> None:
    _fake_r2_dependency(tmp_path)
    script = REPO_ROOT / "scripts/launch_sequence_r1_wave.sh"
    env = {
        **os.environ,
        "CATENA_ARTIFACT_ROOT": str(tmp_path),
        "CATENA_LAUNCH_CHECK_ONLY": "1",
        "CATENA_PYTHON": sys.executable,
        "CATENA_V6_PREFIX": sys.prefix,
    }
    first = subprocess.run(
        ["bash", str(script), str(REPO_ROOT), "3"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert "preflight only; no jobs were started" in first.stdout
    assert not (tmp_path / e13b.EXPERIMENT_ID).exists()

    duplicate = tmp_path / e13b.EXPERIMENT_ID / "20260727T210000.000000Z"
    _write_json(duplicate / "report.json", {"status": "PASS"})
    _write_json(duplicate / "run_manifest.json", {"run_mode": "MAIN"})
    (duplicate / "sequence_main_metrics.jsonl").write_text(
        json.dumps({"variant": "tied", "seed": 503}) + "\n",
        encoding="utf-8",
    )
    second = subprocess.run(
        ["bash", str(script), str(REPO_ROOT), "3"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "Completed E13b-R1 target already exists" in (
        second.stdout + second.stderr
    )
