from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.common import finalize_run, initialize_run


def test_common_runner_writes_postcore_artifact_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"experiment_id": "probe", "value": 3}),
        encoding="utf-8",
    )

    config, run_dir, device = initialize_run(
        experiment_id="probe",
        config_path=str(config_path),
        artifact_root=str(tmp_path / "artifacts"),
        device_request="cpu",
        run_mode="DRY_RUN",
    )
    finalize_run(
        experiment_id="probe",
        artifact_root=str(tmp_path / "artifacts"),
        run_dir=run_dir,
        report={"status": "DRY_RUN"},
    )

    assert config["value"] == 3
    assert str(device) == "cpu"
    assert (run_dir / "config.resolved.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["run_mode"] == "DRY_RUN"
    assert len(manifest["config_file_sha256"]) == 64
    assert len(manifest["resolved_config_sha256"]) == 64
    assert len(manifest["source_fingerprint"]["sha256"]) == 64
    assert len(manifest["report_sha256"]) == 64
