from __future__ import annotations

import importlib.util
from pathlib import Path


def _status_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/post_e21_status.py"
    specification = importlib.util.spec_from_file_location("post_e21_status", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_status_reports_not_run_without_latest(tmp_path: Path) -> None:
    module = _status_module()
    row = module.experiment_status(tmp_path, "e22a_locality_method_selection")
    assert row["state"] == "NOT_RUN"


def test_status_rejects_unsafe_latest(tmp_path: Path) -> None:
    module = _status_module()
    directory = tmp_path / "e22a_locality_method_selection"
    directory.mkdir()
    (directory / "latest.json").write_text(
        '{"run_dir": "/outside/artifacts"}',
        encoding="utf-8",
    )
    row = module.experiment_status(tmp_path.resolve(), directory.name)
    assert row["state"] == "UNSAFE_LATEST"


def test_launcher_defaults_to_nonexecuting_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/launch_post_e21_wave1.sh").read_text(encoding="utf-8")
    assert "CATENA_POST_E21_MAIN_ACK" in text
    assert "Preflight only. No process was started." in text
    assert "--parent-e21-freeze" in text
    assert "--e18-freeze" in text
    assert "--allow-main --dependency-root" in text
    assert "--stage gate" in text
    assert "--prepare-audit" in text
    assert "--allow-scientific-replication" not in text
    assert "E25A_GATE_TERMINAL" in text
    assert "no automatic rerun" in text
