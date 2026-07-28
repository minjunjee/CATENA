from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    REPO_ROOT / "scripts/run_postcore_dry.sh",
    REPO_ROOT / "scripts/launch_postcore_wave1.sh",
    REPO_ROOT / "scripts/launch_sequence_wave.sh",
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_postcore_operational_script_has_valid_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_dry_runner_refuses_cleanup_outside_dedicated_tmp_child(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "must_survive.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_postcore_dry.sh"),
            str(REPO_ROOT),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing destructive dry cleanup" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_dry_runner_can_validate_safe_target_without_changing_it() -> None:
    target = Path("/tmp") / f"catena_postcore_dry_test_{uuid.uuid4().hex}"
    env = {**os.environ, "CATENA_DRY_VALIDATE_ONLY": "1"}
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/run_postcore_dry.sh"),
            str(REPO_ROOT),
            str(target),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Canonical post-core dry target" in result.stdout
    assert "validation only" in result.stdout
    assert not target.exists()


def test_sequence_launcher_detects_an_alive_wave_target_without_launching() -> None:
    marker = [
        "e13b_transactional_sequence_memory",
        "--variant",
        "tied",
        "--seed",
        "503",
    ]
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *marker]
    )
    try:
        env = {
            **os.environ,
            "CATENA_LAUNCH_CHECK_ONLY": "1",
            "CATENA_PYTHON": sys.executable,
            "CATENA_V6_PREFIX": sys.prefix,
        }
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts/launch_sequence_wave.sh"),
                str(REPO_ROOT),
                "3",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "Sequence target is already alive" in result.stderr
        assert "seed=503" in result.stderr
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=10)


def test_wave1_launcher_detects_an_alive_target_without_launching() -> None:
    sleeper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "e10_learned_rank_scaling",
        ]
    )
    try:
        env = {
            **os.environ,
            "CATENA_LAUNCH_CHECK_ONLY": "1",
            "CATENA_PYTHON": sys.executable,
            "CATENA_V6_PREFIX": sys.prefix,
        }
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts/launch_postcore_wave1.sh"),
                str(REPO_ROOT),
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "Target experiment is already alive" in result.stderr
        assert "e10_learned_rank_scaling" in result.stderr
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=10)


def test_launchers_use_explicit_catena_python_modules_and_keep_three_waves() -> None:
    wave1 = (REPO_ROOT / "scripts/launch_postcore_wave1.sh").read_text(
        encoding="utf-8"
    )
    sequence = (REPO_ROOT / "scripts/launch_sequence_wave.sh").read_text(
        encoding="utf-8"
    )
    assert '"$PYTHON_BIN" -m "$module"' in wave1
    assert '"$PYTHON_BIN" -m experiments.e13b_transactional_sequence_memory' in (
        sequence
    )
    for registered_job in (
        'JOBS=("0 tied 101" "1 dual 101" "2 tied 211" "3 dual 211")',
        'JOBS=("0 tied 307" "1 dual 307" "2 tied 401" "3 dual 401")',
        'JOBS=("0 tied 503" "1 dual 503")',
    ):
        assert registered_job in sequence
