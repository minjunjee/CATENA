from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_post_e21_dry.sh"


def _validation_env() -> dict[str, str]:
    return {
        **os.environ,
        "CATENA_POST_E21_DRY_VALIDATE_ONLY": "1",
        "CATENA_PYTHON": sys.executable,
        "CATENA_V6_PREFIX": sys.prefix,
    }


def test_post_e21_dry_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_post_e21_dry_script_accepts_only_fresh_named_tmp_root() -> None:
    target = Path("/tmp") / f"catena_post_e21_dry_test_{uuid.uuid4().hex}"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(target)],
        env=_validation_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Validation only" in result.stdout
    assert not target.exists()


def test_post_e21_dry_script_refuses_existing_root_without_cleanup(
    tmp_path: Path,
) -> None:
    target = Path("/tmp") / f"catena_post_e21_dry_existing_{uuid.uuid4().hex}"
    target.mkdir()
    sentinel = target / "must_survive.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), str(target)],
            env=_validation_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "must not already exist" in result.stderr
        assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    finally:
        sentinel.unlink()
        target.rmdir()

    outside = tmp_path / "catena_post_e21_dry_wrong_parent"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(outside)],
        env=_validation_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "direct fresh /tmp" in result.stderr
    assert not outside.exists()


def test_post_e21_dry_script_order_and_forbidden_modes_are_static() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ordered = (
        "e22a_locality_method_selection",
        "e22b_active_path_locality",
        "e23a_product_poset_screen",
        "e23b_product_poset_confirmatory",
        "e24a_approximate_rank_stress",
        "e24b_behavioral_attainability_stress",
        "e25a_official_gdn2_gate",
        "e25b_text_transaction_anchor",
    )
    positions = [source.index(experiment_id) for experiment_id in ordered]
    assert positions == sorted(positions)
    assert '--selection-run "$POST_E21_E22A_RUN"' in source
    assert "--stage gate" in source
    assert "--allow-scientific-replication" not in source
    assert "--prepare-audit" in source
    assert "AUDIT_PREPARATION" in source
    assert "audit_artifacts" in source
    assert "E25B_V3_HUMAN_AUDIT_POPULATION_LOCK.json" not in source
    assert "--audit-csv" not in source
    assert "--e22b-run" not in source
    assert "rm -rf" not in source
    assert "CATENA_POST_E21_DRY_VALIDATE_ONLY" in source
