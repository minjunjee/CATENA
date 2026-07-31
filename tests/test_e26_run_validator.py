from __future__ import annotations

from pathlib import Path

from catena.lm.artifacts import ArtifactRun


def test_run_validator_rejects_missing_contract_files(tmp_path: Path) -> None:
    from tools.validate_e26_run import validate_run

    root = Path("/tmp") / f"catena_e26_dry_validator_{tmp_path.name}"
    root.mkdir(exist_ok=False)
    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    try:
        validate_run(
            run.run_dir,
            Path(__file__).resolve().parents[1] / "schemas/v8_1",
        )
    except FileNotFoundError as error:
        assert "required artifacts" in str(error)
    else:
        raise AssertionError("Incomplete run unexpectedly validated")
