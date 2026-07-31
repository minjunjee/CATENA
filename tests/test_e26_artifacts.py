import json
from pathlib import Path

from catena.lm.artifacts import ArtifactRun


def test_dry_run_artifact_contract(tmp_path: Path) -> None:
    # ArtifactRun intentionally requires a /tmp direct child.
    root = Path("/tmp") / f"catena_e26_dry_pytest_{tmp_path.name}"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    run.write("protocol_lock.json", {"schema_version": "catena-v8.1"})
    run.finalize(
        {
            "schema_version": "catena-v8.1",
            "experiment": "e26a_operator_data_gate",
            "run_id": run.run_id,
            "run_mode": "DRY_RUN",
            "status": "PASS",
            "scientific_evidence": False,
            "evidence_tier": "NON_EVIDENCE_VALIDATION",
            "disposition": "TEST",
            "allowed_claim": "artifact writer works",
            "forbidden_claims": [],
            "gates": [],
            "artifacts": {},
        },
        "# test",
    )
    assert (run.run_dir / "report.json").is_file()
    assert (run.run_dir / "RESULTS_SUMMARY_KO.md").is_file()
    latest = json.loads((root / "e26a_operator_data_gate" / "latest.json").read_text())
    assert latest["run_id"] == run.run_id
