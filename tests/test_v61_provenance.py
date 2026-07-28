from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from catena.core.provenance_v61 import (
    ManifestValidationRequirements,
    ProvenanceValidationError,
    StrictJSONError,
    loads_json_strict,
    read_jsonl_strict,
    resolve_latest_run,
    resolve_within_root,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    source_tree_fingerprint,
    validate_latest_run,
    validate_run_manifest,
    write_json_strict,
    write_jsonl_strict,
)

RUN_ID = "20260726T120000.000000Z"
SOURCE_SHA256 = "a" * 64


def _make_completed_run(
    tmp_path: Path,
    *,
    experiment_id: str = "e_test",
    run_mode: str = "main",
    main_eligible: bool = True,
    full_eligible: bool = True,
    status: str = "PASS",
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / experiment_id / RUN_ID
    run_dir.mkdir(parents=True)

    config = {"experiment_id": experiment_id, "seed": 17}
    config_path = tmp_path / "configs" / f"{experiment_id}.json"
    write_json_strict(config_path, config)
    source = {"sha256": SOURCE_SHA256, "files": 12}
    eligibility = {"main": main_eligible, "full": full_eligible}
    report = {
        "experiment_id": experiment_id,
        "run_id": RUN_ID,
        "status": status,
        "run_mode": run_mode,
        "eligibility": eligibility,
        "source_fingerprint": source,
    }
    report_path = run_dir / "report.json"
    write_json_strict(report_path, report)
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "experiment_id": experiment_id,
        "run_id": RUN_ID,
        "run_dir": str(run_dir.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "completed_at_utc": "2026-07-26T12:00:00+00:00",
        "status": status,
        "run_mode": run_mode,
        "eligibility": eligibility,
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_canonical_json(config),
        "config_file_sha256": sha256_file(config_path),
        "source_fingerprint": source,
        "source_fingerprint_verified_at_completion": True,
        "report_sha256": sha256_file(report_path),
        "dependencies": [],
    }
    write_json_strict(run_dir / "run_manifest.json", manifest)
    write_json_strict(artifact_root / experiment_id / "latest.json", {"run_dir": RUN_ID})
    return artifact_root, run_dir


def test_strict_hash_helpers_are_deterministic(tmp_path: Path) -> None:
    assert sha256_canonical_json({"b": 2, "a": 1}) == sha256_canonical_json({"a": 1, "b": 2})
    target = tmp_path / "bytes.bin"
    target.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()
    assert sha256_bytes(b"abc") == expected
    assert sha256_file(target) == expected


def test_strict_json_rejects_nonfinite_and_duplicate_keys(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"preserved":true}\n', encoding="utf-8")
    with pytest.raises(StrictJSONError, match="Non-finite"):
        write_json_strict(target, {"metric": float("nan")})
    assert target.read_text(encoding="utf-8") == '{"preserved":true}\n'

    with pytest.raises(StrictJSONError, match="Non-standard"):
        loads_json_strict('{"metric": NaN}')
    with pytest.raises(StrictJSONError, match="Duplicate"):
        loads_json_strict('{"metric": 1, "metric": 2}')


def test_strict_jsonl_is_atomic_and_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    write_jsonl_strict(target, [{"row": 1}, {"row": 2}])
    assert read_jsonl_strict(target) == [{"row": 1}, {"row": 2}]

    with pytest.raises(StrictJSONError, match="row 2"):
        write_jsonl_strict(target, [{"row": 3}, {"row": float("inf")}])
    assert read_jsonl_strict(target) == [{"row": 1}, {"row": 2}]


def test_source_tree_fingerprint_prunes_artifacts_and_caches(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    artifact_file = tmp_path / "artifacts_large" / "run" / "ignored.py"
    cache_file = tmp_path / ".pytest_cache" / "ignored.py"
    artifact_file.parent.mkdir(parents=True)
    cache_file.parent.mkdir(parents=True)
    artifact_file.write_text("first\n", encoding="utf-8")
    cache_file.write_text("first\n", encoding="utf-8")

    before = source_tree_fingerprint(tmp_path)
    artifact_file.write_text("second\n", encoding="utf-8")
    cache_file.write_text("second\n", encoding="utf-8")
    assert source_tree_fingerprint(tmp_path) == before
    assert before.files == 2

    (tmp_path / "model.py").write_text("value = 2\n", encoding="utf-8")
    assert source_tree_fingerprint(tmp_path).sha256 != before.sha256


def test_resolve_within_root_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    assert resolve_within_root(root, "child") == child.resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ProvenanceValidationError, match="escapes"):
        resolve_within_root(root, outside)
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProvenanceValidationError, match="escapes"):
        resolve_within_root(root, link)


def test_latest_pointer_resolves_relative_direct_child(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(tmp_path)
    assert resolve_latest_run(artifact_root, "e_test") == run_dir.resolve()


def test_latest_pointer_rejects_run_outside_experiment(tmp_path: Path) -> None:
    artifact_root, _ = _make_completed_run(tmp_path)
    outside = artifact_root / "other" / RUN_ID
    outside.mkdir(parents=True)
    write_json_strict(
        artifact_root / "e_test" / "latest.json",
        {"run_dir": str(outside.resolve())},
    )
    with pytest.raises(ProvenanceValidationError, match="escapes"):
        resolve_latest_run(artifact_root, "e_test")


def test_generic_manifest_validator_accepts_main_full_run(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(tmp_path)
    requirements = ManifestValidationRequirements(
        expected_experiment_id="e_test",
        accepted_schema_versions=frozenset({3}),
        expected_source_sha256=SOURCE_SHA256,
        expected_source_files=12,
        expected_run_mode="main",
        require_main_eligible=True,
        require_full_eligible=True,
    )
    validated = validate_latest_run(
        artifact_root,
        "e_test",
        requirements=requirements,
    )
    assert validated.run_dir == run_dir.resolve()
    assert validated.main_eligible is True
    assert validated.full_eligible is True
    assert validated.dependency_record()["manifest_sha256"] == sha256_file(
        run_dir / "run_manifest.json"
    )


def test_manifest_validator_rejects_report_tampering(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(tmp_path)
    report_path = run_dir / "report.json"
    report = loads_json_strict(report_path.read_bytes())
    report["unexpected"] = True
    write_json_strict(report_path, report)
    with pytest.raises(ProvenanceValidationError, match="report SHA-256"):
        validate_run_manifest(run_dir, artifact_root)


def test_manifest_validator_rejects_config_payload_tampering(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = loads_json_strict(manifest_path.read_bytes())
    manifest["config"]["seed"] = 18
    write_json_strict(manifest_path, manifest)
    with pytest.raises(ProvenanceValidationError, match="config payload SHA-256"):
        validate_run_manifest(run_dir, artifact_root)


def test_manifest_validator_rejects_config_file_tampering(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(tmp_path)
    manifest = loads_json_strict((run_dir / "run_manifest.json").read_bytes())
    Path(manifest["config_path"]).write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ProvenanceValidationError, match="config file SHA-256"):
        validate_run_manifest(run_dir, artifact_root)


def test_manifest_validator_enforces_source_and_run_eligibility(tmp_path: Path) -> None:
    artifact_root, run_dir = _make_completed_run(
        tmp_path,
        run_mode="dry_run",
        main_eligible=False,
        full_eligible=False,
    )
    with pytest.raises(ProvenanceValidationError, match="source fingerprint"):
        validate_run_manifest(
            run_dir,
            artifact_root,
            requirements=ManifestValidationRequirements(expected_source_sha256="b" * 64),
        )
    with pytest.raises(ProvenanceValidationError, match="run_mode"):
        validate_run_manifest(
            run_dir,
            artifact_root,
            requirements=ManifestValidationRequirements(expected_run_mode="main"),
        )
    with pytest.raises(ProvenanceValidationError, match="main-eligible"):
        validate_run_manifest(
            run_dir,
            artifact_root,
            requirements=ManifestValidationRequirements(require_main_eligible=True),
        )
    with pytest.raises(ProvenanceValidationError, match="full-eligible"):
        validate_run_manifest(
            run_dir,
            artifact_root,
            requirements=ManifestValidationRequirements(require_full_eligible=True),
        )
