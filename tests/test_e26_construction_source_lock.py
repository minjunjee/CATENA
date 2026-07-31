from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import pytest

from catena.core.provenance_v61 import write_json_strict
from catena.lm.construction_source import (
    CONSTRUCTION_SOURCE_FILES,
    CRITICAL_TOOL_VERSIONS,
    REQUIRED_ARTIFACT_BINDINGS,
    build_construction_source_receipt,
)
from catena.lm.data_readiness_v2 import (
    Stage2DataReadinessError,
    _validate_construction_source,
)


def _run(repo: Path, *arguments: str) -> None:
    subprocess.run(("git", "-C", str(repo), *arguments), check=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    for package, expected in CRITICAL_TOOL_VERSIONS.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("construction lock test requires pinned isolated data environment")
        if observed != expected:
            pytest.skip("construction lock test requires pinned isolated data environment")

    source_root = Path(__file__).parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in CONSTRUCTION_SOURCE_FILES:
        source = source_root / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _run(repo, "init", "-b", "construction")
    _run(repo, "config", "user.email", "catena-test@example.invalid")
    _run(repo, "config", "user.name", "CATENA Test")
    _run(repo, "add", ".")
    _run(repo, "commit", "-m", "construction source")

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    bindings: dict[str, Path] = {}
    for label in REQUIRED_ARTIFACT_BINDINGS:
        if label == "data_lock":
            path = repo / "configs/e26_data_lock_v1.yaml"
        else:
            path = artifact_root / f"{label}.json"
            path.write_text(f'{{"label":"{label}"}}\n', encoding="utf-8")
        bindings[label] = path
    payload = build_construction_source_receipt(
        repo_root=repo,
        artifact_bindings=bindings,
    )
    receipt = artifact_root / "construction.json"
    write_json_strict(receipt, payload)
    return repo, receipt, bindings


def test_construction_commit_can_have_clean_report_only_descendant(
    tmp_path: Path,
) -> None:
    repo, receipt, bindings = _fixture(tmp_path)
    _validate_construction_source(receipt, artifact_paths=bindings)
    report = repo / "REPORT.md"
    report.write_text("report only\n", encoding="utf-8")
    _run(repo, "add", "REPORT.md")
    _run(repo, "commit", "-m", "report descendant")
    _validate_construction_source(receipt, artifact_paths=bindings)


def test_construction_rejects_non_descendant_and_builder_change(tmp_path: Path) -> None:
    repo, receipt, bindings = _fixture(tmp_path)
    construction_head = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run(repo, "switch", "--orphan", "unrelated")
    _run(repo, "commit", "--allow-empty", "-m", "unrelated source")
    with pytest.raises(Stage2DataReadinessError, match="not an ancestor"):
        _validate_construction_source(receipt, artifact_paths=bindings)

    _run(repo, "switch", "-C", "changed-builder", construction_head)
    builder = repo / "src/catena/lm/data_lock.py"
    builder.write_text(builder.read_text(encoding="utf-8") + "\n# changed\n")
    _run(repo, "add", str(builder.relative_to(repo)))
    _run(repo, "commit", "-m", "change builder")
    with pytest.raises(Stage2DataReadinessError, match="construction builder changed"):
        _validate_construction_source(receipt, artifact_paths=bindings)
