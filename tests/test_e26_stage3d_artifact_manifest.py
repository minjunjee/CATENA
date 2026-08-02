from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from catena.core.provenance_v61 import sha256_canonical_json

_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "prepare_e26_stage3d_artifact_manifest.py"
)
_SPEC = importlib.util.spec_from_file_location("e26_stage3d_manifest_tool", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def _git_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CATENA Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "catena-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    (repo / "tracked.txt").write_text("locked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "lock"], cwd=repo, check=True)
    return repo


def _stage3c_tree(tmp_path: Path) -> Path:
    root = tmp_path / "stage3c"
    root.mkdir()
    for index, name in enumerate(_TOOL.EXPECTED_RAW_FILES):
        (root / name).write_text(f"raw-{index}\n", encoding="utf-8")
    (root / "RESULTS_SUMMARY_KO.md").write_text("summary\n", encoding="utf-8")
    return root


def test_manifest_binds_exact_eleven_file_raw_payload(tmp_path: Path) -> None:
    payload = _TOOL.build_manifest(
        repo_root=_git_repo(tmp_path),
        artifact_root=_stage3c_tree(tmp_path),
        require_registered_anchors=False,
    )
    assert payload["file_count"] == 11
    assert [row["path"] for row in payload["files"]] == list(_TOOL.EXPECTED_RAW_FILES)
    assert "RESULTS_SUMMARY_KO.md" not in {row["path"] for row in payload["files"]}
    observed = payload.pop("manifest_sha256")
    assert observed == sha256_canonical_json(payload)


def test_manifest_fails_if_predecessor_file_set_changes(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    root = _stage3c_tree(tmp_path)
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set changed"):
        _TOOL.build_manifest(
            repo_root=repo,
            artifact_root=root,
            require_registered_anchors=False,
        )


def test_manifest_requires_clean_committed_source(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    root = _stage3c_tree(tmp_path)
    (repo / "dirty.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean committed"):
        _TOOL.build_manifest(
            repo_root=repo,
            artifact_root=root,
            require_registered_anchors=False,
        )


def test_production_predecessor_anchors_are_explicit() -> None:
    assert _TOOL.REGISTERED_STAGE3C_RESULT_SHA256 == (
        "83fab26e7936654b664653776d501c3fdee6cb7f0ffd78c3d9682ed41d319b56"
    )
    assert _TOOL.REGISTERED_STAGE3C_STATUS_SHA256 == (
        "15b896a33e0fe286c80f2c204b7be2be0fbe6aaf8cdc512fafbd31040f8aabda"
    )
    assert _TOOL.REGISTERED_RAW_RUN_AGGREGATE_SHA256 == (
        "296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc"
    )
    assert _TOOL.REGISTERED_FAILURE_STATUS_SHA256 == (
        "dc7ed1837ccf022fe5110fdb44907c5e340391f0bcc5c92b7d5e26dcf2a95616"
    )
