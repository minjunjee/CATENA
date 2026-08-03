from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tools import apply_e26_final_official_patch as official_patch

_SOURCE_TEMPLATE = b"""import torch

class Layer:
    def __init__(self, b_proj, w_proj, policy_marker):
        self.b_proj = b_proj
        self.w_proj = w_proj
        if policy_marker is not _MISSING:
            self.e26_gate_policy = policy_marker

    def forward(self, hidden_states):
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        return b, w
"""


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "official"
    target = repo / official_patch.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(_SOURCE_TEMPLATE)
    _git(repo, "init", "-b", "pinned")
    _git(repo, "config", "user.email", "catena-test@example.invalid")
    _git(repo, "config", "user.name", "CATENA Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pinned official fixture")
    monkeypatch.setattr(official_patch, "PINNED_OFFICIAL_COMMIT", _git(repo, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        official_patch,
        "PINNED_GDN2_SHA256",
        hashlib.sha256(_SOURCE_TEMPLATE).hexdigest(),
    )
    return repo, target


def _apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: official_patch.PatchMode = "apply",
) -> tuple[Path, Path, Path, dict[str, object]]:
    repo, target = _fixture(tmp_path, monkeypatch)
    patch_path = tmp_path / "e26_final.patch"
    receipt_path = tmp_path / "receipt.json"
    payload = official_patch.apply_e26_final_official_patch(
        repo_root=repo,
        patch_output=patch_path,
        receipt_output=receipt_path,
        mode=mode,
    )
    return target, patch_path, receipt_path, payload


def _load_layer(source: bytes) -> tuple[type[Any], object]:
    missing = object()
    namespace: dict[str, object] = {"_MISSING": missing}
    exec(compile(source, "gdn2.py", "exec"), namespace)
    layer = namespace["Layer"]
    assert isinstance(layer, type)
    return cast(type[Any], layer), missing


class _Projection:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden + self.offset


def test_apply_changes_only_gate_anchor_and_writes_hash_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, patch_path, receipt_path, payload = _apply(tmp_path, monkeypatch)
    patched = target.read_bytes()
    patch_bytes = patch_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert payload == receipt
    assert receipt["status"] == "APPLIED"
    assert receipt["base_file_sha256"] == hashlib.sha256(_SOURCE_TEMPLATE).hexdigest()
    assert receipt["patched_file_sha256"] == hashlib.sha256(patched).hexdigest()
    assert receipt["unified_diff_sha256"] == hashlib.sha256(patch_bytes).hexdigest()
    assert receipt["allowed_policy_values"] == ["dual_gdn2", "projected_tied_gdn2"]
    assert patched.replace(
        official_patch._PATCHED_GATE_BLOCK,
        official_patch._ORIGINAL_GATE_BLOCK,
        1,
    ) == _SOURCE_TEMPLATE
    assert patched.count(b"self.b_proj(hidden_states)") == 1
    assert patched.count(b"self.w_proj(hidden_states)") == 1
    assert b"chunk_gdn2" not in patch_bytes
    assert b"fused_recurrent_gdn2" not in patch_bytes


def test_render_only_never_mutates_official_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _, receipt_path, _ = _apply(tmp_path, monkeypatch, mode="render")

    assert target.read_bytes() == _SOURCE_TEMPLATE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "RENDERED_NOT_APPLIED"


def test_patched_runtime_requires_explicit_known_policy_and_uses_both_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _, _, _ = _apply(tmp_path, monkeypatch)
    layer_type, missing = _load_layer(target.read_bytes())
    hidden = torch.tensor([[-2.0, 1.0]])

    b_proj = _Projection(0.5)
    w_proj = _Projection(-0.25)
    dual = layer_type(b_proj, w_proj, "dual_gdn2")
    dual_b, dual_w = dual.forward(hidden)
    torch.testing.assert_close(dual_b, torch.sigmoid(hidden + 0.5))
    torch.testing.assert_close(dual_w, torch.sigmoid(hidden - 0.25))
    assert (b_proj.calls, w_proj.calls) == (1, 1)

    tied_b_proj = _Projection(0.5)
    tied_w_proj = _Projection(-0.25)
    tied = layer_type(tied_b_proj, tied_w_proj, "projected_tied_gdn2")
    tied_b, tied_w = tied.forward(hidden)
    expected = torch.sigmoid(((hidden + 0.5) + (hidden - 0.25)) / 2.0)
    torch.testing.assert_close(tied_b, expected)
    torch.testing.assert_close(tied_w, expected)
    assert tied_b is tied_w
    assert (tied_b_proj.calls, tied_w_proj.calls) == (1, 1)

    missing_policy = layer_type(_Projection(0.0), _Projection(0.0), missing)
    with pytest.raises(ValueError, match="requires an explicit e26_gate_policy"):
        missing_policy.forward(hidden)

    unknown_b_proj = _Projection(0.0)
    unknown_w_proj = _Projection(0.0)
    unknown = layer_type(unknown_b_proj, unknown_w_proj, "unknown")
    with pytest.raises(ValueError, match="requires an explicit e26_gate_policy"):
        unknown.forward(hidden)
    assert (unknown_b_proj.calls, unknown_w_proj.calls) == (1, 1)


def test_shape_mismatch_fails_closed_after_both_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _, _, _ = _apply(tmp_path, monkeypatch)
    layer_type, _ = _load_layer(target.read_bytes())

    class NarrowProjection(_Projection):
        def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return hidden[..., :1]

    b_proj = _Projection(0.0)
    w_proj = NarrowProjection(0.0)
    layer = layer_type(b_proj, w_proj, "dual_gdn2")
    with pytest.raises(ValueError, match="identical shapes"):
        layer.forward(torch.zeros(2, 3))
    assert (b_proj.calls, w_proj.calls) == (1, 1)


def test_refuses_wrong_commit_or_changed_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, target = _fixture(tmp_path, monkeypatch)
    patch_path = tmp_path / "wrong.patch"
    receipt_path = tmp_path / "wrong.json"
    monkeypatch.setattr(official_patch, "PINNED_OFFICIAL_COMMIT", "0" * 40)
    with pytest.raises(official_patch.E26FinalOfficialPatchError, match="HEAD"):
        official_patch.apply_e26_final_official_patch(
            repo_root=repo,
            patch_output=patch_path,
            receipt_output=receipt_path,
            mode="apply",
        )
    assert target.read_bytes() == _SOURCE_TEMPLATE
    assert not patch_path.exists()
    assert not receipt_path.exists()

    monkeypatch.setattr(official_patch, "PINNED_OFFICIAL_COMMIT", _git(repo, "rev-parse", "HEAD"))
    target.write_bytes(_SOURCE_TEMPLATE + b"# drift\n")
    with pytest.raises(official_patch.E26FinalOfficialPatchError, match="bytes"):
        official_patch.apply_e26_final_official_patch(
            repo_root=repo,
            patch_output=patch_path,
            receipt_output=receipt_path,
            mode="apply",
        )
    assert not patch_path.exists()
    assert not receipt_path.exists()


def test_refuses_double_patch_and_unknown_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, target = _fixture(tmp_path, monkeypatch)
    official_patch.apply_e26_final_official_patch(
        repo_root=repo,
        patch_output=tmp_path / "first.patch",
        receipt_output=tmp_path / "first.json",
        mode="apply",
    )
    with pytest.raises(official_patch.E26FinalOfficialPatchError, match="already"):
        official_patch.apply_e26_final_official_patch(
            repo_root=repo,
            patch_output=tmp_path / "second.patch",
            receipt_output=tmp_path / "second.json",
            mode="apply",
        )
    assert official_patch._PATCH_SENTINEL in target.read_bytes()

    with pytest.raises(official_patch.E26FinalOfficialPatchError, match="Unknown patch mode"):
        official_patch.apply_e26_final_official_patch(
            repo_root=repo,
            patch_output=tmp_path / "third.patch",
            receipt_output=tmp_path / "third.json",
            mode="invalid",  # type: ignore[arg-type]
        )
