from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from catena.core.provenance_v61 import sha256_file

_SPEC = importlib.util.spec_from_file_location(
    "catena_e26_stage3d_g0_tool",
    Path(__file__).resolve().parents[1] / "tools" / "run_e26_stage3d_preflight.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    data_lock = tmp_path / "data_lock.json"
    data_lock.write_text('{"repository":{}}\n', encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "live_repository": {"observed_head": "old-head"},
                "receipt_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "stage3c_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "execution_input_paths": {
                    "data_lock": str(data_lock.resolve()),
                    "frozen_tree_receipt": str(frozen.resolve()),
                },
                "execution_inputs": {
                    "data_lock_sha256": sha256_file(data_lock),
                    "frozen_tree_receipt_sha256": sha256_file(frozen),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_manifest = tmp_path / "artifact_manifest.json"
    artifact_manifest.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    refreshed = {
        "passed": True,
        "receipt_sha256": "b" * 64,
        "live_repository": {"observed_head": "new-head"},
    }
    return {
        "data_lock": data_lock,
        "frozen": frozen,
        "protocol": protocol,
        "artifact_manifest": artifact_manifest,
        "artifact_root": artifact_root,
    }, refreshed


def test_g0_binds_historical_receipt_to_stage3c_before_live_reaudit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, refreshed = _fixture(tmp_path)
    monkeypatch.setattr(
        _TOOL,
        "validate_historical_frozen_invariance_receipt",
        lambda *_args, **_kwargs: refreshed,
    )
    monkeypatch.setattr(
        _TOOL,
        "_verify_stage3c_artifact_manifest",
        lambda **_kwargs: {"passed": True},
    )
    result = _TOOL._verify_g0_frozen_inputs(
        stage3c_protocol_path=paths["protocol"],
        stage3c_artifact_manifest_path=paths["artifact_manifest"],
        stage3c_artifact_root=paths["artifact_root"],
        frozen_receipt_path=paths["frozen"].resolve(),
    )
    assert result["passed"] is True
    assert result["dynamic_head_change_allowed"] is True
    assert (
        result["stage3c_registered_frozen_receipt_sha256"]
        == sha256_file(paths["frozen"])
    )


@pytest.mark.parametrize("target", ["frozen", "data_lock"])
def test_g0_blocks_stage3c_bound_input_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    paths, refreshed = _fixture(tmp_path)
    monkeypatch.setattr(
        _TOOL,
        "validate_historical_frozen_invariance_receipt",
        lambda *_args, **_kwargs: refreshed,
    )
    monkeypatch.setattr(
        _TOOL,
        "_verify_stage3c_artifact_manifest",
        lambda **_kwargs: {"passed": True},
    )
    paths[target].write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        _TOOL._verify_g0_frozen_inputs(
            stage3c_protocol_path=paths["protocol"],
            stage3c_artifact_manifest_path=paths["artifact_manifest"],
            stage3c_artifact_root=paths["artifact_root"],
            frozen_receipt_path=paths["frozen"].resolve(),
        )
