from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from catena.core.provenance_v61 import sha256_file
from catena.lm.hashing import hash_mapping
from catena.lm.readiness import E26ReadinessBlocked, validate_e26a_control_inputs


def _write_control_inputs(
    tmp_path: Path,
    *,
    candidate_capable: bool = True,
) -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/e26a_operator_data_gate.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    capture = tmp_path / "source-capture.json"
    capture.write_text('{"test":true}\n', encoding="utf-8")
    sections = {
        key: config[key]
        for key in (
            "matching",
            "backend_gates",
            "data",
            "gate_population",
            "floor_gate",
            "throughput",
        )
    }
    protocol = {
        "schema_version": "catena-v8.1",
        "experiment": "e26a_operator_data_gate",
        "locked": True,
        "config_hash": hash_mapping(config),
        "full_config_snapshot": config,
        "thresholds": sections,
        "source_fingerprint_path": str(capture),
        "source_hash": sha256_file(capture),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    backend = {
        "schema_version": "catena-v8.1",
        "backend_type": "TORCH_COMPILED",
        "candidate_codegen_capable": True,
        "e26a_candidate_capable": candidate_capable,
        "e26a_gate_capable": False,
        "parity_verified": False,
        "scientific_main_capable": False,
        "fallback_count": 0,
        "graph_break_count": 0,
        "source_commit": "0" * 40,
    }
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps(backend), encoding="utf-8")
    return config_path, protocol_path, backend_path


def test_control_readiness_accepts_candidate_only_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config, protocol, backend = _write_control_inputs(tmp_path)
    result = validate_e26a_control_inputs(
        repo_root=root,
        config_path=config,
        protocol_lock_path=protocol,
        backend_manifest_path=backend,
        require_clean_source=False,
        verify_backend_source=False,
    )
    assert result["candidate_id"] == "d512_ctx4096"


def test_control_readiness_rejects_failed_candidate_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config, protocol, backend = _write_control_inputs(
        tmp_path,
        candidate_capable=False,
    )
    with pytest.raises(E26ReadinessBlocked, match="candidate smoke"):
        validate_e26a_control_inputs(
            repo_root=root,
            config_path=config,
            protocol_lock_path=protocol,
            backend_manifest_path=backend,
            require_clean_source=False,
            verify_backend_source=False,
        )
