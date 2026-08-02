from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from catena.core.provenance_v61 import (
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.lm import e26a_gate
from catena.lm.e26a_gate import E26AGateBlocked, E26AGateInputPaths


class _ReadinessResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)


def _paths(data_lock: Path, readiness: Path) -> E26AGateInputPaths:
    return cast(
        E26AGateInputPaths,
        SimpleNamespace(data_lock=data_lock.resolve(), data_readiness=readiness.resolve()),
    )


def _v3_bundle(tmp_path: Path) -> tuple[E26AGateInputPaths, dict[str, Any], dict[str, Any]]:
    protocol = tmp_path / "repair_protocol.yaml"
    repair = tmp_path / "repair_receipt.json"
    source = tmp_path / "repair_source.json"
    protocol.write_text("schema_version: repair-v1\n", encoding="utf-8")
    write_json_strict(repair, {"schema_version": "repair-v1"})
    write_json_strict(source, {"schema_version": "source-v1"})
    recorded: dict[str, Any] = {
        "schema_version": "catena-e26-scientific-data-readiness-v3",
        "manifest_type": "E26_SCIENTIFIC_DATA_READINESS_V3",
        "scientific_main_input_eligible": True,
        "protocol_lock": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
        "repair_receipt": {"path": str(repair.resolve()), "sha256": sha256_file(repair)},
        "repair_source": {"path": str(source.resolve()), "sha256": sha256_file(source)},
    }
    recorded["readiness_sha256"] = sha256_canonical_json(recorded)
    readiness = tmp_path / "scientific_data_readiness_v3.json"
    write_json_strict(readiness, recorded)
    data_lock: dict[str, Any] = {
        "schema_version": "catena-e26-data-lock-v3-final-preflight",
        "final_repaired_data": {
            "scientific_data_readiness_v3": {
                "path": str(readiness.resolve()),
                "sha256": sha256_file(readiness),
            },
            "readiness_internal_sha256": recorded["readiness_sha256"],
        },
    }
    data_lock["lock_sha256"] = sha256_canonical_json(data_lock)
    data_lock_path = tmp_path / "stage3c_data_lock.json"
    write_json_strict(data_lock_path, data_lock)
    return _paths(data_lock_path, readiness), data_lock, recorded


def test_readiness_dispatch_preserves_v2_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    placeholder = tmp_path / "placeholder.json"
    write_json_strict(placeholder, {})
    paths = _paths(placeholder, placeholder)
    calls: list[dict[str, Any]] = []

    def fake_stage2(*, paths: E26AGateInputPaths, recorded: dict[str, Any]) -> None:
        del paths
        calls.append(recorded)

    monkeypatch.setattr(e26a_gate, "_revalidate_stage2_data_readiness", fake_stage2)
    recorded = {"manifest_type": "E26_SCIENTIFIC_DATA_READINESS_V2"}
    e26a_gate._revalidate_data_readiness(paths=paths, data_lock={}, recorded=recorded)
    assert calls == [recorded]


def test_v3_readiness_is_reconstructed_from_byte_bound_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    observed_calls: list[dict[str, Path]] = []

    def fake_validate(**kwargs: Path) -> _ReadinessResult:
        observed_calls.append(kwargs)
        return _ReadinessResult(recorded)

    monkeypatch.setattr(e26a_gate, "validate_zero_tolerance_data_bundle", fake_validate)
    e26a_gate._revalidate_data_readiness(
        paths=paths,
        data_lock=data_lock,
        recorded=recorded,
    )
    assert observed_calls == [
        {
            "data_lock_path": Path(recorded["protocol_lock"]["path"]),
            "repair_receipt_path": Path(recorded["repair_receipt"]["path"]),
            "source_receipt_path": Path(recorded["repair_source"]["path"]),
        }
    ]


def test_v3_readiness_rejects_internal_canonical_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(recorded),
    )
    tampered = deepcopy(recorded)
    tampered["scientific_main_input_eligible"] = False
    with pytest.raises(E26AGateBlocked, match="readiness_sha256 does not match"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=tampered,
        )


def test_v3_readiness_rejects_file_byte_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(recorded),
    )
    paths.data_readiness.write_text("{}\n", encoding="utf-8")
    with pytest.raises(E26AGateBlocked, match="byte SHA-256 mismatch"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=recorded,
        )


def test_v3_readiness_rejects_stage3c_lock_canonical_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(recorded),
    )
    data_lock["final_repaired_data"]["readiness_internal_sha256"] = "0" * 64
    with pytest.raises(E26AGateBlocked, match="data_lock.lock_sha256 does not match"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=recorded,
        )


def test_v3_readiness_rejects_rebound_readiness_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(recorded),
    )
    rebound = tmp_path / "rebound_readiness.json"
    write_json_strict(rebound, recorded)
    data_lock["final_repaired_data"]["scientific_data_readiness_v3"] = {
        "path": str(rebound.resolve()),
        "sha256": sha256_file(rebound),
    }
    data_lock["lock_sha256"] = sha256_canonical_json(
        {key: value for key, value in data_lock.items() if key != "lock_sha256"}
    )
    with pytest.raises(E26AGateBlocked, match="binds another V3 readiness path"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=recorded,
        )


def test_v3_readiness_rejects_bound_input_byte_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(recorded),
    )
    Path(recorded["repair_source"]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(E26AGateBlocked, match="repair_source byte SHA-256 mismatch"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=recorded,
        )


def test_v3_readiness_rejects_fresh_reconstruction_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, data_lock, recorded = _v3_bundle(tmp_path)
    reconstructed = deepcopy(recorded)
    reconstructed["near_duplicate_flagged_pair_count"] = 1
    reconstructed["readiness_sha256"] = sha256_canonical_json(
        {key: value for key, value in reconstructed.items() if key != "readiness_sha256"}
    )
    monkeypatch.setattr(
        e26a_gate,
        "validate_zero_tolerance_data_bundle",
        lambda **_: _ReadinessResult(reconstructed),
    )
    with pytest.raises(E26AGateBlocked, match="fresh repaired-bundle revalidation"):
        e26a_gate._revalidate_data_readiness(
            paths=paths,
            data_lock=data_lock,
            recorded=recorded,
        )


def test_readiness_dispatch_rejects_unknown_manifest_type(tmp_path: Path) -> None:
    placeholder = tmp_path / "placeholder.json"
    write_json_strict(placeholder, {})
    with pytest.raises(E26AGateBlocked, match="Unsupported data-readiness manifest_type"):
        e26a_gate._revalidate_data_readiness(
            paths=_paths(placeholder, placeholder),
            data_lock={},
            recorded={"manifest_type": "E26_SCIENTIFIC_DATA_READINESS_V4"},
        )
