# ruff: noqa: E402

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from catena.core.io import file_sha256
from catena.post_e21.official_gdn2 import (
    E25aNotConfigured,
    _plugin_manifest,
    official_source_manifest,
)


def test_dry_source_manifest_never_claims_configuration() -> None:
    result = official_source_manifest(
        {
            "environment": {"required_prefix": "${CATENA_E25A_ENV_PREFIX}"},
            "backend": {},
        },
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert result["configured"] is False
    assert result["reference_fallback"] is False


def test_missing_separate_environment_fails_closed() -> None:
    with pytest.raises(E25aNotConfigured, match="not configured"):
        official_source_manifest(
            {
                "environment": {"required_prefix": "${CATENA_E25A_ENV_PREFIX}"},
                "backend": {},
            },
            dry_run=False,
        )


def test_replication_requires_explicit_authorization() -> None:
    module = importlib.import_module("experiments.e25a_official_gdn2_gate")
    with pytest.raises(PermissionError, match="explicit user approval"):
        module._replication(config={}, gate_report=None, authorized=False)


def test_replication_plugin_origin_and_hash_are_both_required() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "src/catena_official_plugins/e25a_replication.py"
    result = _plugin_manifest(
        module_name="catena_official_plugins.e25a_replication",
        source_path="src/catena_official_plugins/e25a_replication.py",
        source_sha256=file_sha256(source),
        label="test replication",
        relative_to_repo=True,
    )
    assert Path(result["path"]) == source.resolve()
    with pytest.raises(ValueError, match="hash mismatch"):
        _plugin_manifest(
            module_name="catena_official_plugins.e25a_replication",
            source_path="src/catena_official_plugins/e25a_replication.py",
            source_sha256="0" * 64,
            label="test replication",
            relative_to_repo=True,
        )


def test_replication_result_requires_real_subset_seed_rows() -> None:
    module = importlib.import_module("experiments.e25a_official_gdn2_gate")
    config = {
        "replication": {
            "subsets": [
                "e02b_magnitude_factorization",
                "e18_magnitude_sequence",
                "e22_locality_if_supported",
            ]
        }
    }
    result = {
        "rows": [
            {"subset": "e02b_magnitude_factorization", "seed": 11},
            {"subset": "e18_magnitude_sequence", "seed": 101},
        ],
        "seed_rows": [
            {"subset": "e02b_magnitude_factorization", "seed": 11},
            {"subset": "e18_magnitude_sequence", "seed": 101},
        ],
        "checks": {
            "e02b_magnitude_factorization": {"passed": True},
            "e18_magnitude_sequence": {"passed": True},
            "e22_locality_if_supported": {"passed": True, "skipped": True},
        },
        "subset_decisions": {
            "e02b_magnitude_factorization": {"include": True},
            "e18_magnitude_sequence": {"include": True},
            "e22_locality_if_supported": {"include": False},
        },
        "passed": True,
        "scientific_evidence": True,
    }
    assert module._validate_replication_result(result, config=config) is True
    rows, seed_rows = module._artifact_rows(
        stage="replication",
        status="PASS",
        source={
            "status": "CONFIGURED",
            "replication_plugin": {"sha256": "a" * 64},
        },
        result=result,
        dependency={"sha256": "b" * 64},
        config=config,
    )
    assert len(rows) == 2
    assert len(seed_rows) == 2
    assert {row["subset"] for row in rows} == {
        "e02b_magnitude_factorization",
        "e18_magnitude_sequence",
    }
    assert all(row["gate_dependency_sha256"] == "b" * 64 for row in rows)


def test_unimplemented_safe_e22_route_blocks_without_fake_rows() -> None:
    module = importlib.import_module("experiments.e25a_official_gdn2_gate")
    config = {
        "replication": {
            "subsets": [
                "e02b_magnitude_factorization",
                "e18_magnitude_sequence",
                "e22_locality_if_supported",
            ]
        }
    }
    result = {
        "rows": [],
        "seed_rows": [],
        "checks": {
            "e22_locality_if_supported": {
                "passed": False,
                "status": "NOT_IMPLEMENTED",
            }
        },
        "subset_decisions": {
            "e02b_magnitude_factorization": {"include": False},
            "e18_magnitude_sequence": {"include": False},
            "e22_locality_if_supported": {
                "include": True,
                "implemented": False,
                "status": "BLOCKED_DEPENDENCY",
            },
        },
        "passed": False,
        "scientific_evidence": False,
        "blocked_dependency": True,
    }
    assert module._validate_replication_result(result, config=config) is False
    rows, seed_rows = module._artifact_rows(
        stage="replication",
        status="BLOCKED_DEPENDENCY",
        source={"status": "CONFIGURED"},
        result=result,
        dependency={"sha256": "b" * 64},
        config=config,
    )
    assert len(rows) == 1
    assert len(seed_rows) == 1
    assert rows[0]["status"] == "BLOCKED_DEPENDENCY"


def test_optional_e22_report_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    plugin = importlib.import_module("catena_official_plugins.e25a_replication")
    assert plugin.validate_safe_e22_report(None)["include"] is False
    report: dict[str, Any] = {
        "experiment_id": "e22b_active_path_locality",
        "execution_status": "PASS",
        "status": "PASS",
        "run_mode": "MAIN",
        "run_scope": "E22B_ACTIVE_PATH_LOCALITY_CONFIRMATORY",
        "evidence_tier": "CONTROLLED_REFERENCE",
        "claim_eligible": True,
        "protocol_lock": {
            "sha256": ("e19dfd26018e53d7ab601d1bd1b0e94c3bd922e1849c35cdcffec7ae38474598")
        },
        "summary": {
            "status": "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
            "supported": True,
            "dry_run_non_evidence": False,
            "seed_count": 8,
            "recovery_pattern_gate_passed": True,
            "capacity_gate_passed": True,
            "recovery_capacity_gate_passed": True,
            "absolute_locality_gate_passed": True,
            "retention_gate_passed": True,
            "locality_retention_gate_passed": True,
            "selected_vs_mean_locality": {"passed": True},
        },
        "claim_gate": {
            "status": "SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED",
            "supported": True,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"experiment_id": "e22b_active_path_locality"}),
        encoding="utf-8",
    )
    decision = plugin.validate_safe_e22_report(report_path)
    assert decision["include"] is True
    assert decision["protocol_lock_sha256"].startswith("e19dfd")
    report["summary"]["locality_retention_gate_passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not open"):
        plugin.validate_safe_e22_report(report_path)
    report["summary"]["locality_retention_gate_passed"] = True
    report["protocol_lock"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not open"):
        plugin.validate_safe_e22_report(report_path)


def test_replication_source_contracts_are_exact() -> None:
    plugin = importlib.import_module("catena_official_plugins.e25a_replication")
    config = load_yaml_config(Path("configs/e25a_official_gdn2_gate.yaml"))
    result = plugin.validate_source_contracts(config)
    assert result["configs"]["e02b"]["data"]["value_dim"] == 32
    assert result["configs"]["e18"]["evaluation"]["gap_events"][-1] == 2048


def test_pinned_local_source_if_present_is_exact() -> None:
    repository = Path("/home/minjun_dev/CATENA_official/gdn2_upstream")
    if not repository.is_dir():
        pytest.skip("local official checkout is not installed")
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == "95709fc250357c2dd109361c353192f2aa5913f9"
    assert file_sha256(repository / "LICENSE") == (
        "eaff393a7abc4ea7cb05795423b531a212b6d2189bcbe30410587d52d70988bb"
    )


def test_protocol_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/E25A_OFFICIAL_GDN2_LOCK.json").is_file()
    assert (root / "docs/E25A_OFFICIAL_GDN2_PROTOCOL_KO.md").is_file()
    assert (root / "environments/e25a_official_gdn2_environment.yaml").is_file()
    assert (root / "environments/e25a_official_gdn2_observed_lock.json").is_file()


def load_yaml_config(path: Path) -> dict[str, Any]:
    from catena.core.config import load_config

    return load_config(path)
