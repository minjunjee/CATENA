from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest
import yaml
from helpers.e26_stage3d_contract_fixture import stage3c_fixture_inputs

from catena.core.provenance_v61 import sha256_canonical_json
from catena.lm.stage3d_fixed_layout import (
    KNOWN_LAYOUT_SENSITIVITY,
    STAGE3C_DISPOSITION,
    Stage3DContractError,
    build_stage3d_protocol_lock,
    fixed_layouts_from_config,
    load_stage3d_config,
    validate_stage3d_protocol_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e26_stage3d_fixed_layout_bf16_admissibility.yaml"


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = stage3c_fixture_inputs(tmp_path)
    inventory = {"source_tree_sha256": "2" * 64, "files": 600}
    payload = build_stage3d_protocol_lock(
        config_path=CONFIG_PATH,
        stage3c_result_path=paths["result"],
        stage3c_protocol_path=paths["protocol"],
        stage3c_artifact_manifest_path=paths["artifact"],
        frozen_e00_e25_receipt_path=paths["frozen"],
        source_commit="3" * 40,
        source_inventory=inventory,
    )
    return payload, paths


def test_stage3d_config_locks_fixed_layouts_and_determinism() -> None:
    config = load_stage3d_config(CONFIG_PATH)
    layouts = fixed_layouts_from_config(config)
    assert [row["accumulation_steps"] for row in layouts] == [16, 32, 16]
    assert all(row["microbatch_sequences"] == 1 for row in layouts)
    assert all(row["target_global_input_tokens"] == 65_536 for row in layouts)
    assert config["determinism"] == {
        "initialization_seed": 260_301,
        "g3_data_seed": 260_701,
        "g4_data_seed": 260_801,
        "prefill_seed": 260_901,
        "prefill_length": 17,
    }


def test_stage3d_protocol_preserves_stage3c_and_conforms_to_schema(tmp_path: Path) -> None:
    payload, _ = _fixture(tmp_path)
    assert validate_stage3d_protocol_lock(payload, config_path=CONFIG_PATH) == payload
    assert payload["stage3c"]["disposition"] == STAGE3C_DISPOSITION
    assert payload["stage3c"]["diagnostic_disposition"] == KNOWN_LAYOUT_SENSITIVITY
    assert payload["scientific_e26a_started"] is False
    schema = json.loads(
        (REPO_ROOT / "schemas/v8_1/e26_stage3d_protocol_lock.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("stage3c", "disposition"), "PASS"),
        (("stage3c", "diagnostic_disposition"), "IGNORED"),
        (("stage3c", "result", "sha256"), "0" * 64),
        (("stage3c", "status", "sha256"), "0" * 64),
        (("stage3c", "raw_registered_aggregate_sha256"), "0" * 64),
        (("stage3c", "raw_registered_files_sha256", "source_lock.json"), "0" * 64),
        (("stage3c", "failure_status", "sha256"), "0" * 64),
        (("stage3c", "artifact_manifest_rehash_aggregate_sha256"), "0" * 64),
        (("inherited_thresholds", "bf16_relative_l2_max"), 0.02),
        (("fixed_layouts", "0", "microbatch_sequences"), 2),
        (("determinism", "g3_data_seed"), 7),
        (("backend_binding", "registered_backend_id"), "changed_backend"),
        (("backend_binding", "runtime_backend_alias"), "changed_alias"),
        (("scientific_e26a_started",), True),
    ],
)
def test_stage3d_protocol_rejects_rehashed_semantic_tampering(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> None:
    payload, _ = _fixture(tmp_path)
    mutated = deepcopy(payload)
    target: Any = mutated
    for key in path[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    key = path[-1]
    if isinstance(target, list):
        target[int(key)] = value
    else:
        target[key] = value
    mutated.pop("protocol_sha256")
    mutated["protocol_sha256"] = sha256_canonical_json(mutated)
    with pytest.raises(Stage3DContractError):
        validate_stage3d_protocol_lock(mutated)


def test_stage3d_protocol_rejects_changed_bound_stage3c_bytes(tmp_path: Path) -> None:
    payload, paths = _fixture(tmp_path)
    paths["artifact"].write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(Stage3DContractError, match="bytes or path changed"):
        validate_stage3d_protocol_lock(payload)


def test_stage3d_config_rejects_registered_backend_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["gates"]["g3_fixed_layout_bf16_admissibility"]["compiled_backend"] = "compiled_scan"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(Stage3DContractError, match="G3/G4 coverage changed"):
        load_stage3d_config(mutated)
