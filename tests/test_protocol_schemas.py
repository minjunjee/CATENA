import json
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
V8_CONFIG_PREFIXES = ("e26", "e27", "e28", "e29", "e30")
PACKET_SCHEMA_NAMES = {
    "backend_manifest.schema.json",
    "data_manifest.schema.json",
    "evaluation_row.schema.json",
    "model_manifest.schema.json",
    "protocol_lock.schema.json",
    "report.schema.json",
    "transaction_episode.schema.json",
}
SCIENTIFIC_DATA_SCHEMA_NAMES = {
    "general_corpus_manifest.schema.json",
    "scientific_data_readiness.schema.json",
    "tokenizer_manifest.schema.json",
}


def test_all_protocol_yaml_parse_and_have_core_keys() -> None:
    paths = [
        path
        for path in sorted((REPO_ROOT / "configs").glob("*.yaml"))
        if path.name.startswith(V8_CONFIG_PREFIXES)
    ]
    assert len(paths) == 14
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "catena-v8.1"
        assert payload["experiment"].startswith("e")
        assert payload["stage"]
        assert payload["registered_dispositions"]


def test_schema_files_are_valid_json_schemas() -> None:
    schema_root = REPO_ROOT / "schemas" / "v8_1"
    paths = sorted(schema_root.glob("*.schema.json"))
    assert {path.name for path in paths} == PACKET_SCHEMA_NAMES | SCIENTIFIC_DATA_SCHEMA_NAMES
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
