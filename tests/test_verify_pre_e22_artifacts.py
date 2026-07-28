from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/verify_pre_e22_artifacts.py"
    specification = importlib.util.spec_from_file_location("verify_pre_e22", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_inventory_scope_stops_at_e21(tmp_path: Path) -> None:
    module = _module()
    for name in ("e00_protocol", "e21b_result", "e22_new"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "report.json").write_text(name, encoding="utf-8")
    payload = module.inventory(tmp_path)
    assert [row["path"] for row in payload["files"]] == [
        "e00_protocol/report.json",
        "e21b_result/report.json",
    ]


def test_compare_detects_changed_and_unexpected() -> None:
    module = _module()
    expected = {
        "files": [{"path": "a", "bytes": 1, "sha256": "a" * 64}],
        "aggregate_sha256": "1" * 64,
    }
    observed = {
        "files": [
            {"path": "a", "bytes": 1, "sha256": "b" * 64},
            {"path": "b", "bytes": 1, "sha256": "c" * 64},
        ],
        "aggregate_sha256": "2" * 64,
    }
    result = module.compare(expected, observed)
    assert result["status"] == "FAIL"
    assert result["changed"] == ["a"]
    assert result["unexpected"] == ["b"]
