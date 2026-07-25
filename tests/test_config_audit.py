from pathlib import Path

from catena.experiments.config_audit import audit_configs


def test_repository_configs_are_internally_consistent():
    root = Path(__file__).resolve().parents[1]
    report = audit_configs(root, write_report=False)
    assert report["passed"], report["errors"]
