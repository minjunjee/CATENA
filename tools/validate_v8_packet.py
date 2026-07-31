#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    root = args.packet_root.resolve()
    errors: list[str] = []

    required = [
        "README_FIRST_KO.md",
        "CODEX_MASTER_TASK_KO.md",
        "SCIENTIFIC_EVIDENCE_LOCK_KO.md",
        "EXPERIMENT_NUMBERING_LOCK_KO.md",
        "overlay/src/catena/lm/model.py",
        "overlay/experiments/e26a_operator_data_gate.py",
        "schemas/report.schema.json",
    ]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    protocols: dict[str, dict] = {}
    for path in sorted((root / "protocol").glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)
            for key in ("schema_version", "experiment", "stage", "registered_dispositions"):
                if key not in payload:
                    raise ValueError(f"missing key {key}")
            if payload["schema_version"] != "catena-v8.1":
                raise ValueError("wrong schema_version")
            protocols[path.name] = payload
        except Exception as exc:  # noqa: BLE001
            errors.append(f"protocol {path.name}: {exc}")

    for path in sorted((root / "schemas").glob("*.schema.json")):
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"schema {path.name}: {exc}")

    overlay_configs = {path.name for path in (root / "overlay/configs").glob("*.yaml")}
    missing_copies = set(protocols) - overlay_configs
    if missing_copies:
        errors.append(f"protocols missing from overlay/configs: {sorted(missing_copies)}")

    experiment_files = {path.stem for path in (root / "overlay/experiments").glob("e*.py")}
    protocol_experiments = {str(payload["experiment"]) for payload in protocols.values()}
    missing_entries = protocol_experiments - experiment_files
    if missing_entries:
        errors.append(f"protocol experiments missing entry points: {sorted(missing_entries)}")

    if args.run_tests:
        environment = dict(**__import__("os").environ)
        source = str(root / "overlay/src")
        environment["PYTHONPATH"] = source + (
            ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        commands = [
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                str(root / "overlay/src"),
                str(root / "overlay/experiments"),
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(root / "overlay/tests"),
            ],
        ]
        for command in commands:
            completed = subprocess.run(command, env=environment, cwd=root / "overlay", check=False)
            if completed.returncode:
                errors.append(f"command failed ({completed.returncode}): {' '.join(command)}")

    result = {
        "packet_root": str(root),
        "protocol_count": len(protocols),
        "schema_count": len(list((root / "schemas").glob("*.schema.json"))),
        "experiment_entry_count": len(experiment_files),
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
