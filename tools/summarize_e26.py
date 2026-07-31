#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CATENA v8.1 report.json files")
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": report.get("experiment"),
                "run_id": report.get("run_id"),
                "status": report.get("status"),
                "scientific_evidence": report.get("scientific_evidence"),
                "disposition": report.get("disposition"),
                "failed_gates": [
                    gate.get("name") for gate in report.get("gates", []) if not gate.get("passed")
                ],
                "path": str(path.resolve()),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
