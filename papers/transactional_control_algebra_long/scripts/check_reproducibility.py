from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PAPER_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PAPER_ROOT / "scripts/generate_main_figures.py"
CHECKED_IN = PAPER_ROOT / "figures"
FREEZE_NAME = "source_data_freeze.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def compare_file(expected: Path, observed: Path) -> None:
    if not expected.is_file():
        raise FileNotFoundError(f"Missing checked-in output: {expected}")
    if not observed.is_file():
        raise FileNotFoundError(f"Regeneration omitted output: {observed}")
    if expected.read_bytes() != observed.read_bytes():
        raise ValueError(
            "Reproducibility mismatch: "
            f"{expected.relative_to(CHECKED_IN)} "
            f"checked_in={file_sha256(expected)} "
            f"regenerated={file_sha256(observed)}"
        )


def check(artifact_root: Path) -> dict[str, Any]:
    checked_freeze_path = CHECKED_IN / FREEZE_NAME
    checked_freeze = load_json(checked_freeze_path)
    output_records = checked_freeze.get("generated_outputs")
    if not isinstance(output_records, dict) or not output_records:
        raise ValueError("Checked-in freeze has no generated outputs")

    with tempfile.TemporaryDirectory(prefix="catena_longpaper_figures_") as raw:
        regenerated = Path(raw) / "figures"
        command = [
            sys.executable,
            str(GENERATOR),
            "--artifact-root",
            str(artifact_root),
            "--output-dir",
            str(regenerated),
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        for relative in sorted(output_records):
            compare_file(CHECKED_IN / relative, regenerated / relative)
        compare_file(
            checked_freeze_path,
            regenerated / FREEZE_NAME,
        )

    return {
        "status": "PASS",
        "artifact_root": str(artifact_root.resolve()),
        "generator": str(GENERATOR),
        "checked_output_count": len(output_records) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate CATENA long-paper figures in a temporary directory "
            "and compare them byte-for-byte with the checked-in outputs."
        )
    )
    parser.add_argument(
        "--artifact-root",
        default="/data/minjun_dev/CATENA/artifacts",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check(Path(args.artifact_root)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
