from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pre_e26_base_has_no_tracked_modifications(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "source-freeze.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools/verify_pre_e26_source.py"),
            "--root",
            str(root),
            "--base-commit",
            "adfdeaf9e87a8602a8e334915d87acb9ff25af39",
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert output.is_file()
