from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    command = ["bash", str(root / "scripts" / "run_postcore_dry.sh"), str(root)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
