from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_ROOT = ROOT / "papers" / "transactional_control_algebra_long" / "tex"


def test_longpaper_tex_scaffold_static_contract() -> None:
    subprocess.run(
        [sys.executable, str(TEX_ROOT / "scripts" / "check_scaffold.py")],
        cwd=ROOT,
        check=True,
    )


def test_longpaper_tex_build_uses_anonymous_review_mode() -> None:
    main = (TEX_ROOT / "main.tex").read_text(encoding="utf-8")
    assert r"\usepackage[review]{acl}" in main
    assert r"\author{Anonymous submission}" in main
    assert r"\bibliographystyle{acl_natbib}" in main


def test_longpaper_vector_figure_handoff() -> None:
    subprocess.run(
        [
            sys.executable,
            str(TEX_ROOT / "scripts" / "prepare_figures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
