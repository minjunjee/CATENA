#!/usr/bin/env python3
"""Static checks for anonymity, provenance, macros, citations, and boundaries."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

STYLE_HASHES = {
    "acl.sty": "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de",
    "acl_natbib.bst": "6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e",
}

EXPECTED_INPUTS = (
    "01_introduction",
    "02_control_algebra",
    "03_experimental_design",
    "04_geometry",
    "05_lattice",
    "06_sequence",
    "07_boundaries",
    "08_related_conclusion",
    "appendix",
)

FORBIDDEN_POSITIVE_CLAIMS = (
    "every cell improved",
    "uniform persistence",
    "stress sesoi maintained",
    "accurate preservation",
    "official gate passed",
    "semantic compositional transfer is supported",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for payload in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        keys.update(key.strip() for key in payload.split(",") if key.strip())
    return keys


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    tex_root = Path(__file__).resolve().parents[1]
    paper_root = tex_root.parent
    main = (tex_root / "main.tex").read_text(encoding="utf-8")
    section_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((tex_root / "sections").glob("*.tex"))
    )
    all_tex = main + "\n" + section_text

    if r"\usepackage[review]{acl}" not in main:
        fail("main.tex must use official ACL review mode")
    if r"\author{Anonymous submission}" not in main:
        fail("main.tex must retain the anonymous literal author")
    if re.search(r"\\(?:thanks|email)\b", all_tex, flags=re.IGNORECASE):
        fail("possible identifying author metadata found")

    for stem in EXPECTED_INPUTS:
        needle = rf"\input{{sections/{stem}}}"
        if needle not in main:
            fail(f"missing section input: {needle}")

    lowered = all_tex.lower()
    for phrase in FORBIDDEN_POSITIVE_CLAIMS:
        if phrase in lowered:
            fail(f"forbidden positive claim phrase: {phrase}")

    subprocess.run(
        [sys.executable, str(tex_root / "scripts" / "generate_results_macros.py"), "--check"],
        check=True,
    )

    provenance = json.loads(
        (tex_root / "vendor" / "acl" / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    if provenance.get("commit") != "d5adc823ff0f80f98c80405ca0ab66c68e684409":
        fail("unexpected ACL style commit")
    for name, expected in STYLE_HASHES.items():
        path = tex_root / "vendor" / "acl" / name
        if digest(path) != expected:
            fail(f"official ACL style hash mismatch: {name}")
        if provenance["files"][name]["sha256"] != expected:
            fail(f"ACL provenance hash mismatch: {name}")

    manifest = json.loads(
        (paper_root / "data" / "source_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("data_sources") or not manifest.get("provenance_anchors"):
        fail("paper source manifest is empty")

    bib_text = (paper_root / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    missing = sorted(citation_keys(all_tex) - bib_keys)
    if missing:
        fail(f"missing bibliography keys: {missing}")

    figure_sources = [
        paper_root / "figures" / "figure1_geometry.svg",
        paper_root / "figures" / "figure2_control_lattice.svg",
        paper_root / "figures" / "figure3_sequence_transfer.svg",
    ]
    if any(not path.exists() for path in figure_sources):
        fail("one or more canonical SVG figures are missing")

    print("PASS: anonymous ACL/REALM TeX scaffold static checks")
    print(f"PASS: {len(citation_keys(all_tex))} cited keys resolve")
    print(f"PASS: {len(STYLE_HASHES)} official style hashes verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
