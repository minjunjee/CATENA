#!/usr/bin/env python3
"""Convert frozen SVG figures to vector PDF without raster fallback."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FIGURES = (
    "figure1_geometry",
    "figure2_control_lattice",
    "figure3_sequence_transfer",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def converter() -> tuple[str, list[str]] | None:
    if shutil.which("rsvg-convert"):
        return "rsvg-convert", ["rsvg-convert", "-f", "pdf", "-o"]
    if shutil.which("inkscape"):
        return "inkscape", ["inkscape", "--export-type=pdf", "--export-filename"]
    if importlib.util.find_spec("cairosvg") is not None:
        return "cairosvg", []
    return None


def converter_version(kind: str) -> str:
    if kind == "rsvg-convert":
        result = subprocess.run(
            ["rsvg-convert", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    if kind == "inkscape":
        result = subprocess.run(
            ["inkscape", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    if kind == "cairosvg":
        import cairosvg  # type: ignore[import-not-found]

        return str(cairosvg.__version__)
    raise ValueError(kind)


def convert(kind: str, prefix: list[str], source: Path, target: Path) -> None:
    temporary = target.with_suffix(".tmp.pdf")
    if temporary.exists():
        temporary.unlink()
    if kind == "rsvg-convert":
        subprocess.run([*prefix, str(temporary), str(source)], check=True)
    elif kind == "inkscape":
        subprocess.run(
            [*prefix, str(temporary), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
    elif kind == "cairosvg":
        import cairosvg  # type: ignore[import-not-found]

        cairosvg.svg2pdf(url=str(source), write_to=str(temporary))
    else:  # pragma: no cover - guarded by converter()
        raise ValueError(kind)
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"converter produced no PDF for {source}")
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing PDF and conversion-manifest hashes only",
    )
    args = parser.parse_args()

    tex_root = Path(__file__).resolve().parents[1]
    source_dir = tex_root.parent / "figures"
    output_dir = tex_root / "generated" / "figures"
    manifest_path = output_dir / "conversion_manifest.json"

    if args.check:
        if not manifest_path.exists():
            print("missing vector figure conversion manifest", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in FIGURES:
            source = source_dir / f"{name}.svg"
            target = output_dir / f"{name}.pdf"
            record = manifest["figures"].get(name, {})
            if not source.exists() or not target.exists():
                print(f"missing source or PDF for {name}", file=sys.stderr)
                return 1
            if record.get("source_svg_sha256") != sha256(source):
                print(f"source SVG hash changed for {name}", file=sys.stderr)
                return 1
            if record.get("output_pdf_sha256") != sha256(target):
                print(f"output PDF hash changed for {name}", file=sys.stderr)
                return 1
        print(f"PASS: {len(FIGURES)} vector figure conversions verified")
        return 0

    selected = converter()
    if selected is None:
        print(
            "No vector converter found. Install librsvg (rsvg-convert), "
            "Inkscape, or CairoSVG. ImageMagick raster fallback is "
            "intentionally disabled.",
            file=sys.stderr,
        )
        return 2

    kind, prefix = selected
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, str]] = {}
    for name in FIGURES:
        source = source_dir / f"{name}.svg"
        target = output_dir / f"{name}.pdf"
        if not source.exists():
            raise FileNotFoundError(source)
        convert(kind, prefix, source, target)
        records[name] = {
            "source_svg": str(source.relative_to(tex_root.parent)),
            "source_svg_sha256": sha256(source),
            "output_pdf": str(target.relative_to(tex_root)),
            "output_pdf_sha256": sha256(target),
        }
    manifest = {
        "schema_version": 1,
        "converter": kind,
        "converter_version": converter_version(kind),
        "claim": "format-only vector conversion; scientific source is unchanged",
        "figures": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE: {len(records)} vector PDFs using {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
