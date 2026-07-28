# Anonymous ACL/REALM long-paper TeX scaffold

This directory is an additive review-mode typesetting handoff for
`../PAPER_SCAFFOLD.md`. It does not change experiment code, configs, metrics,
or artifacts.

## Scientific-source contract

- Numeric result commands are generated from
  `../figures/RESULTS_MACROS.md`.
- Figures originate from the hash-checked SVGs in `../figures/`.
- Artifact paths and SHA-256 values remain in
  `../data/source_manifest.json`.
- Writing boundaries remain governed by `../CLAIM_BOUNDARIES.md`.
- `main.tex` uses `\usepackage[review]{acl}` and contains no author identity.

## Static check

From this directory:

```bash
python scripts/generate_results_macros.py
python scripts/check_scaffold.py
```

The checker verifies anonymous review mode, section inventory, macro
freshness, bibliography keys, source-manifest presence, claim-phrase
guardrails, and the hashes of the pinned official ACL style files.

## Vector figure preparation

The TeX source expects PDFs under `generated/figures/`. Generate them with:

```bash
python scripts/prepare_figures.py
python scripts/prepare_figures.py --check
```

The converter accepts `rsvg-convert`, Inkscape, or CairoSVG. It intentionally
does not fall back to ImageMagick because that path may rasterize the plots.
The original SVGs are never modified.

## Build

Install an ACL-compatible TeX distribution with `latexmk`, `pdflatex`, and
BibTeX, then run:

```bash
bash build.sh
```

The current `catena-v6` environment does not contain a TeX engine, so the
repository-side validation is static until a TeX toolchain is supplied.
`build.sh` pins TeX and BibTeX search paths to the official style snapshot in
`vendor/acl`.

Before submission, manually verify:

1. at most eight pages of main content under the current REALM call;
2. anonymous double-blind metadata and PDF properties;
3. A4 size, embedded fonts, vector figures, and legible two-column labels;
4. references/appendix placement under the current call;
5. that the venue-required ACL style pin still matches `vendor/acl`.
