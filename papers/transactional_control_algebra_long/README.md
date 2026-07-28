# Transactional Control Algebra — long-paper workspace

This directory is an additive, artifact-backed scaffold for an eight-page
long paper. It does not modify the completed CATENA papers, scientific source,
or experiment artifacts.

## Contents

- `PAPER_SCAFFOLD.md`: section-level eight-page manuscript plan and draft text.
- `FORMAT_AND_SUBMISSION.md`: official REALM/ACL format references and the
  current Markdown-first workflow.
- `CLAIM_BOUNDARIES.md`: claims that are open, closed, or explicitly out of
  scope.
- `RELATED_WORK_PRIMARY_SOURCES.md`: verified primary-source map with
  manuscript-use guardrails.
- `../../docs/E15A_R1_OFFICIAL_PARITY_FAILURE_AUDIT_KO.md`: no-patch audit of
  the two failed official-kernel parity checks.
- `FIGURE_CAPTIONS.md`: manuscript-ready captions and provenance notes.
- `data/source_manifest.json`: canonical artifact paths and expected SHA-256
  digests.
- `tex/`: anonymous ACL review-mode LaTeX scaffold, pinned official style
  provenance, static checks, and vector-figure/build wrappers.
- `scripts/generate_main_figures.py`: deterministic source-data extraction and
  SVG generation using only the Python standard library and NumPy.
- `scripts/check_reproducibility.py`: clean regeneration and byte-for-byte
  reproducibility check.
- `figures/source_data_freeze.json`: validated inputs, generator version, and
  generated-output digests.

## Rebuild

Run from the repository root in the existing `catena-v6` environment:

```bash
python papers/transactional_control_algebra_long/scripts/generate_main_figures.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts

python papers/transactional_control_algebra_long/scripts/check_reproducibility.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

The generator fails before writing scientific outputs if any canonical input
is missing or its hash differs from `data/source_manifest.json`. Scientific
numbers in the figures and `figures/RESULTS_MACROS.md` are derived from those
inputs; they are not copied into the plotting code.

## Submission handoff

The additive `tex/` directory now contains the anonymous long-paper
typesetting scaffold and a hash-pinned snapshot of the two required official
ACL style files. No TeX compiler is installed in the current `catena-v6`
environment, so run its static checker here and compile in an ACL-compatible
TeX environment:

```bash
python papers/transactional_control_algebra_long/tex/scripts/check_scaffold.py
```

See `tex/README.md` for vector conversion, compilation, page-count,
font-embedding, and anonymity checks.
