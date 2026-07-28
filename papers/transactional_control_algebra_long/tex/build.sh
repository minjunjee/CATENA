#!/usr/bin/env bash
set -euo pipefail

tex_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$tex_root"

python scripts/generate_results_macros.py
if ! python scripts/prepare_figures.py --check; then
  python scripts/prepare_figures.py
fi
python scripts/check_scaffold.py
python scripts/prepare_figures.py --check

export TEXINPUTS="$tex_root/vendor/acl:${TEXINPUTS:-}"
export BSTINPUTS="$tex_root/vendor/acl:${BSTINPUTS:-}"
export BIBINPUTS="$tex_root/..:${BIBINPUTS:-}"

mkdir -p build

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  bibtex build/main
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
else
  echo "No TeX toolchain found. Install latexmk+pdflatex (recommended)." >&2
  exit 2
fi

echo "Built $tex_root/build/main.pdf"
