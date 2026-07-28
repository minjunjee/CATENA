# Format and submission references

Verified on 2026-07-28. The workshop call for papers is authoritative if any
general ACL guidance differs.

## Official facts used by this scaffold

- The [REALM 2026 call for papers](https://realm-workshop.github.io/call_for_papers/)
  permits long papers with up to eight pages of main content, with references
  and appendices outside that content limit, and directs authors to the ACL
  2026 style. The same page currently lists the direct-submission deadline as
  **2026-08-05 23:59 Anywhere on Earth (UTC−12)** and requires anonymous
  double-blind submission.
- The [ACL publication formatting guide](https://acl-org.github.io/ACLPUB/formatting.html)
  says that a workshop's call for papers determines its page limit and
  submission details. It also documents the ACL layout requirements, including
  A4 PDF output, an 11-point two-column body, figure captions, and a preference
  for vector graphics.
- The maintained [official ACL style-file repository](https://github.com/acl-org/acl-style-files)
  is the source to use when constructing the eventual LaTeX submission.
- If the paper is submitted through ARR, consult the current
  [ARR author instructions](https://aclrollingreview.org/authors) in addition
  to the workshop-specific call.

These links record format facts only. They do not imply that a submission
venue, deadline, anonymity policy, or archival status has been selected.

## Current repository choice

`latexmk`, `pdflatex`, and `xelatex` remain unavailable in the current
`catena-v6` environment. The additive `tex/` handoff therefore:

1. keeps the eight-page content budget in `PAPER_SCAFFOLD.md`;
2. imports only generated result tokens and deterministic vector figures;
3. separates references and appendices from the eight-page plan;
4. uses anonymous `review` mode; and
5. vendors `acl.sty` and `acl_natbib.bst` from the official ACL style
   repository at commit
   `d5adc823ff0f80f98c80405ca0ab66c68e684409`, with file hashes in
   `tex/vendor/acl/PROVENANCE.json`.

The pin is a reproducible handoff snapshot, not a claim that it will remain
the venue-required revision. Re-check it against the current REALM call
immediately before submission.

## Final-format checklist

- Re-check the workshop call and official style repository immediately before
  submission.
- Confirm that the OpenReview profiles needed for submission and reviewer
  nomination are active before the deadline.
- Use the exact style version required by the selected venue.
- Preserve anonymous-review requirements until the venue permits
  de-anonymization.
- Confirm eight or fewer main-content pages in the compiled PDF.
- Keep references and appendices in the locations allowed by the current CFP.
- Convert SVGs to the vector format accepted by the TeX pipeline without
  rasterizing data marks or text.
- Verify embedded fonts, A4 page size, legible two-column figure labels,
  bibliography integrity, and accessible color contrast.
