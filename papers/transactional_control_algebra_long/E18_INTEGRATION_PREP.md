# E18 long-paper integration record

E18b is now integrated from the provenance-valid MAIN aggregate
`20260728T074753.618843Z`. Its top-level freeze is
`E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json`, SHA-256
`39416476994963900305e04682dabd458719a0a94c92820405feeb639c33e67c`.
The source manifest, generated Figure 2, result macros, and deterministic
source-data freeze have been regenerated and validated.

Completed integration contract:

1. Both records below are in `data/source_manifest.json` with their observed
   relative paths and SHA-256 values:

   - `e18b_report` → `report.json`
   - `e18b_paired_metrics` →
     `sequence_control_lattice_paired_metrics.jsonl`

2. The validated E18 top-level freeze is under `provenance_anchors`.
3. All paper outputs were regenerated; the byte-for-byte reproducibility
   check passed for 10/10 generated outputs.
4. Figure 2 and the manuscript use the frozen `SUPPORTED` disposition subject
   to the audited claim boundary below.

The generator treats the two E18 data records as all-or-none. With both
records present, Figure 2 is a shared-scale two-panel comparison: static E12
and repeated structured-sequence E18b. The E18 derivation independently checks
the five-seed paired grid,
adjacent-controller identities, report means, stress-direction fractions,
evidence tier, and claim disposition.

E18 must be described as a controlled execution-capacity probe with oracle
erase/write addresses, oracle candidates, **explicit oracle demand
descriptors** (family, operation, and channel mask where applicable), and a
**model-visible verified-event bit**. The registered primary estimand is the
matched adjacent controller's **mean gain over the update-by-gap grid**. It
does not require a positive gain in every cell. The 2,048-event stress gate
requires a positive direction in 5/5 paired seeds, but has no separate SESOI.

Accordingly, do not write `every cell`, `uniform persistence`, `stress SESOI
maintained`, or `accurate preservation`. Simpler-demand and retention gates
are adjacent cell-mean non-inferiority margins, not absolute-quality gates.
E18 also does not establish semantic demand inference, learned relevance,
learned localization/candidates, arbitrary event interleaving, natural
language, pretrained-model, official-backend, planning, or agent evidence.
The full scientific audit is in
`../../docs/E18_CLAIM_BOUNDARY_AUDIT_KO.md`.
