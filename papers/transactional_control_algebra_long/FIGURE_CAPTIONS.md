# Main figure captions

All annotations are generated from the hash-pinned sources listed in
`data/source_manifest.json`. Exact manuscript values are available in
`figures/RESULTS_MACROS.md`; captions intentionally avoid manually duplicated
scientific numbers.

## Figure 1 — Geometry predicts the minimal control class

**Transactional control geometry across reachability, rank, and basis
families.** (A) Operation-adjusted out-of-sample prediction by three
reachability summaries in H1; bounded behavioral feasible regret is the
strongest predictor. (B) For every E10b seed and intrinsic-rank condition, the
minimum learned rank meeting the prospectively floor-aware quality criterion
is plotted against intrinsic rank; the diagonal is exact rank tracking.
(C) Held-out empirical application error versus analytic
joint-diagonalization regret for the prospectively stratified E03b families;
the diagonal denotes identity calibration. Panel annotations are derived from
the canonical reports, and points are derived from the corresponding frozen
cell-level files. These controlled-reference results establish a geometry
principle, not official-backend or language-model performance.

Source data: `figures/source_data/figure1.json`.

## Figure 2 — Architecture-demand control lattice

**Matched adjacent gains in static and repeated controlled probes.** Panel A
shows E12; Panel B shows E18b's registered mean affected-MSE gain over the
update-by-gap grid. Bars are frozen seed means and dots are paired seed-level
gains. From left to right, the transitions add independent erase/write
magnitude, value-channel granularity, separate erase/write addresses, and
state-conditioned control. E18b passed the registered relative cell-mean
simpler-demand and retention non-inferiority guards; its 8-update,
2,048-distractor stress gain was positive in 5/5 seeds, without a separate
stress SESOI. E18 supplies oracle addresses, candidates, demand descriptors,
and a model-visible verified-event bit. The result therefore concerns
controlled execution capacity: it does not establish positive gain in every
grid cell, minimal-controller sufficiency, semantic demand inference, or
learned event relevance.

Source data: `figures/source_data/figure2.json`.

## Figure 3 — Structured sequence transfer

**Independent erase/write control retains its advantage across repeated
structured updates and distractor gaps.** Each heat-map cell is the seed-mean
affected-MSE gain of dual over tied control for one update-count by gap-length
condition in E13c-R1. Cell annotations show the derived mean and the observed
seed range; all grid values come from the frozen paired metrics. The result is
evidence for transfer behind a shared structured event encoder. It is not
natural-language, pretrained-LM, official-backend, or general agent-planning
evidence.

Source data: `figures/source_data/figure3.json`.
