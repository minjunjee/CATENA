# Codex contract - CATENA v6.1

## Scientific invariants

- Keep `R_span`, bounded `R_feas`, and readout-level `R_beh` distinct.
- H1 primary is unseen-geometry behavioral-regret calibration with operation fixed effects.
- OracleCandidate and RecurrentRead are separate; RecurrentRead is candidate-recovery/content interference, not addressing.
- H2 passes only if asymmetric absolute gain, symmetric equivalence, positive interaction, retention non-inferiority, and tuning robustness all pass.
- Tied and dual use the same two-output head; tied is a diagonal projection of the same logits.
- H3 compares axis-commuting, common-rotated commuting, and noncommuting families. It is a joint-diagonalizability test, not a generic rotation ablation.
- H4 requires all paired seeds, operation counterfactuals with the same base state/candidates, dose, transplant, and nontrivial rescue.
- H5 input has no operation one-hot or oracle demand. Its primary split is held-out SUPERSEDE in seen domains/templates.
- H6/RQ-T cannot raise the REALM claim ceiling without official evidence.

## Files frozen while E00/E01 are active

Do not modify the paths listed in `docs/POST_E01_PATCH_INJECTION_KO.md`. v6.1 adds E01b instead of rewriting the running E01.

## Repository rules

- One Python entry point per experiment.
- Preserve raw per-episode JSONL and per-seed effects.
- Never silently fall back from official to mock/reference backends.
- Never revise claim thresholds after test artifacts exist.
- E08 is the claim-freeze authority.

<!-- CATENA_POSTCORE_EXTENSION_BEGIN -->
## Post-core extension (E10-E16)

Before changing or running post-core experiments, read `CODEX_POSTCORE.md` and then:

1. `docs/NEXT_ACTIONS_KO.md`
2. `docs/POSTCORE_DEPENDENCY_DAG_KO.md`
3. `docs/POSTCORE_ARTIFACT_CONTRACT_KO.md`
4. `docs/POSTCORE_CLAIM_GATES_KO.md`
5. `docs/CODEX_POSTCORE_TASKS.md`

Completed H1-H5 reports are immutable. H5 is closed for the current submission. New scientific work begins at E10 and must never rewrite prior artifacts.
<!-- CATENA_POSTCORE_EXTENSION_END -->
