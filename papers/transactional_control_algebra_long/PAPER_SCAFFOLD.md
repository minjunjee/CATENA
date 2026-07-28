# Transactional Control Algebra: Matching Memory-Control Architecture to Update-Demand Geometry

Eight-page long-paper scaffold. Provisional title and prose; scientific
numbers must be inserted from `figures/RESULTS_MACROS.md`, never transcribed
from memory.

## Abstract

Memory systems must transform a current state in response to transactions, but
controller design is usually justified by architecture precedent rather than
the geometry of the required updates. We study a controlled transactional
algebra in which reachable update sets, operator rank, basis sharing, and
operation-specific control freedoms can be measured directly. Bounded
behavioral reachability predicts unseen update error; prospectively graded
families show that learned rank and joint diagonalizability determine which
operator classes are sufficient. An architecture-demand lattice then isolates
when erase/write factorization, value-channel granularity, address
decoupling, and state conditioning are selectively necessary. Finally, paired
structured-sequence experiments test whether the tied-versus-dual distinction
persists across repeated updates and long distractor gaps. A fixed-codebook
decomposition further isolates learned address selection from current-state
erase-candidate recovery. Together these results support a design rule: the
geometry and algebra of update demands determine the minimal memory-control
architecture. The evidence is controlled-reference and structured-sequence
evidence; semantic factorization is closed for this submission, and the
preregistered official-operator gate did not pass.

## Eight-page content budget

| Main-content section | Pages | Primary payload |
|---|---:|---|
| 1. Introduction | 0.85 | Problem, thesis, contributions |
| 2. Transactional control algebra | 1.10 | Reachability, rank, joint basis, lattice |
| 3. Experimental design | 0.95 | Protocol lineage, pairing, artifacts, estimands |
| 4. Geometry and learned sufficiency | 1.25 | H1, E10b, E03b; Figure 1 |
| 5. Architecture-demand lattice | 1.10 | E12 selective gains; Figure 2 |
| 6. Structured sequence transfer | 1.05 | E13a/R1 and E13c-R1; Figure 3 |
| 7. Localization, boundaries, and systems proxy | 0.95 | E19, H5, E14, E15, E20 |
| 8. Related work and conclusion | 0.75 | Positioning and design rule |
| **Total main content** | **8.00** | References and appendices separate |

## 1. Introduction

### Motivation

Frame transaction assimilation as a constrained update problem. A memory
architecture does not merely need enough parameters; its controller must be
able to reach the operation family required by the environment. Existing
comparisons often entangle representational choice, update rank, address
selection, and optimization.

### Thesis

> The geometry and algebra of update demands determine the minimal
> memory-control architecture.

### Contributions

1. A behavioral reachability account that predicts unseen update error.
2. A prospective learned-rank result and a calibrated
   joint-diagonalizability result connecting demand geometry to controller
   sufficiency.
3. A selective architecture-demand lattice spanning magnitude, value,
   address, and state-conditioned freedom.
4. A paired structured-sequence bridge showing that independent erase/write
   control remains useful under repeated events and distractor gaps.
5. A controlled decomposition showing complementary recovery from learned
   address selection and current-state erase-candidate reads.
6. Immutable artifact provenance and deterministic main-figure regeneration.

State the evidence boundary in the introduction: this is controlled-reference
and structured relational evidence, not a passed official GDN2/KVEraser or
pretrained-language-model comparison. E20 is separately labeled as a
controlled in-process systems proxy.

## 2. Transactional control algebra

### 2.1 Reachable update sets

Define a state update target, the architecture-constrained reachable set, and
behavioral regret after readout. Explain why state-space proximity can differ
from behaviorally relevant proximity.

### 2.2 Control rank

Define descriptor-conditioned target operators and the best-rank reachable
floor. Distinguish intrinsic target rank, learned controller rank, and
excess-over-oracle. The claim concerns the minimum learned rank satisfying the
prospectively floor-aware quality rule.

### 2.3 Shared bases and demand algebra

Define commuting, common-rotated commuting, and noncommuting operator
families. A shared diagonal controller is sufficient only when a common basis
can jointly diagonalize the demand family; representation learning can absorb
a common rotation but not arbitrary transaction-dependent noncommutativity.

### 2.4 Architecture lattice

Introduce the nested reachable sets:

```text
tied scalar
  ⊂ dual scalar
  ⊂ diagonal value control
  ⊂ separate erase/write address control
  ⊂ state-aware control
```

Each inclusion adds a specific freedom. Predict selective gains only when the
demand family requires the added degree of freedom, alongside
simpler-demand and retention guardrails.

## 3. Experimental design

### 3.1 Evidence lineage

Summarize completed controlled core H1–H4, then motivate post-core E10b, E12,
and E13c-R1. Preserve the distinction between original inconclusive protocols
and prospective repairs in the appendix provenance table.

### 3.2 Fair comparisons

Document paired seeds, identical data order and optimization budgets, maximal
parameter-surface matching where applicable, held-out operators or
transactions, and seed-level statistical units. Do not substitute episode
counts for independent seeds.

### 3.3 Reproducibility contract

Point readers to the immutable run IDs and hashes in
`data/source_manifest.json`. Explain that figure generation first verifies all
inputs and then derives plot-ready JSON and SVG. Mention that report-level
claim gates, not figure appearance, determine status.

## 4. Geometry and learned sufficiency

Place `figures/figure1_geometry.svg` across both columns.

### 4.1 Behavioral reachability

Report `H1_BEHAVIORAL_OOS_R2` and the calibrated slope token from the generated
macro table. Explain operation adjustment and why behavioral feasible regret
outperforms state-only alternatives.

### 4.2 Learned rank scaling

Report `E10B_RANK_MATCH_FRACTION`. Emphasize that E10b prospectively repaired
the floor-definition issue and tests a learned controller, not only a
low-rank oracle.

### 4.3 Joint-diagonalization calibration

Report `E03B_JD_R2` and `E03B_JD_SLOPE`. Connect graded analytic regret to
held-out application error. Preserve the historical fact that the original
E03 established the categorical boundary but lacked predictor range; E03b was
a separate prospective calibration.

## 5. Architecture-demand lattice

Place `figures/figure2_control_lattice.svg` across both columns.

### 5.1 Static controlled lattice

Explain the four adjacent comparisons and use the generated E12 macro tokens.
The main inferential point is selectivity: a richer reachable set helps on the
matched demand family while guardrails test non-inferiority on simpler demands
and unaffected retention. Avoid describing raw controller size as the causal
explanation.

Include a compact table in final typesetting:

| Added freedom | Matched demand | Primary contrast | Guardrails |
|---|---|---|---|
| Erase/write magnitude | Asymmetric same-address update | tied → dual scalar | symmetric demands, retention |
| Value channels | Partial value-subspace update | dual scalar → diagonal value | scalar-demand non-inferiority |
| Address decoupling | Different erase/write addresses | diagonal value → separate address | prior-demand non-inferiority |
| State conditioning | Same observation, state-dependent update | separate address → state-aware | prior-demand non-inferiority |

### 5.2 Repeated-sequence lattice (E18b)

The E18b aggregate, compact summaries, and top-level freeze are validated and
hash-pinned in the paper manifest. Describe the setting before the result:

> E18 supplies oracle erase/write addresses, oracle candidates, explicit
> oracle demand descriptors (family, operation, and channel mask where
> applicable), and a model-visible verified-event bit.

Use the four generated E18 gain macros as **registered-grid mean adjacent
gains**. All four registered adjacent conjunctions passed. State that
simpler-demand and retention checks are maximum cell-mean adjacent
non-inferiority guardrails, not absolute-accuracy claims. For the
`updates=8, gap=2048` stress cell, the strongest confirmatory wording is
`positive in 5/5 paired seeds` with exact sign-flip `p=0.03125`.

Do not write `every cell improved`, `uniform persistence`, `stress SESOI
maintained`, or `accurate preservation`. The primary gain can average across
update×gap cells, and the stress gate has no separate SESOI. `Selective` means
matched adjacent gain plus simpler-demand/retention non-inferiority; it does
not mean the added freedom has zero benefit on all other harder demands.

## 6. Structured sequence transfer

Place `figures/figure3_sequence_transfer.svg` across both columns.

### 6.1 Calibration and paired training

Describe E13a as a preregistered floor/throughput gate and E13c-R1 as the
aggregate over paired tied/dual cells. Give the update-count and distractor-gap
grid from the generated source data.

### 6.2 Result

Use `E13C_OVERALL_GAIN` and `E13C_STRESS_GAIN`. Discuss the complete grid and
seed ranges rather than highlighting only the stress endpoint. Restrict the
claim to repeated structured relational transactions behind a shared event
encoder.

## 7. Localization, boundaries, and systems proxy

### 7.1 Learned localization/candidate decomposition

E19 removes the oracle assumptions one at a time in a fixed-slot random-code
setting. Report `E19_SEPARATE_ADDRESS_GAIN`, `E19_STATE_READ_GAIN`, and
`E19_FULL_ONLY_GAIN`: separate-address control recovers the learned-address
condition, state-aware control recovers the state-read condition, and only the
full controller recovers when both bottlenecks are active. Address accuracy
and candidate error reach their registered capable floors without retention
damage. This is learned localization over a fixed codebook, not semantic or
novel-entity generalization.

### 7.2 Semantic factorization is closed

E05a-R1 repaired relation representation, but the parameter-matched shared
controller also approached the oracle neighborhood. The registered absolute
factorized advantage and direction gate did not pass. Therefore H5 is closed
for this submission—terminated, not refuted—and no E05b claim is made.

### 7.3 Structured-sequence localization transfer did not open

E21 tested learned localization and current-state candidate recovery over
repeated structured events with a fixed identifier schema and explicit
demand/provenance fields. The original E21b aggregate is frozen as
`INCONCLUSIVE_GATE_IMPLEMENTATION`; its active-guardrail implementation did
not identify the intended state-read and cellwise non-inferiority tests. A
prospectively locked aggregate-only repair retained the source runs and
primary estimands. In E21b-R1, all three primary contrasts passed their
registered effect and 5/5 paired-seed direction gates, but the capable
affected-error floor and cellwise non-target non-inferiority gate failed. The
full status is therefore `NOT_SUPPORTED`, and the positive primary contrasts
must not be promoted to a selective-recovery or sequence-transfer claim.

This is a negative claim boundary, not evidence for semantic/natural-language,
novel-identifier, pretrained or recurrent-LM, agent/planning,
official-backend, or runtime transfer.

### 7.4 Proxy and backend boundaries

E14 supports only a structured synthetic plan-state proxy; it is not evidence
for general agent planning. E15a-R1 executed pinned official GDN2 and FLA
kernels, but only four of six registered checks passed: tied-reduction parity
and BF16 parity exceeded their preregistered tolerances. A post-run no-patch
audit localized tied divergence to the official kernels' different
IEEE-versus-TF32 solve policies at the first inter-subchunk boundary, not to
probe wiring. The outcome remains `FAIL`, used no reference fallback, and
provides no official GDN2/KDA architecture-transfer or language-model
evidence. E15b KVEraser remains unconfigured.

### 7.5 Quality-constrained systems proxy

E20 compares one-time device-resident assimilation with three in-process
external-state proxies over the registered follow-up-query grid. Report the
three generated break-even values and quality guardrails, while making the
timing boundary explicit: persistent internal-state placement is outside the
timed region, the assimilation update is inside it, and external baselines
include their registered CPU-to-device transfers. This is neither a storage
service nor a production serving benchmark.

### 7.6 Implication

Separate the capacity question from the inference question. The controlled
algebra predicts which update freedoms are necessary once demand is specified;
semantic demand inference and official-system transfer remain distinct future
problems.

## 8. Related work and conclusion

### Related-work buckets

- Fast weights, associative memories, and recurrent state updates.
- Memory editing, erase/write control, and transaction assimilation.
- Low-rank adaptation and operator-valued prediction.
- Simultaneous diagonalization, commuting operator families, and structured
  linear algebra.
- Causal or functional intervention on learned control mechanisms.

Use the verified primary-source map in
`RELATED_WORK_PRIMARY_SOURCES.md` when constructing the final bibliography.
Do not turn a related-work citation into evidence that CATENA evaluated that
system.

### Conclusion

Restate the design rule and the three empirical bridges: error to reachability,
minimal class to rank/basis algebra, and controlled probes to repeated
structured sequences. End with the bounded future test: official
quality-controlled backends and semantic demand inference without changing
the already frozen controlled-core claims.

## References

Outside the eight main-content pages under the current REALM CFP. Populate
after the related-work audit.

## Appendix plan

- Complete protocol and prospective-repair chronology.
- Full seed-level tables and confidence/sign-flip procedures.
- Controller parameter surfaces and compute accounting.
- E12 non-inferiority and retention guardrails.
- E18 registered-grid mean contrasts, full cell table, explicit oracle demand
  descriptors/verified-bit boundary, and scientific claim-boundary audit.
- E13a gate, E13b cell inventory, and E13c full metrics.
- H5 closed-status report and leakage-control history.
- E14 proxy-only, E15a failed-gate and no-patch audit, and E15b unconfigured
  boundary records.
- E19 fixed-codebook learned-localization and current-state candidate
  decomposition.
- E21 original gate-implementation disposition, R1 primary contrasts, failed
  capable affected-error floor, and failed cellwise non-target guardrail.
- E20 raw timing repeats, quality guardrails, and timing-boundary diagram.
- Artifact paths, SHA-256 hashes, and reproduction commands.
