# Related-work primary-source map

Verified 2026-07-28 (UTC). This is an additive writing aid for the
Transactional Control Algebra long paper. Links point to primary papers on
official publisher or proceedings pages, arXiv, or OpenReview; venue years are
used for published papers, and explicitly marked preprints use their arXiv
year.

> **Evidence boundary — context only, not CATENA evidence.** These papers
> position the manuscript; they are not CATENA inputs, artifacts, replications,
> or evaluations. In particular, citing DeltaNet, Gated DeltaNet, KDA, GDN2,
> KVEraser, Transformers, or language models does not establish an
> official-backbone or language-model claim. CATENA claims remain limited by
> `CLAIM_BOUNDARIES.md` and the hash-pinned artifact record.

## Fast weights and recurrent linear attention

- [Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks](https://doi.org/10.1162/NECO.1992.4.1.131) — Schmidhuber (1992), *Neural Computation*. Relevance: the slow network's context-dependent programming of a rapidly changing weight memory is an early formulation of descriptor-conditioned memory operators.
- [Using Fast Weights to Attend to the Recent Past](https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html) — Ba et al. (2016), NeurIPS. Relevance: its fast associative matrix provides a concrete precedent for treating transient recurrent state as an editable memory rather than as fixed model parameters.
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html) — Katharopoulos et al. (2020), ICML. Relevance: the recurrent formulation of causal linear attention supplies the fixed-size matrix-state bridge between attention and transactional state updates.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) — Schlag, Irie, and Schmidhuber (2021), ICML. Relevance: the paper makes the fast-weight interpretation explicit and replaces additive writes with a delta-rule instruction that can correct an existing key-value mapping.

## DeltaNet, KDA, and GDN2

- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d13a3eae72366e61dfdc7eea82eeb685-Abstract-Conference.html) — Yang et al. (2024), NeurIPS. Relevance: it identifies the delta-rule recurrent update as DeltaNet and shows how that matrix-state mechanism can be trained in parallel over sequence length.
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://openreview.net/forum?id=r8H7xhYPwz) — Yang, Kautz, and Hatamizadeh (2025), ICLR. Relevance: Gated DeltaNet combines adaptive forgetting with delta-rule correction, making erase and targeted update mechanisms explicit architectural objects.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Team et al. (2025), arXiv preprint. Relevance: the paper introduces Kimi Delta Attention (KDA), whose finer-grained decay and diagonal-plus-low-rank transition structure provide the direct KDA comparison point.
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791) — Hatamizadeh, Choi, and Kautz (2026), arXiv preprint. Relevance: GDN2 separates channel-wise erase and write gates and states tied reductions to KDA and Gated DeltaNet, making it the closest external architectural comparison to CATENA's erase/write factorization.
- [Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention](https://arxiv.org/abs/2606.26560) — Li et al. (2026), arXiv preprint. Relevance: EDA gives erasure an address independent of the delta-rule write address, directly matching the address-decoupling axis in CATENA's architecture-demand lattice.
- [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) — Dutta (2026), arXiv preprint. Relevance: CARVE lets the erase gate consult the stored recurrent state, providing a direct external example of state-conditioned rather than input-only control.

## Memory editing and KV erasure

The first three papers below edit model parameters; KVEraser instead edits a
runtime KV cache. Neither operation should be conflated with CATENA's
controlled recurrent-state transactions.

- [Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309) — Mitchell et al. (2021 preprint; ICLR 2022). Relevance: MEND learns an auxiliary editor that turns an edit example into a localized, low-rank-transformed parameter update.
- [Locating and Editing Factual Associations in GPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) — Meng et al. (2022), NeurIPS. Relevance: ROME couples causal tracing with a rank-one parameter edit, providing a useful contrast between localized model-weight editing and online memory-state control.
- [Mass-Editing Memory in a Transformer](https://openreview.net/forum?id=MkbcAHIYgyS) — Meng et al. (2023), ICLR. Relevance: MEMIT extends direct parameter editing to many associations and foregrounds the efficacy-specificity trade-off that any claimed edit must guard.
- [KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing](https://arxiv.org/abs/2606.17034) — Li et al. (2026), arXiv preprint. Relevance: KVEraser replaces KV states for an erased interval with learned steering states while reusing the remaining cache, making it the direct runtime-cache erasure comparison.

## Low-rank operator prediction and adaptation

- [HyperNetworks](https://arxiv.org/abs/1609.09106) — Ha, Dai, and Le (2016), arXiv preprint. Relevance: a network that generates another network's weights is a general precedent for predicting transaction- or descriptor-conditioned operators.
- [Dynamic Filter Networks](https://proceedings.neurips.cc/paper/2016/hash/8bf1211fd4b7b94528899de0a43b9fb3-Abstract.html) — Jia et al. (2016), NeurIPS. Relevance: its input-conditioned filter-generating network is an early example of predicting an operator on the fly rather than selecting a fixed transformation.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (2021 preprint; ICLR 2022). Relevance: LoRA establishes low-rank factorization as an effective parameter-adaptation constraint, while differing from CATENA's per-transaction state operator in both timescale and target.
- [Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309) — Mitchell et al. (2021 preprint; ICLR 2022). Relevance: MEND's low-rank gradient decomposition links operator prediction to localized adaptation, but it remains a parameter editor rather than recurrent-memory evidence.

## Joint diagonalization and commuting operators

- [Numerical Methods for Simultaneous Diagonalization](https://epubs.siam.org/doi/10.1137/0614062) — Bunse-Gerstner, Byers, and Mehrmann (1993), *SIAM Journal on Matrix Analysis and Applications*. Relevance: its stable unitary diagonalization of commuting normal-matrix pairs supplies the exact shared-basis setting behind the manuscript's commuting-family case.
- [Jacobi Angles for Simultaneous Diagonalization](https://epubs.siam.org/doi/10.1137/S0895479893259546) — Cardoso and Souloumiac (1996), *SIAM Journal on Matrix Analysis and Applications*. Relevance: the Jacobi-style reduction of aggregate off-diagonal mass is a classical computational precedent for an approximate joint-diagonalization objective.
- [Almost-commuting matrices are almost jointly diagonalizable](https://arxiv.org/abs/1305.2135) — Glashoff and Bronstein (2013), arXiv preprint. Relevance: the paper relates commutator size to approximate joint diagonalizability for self-adjoint matrices, offering mathematical context—but not validation—for CATENA's graded demand-family calibration.

## Causal mediation and internal interventions

- [Direct and Indirect Effects](https://ftp.cs.ucla.edu/pub/stat_ser/R273-U.pdf) — Pearl (2001), UAI. Relevance: the paper formalizes direct, indirect, and path-specific effects that ground the manuscript's careful use of mediation language.
- [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/abs/2004.12265) — Vig et al. (2020), arXiv preprint. Relevance: it operationalizes neural components as mediators under controlled interventions, providing direct methodological context for mechanism-level mediation tests.
- [Causal Abstractions of Neural Networks](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) — Geiger et al. (2021), NeurIPS. Relevance: interchange interventions test whether aligned neural representations realize hypothesized causal variables, closely contextualizing transplant-style functional interventions.

## Citation-key map for the paper scaffold

- Fast weights and recurrent attention: `schmidhuber1992fastweights`, `ba2016fastweights`, `katharopoulos2020transformersrnn`, `schlag2021fastweight`.
- Delta-rule architecture family: `yang2024deltanet`, `yang2025gateddeltanet`, `kimiteam2025kimilinear`, `hatamizadeh2026gdn2`, `li2026eda`, `dutta2026carve`.
- Model/cache editing: `mitchell2022mend`, `meng2022rome`, `meng2023memit`, `li2026kveraser`.
- Operator prediction and low rank: `ha2016hypernetworks`, `jia2016dynamicfilters`, `hu2022lora`; reuse `mitchell2022mend` for MEND.
- Joint diagonalization: `bunsegerstner1993simultaneous`, `cardoso1996jacobi`, `glashoff2013almost`.
- Causal interventions: `pearl2001direct`, `vig2020causalmediation`, `geiger2021causalabstractions`.

## Manuscript-use guardrail

- Cite these sources for definitions, historical lineage, or architectural comparison only.
- Do not import their reported metrics into CATENA tables or artifact manifests.
- Do not describe CATENA's controlled-reference or structured-sequence results as a replication, evaluation, or validation of any cited official architecture.
