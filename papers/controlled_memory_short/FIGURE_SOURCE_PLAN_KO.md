# CATENA controlled-memory short paper figure source plan

> 목적: 4쪽 본문의 Figure 1과 Figure 2를 immutable artifact에서 재생성하기
> 위한 source-of-truth와 panel contract를 고정한다. 이 문서는 figure 자체를
> 만들거나 실험 artifact를 변경하지 않는다.

## 1. 공통 제작 원칙

- Canonical root는 `/data/minjun_dev/CATENA/artifacts`다. Repository에서는
  `artifacts/...` symlink 경로로 동일 파일을 읽을 수 있다.
- `latest.json`을 사용하지 않는다. 모든 reader는 아래 exact run ID와 SHA-256을
  먼저 검증하고 hash mismatch에서 즉시 실패해야 한다.
- Raw row를 수정하지 않는다. Plot용 집계는 deterministic derived table로
  만들며 source path, source hash, filter, group keys, aggregation을 metadata에
  남긴다.
- Registered report의 CI와 p-value만 inferential annotation으로 사용한다.
  새 CI, 새 hypothesis test 또는 episode를 독립 replicate로 취급한
  pseudo-replication을 추가하지 않는다.
- 색상은 controller/component에 고정한다.
  - tied/shared scalar: gray
  - dual/factorized: blue
  - erase: orange
  - write: teal
  - oracle upper bound: black outline
  - prospective repair/calibration: solid
  - original failed/inconclusive disposition: dashed annotation
- 모든 figure footer 또는 caption에
  `CONTROLLED_REFERENCE · OracleCandidate/address · no official-backend/LM claim`
  을 넣는다.
- E03 rank-8은 항상 `oracle upper bound`라고 쓴다. E04 independent donor는
  globally identifiable natural mediator라고 쓰지 않는다.

## 2. Figure 1 — Reachability and shared-basis geometry

### 전달할 메시지

목표 update와 bounded reachable behavior 사이의 거리가 learned error를
예측하고, diagonal control의 충분성은 현재 coordinate axis가 아니라 demand
family가 shared basis에서 jointly diagonalizable한지에 달려 있다.

### 권장 배치

```text
┌──────────────────────┬────────────────────────────┐
│ A. Reachable set     │ C. Three operator families │
│    geometry sketch   │    and residuals           │
├──────────────────────┼────────────────────────────┤
│ B. H1 OOS relation   │ D. E03b graded calibration │
└──────────────────────┴────────────────────────────┘
```

### Panel A — Behavioral reachable-set schematic

Data-free vector schematic다.

- Target behavior \(b^\star\), bounded reachable set
  \(\mathcal B_\theta\), nearest reachable point
  \(\Pi_{\mathcal B_\theta}(b^\star)\), distance
  \(R_{\mathrm{beh}}\)를 표시한다.
- State-space span과 behaviorally bounded set을 다른 선으로 표시하되, H1이
  검정한 것은 box-constrained equal-weight **behavioral MSE lower bound**임을
  label에 쓴다.
- 이 panel의 기하 그림은 설명용이며 empirical scale 또는 observed
  confidence region으로 표현하지 않는다.

### Panel B — H1 unseen-geometry calibration

Source:

```text
path:
artifacts/e01b_constrained_behavioral_reachability/
20260726T152354.081239Z/episode_geometry_metrics.jsonl
sha256:
c2a5afff71804da645d884c832b8c6dec6bab1b4eaefe0cf16d067f28d9547ac
rows: 239616
```

Filter:

```text
candidate_mode == "oracle_candidate"
constraint == "tied"
split == "test"
x = behavior_feasible_mse
y = learned_behavior_mse
color/facet = operation
```

Plot contract:

- Operation별 저투명도 point 또는 hexbin을 쓰고 \(y=x\) reference를 표시한다.
- 그림에 새 pooled regression line을 넣지 않는다.
- Report에서 가져온 registered annotation만 쓴다.

```text
conditional unseen-geometry OOS R² = 0.9976154443
operation-adjusted slope = 1.0086329002
8-seed exact sign-flip p = 0.00390625
```

Conditional \(R^2\)의 정의는
`1-SSE(full operation+predictor)/SSE(operation-only)`임을 caption에 쓴다.
보이는 raw scatter의 \(R^2\)로 오해하게 만들지 않는다.

Report source:

```text
path:
artifacts/e01b_constrained_behavioral_reachability/
20260726T152354.081239Z/report.json
sha256:
8e1d16ca7763cec1e4e5b13d2b0f163f4015c8058ed7764871a6fbb5fa5ea6d6
```

### Panel C — Joint-diagonalizability regimes

왼쪽 절반은 세 family의 data-free matrix schematic, 오른쪽 절반은 원 E03의
8-seed mean residual이다.

Source:

```text
path:
artifacts/e03_granularity_orientation/
20260726T161535.271015Z/operator_family_metrics.jsonl
sha256:
54efda16cc92da17e0a40472de0075b7fec252c3451fde682304f7f63fa29dc4
rows: 24
```

Group:

```text
group = family
unit = seed
fields =
  fixed_diagonal_regret
  learned_basis_diagonal_regret
  low_rank_regret
  commutator_norm
```

표시할 mean:

| Family | Fixed diagonal | Learned shared basis | Rank-8 oracle |
|---|---:|---:|---:|
| Axis commuting | 0 | 0 | 0 |
| Common-rotated commuting | 0.00550276 | \(7.52\times10^{-22}\) | \(5.52\times10^{-33}\) |
| Noncommuting | 0.00553512 | 0.00551690 | \(5.40\times10^{-33}\) |

강조 contrast:

```text
fixed-basis rotation penalty = 0.0055027558
shared-basis recovery = 0.0055027558
noncommuting shared-basis gap = 0.0055169034
rank-8 oracle recovery = 0.0055169034
each exact sign-flip p = 0.00390625
```

원 E03의 categorical package는 supported지만 quantitative calibration은
failed였다는 작은 dashed note를 둔다.

```text
Original E03 calibration:
R² = 0.863588, slope = 0.895268, intercept = 5.7756e-4
Disposition: FAILED; predictor-range restriction; not relabelled
```

Report source:

```text
path:
artifacts/e03_granularity_orientation/
20260726T161535.271015Z/report.json
sha256:
ee0114f45d5facbc3ccdd0e3a0235531e1de078f29fd7a949420df0899fa98c0
```

### Panel D — E03b graded JD calibration

Source:

```text
path:
artifacts/e03b_graded_jd_calibration/
20260726T180514.626996Z/family_calibration_metrics.jsonl
sha256:
47d264beb42e5826646cb69c62db0824688c9af0a01d2b7311923f67b14ee086
rows: 48
```

Mapping:

```text
x = heldout_analytic_regret
y = heldout_empirical_application_error
color = bin
point = one family mean over 8 held-out operators
reference = y=x
fit annotation = registered report fit, not a newly fit line
```

Annotation:

```text
6 bins × 8 families
predictor range = 0.0053038816
R² = 0.9999981686
slope = 1.0002589211
intercept = 1.8464323e-7
```

E03b는 analytic held-out predictor로 stratified/selected된 locked-bin
conditional Monte Carlo calibration이다. Untouched natural-distribution test
또는 certified global JD optimum으로 표현하지 않는다.

Lock/report sources:

```text
preprobe lock:
artifacts/e03b_graded_jd_calibration/
20260726T180514.626996Z/preprobe_lock.json
sha256:
4167a5b6b6304a4d818818ac20e2a8dd5f9f722b4864dc5b8bb54f966f496a56

report:
artifacts/e03b_graded_jd_calibration/
20260726T180514.626996Z/report.json
sha256:
fc88dd3923bbcfb63b99953a9f839b25cbee956f155ce1dd4c397ea82a71772c
```

### Figure 1 caption 초안

> **Behavioral reachability and shared-basis geometry predict controlled memory
> error.** (A) The relevant lower bound is the distance to the bounded reachable
> behavior set. (B) In the OracleCandidate/tied condition, this bound predicted
> operation-adjusted unseen-geometry error with conditional
> \(R^2=0.9976\). (C) A learned shared basis removed the residual for a commonly
> rotated commuting family but not for noncommuting demands; rank-8 is an oracle
> richer-control upper bound. (D) In the separately preregistered graded
> construction, estimated held-out JD regret calibrated to isotropic application
> error across six locked bins (\(R^2=0.999998\)). Results are
> `CONTROLLED_REFERENCE`, not official-backend or language-model evidence.

## 3. Figure 2 — Factorization gap and functional intervention

### 전달할 메시지

Factorized erase/write control은 asymmetric operations에서 tied-control gap을
제거하면서 symmetric operations에서는 동등했고, frozen-checkpoint dose,
transplant, rescue, scalarization이 이 두 component의 operation-matched 기능과
architecture-gap mediation을 보였다.

### 권장 배치

```text
┌──────────────────────┬────────────────────────────┐
│ A. H2 operation gap  │ B. Retained-dose curves    │
├──────────────────────┼────────────────────────────┤
│ C. Transplant/rescue │ D. Gap scalarization       │
└──────────────────────┴────────────────────────────┘
```

### Panel A — E02b operation-specific tied-minus-dual gap

Primary plotting source는 원 E02가 아니라 fresh OOD E02b다.

```text
path:
artifacts/e02b_prospective_absolute_supersede/
20260726T180207.055493Z/prospective_episode_metrics.jsonl
sha256:
ee544ac88ca632f1412e54a7c3f9e0f8e235ad8721cbc9abffb333b2b6a17171
rows: 16384
```

Aggregation:

```text
first unit = checkpoint_seed × operation mean
display = 8 seed points + grand mean
y = tied_minus_dual_affected_mse
operations = preserve, add, invalidate, supersede
```

Raw grand means:

| Operation | Tied MSE | Dual MSE | Tied − dual |
|---|---:|---:|---:|
| PRESERVE | \(2.2617\times10^{-8}\) | \(2.3623\times10^{-9}\) | \(2.0255\times10^{-8}\) |
| ADD | 0.0163484842 | \(2.6683\times10^{-9}\) | 0.0163484815 |
| INVALIDATE | 0.0163379389 | \(2.4553\times10^{-9}\) | 0.0163379365 |
| SUPERSEDE | \(2.9892\times10^{-8}\) | \(2.5536\times10^{-9}\) | \(2.7339\times10^{-8}\) |

Registered annotations:

```text
asymmetric normalized gain = 0.9999997598
positive asymmetric−symmetric interaction = 0.0163431852
PRESERVE/SUPERSEDE absolute equivalence margin = ±0.0005
fresh gates = 5/5; full registered conjunction = 6/6
```

Caption note:

> Original E02 remained `INCONCLUSIVE` because the preregistered relative
> SUPERSEDE denominator vanished. E02b was a separate prospective absolute
> equivalence repair on fresh OOD geometry; one of six gates was the inherited
> immutable tuning-direction fact.

Report and disposition:

```text
E02b report:
artifacts/e02b_prospective_absolute_supersede/
20260726T180207.055493Z/report.json
sha256:
032c1b015851b44555ce666ed1d50908332b13f0ad65608355559c000f1d3a52

Original E02 report:
artifacts/e02_magnitude_factorization/
20260726T153504.455509Z/report.json
sha256:
f3df03e231598d6eda11ebf71825ab418cc9a59ac9a96a299caff617291e4211
```

### Panel B — Operation-matched retained-component dose

Source:

```text
path:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/seed_level_effects.jsonl
sha256:
2fe7e595fe6d45467d7441797f8b2774be0cb681a809aacb49f6f71f81a05e2f
rows: 8
```

Mapping:

```text
x = retained physical component dose [0, .25, .5, .75, 1]
y = affected_read_mse
solid = relevant component
faint dashed = irrelevant component
thin line = checkpoint seed
thick line = mean of 8 seed means
```

Confirmatory relevant cells:

```text
ADD/write
INVALIDATE/erase
SUPERSEDE/erase
SUPERSEDE/write
```

Dose 1 is the unmodified dual baseline and dose 0 is component removal. All
32 relevant seed×cell monotonicity scores were 1.0. SUPERSEDE joint-dose is
secondary and should be omitted from the main panel or drawn as a clearly
labelled secondary dotted line.

### Panel C — Same versus cross transplant and rescue

Forest plot은 raw condition means가 아니라 registered cross-minus-same
contrasts를 표시한다.

| Contrast | Estimate | Registered 95% CI |
|---|---:|---:|
| Transplant, cross − same | 0.0624816855 | [0.0624816784, 0.0624816926] |
| Rescue, cross − same | 0.0312407527 | [0.0312407491, 0.0312407563] |
| Raw same-donor recovery | 0.0312499992 | [0.0312499989, 0.0312499995] |
| Same transplant − baseline | \(1.02\times10^{-14}\) | [\(-6.17\times10^{-15}\), \(2.60\times10^{-14}\)] |
| Same rescue − baseline | 0 | [0, 0] |

Positive-effect SESOI \(0.001\)과 equivalence margin
\(\pm0.0005\)를 서로 다른 visual guide로 표시한다. Same/cross transplant는
동일한 outcome-independent adjacent-two-cycle donor base를 공유하며, rescue의
oracle condition은 assay positive control일 뿐 confirmatory donor와 합치지
않는다.

Inferential values source:

```text
path:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/report.json
sha256:
7111e23ab70558a5130dff6937a3264c3ec6e4b5ea342183e52ea28f1bb36444
```

### Panel D — Post-hoc scalarization and gap recreation

두 estimate와 registered CI를 paired point-range로 표시한다.

| Quantity | Estimate | Registered 95% CI |
|---|---:|---:|
| Fresh tied − dual total effect | 0.0156250000 | [0.0156249998, 0.0156250002] |
| Post-hoc scalarization mediated effect | 0.0156249986 | [0.0156249984, 0.0156249987] |

옆에 ratio만 간결하게 표기한다.

```text
gap recreation fraction = 0.9999999057
95% CI = [0.9999998970, 0.9999999143]
registered minimum = 0.50
symmetric scalarization difference = -1.0466e-12
```

Scalarization은 post-hoc intervention이며 새 controller training이 아니다.
“100% causal identification” 대신 “registered gap-recreation conjunction
passed within the frozen probe”라고 쓴다.

### Figure 2 공통 E04 source contract

```text
run:
20260727T054917.678326Z

preintervention lock:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/preintervention_lock.json
sha256:
3b1f05873f084bc0073466433e9cfffbe5cc86059436e380a5f02329182278e3

intervention metrics:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/intervention_metrics.jsonl
sha256:
df5280041c50c1faa1656ab23fd28c5115ba31dde17e203bf89c05a072ec4b96
rows: 61440

report:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/report.json
sha256:
7111e23ab70558a5130dff6937a3264c3ec6e4b5ea342183e52ea28f1bb36444

manifest:
artifacts/e04_functional_mediation/
20260727T054917.678326Z/run_manifest.json
sha256:
7ae767b8fb7226588fd770194783308ff89a24b52b6f67389c55dab0e044ff7d

artifact freeze:
artifacts/E04_ARTIFACT_FREEZE_V1.json
sha256:
6d225b673da998cef9131af0b2d49fc699f89af2159f40c302898144c2765b30
```

### Figure 2 caption 초안

> **Factorized erase/write control closes asymmetric reachability gaps and
> supports operation-matched functional mediation.** (A) On the prospectively
> locked E02b OOD set, tied-minus-dual error was large for ADD and INVALIDATE
> but within the absolute-equivalence margin for PRESERVE and SUPERSEDE.
> Original E02 remains inconclusive; E02b is a separate prospective repair.
> (B) Removing the operation-relevant component produced monotonic damage,
> whereas irrelevant components were equivalent. (C) Same-operation
> transplant/rescue preserved or restored behavior, while cross-operation
> donors produced registered gaps. (D) Post-hoc scalarization recreated
> \(0.999999906\) of the fresh tied-dual gap. All results use frozen
> OracleCandidate/address controlled-reference checkpoints and do not establish
> semantic alignment, an official backend, or language-model behavior.

## 4. Pre-render validation checklist

- [ ] 모든 source SHA-256이 이 문서와 일치한다.
- [ ] Figure script에 exact run ID가 literal로 들어가고 `latest.json`을 읽지 않는다.
- [ ] H1 annotation이 raw pooled \(R^2\)가 아니라 conditional OOS \(R^2\)다.
- [ ] 원 E02는 `INCONCLUSIVE`, E02b는 separate prospective `SUPPORTED`로 보인다.
- [ ] 원 E03 quantitative gate는 `FAILED`, E03b는 separate prospective
      `SUPPORTED`로 보인다.
- [ ] E03 rank-8 label에 `oracle upper bound`가 포함된다.
- [ ] Dose x-axis가 “retained component dose”이며 dose 1이 baseline이다.
- [ ] SUPERSEDE joint-dose가 confirmatory처럼 보이지 않는다.
- [ ] Transplant/rescue error bar는 report의 registered CI다.
- [ ] E04 scalarization을 새 training 또는 semantic identification으로 표현하지 않는다.
- [ ] 두 caption 모두 `CONTROLLED_REFERENCE`, oracle candidate/address,
      no official-backend/LM claim을 포함한다.
- [ ] E05 semantic-effect 수치는 넣지 않는다. E05a가 `PASS / NO_GO`로
  E05b를 열지 않았다는 disposition은 limitations에만 둔다.
