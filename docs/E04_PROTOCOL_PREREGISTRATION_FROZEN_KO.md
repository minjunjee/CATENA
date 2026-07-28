# E04 prospective functional-mediation protocol

## 상태와 lineage

- 이 문서가 동결되기 전 E04 main/dry artifact는 `0`개였다.
- 원본 E02 `20260726T153504.455509Z`는 실행 `PASS`, confirmatory
  `INCONCLUSIVE`, original H2 claim `false`로 유지한다.
- E02b `20260726T180207.055493Z`의 prospective repair는 `SUPPORTED`,
  `main/full=true`다.
- E04는 E02b가 새 checkpoint를 만든 것으로 취급하지 않는다. Intervention에는
  원본 E02의 hash-pinned strict tied/dual checkpoint 8쌍만 사용한다.
- 변경 source 전체는 E04 실행 전에 새 E00으로 다시 고정한다.

## 연구 질문과 허용 claim

E04는 frozen `CONTROLLED_REFERENCE` OracleCandidate probe에서 H2의
tied-versus-dual asymmetric correction advantage가 operation-matched
erase/write component에 의해 기능적으로 매개되는지 검정한다.

허용되는 최대 표현:

> Frozen controlled-reference OracleCandidate probe에서 fresh asymmetric
> tied-versus-dual correction advantage가 operation-matched erase/write
> component의 dose, transplant, rescue, post-hoc scalarization 결과에 의해
> functionally mediated되었다.

Semantic causal alignment, 자연어 transaction 이해, official GDN2/KDA,
pretrained language model, agent behavior, 일반 architecture causality는
주장하지 않는다.

## 고정 데이터와 추론 단위

- Checkpoint/inference seeds: `11,22,33,44,55,66,77,88`
- Main: seed마다 128 counterfactual quartets
- Dry: 첫 checkpoint seed에서 별도 namespace의 8 quartets
- Main relative seed namespace: `75000..75127`
- Dry relative seed namespace: `92500..92507`
- 각 quartet은 같은 `state`, address, keys/values, old/new value,
  erase/write candidate를 공유하며 operation만 바뀐다.
- Geometry는 기존 E04 config의 `(old norm,new norm,angle)=(1,1,90°)`다.
- 통계적 inference unit은 8개의 frozen checkpoint seed다. Episode row를
  독립 replicate로 취급하지 않는다.
- Seed 내부 uncertainty는 adjacent two-cycle donor pair block을 유지한
  5,000회 paired bootstrap으로 보고하고, seed inference는 exact sign-flip으로
  분리한다.

## Outcome과 물리적 dose

Primary outcome은 affected-read raw MSE다. Dual baseline gate
`(e,w)`에서 물리적 channel `c`를 dose 0으로 만든 damage를

```text
D(o,c) = Y(o, channel c dose=0) - Y(o, dual baseline)
```

로 정의한다. Erase와 write는 모든 operation에서 각각
`0,.25,.5,.75,1`로 평가한다. `SUPERSEDE`의 joint dose curve도 기록하지만
개별 erase/write 필요성 gate를 대신하지 않는다.

Relevant cells:

- `ADD/write`
- `INVALIDATE/erase`
- `SUPERSEDE/erase`
- `SUPERSEDE/write`

Negative-control cells:

- `ADD/erase`
- `INVALIDATE/write`
- `PRESERVE/erase`
- `PRESERVE/write`

Primary per-base interaction:

```text
I = 0.5 * [
  D(ADD,write) - D(ADD,erase)
  + D(INVALIDATE,erase) - D(INVALIDATE,write)
]
```

Relevant cell은 endpoint damage의 practical positivity와 seed별 mean curve의
monotonic fraction을 모두 통과해야 한다. 따라서 flat curve는 통과할 수 없다.

## Donor, transplant, rescue

- Main/dry base count는 짝수여야 한다.
- Donor는 output이나 error를 보지 않는 adjacent two-cycle
  `0<->1, 2<->3, ...`로 고정한다.
- Same-operation과 cross-operation arm은 동일한 independent donor quartet을
  공유하며 donor operation만 다르다.
- Confirmatory transplant/rescue operation은 norm-degenerate
  `PRESERVE/SUPERSEDE`를 제외한 `ADD/INVALIDATE`다.
- Cross operation은 `ADD<->INVALIDATE`다.
- Donor vector는 clipping 없이 scalar rescale한다. Applied gate가 `[0,1]^2`
  안에 있고 recipient와 L2 norm 차이가 `<=1e-6`일 때만 design-valid다.
- Transplant는 `Y_cross - Y_same`의 practical positivity와
  `Y_same - Y_baseline`의 equivalence를 모두 요구한다.
- Rescue는 relevant channel을 zero한 뒤 donor의 relevant component만 복원한다.
  `Y_cross_rescue - Y_same_rescue` positivity,
  `Y_same_rescue - Y_baseline` equivalence, raw same-donor recovery positivity를
  각각 요구한다.
- Oracle rescue는 baseline equivalence를 요구하는 assay positive control이다.
- Recipient 원 gate 복원은 exact sanity check이며 confirmatory rescue로 세지 않는다.
- Episode별 `max(same,oracle)` 또는 outcome-based donor selection은 금지한다.
- Recovery ratio가 필요할 때 episode ratio를 평균하지 않고 seed mean error의
  ratio를 사용하며 denominator headroom은 `>0.001`이어야 한다.

## Architecture-gap mediation

Asymmetric `ADD/INVALIDATE`에서 새 E04 holdout의 seed별

```text
TE = mean(Y_tied - Y_dual)
ME = mean(Y_scalarized_dual - Y_dual)
Q  = ME / TE
```

를 계산한다. `scalarized_dual`은 frozen dual output을 사후에
`beta=(e+w)/2`, `(beta,beta)`로 투영한 조건이다.

- `TE > 0.001`
- `ME > 0.001`
- 모든 seed에서 `TE > 0.001`
- `Q >= 0.50`
- `PRESERVE/SUPERSEDE` scalarization damage는 `±0.0005` equivalence
- same-operation transplant/rescue의 retention change는 `0.0005`
  non-inferiority

를 모두 요구한다.

## 사전 threshold와 inference

| 항목 | 고정값 |
|---|---:|
| Alpha | 0.05 |
| Positive-effect SESOI `delta` | raw affected MSE 0.001 |
| Equivalence / NI margin `m` | 0.0005 |
| Rescue denominator headroom | 0.001 |
| Dose monotonic fraction | 0.75 |
| Scalarization gap fraction `Q` | 0.50 |
| Bootstrap | paired donor-block 5,000회, 95% CI |

Unit-norm value와 `d=32`에서 required channel 하나를 완전히 누락할 때의
canonical affected-read MSE는 `1/32=0.03125`다. `delta=0.001`은 그
약 3.2%이며, 기존 E02 raw equivalence margin `0.0005`의 두 배다.
E04 outcome에서 threshold를 선택하지 않았다.

Positive gate는 bootstrap lower CI가 threshold보다 크고
`seed_effect-threshold`의 one-sided exact sign-flip `p<=0.05`여야 한다.
Equivalence gate는 95% CI 전체가 `[-m,m]` 안에 있고 seed-level exact
TOST 두 검정이 모두 `p<=0.05`여야 한다. Non-inferiority는 CI upper가
`m` 이하이고 `seed_effect-m`의 `less` sign-flip이 `p<=0.05`여야 한다.

전체 H4는 모든 등록 gate의 intersection-union conjunction이다. 이 하나의
전체 claim에는 multiplicity 보정을 추가하지 않는다. 개별 operation/component를
독립 confirmatory claim으로 분리할 경우 Holm 보정이 필요하다.

## 상태 분리

```text
execution_status
functional_specificity_status
architecture_gap_mediation_status
full_h4_claim_open
```

- Dry run은 수치와 관계없이 `NOT_EVALUATED_DRY_RUN`.
- Main shape/dependency/design이 불완전하면 `INCONCLUSIVE`.
- 완전한 main에서 모든 conjunction gate가 통과하면 `SUPPORTED`.
- 완전한 main에서 평가 가능한 gate가 하나라도 실패하면 `NOT_SUPPORTED`.
- Evidence tier는 항상 `CONTROLLED_REFERENCE`,
  `scientific_evidence=false`다.
