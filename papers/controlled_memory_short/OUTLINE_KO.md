# CATENA controlled-memory short paper 개요

> 상태: H1–H4 결과를 바탕으로 한 4쪽 본문 설계안이다. 기존
> `papers/research_plan/`은 수정하지 않는다. 원본 E05a와 사전등록된 단 한
> 번의 E05a-R1 design repair는 모두 execution `PASS`, design `NO_GO`였고
> E05b는 실행되지 않았다. 따라서 semantic anchor는 이 문서의 H1–H4
> evidence package에 포함하지 않는다.

## 1. 논문의 경계

### Working title

**Behavioral Reachability and Operator Geometry Govern Controlled Memory Correction**

짧은 대안:

**When Is a Memory Update Reachable?**

### 한 문장 중심 주장

Oracle candidate와 oracle address가 주어진 controlled finite-memory probe에서,
학습 오차는 bounded behavioral reachability와 정렬되고, 비대칭 erase/write
demand에는 factorized control이 필요하며, shared diagonal control의 충분성은
studied demand family의 joint diagonalizability에 의해 결정되고, 이
erase/write 성분은 등록된 intervention에서 operation-matched 기능을 보였다.

### 반드시 반복할 evidence qualifier

```text
evidence_tier: CONTROLLED_REFERENCE
oracle_candidate: true
oracle_address: true
official_backend_claim_eligible: false
language_model_claim_eligible: false
architecture_transfer_claim_eligible: false
scientific_evidence_artifact_flag: false
```

`scientific_evidence=false`는 artifact의 official-backend claim flag다. 이를
근거로 현재 결과를 mock이라고 부르지는 않되, official GDN2/KDA, pretrained
language model, semantic transaction understanding, agent behavior 또는 runtime
우위로 확장하지 않는다.

## 2. 결과 원장

| 가설 | 보존해야 할 원 판정 | 현재 additive evidence | 본문 핵심 수치 |
|---|---|---|---|
| H1 behavioral reachability | E01b `SUPPORTED` | 별도 repair 없음 | conditional unseen-geometry OOS \(R^2=0.997615\); operation-adjusted slope \(1.008633\); 8-seed exact sign-flip \(p=0.00390625\) |
| H2 erase/write factorization | 원 E02 execution `PASS`, confirmatory `INCONCLUSIVE`; evaluable gate 5/5, original H2 closed | E02b prospective repair `SUPPORTED`; fresh 5/5 + inherited tuning-direction 1/1 = 6/6 | OOD asymmetric normalized gain \(0.999999760\); SUPERSEDE raw difference \(2.7339\times10^{-8}\) within \(\pm5\times10^{-4}\); interaction \(0.016343185\) |
| H3 joint diagonalizability | 원 E03 categorical `SUPPORTED`, quantitative calibration `FAILED`, original full H3 closed | E03b prospective calibration `SUPPORTED`; 원 E03를 재판정하지 않음 | E03 contrasts \(0.00550276\), \(0.00551690\); E03b \(R^2=0.999998169\), slope \(1.000258921\), intercept \(1.8464\times10^{-7}\) |
| H4 functional mediation | E04 `SUPPORTED` | functional 4/4, architecture-gap 6/6 | relevance interaction \(0.0312499999\); transplant \(0.0624816855\); rescue \(0.0312407527\); gap recreation \(0.999999906\) |
| H5 semantic anchor | 원 E05a `PASS / NO_GO`, R1 `PASS / NO_GO` | R1은 fresh-namespace fit을 수리했지만 shared headroom 부재; H5 `TERMINATED_NOT_REFUTED`, E05b 미실행 | 본문 claim/figure에 semantic-effect 수치 입력 금지 |

### H1

Primary condition은 `OracleCandidate / tied`다. Operation identity를 통제한
unseen-geometry error에 대해 box-constrained equal-weight behavioral lower
bound가 conditional OOS \(R^2=0.9976154443\)를 보였다. State-feasible 및
state-span predictor의 대응 값은 각각 \(0.9772914533\),
\(0.9754379351\)였다. 이 차이는 behavioral reachable set이 nominal state
span보다 empirical error를 더 잘 설명한다는 controlled-probe 결과다.

### H2

원 E02에서 ADD와 INVALIDATE의 tied affected-read MSE는 약 \(1/64\), dual은
수치적 0에 도달했다. PRESERVE와 SUPERSEDE는 두 controller 모두 수치적 0에
가까웠다. 이 때문에 원래 SUPERSEDE relative-equivalence 분모인
tied-to-oracle headroom이 사전 최소치보다 작아 gate가 구조적으로
식별되지 않았다. 원 E02를 `SUPPORTED`로 바꾸지 않는다.

E02b는 threshold를 낮춘 재분석이 아니라, 실행 전에 absolute SUPERSEDE
equivalence를 동결하고 새 norm/angle OOD geometry에서 frozen strict
checkpoint를 평가한 prospective repair다. 새 데이터 gate 5/5와 원 E02의
immutable equal-budget tuning-direction fact를 합친 6/6 conjunction이
통과했다. Tuned checkpoint는 새 holdout에서 replay되지 않았다.

### H3

원 E03에서 common-rotated commuting family는 fixed diagonal에서
\(0.00550276\) residual을 남겼지만 learned shared basis는 numerical zero로
복구했다. Noncommuting family는 shared basis에서 \(0.00551690\) residual을
남겼고, transaction-conditioned rank-8 **oracle upper bound**만 numerical
zero로 제거했다. Rank-8을 learned 또는 parameter-matched comparator로
표현하지 않는다.

원 E03 calibration은 nonzero predictor 범위가 약 \(1.27\times10^{-4}\)에
제한되어 사전 gate를 통과하지 못했다. E03b는 application probe 전에
analytic predictor로 6 bins × 8 families를 고정해 range를
\(0.005303882\)로 넓혔다. 그 별도 prospective 실험에서 family-mean JD
regret가 isotropic application error를 \(R^2=0.999998169\)로
calibration했다. 이는 global JD minimum의 인증이 아니다.

### H4

E04는 원 E02 strict checkpoint 8쌍을 재학습 없이 사용한 checkpoint-only
실험이다. ADD/write, INVALIDATE/erase, SUPERSEDE/erase와 write의 zero-dose
damage는 각각 약 \(1/32\), irrelevant channel과 PRESERVE effect는
equivalence margin 안이었다. Relevant 32 seed×cell dose curve가 모두
monotonicity score 1.0을 보였다.

Same-operation transplant는 baseline과 동등했고 cross-operation transplant는
\(0.0624816855\) 더 나빴다. Independent-donor same-operation rescue는 baseline을
복원했고 cross-operation rescue gap은 \(0.0312407527\)였다. Post-hoc
scalarization mediated effect \(0.0156249986\)은 fresh tied-minus-dual total
effect \(0.0156250000\)의 \(0.999999906\)을 재현했다. 이는 frozen
OracleCandidate probe 안의 functional mediation이며 semantic causal alignment가
아니다.

## 3. 4쪽 본문 배치

| 지면 | 내용 | 목표 |
|---|---|---|
| 1쪽 | 제목, 120–150-word abstract, 문제 정의, controlled probe, 세 가지 geometry quantity | “필요한 control 자유도는 demand geometry에서 예측할 수 있는가?”를 설정 |
| 2쪽 | H1 + H3, Figure 1 | behavioral reachability의 error calibration과 JD family boundary를 한 흐름으로 제시 |
| 3쪽 | H2 + H4, Figure 2 | asymmetric factorization gap을 functional dose/transplant/rescue/scalarization으로 연결 |
| 4쪽 | 짧은 synthesis, limitations, reproducibility/protocol disposition, conclusion | E05a가 shortcut을 검출했고 R1은 이를 수리했지만 factorization headroom이 없어서 E05b를 차단했다고 표시 |

### Abstract 골격

1. Persistent memory correction은 목표 update와 controller가 도달 가능한
   update set이 어긋날 때 실패할 수 있다는 문제를 제기한다.
2. Oracle candidate/address controlled finite-memory probe에서 behavioral
   reachability가 operation-adjusted unseen error를
   \(R^2=0.9976\)으로 예측했다고 보고한다.
3. 비대칭 erase/write operation에서는 tied control의 analytic residual이
   남고 factorized control이 이를 제거했으며, shared diagonal control은
   commuting family를 복구하지만 noncommuting family에는 residual을
   남겼고, graded JD regret는 별도 prospective 실험에서 application error를
   \(R^2=0.999998\)로 calibration했다고 쓴다.
4. Frozen-checkpoint intervention이 component-specific dose, transplant,
   rescue와 scalarization gate를 모두 통과했다고 보고한다.
5. 마지막 문장에서 이 결과가 `CONTROLLED_REFERENCE`이며 official
   backend/LM 또는 semantic understanding evidence가 아님을 명시한다.

### 1. Introduction

- Memory controller의 nominal gate 수가 아니라 **behaviorally reachable update
  set**을 설계 단위로 제안한다.
- 질문을 세 단계로 제한한다.
  1. Reachability distance가 학습 오차를 예측하는가?
  2. 어떤 demand가 factorized magnitude 또는 richer basis family를
     요구하는가?
  3. 학습된 erase/write component가 operation-matched 기능을 수행하는가?
- Contribution은 H1–H4 네 결과로 끝낸다. E05는 “a one-shot
  representation repair recovered fresh-namespace fit but left no
  preregistered factorization headroom, so the semantic confirmatory stage was
  not opened” 한 문장만 둔다.

### 2. Controlled probe and geometric predictions

핵심 update를 다음처럼 최소 표기한다.

\[
\Delta S_{e,w}=-eA+wB,\qquad
\Delta S_{\beta}=\beta(-A+B).
\]

Behavioral regret는 목표 행동과 bounded reachable behavior 사이의 거리로,
JD regret는 한 shared basis에서 demand operator family의 off-diagonal
residual로 정의한다. Oracle candidate/address와 operation identity가
H1–H4 probe에 제공된다는 조건을 첫 문단에서 밝힌다.

### 3. Reachability and basis-family sufficiency

- Figure 1A–B: reachable-set distance와 H1 conditional OOS calibration.
- Figure 1C: axis commuting, common-rotated commuting, noncommuting의
  categorical boundary.
- Figure 1D: E03b의 six-bin prospective calibration.
- 원 E03의 failed calibration과 E03b 결과를 같은 문장에서 구분한다.

### 4. Factorization and functional mediation

- Figure 2A: 새 OOD geometry에서 operation별 tied-minus-dual gap.
- Figure 2B: retained component dose에 따른 operation-matched error curve.
- Figure 2C: same/cross transplant와 rescue contrast.
- Figure 2D: fresh architecture gap과 post-hoc scalarization mediated effect.
- 원 E02가 `INCONCLUSIVE`이고 E02b가 별도 prospective repair임을 caption과
  본문에 모두 기록한다.

### 5. Limits and conclusion

전체 흐름 해석은 다음 두 문장으로 제한한다.

> Studied controlled updates에서 error magnitude와 controller sufficiency는
> reachable control geometry로 예측됐고, erase/write intervention은 그
> factorization의 operation-matched 기능을 보였다. Operation label과 oracle
> demand를 제거하려던 원 E05a는 fresh-namespace generalization 조건을
> 만족하지 못했다. 사전등록된 한 번의 R1은 관계 표현으로 두 controller를
> oracle 근처까지 복구했지만 factorized-specific headroom과 SESOI를
> 만족하지 못해 semantic confirmatory stage를 열지 않았으며, 현재
> 결과만으로 official backend, language model 또는 semantic understanding을
> 주장할 수 없다.

## 4. 과장 방지 문구

### 사용할 수 있는 표현

- “within the controlled-reference OracleCandidate probe”
- “the studied synthetic demand families”
- “prospectively repaired on a fresh held-out geometry set”
- “operation-matched functional mediation”
- “rank-8 oracle upper bound”
- “estimated train-fit out-of-sample JD regret”

### 사용하지 않을 표현

- “H2 original experiment was supported”
- “the original E03 calibration passed”
- “rank-8 learned controller”
- “semantic causal gate”
- “natural-language transaction understanding”
- “official GDN2/KDA superiority”
- “pretrained language-model or agent evidence”
- “runtime/latency Pareto superiority”
- “joint diagonalizability is globally certified”

## 5. 실험별 개발 이슈 원장

이 표는 authoring/reproducibility용이다. 본문에는 각 repair의 과학적
disposition만 압축하고, 구현 오류의 세부는 artifact note 또는 supplement로
보낸다.

| 실험 | 개발 또는 protocol 이슈 | 보존·수리 방식 | 논문에서의 처리 |
|---|---|---|---|
| E01b | 초기 run `20260726T150542.747333Z`가 CUDA episode와 CPU geometry feature의 device mismatch로 metric 전 중단 | 불완결 directory 보존; feature 생성 순서 수정 및 GPU regression test 후 새 UTC main | 실패 run은 과학적 결과가 아님; main `20260726T152354.081239Z`만 사용 |
| E02 | SUPERSEDE tied-to-oracle headroom이 사실상 0이라 preregistered relative-equivalence gate가 계산 불가 | 원 artifact와 `INCONCLUSIVE` 판정 고정 | “5/5 evaluable gates passed; original H2 not opened” |
| E02b | 원 E02에 완전한 tuned checkpoint pair가 없어 fresh OOD에서 tuning replay 불가 | 5개 fresh gate와 immutable E02 tuning-direction 1개를 사전 구분 | 6/6 중 1개가 inherited fact임을 명시 |
| E03 | 초기 dry `20260726T161333.380725Z`에서 n=2 descriptive Pearson 처리 누락; main calibration은 predictor range restriction으로 실패 | dry 보존, `n<3`를 unevaluable로 처리; main gate 불변 | categorical support와 quantitative failure를 분리 |
| E03b | Freeze 전 독립 감사에서 spectral aggregate proxy가 `min_Q` estimand가 아님을 확인 | 원 E03 train-only QR/Adam estimator로 교체한 뒤 protocol freeze; spectral은 comparator-only | global JD optimum 인증 금지; locked-bin conditional calibration으로 표현 |
| E04 | 최초 dry `20260727T054548.378221Z`에서 integer seed JSON key가 strict writer를 막음 | failed dry 보존; threshold/gate를 바꾸지 않고 key serialization만 동결 수정; completed dry 후 main | main `20260727T054917.678326Z`만 inference; analytic unit-test 성격 명시 |
| E05a | Leakage/control 9개 조건은 통과했지만 factorized/shared affected oracle excess와 affected parity 3개 조건이 실패; train→fresh-namespace gap 17.0×/24.6× | main `20260727T081532.073522Z`와 `NO_GO` 동결; E05b registry/audit 미생성; 동일 protocol 재시도 금지 | H5를 `NOT TESTED / NOT OPENED`로 기록하고 shortcut-sensitive semantic generalization을 limitation으로만 보고 |
| E05a-R1 | 첫 dry에서 기존 fieldwise donor search가 balanced factorial의 valid derangement를 못 찾아 학습 전 중단; R1 pairing으로 구현 수리. Main에서는 shared도 oracle 근처라 headroom 부재, absolute gain 0.0000118 < SESOI 0.001, 7/8 방향 | incomplete dry 보존; completed dry와 main을 새 UTC run으로 수행; main `PASS / NO_GO`와 artifact hash 동결; audit/E05b-R1 미생성 | H5 `TERMINATED_NOT_REFUTED`; 관계 표현 일반화 수리는 development diagnostic, factorized semantic advantage는 주장하지 않음 |

## 6. Immutable source ledger

Canonical artifact root는
`/data/minjun_dev/CATENA/artifacts`이며 repository의 `artifacts`는 이 위치를
가리키는 symlink다. `latest.json`은 인용하지 않고 아래 exact UTC run만
사용한다.

| Evidence | Exact source | SHA-256 |
|---|---|---|
| H1 report | `artifacts/e01b_constrained_behavioral_reachability/20260726T152354.081239Z/report.json` | `8e1d16ca7763cec1e4e5b13d2b0f163f4015c8058ed7764871a6fbb5fa5ea6d6` |
| H1 manifest | `artifacts/e01b_constrained_behavioral_reachability/20260726T152354.081239Z/run_manifest.json` | `31e6651e187058fe7e26a791bd14c6d0f5a4d613101a107c809514ac53b23847` |
| H1 raw metrics | `artifacts/e01b_constrained_behavioral_reachability/20260726T152354.081239Z/episode_geometry_metrics.jsonl` | `c2a5afff71804da645d884c832b8c6dec6bab1b4eaefe0cf16d067f28d9547ac` |
| Original E02 report | `artifacts/e02_magnitude_factorization/20260726T153504.455509Z/report.json` | `f3df03e231598d6eda11ebf71825ab418cc9a59ac9a96a299caff617291e4211` |
| Original E02 manifest | `artifacts/e02_magnitude_factorization/20260726T153504.455509Z/run_manifest.json` | `1ea4ad867ffbc86bca3f4ee8f3eceb698089f973db250920a0eeb3dc39641d4c` |
| E02b report | `artifacts/e02b_prospective_absolute_supersede/20260726T180207.055493Z/report.json` | `032c1b015851b44555ce666ed1d50908332b13f0ad65608355559c000f1d3a52` |
| E02b manifest | `artifacts/e02b_prospective_absolute_supersede/20260726T180207.055493Z/run_manifest.json` | `c6d06a6e174e3162c6cec9ee03048a3a0f725f71de829fa83811527152f8cfdd` |
| E02b claim registry | `artifacts/E02B_CLAIM_STATUS.json` | `ed065cd52688d63c14fc6d59671aaeddae3841c3dadab7507ffd6938813e54b1` |
| Original E03 report | `artifacts/e03_granularity_orientation/20260726T161535.271015Z/report.json` | `ee0114f45d5facbc3ccdd0e3a0235531e1de078f29fd7a949420df0899fa98c0` |
| Original E03 manifest | `artifacts/e03_granularity_orientation/20260726T161535.271015Z/run_manifest.json` | `be56f91b5fc3a5992a02dc2b13b70f82b0b6bd6c5a7e1231f0e0ddee6987f327` |
| E03b report | `artifacts/e03b_graded_jd_calibration/20260726T180514.626996Z/report.json` | `fc88dd3923bbcfb63b99953a9f839b25cbee956f155ce1dd4c397ea82a71772c` |
| E03b manifest | `artifacts/e03b_graded_jd_calibration/20260726T180514.626996Z/run_manifest.json` | `6174f6d7a73735db6cf6136c0218f46adb51d2312f7931d7f982d9993eb5facb` |
| E03b claim registry | `artifacts/E03B_CLAIM_STATUS.json` | `ac796ad02d0dca72671f00aa28c2aa0ab704253290920d28abe78b3f67e967be` |
| E04 report | `artifacts/e04_functional_mediation/20260727T054917.678326Z/report.json` | `7111e23ab70558a5130dff6937a3264c3ec6e4b5ea342183e52ea28f1bb36444` |
| E04 manifest | `artifacts/e04_functional_mediation/20260727T054917.678326Z/run_manifest.json` | `7ae767b8fb7226588fd770194783308ff89a24b52b6f67389c55dab0e044ff7d` |
| E04 artifact freeze | `artifacts/E04_ARTIFACT_FREEZE_V1.json` | `6d225b673da998cef9131af0b2d49fc699f89af2159f40c302898144c2765b30` |
| E04 claim registry | `artifacts/E04_CLAIM_STATUS.json` | `e6eabc9e20a44deda64569c9e9af204c8825924b4f3b07a03fa9f310e9674abf` |
| E05a report | `artifacts/e05a_semantic_protocol_lock/20260727T081532.073522Z/report.json` | `34bab0288d5bbe82e1debcfa81e51493f4fa280475b86f0d097cb2a4aff8057c` |
| E05a manifest | `artifacts/e05a_semantic_protocol_lock/20260727T081532.073522Z/run_manifest.json` | `c2571fa8c4ec184068dff3bb002dc08be1c503c147348bb19ada1fd1199b5e2b` |
| E05a artifact freeze | `artifacts/E05A_ARTIFACT_FREEZE_V1.json` | `f6e6edebd303fb1b6d48cff9630516a8864dc317386778202da58a2a6c189122` |
| E05a-R1 report | `artifacts/e05a_r1_semantic_design_repair/20260727T142609.591935Z/report.json` | `fdb1a397ccc526f9546b473d63e2ab3351529f184879baaefb3f686b505f6bb3` |
| E05a-R1 manifest | `artifacts/e05a_r1_semantic_design_repair/20260727T142609.591935Z/run_manifest.json` | `e8c75ef56a78c64169aa5c03e46b0267d1805b20b3dc4fc8dc48d588f2f0e2fd` |
| E05a-R1 artifact freeze | `artifacts/E05A_R1_ARTIFACT_FREEZE_V1.json` | `b5cdf6036d25060d2bb05d77cd712769c141bf8a8ced0e6944ad78c95ed12aad` |
| E05a-R1 claim registry | `artifacts/E05A_R1_CLAIM_STATUS.json` | `6847805450e30d08c5b6216865a0ea9c0459b8f7a7c485f3973604b84829b238` |
