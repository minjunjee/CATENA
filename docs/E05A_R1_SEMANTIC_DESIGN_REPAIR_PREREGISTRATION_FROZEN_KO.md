# E05a-R1 Semantic Design Repair — 사전등록 동결본

동결일: 2026-07-27 UTC  
실험 ID: `e05a_r1_semantic_design_repair`  
역할: H5-lite main을 열기 위한 design-validity 재검증이며, 그 자체는 H5 claim evidence가 아니다.

## 1. 원본 E05a의 보존과 수리 범위

원본 E05a run `20260727T081532.073522Z`의 판정은 `PASS / NO_GO`로
영구 보존한다. 원본은 12개 등록 검정 중 9개를 통과했으나 다음 세
조건을 통과하지 못했다.

| 실패 조건 | 원본 추정값 | 등록 기준 |
| --- | ---: | ---: |
| Factorized−oracle affected excess | 0.00599550 | CI 상한 ≤ 0.0005 |
| Shared−oracle affected excess | 0.00721133 | CI 상한 ≤ 0.0005 |
| Shared−factorized affected parity | 0.00121583 | CI가 ±0.0005 안 |

원본 네 seed의 shared−factorized 차이는 모두 factorized에 유리했지만,
원본 protocol의 GO 조건은 sign-flip 검정이 아니었다. 따라서 실패
원인은 seed 수만의 해상도 부족이 아니라 fresh-namespace
representation generalization 실패인 `Case B`로 고정한다.

E05a-R1은 기존 네 seed나 episode를 추가 표본처럼 합치지 않는다.
기존 threshold를 낮추지 않고, 결과를 본 뒤 seed를 추가하지 않으며,
별도의 여덟 seed와 숫자 namespace를 사용하는 단 한 번의 prospective
representation repair이다. R1이 `NO_GO`이면 이번 제출에서 H5를
종료하고 세 번째 semantic repair를 하지 않는다.

## 2. R1 표현

Gate encoder는 다음 6개의 고정 relational feature만 받는다.

\[
x =
\left[
\frac{p_{\mathrm{to}}-t}{32},
\frac{e_{\mathrm{from}}-t}{32},
\frac{e_{\mathrm{to}}-t}{32},
\frac{v_e-v_p}{4},
\mathbf{1}[s_e=s_c],
\mathbf{1}[s_e\ne s_c]
\right].
\]

수치 feature에는 clipping, threshold, sign 변환을 하지 않는다.
Scope feature는 opaque token 자체가 아니라 equality partition만
표현한다. 이 encoder에는 state, state read, address, entity/value
token, domain/template, source/provenance, 절대 날짜·version, seed/hash,
operation 또는 oracle erase/write가 들어가지 않는다.

State read는 gate input과 분리한다. 공개된 oracle address에서 읽은
old state content는 erase candidate
\(A=\operatorname{outer}(a,a^\top S)\)를 계산하는 update 경로에만
사용한다. Incoming value는 write candidate
\(B=\operatorname{outer}(a,x_{\mathrm{new}})\)를 계산하는 경로에만
사용한다.

## 3. 데이터 균형과 namespace

R1 main seed는 다음 여덟 개로 고정한다.

```text
1103, 2207, 3301, 4409, 5501, 6607, 7703, 8807
```

Dry-run seed `9919`는 main 추론에서 제외한다. Main numeric namespace
root는 `6_000_000_000_000`이며 `r1_train=11`,
`r1_validation=12`만 연다. Audit root `6_100_000_000_000`과
E05b-R1 root `7_000_000_000_000`은 이 단계에서 예약만 하고 어떤
row도 생성하지 않는다.

학습·검증 operation은 `PRESERVE`, `ADD`, `INVALIDATE`다. 각
operation×domain×template cell의 크기는 main train 330, validation
154이다. `PRESERVE`와 `INVALIDATE`의 write-false 예는 다음 11개의
가능한 failure stratum을 정확히 같은 수로 포함한다.

```text
time ∈ {active, expired, future}
version ∈ {positive, nonpositive}
scope ∈ {same, different}
```

단, `(active, positive, same)`은 write-true이므로 write-false
stratum에서 제외한다. 날짜·version margin은 각 stratum 안에서
결정론적 별도 stream으로 생성하고, namespace seed의 단순 modulo가
operation shortcut이 되지 않게 한다. Train과 validation의 entity,
private old-value, new-value vocabulary 교집합은 비어 있어야 한다.

## 4. 모델과 학습 예산

Factorized와 shared controller는 parameter tensor 수, dense compute
graph, 초기값, batch schedule을 맞춘다. 유일한 모델 차이는 기존
동결 architecture의 path routing이다. 두 모델 모두 behavioral
affected-read와 unaffected-retention loss만 사용하며 gate label이나
target-state loss를 사용하지 않는다. Main은 2,400 fixed step을
실행하고 최종 step checkpoint만 평가한다.

## 5. Primary estimand와 seed-level 추론

각 seed \(s\)에서 operation별 episode 평균을 먼저 계산한 뒤
`ADD`와 `INVALIDATE`에 같은 가중치를 준다.

\[
G_s =
\frac{1}{2}
\sum_{o\in\{\mathrm{ADD},\mathrm{INVALIDATE}\}}
\left(
E_{\mathrm{shared},s,o}
-
E_{\mathrm{factorized},s,o}
\right).
\]

여덟 \(G_s\)가 독립 통계 단위다. Primary 95% CI는 여덟 seed mean을
같은 가중치로 paired cluster-resample하는 5,000회 percentile
bootstrap이며 episode를 primary inference unit로 재표집하지 않는다.

Primary gate는 다음 네 조건의 conjunction이다.

1. \(\bar G \ge 0.001\)
2. two-sided 95% seed-cluster CI의 하한 \(>0\)
3. \(G_s\)에 대한 unshifted one-sided exact sign-flip \(p\le0.05\)
4. 8/8 seed에서 \(G_s>0\)

SESOI는 point estimate에 적용하고 CI는 0과의 방향 안정성에
적용한다. 이 둘을 사후에 서로 바꾸지 않는다.

## 6. 추가 GO 조건

모든 조건을 동시에 만족해야 `GO`다.

| 구분 | 고정 조건 |
| --- | --- |
| Oracle | affected와 retention seed-cluster CI 상한 ≤ \(10^{-8}\) |
| Factorized asymmetric fit | `ADD/INVALIDATE` oracle excess CI 상한 ≤ 0.0005 |
| Shared headroom | `ADD/INVALIDATE` oracle headroom CI 하한 > 0.001이고 8/8 seed > 0.001 |
| `PRESERVE` | Factorized와 shared 각각 oracle excess CI 상한 ≤ 0.0005 |
| Retention | P/A/I에서 각 모델의 oracle excess CI 상한 ≤ 0.0005 |
| Retention NI | Factorized−shared retention CI 상한 ≤ 0.0005이고 shifted exact NI \(p\le0.05\) |
| Five controls | 각 control의 A/I factorized 대비 degradation CI 하한 > 0.001, shifted exact sign-flip \(p\le0.05\), 8/8 raw direction 양수 |
| Static | forbidden access, namespace/stratum balance, matched parameter/compute 모두 통과 |

Five controls는 shuffled semantics, same-operation/same-base-state/norm-matched
wrong address, transaction-only, state-only, coherent wrong semantics다.
Wrong-address mapping은 outcome을 보지 않으며 erase/write candidate norm
mismatch가 각각 \(10^{-6}\) 이하여야 한다.

Shared headroom이 없으면 primary superiority를 검정할 실질적 여지가
없으므로 `NO_GO / INSUFFICIENT_ORACLE_HEADROOM`으로 기록한다. 실행
오류만 `FAIL`이며, 어떤 `NO_GO`도 H5 반증으로 바꾸지 않는다.

## 7. Human audit와 E05b-R1 경계

R1 run은 audit item이나 E05b row를 자신의 run directory에 추가하지
않는다. `GO` 이후에만 별도 실험
`e05a_r1_semantic_audit_pool_lock`이 독립 300-item audit pool을 만들고
동결한다. 두 사람의 독립 review와 adjudication은 다시 별도 실험
`e05a_r1_semantic_audit_adjudication`으로 저장한다.

Audit label은 의미 보존, operation leakage, address/entity ambiguity,
old-value lexical leakage, gold target consistency로 분리한다. 고정
기준은 각각 meaning ≥0.95, leakage/ambiguity/old-value leakage ≤0.02,
gold consistency ≥0.98, 각 label raw agreement ≥0.80이다. Cohen’s
\(\kappa\)는 보고하되, 양 평가자에서 사건 수가 0이라 정의되지 않는
label은 finite \(\kappa\)를 요구하지 않는다. AI review는 두 human
reviewer를 대체할 수 없다.

원본 E05b의 seen-operation parity 조건은 R1의 positive-gain estimand와
양립하지 않는다. 따라서 원본 E05b config를 수정하거나 재사용하지
않는다. Audit `PASS` 후에만 새 `e05b_r1_registry_lock`이 root
`7_000_000_000_000`에서 실제 registry를 생성하고, 그 다음
`e05b_r1_semantic_anchor`가 실행될 수 있다. Downstream checkpoint
seed는 다음으로 사전 고정한다.

```text
1201, 2309, 3413, 4517, 5623, 6733, 7841, 8951
```

E05b-R1 primary는 seen-domain/template held-out `SUPERSEDE`에서
shared−factorized affected-read MSE이며 SESOI 0.001과 retention
non-inferiority margin 0.0005를 유지한다. 상세 registry와 secondary
split은 R1 `GO` 및 audit `PASS` 전에는 물질화하지 않는다.

## 8. 판정과 허용 해석

```text
execution_status: PASS | FAIL
e05a_r1_design_status: GO | NO_GO
h5_claim_open: false
e05b_r1_execution_allowed: false until human audit PASS
```

R1 `GO`는 semantic design validity가 안정화되어 human audit을 열 수
있다는 뜻일 뿐 H5를 지지하지 않는다. R1 `NO_GO`는 H5 반증이 아니라
`DESIGN_VALIDITY_NOT_ESTABLISHED`이며, 이번 제출에서 H5를 종료한다.
H1–H4 controlled core의 판정은 어느 경우에도 바뀌지 않는다.

모든 결과는 `CONTROLLED_REFERENCE` tier이며
`scientific_evidence=false`다.
