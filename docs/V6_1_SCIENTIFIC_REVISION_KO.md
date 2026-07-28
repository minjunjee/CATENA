# v6.1 scientific revision

이번 revision은 최신 검토의 세 핵심 지적을 중심으로 v6.0의 confirmatory logic을 보강한다.

## 1. H1: span이 아니라 constrained behavioral reachability

- `R_span`: unconstrained local Jacobian span의 구조적 한계
- `R_feas`: 실제 post-sigmoid gate box `[0,1]^p`에서의 bounded regret
- `R_beh`: affected correction과 unaffected retention을 각각 1/2로 가중한 behavioral regret
- raw learned error와 같은 MSE 단위로 비교
- operation fixed effects를 포함한 train-geometry fit, unseen-geometry test
- OOS R²는 operation-only baseline 대비 conditional R²로 계산
- exact sign-flip은 unseen test geometry의 operation-adjusted seed slope에 적용
- calibration slope는 사전 margin이 없어 descriptive primary이며 claim gate는 아님
- OracleCandidate가 confirmatory, RecurrentRead gap은 candidate-recovery/content interference

## 2. H2: positive DID만으로는 통과하지 않음

H2는 다음을 모두 만족해야 한다.

1. ADD/INVALIDATE에서 tied-to-oracle gap의 사전 고정 SESOI 이상을 dual이 닫음
2. PRESERVE raw effect가 equivalence interval 안에 있음
3. SUPERSEDE relative effect가 equivalence interval 안에 있음
4. asymmetric-minus-symmetric DID가 양수
5. unaffected retention이 one-sided non-inferior
6. equal-budget validation tuning이 strict same-recipe 방향을 뒤집지 않음

Tied와 dual은 동일 two-output head를 사용한다. Tied 조건만 두 logit의 평균을 사용해 diagonal subspace로 projection한다.

## 3. H3: axis-vs-rotation이 아니라 joint diagonalizability

세 demand family를 사용한다.

- axis-aligned commuting
- common-rotated commuting
- noncommuting

Fixed diagonal, learned shared-basis diagonal, transaction-conditioned low-rank oracle, full-matrix oracle를 비교한다. Low-rank 결과는 richer-control upper bound이며 parameter-matched learned model로 해석하지 않는다.

## 4. H4: counterfactual functional mediation

- 같은 `S,A,B`에서 operation만 바꾼 quartet
- E02의 8 paired-seed checkpoint 전체
- relevant/irrelevant dose intervention
- independent same-operation donor와 same-base cross-operation donor의 norm-matched transplant
- relevant component만 복원하는 nontrivial rescue
- post-hoc scalarization mediation fraction

Primary는 operation x intervention relevance interaction이다.

## 5. H5-lite와 workshop scope

- operation label, oracle `(e,w)`, exact mask를 입력하지 않음
- held-out SUPERSEDE in seen domain/template가 primary
- paraphrase, domain, combined stress는 분리
- oracle address + hidden old value 조건만 primary
- 300-item stratified two-reviewer audit 필요

REALM critical path는 H1-H4와 작은 H5 anchor다. H6와 RQ-T는 workshop claim ceiling을 올리지 않는다.
