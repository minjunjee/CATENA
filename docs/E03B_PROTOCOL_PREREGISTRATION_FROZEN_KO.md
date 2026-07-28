# E03b graded JD calibration prospective protocol

동결 시각: `2026-07-26T17:58:12.858533Z`

상태: **FROZEN BEFORE ANY E03b DRY-RUN, MAIN-RUN, OR EMPIRICAL PROBE**

E03b는 기존 E03을 재판정하지 않는다. 원 E03의 categorical claim은
`SUPPORTED`, quantitative calibration은 `FAILED /
PREDICTOR_RANGE_RESTRICTION`, full H3 claim은 `false`로 유지한다.

## 1. 기존 E03 불변 계약

```text
source_run: 20260726T161535.271015Z
execution_status: PASS
categorical_geometry_status: SUPPORTED
quantitative_calibration_status: FAILED
quantitative_diagnosis: PREDICTOR_RANGE_RESTRICTION
full_h3_claim_open: false
```

| Source | SHA-256 |
|---|---|
| Original run manifest | `be56f91b5fc3a5992a02dc2b13b70f82b0b6bd6c5a7e1231f0e0ddee6987f327` |
| Original report | `ee0114f45d5facbc3ccdd0e3a0235531e1de078f29fd7a949420df0899fa98c0` |
| Operator-family metrics | `54efda16cc92da17e0a40472de0075b7fec252c3451fde682304f7f63fa29dc4` |
| Control frontier | `59c1aba1bffa039b85a61aa51714951924869ad8164787361e3f982927c1f035` |
| Additive E03 claim registry | `fb53ae74d4702aefad4a5503129552cfe34a0f75079bc0bbb9fd9b4a2deb26f6` |
| Preserved E03 runner | `bf45a0091bd989ca0efcf67e48547a23d5ab3c656885f75d5b66d84e2d5fbaf8` |
| Preserved JD core | `da3f01721b28d36d50464450b2dd790d0cac43e37d22cefd06d0fc4ff0f02e2e` |

## 2. E03b estimand

\[
\widehat Q_{\mathrm{train}}
\approx
\operatorname*{argmin}_{Q^\top Q=I}
\frac{1}{24d^2}
\sum_{\tau\in\mathrm{train}}
\left\|
\operatorname{offdiag}(Q^\top P_\tau Q)
\right\|_F^2 ,
\]

\[
\widehat R_{\mathrm{JD,OOS}}
=
\frac{1}{8d^2}
\sum_{\tau\in\mathrm{heldout}}
\left\|
\operatorname{offdiag}
(\widehat Q_{\mathrm{train}}^\top P_\tau\widehat Q_{\mathrm{train}})
\right\|_F^2 .
\]

명칭은 **estimated train-fit out-of-sample JD regret**다. Numerical
multi-restart fit이며 global minimum certificate가 아니다. Basis 선택에는
24개 train operator만 사용하고, 8개 held-out operator와 application error는
restart, step, initializer 또는 checkpoint 선택에 사용하지 않는다.
다만 held-out analytic regret는 bin stratification에 사용되고, 같은 held-out
operator에서 empirical application error를 측정한다. 따라서 이는 untouched
independent-test calibration이 아니라 **predictor-stratified conditional
Monte Carlo calibration over six locked bins**이다.

## 3. Generator와 graded support

\[
Q_\tau(\alpha)=Q\exp(\alpha A_\tau),\qquad
P_\tau(\alpha)=Q_\tau(\alpha)D_\tau Q_\tau(\alpha)^\top .
\]

Rank-8 mask는 32개 coordinate 전체에서 uniform하게 중복 없이 생성한다.
각 family는 24 train / 8 held-out operator를 가지며 float64 CPU에서 평가한다.

Excluded analytic-only pilot은 empirical probe를 한 번도 호출하지 않았다.

| Pilot item | Frozen value |
|---|---|
| Pilot document | `docs/E03B_ANALYTIC_PILOT_KO.md` |
| Pilot SHA-256 | `21d60d79738e5bd05034312ec630a694d6485f00be1bc824344088eebc9fe94c` |
| Main alpha anchors | `0.06, 0.12, 0.18, 0.26, 0.40, 0.55` |
| Main seed namespace | starts at `500001`; not inspected in pilot |
| Dry seed namespace | starts at `950001`; not inspected in pilot |
| Excluded development dry seeds | `900001`–`900024` |
| Candidate traversal | alpha-major, then replicate |
| Main / dry replicates per alpha | 16 / 4 |
| Finite hard cap | 432 main / 108 dry candidates |

각 bin의 첫 valid candidate를 analytic regret만 보고 deterministic하게
선택한다. Application result를 보고 family를 선택할 수 없다.

## 4. JD optimization adequacy

| Item | Frozen value |
|---|---:|
| Optimizer | QR-parameterized Adam |
| Main steps / restarts | 1,000 / 4 |
| Dry steps / restarts | 250 / 2 |
| Learning rate | 0.03 |
| CPU threads | 1 |
| Maximum orthogonality error | `1e-10` |
| Minimum competitive restarts | 2 |
| Maximum restart-gap uncertainty | `1e-5` |
| Pilot saturation OOS uncertainty floor | `4.61e-6` |
| Minimum boundary clearance | `max(1e-5, 10 × effective uncertainty)` |

Identity와 fixed weighted-spectral solution은 train-only comparator다. Primary
JD fit이 두 comparator 중 더 좋은 objective를 허용오차 `1e-12` 안에서
달성해야 한다. Spectral fallback은 금지한다. Pilot의 2,000-step×8-restart
비교에서 관찰된 최대 held-out drift `4.61e-6`를 uncertainty floor로 직접
사용하므로 모든 selected candidate는 bin boundary에서 최소 `4.61e-5`
떨어져야 한다.

여기서 restart gap, saturation drift, effective uncertainty와 boundary
clearance는 사전 고정한 **diagnostic selection rule**이다. 최적화 오차의
수학적으로 인증된 상한 또는 robust-optimization certificate가 아니다.

## 5. Design-validity gate

| Bin | Registered regret interval |
|---|---|
| 01 | `[1e-5, 2.5e-4)` |
| 02 | `[2.5e-4, 7.5e-4)` |
| 03 | `[7.5e-4, 1.5e-3)` |
| 04 | `[1.5e-3, 3.0e-3)` |
| 05 | `[3.0e-3, 4.5e-3)` |
| 06 | `[4.5e-3, 6.5e-3]` |

Main은 각 bin 8개, 총 48개 unique family와 predictor range `≥0.004`를
요구한다. Exact-zero family는 calibration에서 제외한다. Design 또는
optimizer support가 부족하면 empirical probe를 호출하지 않고
`NOT_EVALUATED / REGISTERED_DESIGN_OR_OPTIMIZATION_SUPPORT_INCOMPLETE`로
종료한다.

## 6. Pre-probe lock

Application probe 전에 다음을 JSON/JSONL로 물리적으로 기록하고 hash한다.

1. Complete candidate registry와 analytic audit
2. Selected 48 family identity와 bin
3. Train/held-out operator tensor hashes
4. Fixed JD basis 값과 hash
5. Train 및 held-out analytic regret
6. Probe seed와 probe count

Empirical phase 직전과 직후, 그리고 finalize 직전에 disk lock과 mutable
runtime object를 다시 대조한다. Config도 initialize 직후와 finalize 직전에
resolved path, canonical payload, exact file bytes를 재검증한다.
Gaussian probe tensor 자체를 사전 저장하지는 않으며, 고정 생성법과
`probe_seed`, `probe_count`를 lock해 재현한다.

## 7. Calibration gate와 해석

Regression 단위는 8개 held-out operator를 평균한 **48개 independent family
mean**이다. 384개 operator row를 독립 표본으로 사용하지 않는다. Equal-bin,
unweighted OLS를 사용하며 원 E03 gate를 완화하지 않는다.

\[
R^2\ge0.99,\qquad
0.95\le\mathrm{slope}\le1.05,\qquad
|\mathrm{intercept}|\le10^{-4}.
\]

평가된 회귀가 이 gate를 통과하면 E03b quantitative calibration은
`SUPPORTED`, 실패하면 `FAILED`다. Positive 결과는 registered synthetic
projector와 isotropic probe에서 analytic normalization이 application MSE를
calibration한다는 제한된 주장만 허용한다. Isotropic probe에서는 이 관계가
대수적 기대값이므로 semantic 또는 natural-state generalization으로 확대하지
않는다.

## 8. Evidence와 config lock

```text
evidence_tier: CONTROLLED_REFERENCE
scientific_evidence: false
official_backend_claim_eligible: false
language_model_claim_eligible: false
architecture_transfer_claim_eligible: false
```

| Config digest | SHA-256 |
|---|---|
| Canonical payload | `46315fbdc9ad01646206ec927f24e622fbcfedd1c8acdb22f70e18030566f8ec` |
| File bytes | `7e258f75aa406d058f748779ff5aeb23d0b93f19c9c5e1eb388df7578bdd7df9` |
| Candidate registry payload | `92be93d2c71a90a688859e783ce26438414b0d8852b7f672df719ac782826671` |

동결 시점에 E03b run directory는 0개다. 이 문서의 hash를 별도 lock 문서와
runner에 고정한 후, 전체 source tree는 새 E00 run으로 다시 fingerprint한다.
