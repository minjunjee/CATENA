# E03b analytic-only JD pilot

기록 시각: `2026-07-26T17:42:36.360555Z`

상태: **DEVELOPMENT PILOT — EMPIRICAL APPLICATION PROBE CALLS = 0**

이 pilot은 E03b의 basis estimator를 spectral proxy에서 원 E03과 동일한
train-only QR/Adam multi-restart joint-diagonalization objective로 교체하기
위한 설계 자료다. 기존 E03 artifact와 main E03b candidate seed는 사용하지
않았다.

## 1. Estimand와 solver

\[
\widehat Q_{\mathrm{train}}
=
\operatorname*{argmin}_{Q^\top Q=I}
\frac{1}{24d^2}
\sum_{\tau\in\mathrm{train}}
\left\|
\operatorname{offdiag}(Q^\top P_\tau Q)
\right\|_F^2
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

이는 **estimated train-fit out-of-sample JD regret**이며 global minimum
certificate가 아니다. Held-out operator와 empirical application error는
restart, step 또는 basis 선택에 사용하지 않는다.

| 항목 | Pilot 값 |
|---|---:|
| Dimension / projector rank | 32 / 8 |
| Train / held-out operators | 24 / 8 |
| Optimizer | QR-parameterized Adam |
| Learning rate | 0.03 |
| Primary budget | 1,000 steps × 4 restarts |
| Saturation comparator | 2,000 steps × 8 restarts |
| Dtype / device | float64 / CPU |
| Torch threads per pilot worker | 1 |
| Empirical probe calls | 0 |

## 2. Alpha support pilot

Paired excluded seeds `300001`–`300008`에서 primary budget만 사용했다.

| α | Held-out regret 최소 | Held-out regret 최대 | 용도 |
|---:|---:|---:|---|
| 0.03 | 0.0000326627 | 0.0000375909 | low fallback |
| 0.06 | 0.000129654 | 0.000149010 | bin 01 anchor |
| 0.09 | 0.000287980 | 0.000330225 | fallback |
| 0.12 | 0.000502777 | 0.000574722 | bin 02 anchor |
| 0.16 | 0.000865585 | 0.000983974 | fallback |
| 0.18 | 0.001074459 | 0.001217338 | bin 03 anchor |
| 0.20 | 0.001298173 | 0.001465346 | fallback |
| 0.24 | 0.001779057 | 0.001991139 | fallback |
| 0.26 | 0.002030628 | 0.002261879 | bin 04 anchor |
| 0.30 | 0.002542470 | 0.002802345 | fallback |
| 0.40 | 0.003771290 | 0.004025367 | bin 05 anchor |
| 0.55 | 0.005320299 | 0.005462128 | bin 06 anchor |

따라서 main candidate stream의 첫 여섯 α는 application error를 보지 않고
`[0.06, 0.12, 0.18, 0.26, 0.40, 0.55]`로 고정한다.

## 3. Optimization saturation

아래 비교에서도 empirical probe는 호출하지 않았다.

| Seed | α | 1,000×4 대비 2,000×8 train 개선 | Held-out regret 변화 |
|---:|---:|---:|---:|
| 300001 | 0.15 | 0 | 0 |
| 300006 | 0.40 | 수치적으로 0 | 0 |
| 300006 | 0.55 | 약 `1.2e-14` | 약 `4.2e-11` |
| 300008 | 0.55 | 약 `1.71e-8` | 약 `4.61e-6` |

관찰된 최대 held-out 변화 `4.61e-6`는 실행 전 고정할 optimizer uncertainty
상한 `1e-5`보다 작다. 따라서 primary budget은 원 E03과 같은
`1,000×4`로 유지한다. Candidate의 bin-boundary clearance는 restart gap만
사용하지 않고 이 관찰값을 uncertainty floor로 포함하여 최소
`10 × 4.61e-6 = 4.61e-5`를 요구한다. 이 비교는 global optimum을 인증하지
않는다.

## 4. 개발용 dry seeds의 처리

`900001`–`900024`는 250-step × 2-restart smoke budget에서 여섯 anchor의
analytic support와 optimizer diagnostics만 확인했다. 이 seed들은 E03b
dry/main candidate registry에서 제외한다. 실제 dry-run은 새로운
`950001` 시작 namespace를 사용한다.

## 5. 불변성

- Original E03 run `20260726T161535.271015Z`는 변경하지 않았다.
- Original E03 quantitative calibration은 계속 `FAILED /
  PREDICTOR_RANGE_RESTRICTION`이다.
- E03b main seed namespace `500001` 이후는 이 pilot에서 생성·평가하지 않았다.
- Main/dry empirical application probe는 한 번도 호출하지 않았다.
- 이 pilot 뒤 config, candidate registry, protocol 문서를 별도로 hash/freeze한
  다음에만 E03b dry/main을 실행한다.
