# E11 / E11b — Representation-Control Co-adaptation 결과

## 판정

| 실험 | Execution | Claim |
|---|---|---|
| E11 original (`20260727T180703.763554Z`) | PASS | `NOT_OPENED_SCALE_RESTRICTION` |
| E11b prospective repair (`20260727T183004.928280Z`) | PASS | **SUPPORTED** |

원본 E11은 8개 seed의 모든 directional contrast가 같은 방향이었지만,
등록된 raw-MSE SESOI `0.001`이 held-out target operator의 평균 전체
에너지 `0.000595`보다 컸다. 원본 gate와 artifact는 그대로 유지하고,
원본 row를 재사용하지 않은 fresh 8-seed E11b에서 scale-invariant gate를
사전 고정했다.

## 주요 결과

| E11b metric | 결과 | Gate |
|---|---:|---:|
| Axis fixed/shared 최대 차이 ÷ target energy | 0.0000437 | ≤ 0.01 |
| Common rotation recovered headroom | 0.997103 | ≥ 0.95 |
| Common shared residual ÷ target energy | 0.002655 | ≤ 0.01 |
| Noncommuting shared-vs-common gap ÷ target energy | 0.179384 | ≥ 0.10 |
| Noncommuting shared residual ÷ target energy | 0.182039 | ≥ 0.10 |
| Low-rank recovery of shared residual | 0.988539 | ≥ 0.90 |
| Low-rank residual ÷ target energy | 0.002087 | ≤ 0.01 |
| Directional exact sign-flip | 3개 모두 p=0.00390625 | ≤ 0.05 |

모든 10개 gate가 통과했다. Common-rotated commuting family는 learned
shared basis가 fixed-basis penalty의 99.7%를 회복했다. 동일한 shared-basis
class는 transaction-dependent noncommuting family에서 target energy의
18.2% residual을 남겼고, learned rank-8 controller가 그 residual의 98.9%를
제거했다.

## 개발 중 확인된 사항

- 원본 E11의 실패는 방향성 부재가 아니라 raw metric scale과 SESOI support의
  불일치였다. 원본을 사후 성공으로 바꾸지 않았다.
- E11b는 global operator scale에 불변인 target-energy/headroom ratio를
  사용하고 fresh seed만 평가했다.
- Low-rank controller는 diagonal controller와 parameter matched가 아니다.
  결과는 richer learned control의 recovery evidence이지 parameter-efficiency
  비교가 아니다.
- E11b는 72개 checkpoint를 저장했다. 실행 시점의 source/config는
  preregistration lock으로 고정했으며, 공통 runner의 확장 provenance
  계약은 이 run 직후 보강됐다.

## 짧은 연구 해석

Shared representation learning은 하나의 공통 회전은 흡수하지만,
transaction마다 basis가 달라지는 noncommuting demand family를 하나의
shared diagonal basis로 환원하지 못했다. 이 residual은 learned low-rank
control로 거의 제거됐다.

## Artifact

- Original E11:
  `/data/minjun_dev/CATENA/artifacts/e11_representation_control_coadaptation/20260727T180703.763554Z`
- Prospective E11b:
  `/data/minjun_dev/CATENA/artifacts/e11b_scale_normalized_coadaptation/20260727T183004.928280Z`
- Protocol lock:
  `docs/E11B_SCALE_NORMALIZED_COADAPTATION_LOCK.json`
- Freeze registry:
  `/data/minjun_dev/CATENA/artifacts/E11_POSTCORE_ARTIFACT_FREEZE_V1.json`
