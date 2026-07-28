# E11b prospective scale-normalized co-adaptation repair

## 원본 보존

원본 E11 run `20260727T180703.763554Z`는 변경하지 않는다. 실행은
`PASS`였으나 세 raw-MSE SESOI `0.001`이 생성 family의 target operator
energy support보다 커 full E11 claim은 열리지 않았다. 원본 row와 seed는
E11b 평가에 재사용하지 않는다.

## 수리 목적

Threshold를 원본 결과에 맞춰 낮추는 대신, 모든 주요 contrast를 각
family의 held-out target energy 또는 해당 controller headroom으로
정규화한다. 이 gate는 operator의 임의적인 global scale에 불변이다.

## Fresh design

- Seeds: `907, 1009, 1103, 1201, 1301, 1409, 1511, 1601`
- Train/test descriptor와 operator family를 fresh seed로 재생성
- Architecture, optimizer, steps, dimension, active rank는 원본 E11과 동일
- Original E11 row/checkpoint 재사용 없음
- Exact one-sided sign-flip의 inferential unit은 8개 training seed

## 고정 gate

| Gate | 기준 |
|---|---:|
| Axis fixed/shared equivalence | target-energy 대비 최대 1% |
| Common-rotation recovery | fixed headroom의 평균 95% 이상 |
| Common shared residual | target-energy 대비 평균 1% 이하 |
| Noncommuting shared-vs-common gap | target-energy 대비 평균 10% 이상 |
| Noncommuting shared residual | target-energy 대비 평균 10% 이상 |
| Low-rank recovery | shared residual의 평균 90% 이상 |
| Low-rank residual | target-energy 대비 평균 1% 이하 |
| 세 directional contrast | 각각 exact sign-flip p ≤ 0.05 |

모든 gate의 conjunction만 E11b를 `SUPPORTED`로 판정한다.

## Claim boundary

성공해도 controlled reference operator family에서의 representation/control
co-adaptation만 주장한다. 원본 E11을 성공으로 재판정하지 않으며,
parameter-matched low-rank 비교, pretrained language model, official backend
transfer는 주장하지 않는다.
