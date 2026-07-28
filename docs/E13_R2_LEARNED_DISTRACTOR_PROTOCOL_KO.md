# E13-R2 — Learned Distractor-Path Prospective Repair

## 원본 보존과 판정

E13a-R1 run `20260727T183609.755945Z`와 `GO`는 수정하지 않는다. 해당
결과는 paired floor와 resource calibration에는 유효하지만, 원본
sequence implementation의 hard-masked distractor path에만 적용된다.
Repaired E13b-R1의 dependency로 사용하지 않는다.

E13b main은 아직 한 번도 실행하지 않았다. 아래 repair는 E13b-R1
training/evaluation 전에 고정한다.

## 발견된 식별 결함

원본 generator는 verified update를 모두 먼저 배치하고 distractor를
마지막에 붙였다. Model은 distractor의 `verified=0`을 gate에 직접 곱한 뒤
`update_mask=False`인 state assignment도 다시 차단했다. 따라서 같은
update prefix에 대해 full sequence와 distractor를 제거한 sequence의
최종 state가 tied/dual 모두 bitwise-identical했다.

또한 gap마다 다른 evaluation seed를 사용했고 tensor RNG 소비량도
gap에 따라 달라졌다. 원본 E13c의 gap별 차이는 동일 transaction에 대한
gap effect로 식별될 수 없었다.

## R2 고정 수리

| 항목 | R2 계약 |
|---|---|
| Namespace | `e13a_r2_*`, `e13b_r1_*`, `e13c_r1_*` |
| Distractor layout | 총 `gap_events`개를 첫 verified update 뒤, 다음 verified update 전에 한 block으로 배치 |
| `updates=1` | 유일한 update 뒤에 block 배치 |
| Verified bit | encoder input으로만 제공 |
| `update_mask` | target/audit metadata로만 보존; model forward 접근 금지 |
| Base transaction RNG | gap 길이와 독립 |
| Gap comparison | 같은 initial state, verified update stream과 target 사용 |
| Distractor RNG | base transaction과 분리된 deterministic stream |

GRU 또는 recurrent event hidden은 이번 repair에 추가하지 않는다. 질문은
structured finite state가 실제로 학습한 no-op control을 통해 긴
distractor block을 견디는지이며, recurrent representation 자체는 별도
연구 branch다.

## E13a-R2 GO gate

R1의 수치 gate를 그대로 유지한다.

| Gate | 기준 |
|---|---:|
| Paired initialization/data/eval contract | 모두 일치 |
| Dual affected exact match | `>=0.95` |
| Dual affected MSE | `<=0.001` |
| Tied−dual affected gain | `>=0.001` |
| Tied/dual retention MSE | 각각 `<=0.001` |
| Forward throughput | 각각 `>=100 examples/s` |
| E13b-R1 projected single run | `<=12 hours` |
| E13b-R1 projected wave | `<=12 hours` |
| Three waves total | `<=36 hours` |
| Unmasked-path counterfactual delta | `>1e-8` |
| Interleaving/base-pairing/source contract | 모두 PASS |

E13a-R2만 repaired E13b-R1을 열 수 있다.

## E13b-R1/E13c-R1 primary gate

기존 전체-grid mean contrast를 보존하면서 long-gap claim을 별도로
식별한다. Statistical unit은 다섯 paired training seed다.

| Gate | 기준 |
|---|---:|
| 전체 grid mean tied−dual gain | `>=0.001` |
| 전체 grid exact sign-flip | one-sided `p<=0.05` |
| 전체 grid positive seed fraction | `1.0` |
| Dual−tied mean retention degradation | `<=0.0005` |
| Stress cell | `updates=8`, `gap=2048` |
| Stress tied−dual affected gain | `>=0.001`, one-sided exact sign-flip `p<=0.05`, 5/5 positive |
| Dual stress absolute retention MSE | `<=0.001` |
| Dual affected MSE: gap 2048−gap 0 | 모든 seed의 최대 degradation `<=0.0005` |
| Active-path assay | max-gap distractor의 verified bit만 0→1로 바꿀 때 dual retention harm의 seed 최솟값 `>=0.001` |
| Base transaction digest | 같은 seed×variant×updates의 모든 gap에서 동일 |

Active-path assay는 target이나 address를 바꾸지 않는다. Distractor의
semantic verified field만 진짜 event처럼 보이게 하여, 정상 조건의
long-gap 보존이 hard mask가 아니라 학습된 semantic no-op 경로에
의존하는지 확인한다.

## Claim boundary

성공 시 허용되는 주장은 다음으로 제한한다.

> Shared structured event encoder 뒤에서 independent erase/write control의
> repeated-update 이점이 learned distractor rejection과 2,048-event gap
> stress에서도 유지된다.

Natural language, learned addressing, recurrent LM, agent planning 또는
official backend transfer는 열지 않는다.
