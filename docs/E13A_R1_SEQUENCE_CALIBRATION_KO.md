# E13a-R1 — Prospective paired sequence calibration repair

## 원본 판정

원본 E13a artifact와 `go_for_e13b` 값은 수정하지 않는다. 원본은 tied와
dual에 서로 다른 initialization·training-data·evaluation seed를 사용했고,
training wall time에 첫 CUDA warm-up이 섞였으며, exact-match가 affected
entity가 아니라 전체 entity에 대해 계산됐다. 따라서 원본 E13a는
`CALIBRATION_PILOT_ONLY`이며 E13b dependency로 사용하지 않는다.

## R1 고정 수리

- tied/dual은 같은 parameter initialization hash를 가져야 한다.
- training batch generator와 evaluation generator seed를 완전히 공유한다.
- 두 조건의 parameter count가 같아야 한다.
- floor는 `affected_entity_exact_match`와 affected MSE를 함께 사용한다.
- retention은 실제 unaffected entity count가 양수인지 먼저 확인한 뒤
  margin을 적용한다.
- throughput은 training/data generation과 분리된 동일 batch forward만
  측정한다. 두 모델을 모두 warm-up한 뒤 반복 순서를 교대로 바꾸고 median
  latency를 기록한다.
- 실행 가능성은 작은 calibration model의 throughput으로 추정하지 않는다.
  E13b와 동일한 hidden size, vocabulary, `updates+gap`, batch size에서
  짧은 warm-up·training-step probe를 별도로 수행한다. 이 측정으로 30,000
  step 단일 run, 병렬 wave, 3개 순차 wave의 wall-clock을 투영한다.
- checkpoint, config file, resolved config의 SHA-256을 report와 raw row에
  고정한다.

## Gate

기존 threshold의 의미를 보존한다.

| Gate | 기준 |
|---|---:|
| Paired initialization/data/eval contract | 모두 일치 |
| Dual affected-entity exact match | `>=0.95` |
| Dual affected MSE | `<=0.001` |
| Tied−dual affected MSE gain | `>=0.001` |
| Tied/dual unaffected retention MSE | 각각 `<=0.001` |
| Forward-only throughput | 각각 `>=100 examples/s` |
| E13b-scale projected single run | `<=12 hours` |
| E13b-scale projected one wave | `<=12 hours` |
| Three sequential waves | `<=36 hours` |

모두 통과한 새 R1 run만 E13b를 열 수 있다. Dry-run과 원본 E13a pilot은
E13b를 열 수 없다.
