# E18 Sequence-Level Architecture–Demand Lattice Protocol

## 목적과 상태

E18은 E12의 architecture–demand lattice를 E13-R2식 model-visible
distractor sequence로 옮기는 독립 prospective 실험이다. E12/E13 source,
artifact, metric과 판정은 수정하지 않는다.

```text
E18a: one variant × one seed per MAIN run
E18b: 25 completed runs의 five-seed paired aggregate
current_status: PROTOCOL_LOCKED_BEFORE_EVALUATION
main_gpu_execution: NOT_STARTED
```

## Controller lattice와 demand

모든 controller는 동일한 entity encoder, state size와
`2 + 2 × value_dim` maximal head를 등록한다. Variant는 forward projection만
달리한다.

| Adjacent contrast | 추가 자유도 | 대응 demand |
|---|---|---|
| `tied_scalar → dual_scalar` | erase/write magnitude 분리 | `magnitude_factorization` |
| `dual_scalar → diagonal_value` | value-channel gate | `value_granularity` |
| `diagonal_value → separate_address` | erase/write address 분리 | `address_decoupling` |
| `separate_address → state_aware` | current-state read | `state_conditioning` |

각 seed에서 모든 variant는 동일 initialization seed, maximal parameter
surface, round-robin family order, batch seed, AdamW optimizer와 학습 step을
사용한다. Oracle address와 candidate는 제공된다.

## Sequence와 distractor 계약

| 항목 | 고정값 |
|---|---|
| Train | updates `4`, distractor gap `128` |
| Training budget | `3,000` steps, batch `128` |
| Test updates | `1, 4, 8` |
| Test gaps | `0, 128, 512, 2048` |
| Seeds | `101, 211, 307, 401, 503` |
| Variants | 5 |
| Demands | 4 |
| Main source rows | run당 48, 전체 1,200 |

총 `gap_events` distractor는 첫 verified update 뒤에 하나의 block으로
배치한다. Verified bit는 model semantic input의 마지막 field이며
`update_mask`는 target/audit metadata로만 존재한다. 동일
seed×demand×updates의 initial state, verified event, candidate와 target은
gap 및 controller variant에 걸쳐 동일해야 한다.

Stress cell(`updates=8`, `gap=2048`)에서는 distractor의 verified bit만
`0→1`로 바꾼다. Address, candidate, target과 base digest는 바꾸지 않는다.
또한 random initialization에서 full-gap과 no-gap output의 최대 차이가
`1e-8`보다 커야 한다. 두 assay는 distractor가 hard mask가 아니라
model-visible active path에 있음을 확인한다.

## E18b confirmatory gate

Statistical unit은 5개 paired training seed다. 각 adjacent pair에 아래
조건을 모두 요구한다.

| Gate | 사전 기준 |
|---|---:|
| 대응 demand의 전체 test-grid mean `baseline − treatment` affected gain | `>= 0.001` |
| 모든 더 단순한 demand에서 최대 treatment degradation | `<= 0.0005` |
| 모든 demand/test cell에서 최대 retention degradation | `<= 0.0005` |
| 대응 demand stress gain 방향 | 5/5 positive |
| Stress one-sided exact sign-flip | `p <= 0.05` |

전체 conjunction에는 다음 design-validity gate도 포함한다.

| Gate | 사전 기준 |
|---|---:|
| Stress active-path retention harm 최소값 | `>= 0.001` |
| Full 25-run / 1,200-row grid | exact match |
| Variant 간 initialization, parameter, optimizer, evaluation seed와 base digest | paired |
| Report, manifest, metric, checkpoint와 protocol-lock hash | 모두 검증 |

Metric, SESOI, margin, stress cell, seed와 claim wording은 main evaluation
뒤에 변경하지 않는다. CPU dry-run은 schema와 실행 경로만 확인하며 claim을
열 수 없다.

## Main 전 development budget calibration

등록 seed와 겹치지 않는 development seed `99001`에서 다섯 variant를
각각 3,000 step 학습하고 train geometry cell만 확인했다. 이 assay에는
main seed, main artifact 또는 stress metric을 사용하지 않았다. 네 대응
adjacent gain은 `0.00661`–`0.01648`로 SESOI의 6.6배 이상이었고, 최대
더-단순-demand degradation은 `0.000158`, 최대 retention degradation은
`1.33e-6`이었다. 처리량은 1,030–1,163 examples/s였다.

따라서 초기 구현 기본값 30,000 step은 필요 이상으로 큰 개발 placeholder로
판정하고, main 실행 전 training budget을 3,000 step으로 고정했다.
Controller, data, seed, metric, gate, evaluation grid에는 변화가 없다.
원자료와 결정 규칙은
`E18_SEQUENCE_CONTROL_LATTICE_DEVELOPMENT_CALIBRATION.json`에 보존한다.

## Claim boundary

성공 시 허용되는 주장은 다음으로 제한한다.

> In controlled structured transaction sequences with oracle addresses and
> candidates, each added memory-control freedom provides a selective benefit
> on the registered demand family that requires it, including the 2,048-event
> stress.

Evidence tier는 `CONTROLLED_REFERENCE`, `scientific_evidence=false`다.
Natural language, learned candidate/addressing, recurrent language model,
agent/planning 또는 official-backend transfer는 주장하지 않는다.
