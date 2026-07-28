# E13a-R2 Learned Distractor Calibration 결과

## 판정

```text
execution_status: PASS
e13a_r2_calibration_status: GO_FOR_E13B_R1
e13a_r1_repaired_dependency_eligible: false
sequence_claim_open: false
```

Main run은 `20260727T190642.222102Z`다. E13a-R1은 hard-masked pipeline
calibration으로만 보존하고, repaired sequence main은 R2만 dependency로
사용한다.

## 주요 결과

| 지표 | Tied | Dual | Gate |
|---|---:|---:|---|
| Affected MSE | 약 0.00407916 | 0.0000001360 | PASS |
| Tied−dual affected gain | \- | 0.0040790231 | `>=0.001` PASS |
| Dual affected exact match | \- | 1.000000 | `>=0.95` PASS |
| Retention | margin 이내 | margin 이내 | PASS |
| Parameter count | 123,394 | 123,394 | Matched |

Learned-distractor structural contract도 전부 통과했다.

| Contract | 결과 |
|---|---|
| Base transaction digest across gaps | Matched |
| Distractor block interleaving | Matched |
| Model-visible input에서 `update_mask` 제외 | PASS |
| Verified field semantic-input-only | PASS |
| Random-init full-vs-no-gap path delta | 1.0 (`>1e-8`) |

E13b-R1 scale의 measured training throughput은 tied `1544.4`, dual
`1520.6` examples/s였다. 30,000-step 예상 최장 run은 약 42.1분,
세 wave 합계는 약 126.3분으로 등록 runtime cap을 통과했다.

## 개발 중 확인한 특기사항

- 원본 distractor는 update 뒤에만 배치되고 두 개의 oracle mask로 완전히
  제거돼 long-gap claim이 식별 불가능했다.
- R2는 첫 update와 다음 update 사이에 총 `gap_events` 길이의 한 block을
  넣고, verified bit를 encoder input으로만 사용한다.
- 같은 seed의 initial state, verified update stream과 target은 gap 길이에
  무관하게 동일하다.
- R2는 calibration gate다. Repeated-update/long-gap claim은 E13b-R1
  5-seed main과 E13c-R1 stress aggregate가 통과해야 열린다.

## Immutable provenance

| 항목 | SHA-256 |
|---|---|
| Report | `9071aee334da3170d79e23b5b2cbf57cffe041bb640a157c400687ddd2565218` |
| Run manifest | `36a4a6a03d6dc14287ee7e6ef58b68dc7ba7e07817803dc50e89f81c76c12d43` |
| Calibration metrics | `b4c7481eacea1ec226203d657528911ce70a1dc91d25921a0cb7c1c2a6bb5400` |
| Scale metrics | `188ea7bc2824506fb07d8f5e05cfd4b11ab588fc8f0f86ea50c651c9d4800278` |
| Tied checkpoint | `4b3987b3872921c920e29e5985b06f98bed10008f12ed656587cab743572491b` |
| Dual checkpoint | `b991fd404d43d66aae2c605ef5ce5cc2227aabf584dc67c429ae51d1406654b9` |
| Protocol lock | `f7fd2b0a893d5bf5099ce4377421f0a744b693159fa2a530546f8eb340101314` |

Evidence tier는 `CONTROLLED_REFERENCE`다.
