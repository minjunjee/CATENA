# E26 Data Repair R1 결과 요약

## 판정

```text
execution_status: COMPLETED
disposition: ZERO_PROTECTED_TRAIN_FLAGS
scientific_data_readiness_v3: PASS
scientific_main_input_eligible: true
gpu_preflight_started: false
scientific_e26a_started: false
```

Stage-2 원본 `BLOCKED_DATA_SOURCE`와 `FAIL_PENDING_MANUAL_AUDIT`는 변경하지
않았다. 별도 prospective R1에서 frozen detector가 flag한 모든
`general_train` endpoint를 의미 판정 없이 제외하고, 동일한 pinned
FineWeb-Edu revision과 deterministic content-SHA 순서로 용량을 보충했다.

| 항목 | 결과 |
|---|---:|
| Frozen flagged pairs / unique train exclusions | 541 / 541 |
| V1 train에 실제 포함됐던 제외 문서 / token | 56 / 80,318 |
| V1 train-token 제외 비율 | 0.02007922% |
| Deterministic backfill 문서 | 82 |
| Repaired train 문서 / token | 359,074 / 400,000,467 |
| Monotonic repair rounds / 추가 제외 | 1 / 0 |
| Final protected–train near-duplicate flags | **0** |
| Validation/test identity guard | PASS |
| Paired first/resume replay 및 80:20 mix | PASS |

최종 train token SHA는
`efc6f85b56328048e4f462addab34c973046ebf5ff3f3247852f32497b158f9c`,
selection SHA는
`3d08c14e7dc610e47ab8d85e4e6336d0e3be6556b4e8d000a1a2ad0ac623855c`로
prospective diagnostic lock과 정확히 일치했다. 독립 validator가 readiness를
다시 계산한 결과도 PASS였으며 readiness SHA는
`fd22ea54413905122545393ae0832981e2c2f9d6c594a58cdb9f1be337dd427f`다.

## Provenance와 artifact

- Source commit: `2721cc4e2072511d15a4f974b86e63cb4cbe0a42`
- Protocol SHA: `6c6cb0fce46f4f43b564350ce2e799b2d3e2c99cbe5037eb558473b3d62012b0`
- Data root: `/data/minjun_dev/CATENA/e26_data_v2_zero_tolerance_6c6cb0fce46f`
- Repair receipt internal SHA: `492960c0d761574b0e74ff631f78e15b82bdb515826bd28a5d5b86ea44230e28`
- Final audit internal SHA: `e09f176325d5cd2417bc36c7ced2402b9225523a4e327659b3875d2093a5a114`
- E00–E25 invariant: 2,062 files, aggregate SHA
  `46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b`

## 해석과 claim 경계

Human/AI label 없이 frozen detector 기준 contamination gate를 결정적으로
해결했다. 이는 E26 scientific input provenance만 연다. LM 성능,
Dual–Projected-Tied 차이, numerical/resource readiness, official operator 또는
E26a GO를 의미하지 않는다. 다음 단계는 이 exact readiness SHA에 대한
final-data numerical/restart/resource preflight이며 별도 실행 승인이 필요하다.
