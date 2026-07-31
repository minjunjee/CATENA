# E26a numerical/restart audit 상태

## 현재 판정

```text
implementation_contract: COMPLETE
cpu_regression: PASS
canonical_target_gpu_audit: NOT_RUN
disposition: BLOCKED_DATA_SOURCE
scientific_evidence: false
```

Stage-2에는 다음 계약이 구현됐다.

- monolithic, `[1, remaining]`, 두 개의 fixed irregular partition과 8개
  fixed-random partition에서 FP32/BF16 logits, 모든 recurrent state,
  attention K/V ring, write index, sequence offset와 gradient 비교
- 세 prospective model candidate와 두 matched variant 전체의 numerical coverage
- `general→transaction`, `transaction→general` 경계 각각에서
  continuous 대 새-process checkpoint resume
- model, AdamW, scheduler, Python/NumPy/Torch/CUDA RNG, paired data cursor,
  token exposure, runtime state와 backend manifest 비교
- 동일 global token batch의 microbatch/gradient-accumulation equivalence
- selected candidate의 실제 context와 65,536-token batch에서 BF16
  accumulation-1 대 selected/더 작은 preregistered layout 비교
- worker가 실제 관측한 CUDA UUID와 물리 GPU inventory 결합
- 마지막 evaluation probe가 아니라 실제 training segment 직후 compiled graph
  identity를 checkpoint에 포함하고 resume graph와 비교

현재 training stream의 각 emitted sequence는 `reset_state=true`이며 optimizer
step 사이 recurrent state carry를 사용하지 않는다. 따라서 checkpoint runtime-state
검사는 hybrid state의 직렬화 및 evaluator-state 보존을 검증하고, optimizer-step
sequence 사이 carry semantics를 주장하지 않는다. E26c가 향후 stateful chunk
continuation을 사용한다면 별도 continuation-resume audit가 필요하다.

CPU unit/integration test는 이 coverage와 fail-closed tamper path를 검증한다.
기존 100-step smoke PASS는 보존하지만, 이는 sequence 256 random-token
`NON_EVIDENCE_VALIDATION`이며 위 target-context Stage-2 audit의 대체물이 아니다.

## 이번 단계에서 GPU audit를 실행하지 않은 이유

사전등록 near-duplicate gate가 541개 pair를 검출했다. Human adjudication 후
문서 제외가 필요하면 general memmap, corpus manifest, schedule, protocol,
numerical/restart locked hash가 모두 달라진다. 따라서 현재 data로 GPU audit를
실행해 영수증을 승격하는 것은 provenance상 무효다.

```text
arbitrary_partition_result: NOT_MEASURED_ON_FINAL_ELIGIBLE_DATA
checkpoint_resume_result: NOT_MEASURED_ON_FINAL_ELIGIBLE_DATA
cursor_resume_result: NOT_MEASURED_ON_FINAL_ELIGIBLE_DATA
grad_accum_result: NOT_MEASURED_ON_FINAL_ELIGIBLE_DATA
resource_preflight_result: NOT_MEASURED
```

이는 numerical failure가 아니라 upstream dependency block이다. Data repair가
필요하면 새 data namespace와 hash를 먼저 고정한 뒤, 한 번의 fresh `/tmp`
non-evidence preflight로 위 표를 채운다. Threshold, seed, metric 또는
candidate order는 변경하지 않는다.

## 실행 경계

Canonical script는 resource-preflight file의 사용자가 승인한 정확한 SHA까지
요구하고, clean worktree·optimized backend·data/numerical/restart/frozen
receipt 중 하나라도 다르면 artifact 생성 전에 중단한다. 이번 단계에서는
그 script를 실행하지 않았다.
