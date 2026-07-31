# E26 Stage-2 resource-preflight lock

Target-context throughput 측정은 scientific E26a가 아니라
`NON_EVIDENCE_VALIDATION`이다. 측정 결과는 다음 비순환 DAG의 downstream
receipt로 고정한다.

```text
prospective protocol + numerical/restart/backend receipts
        -> resource_preflight.json
        -> explicit user-approved file SHA-256
        -> canonical E26a admission
```

Protocol lock이 나중에 생성되는 resource receipt를 역방향으로 hash하지 않는다.
대신 canonical CLI는 `--resource-preflight`와
`--expected-resource-preflight-sha256`를 모두 요구한다. 후자는 사용자가 승인한
exact file bytes를 고정하며 embedded `receipt_sha256`만 다시 계산해 selection을
바꾸는 것을 막는다.

각 isolated GPU worker는 다음을 스스로 확인한다.

- clean source commit과 execution-source inventory
- config 및 모든 upstream input file SHA-256
- candidate config SHA-256
- numerical/restart coverage와 backend promotion chain
- `CUDA_VISIBLE_DEVICES`, PyTorch가 관찰한 단일 GPU UUID, parent inventory의 일치
- 실제 target context/global-token batch에서 accumulation-1 BF16 reference,
  selected layout 및 더 작은 preregistered layout 사이의 equivalence

Parent는 worker receipt의 canonical hash뿐 아니라 candidate/config binding,
worker report bytes와 report content, run manifest, observed physical UUID를 다시
검증한다. Aggregate receipt는 모든 candidate measurement와 resource projection,
worker receipt/report/spec SHA 및 deterministic selection을 포함한다.

Canonical E26a는 throughput을 fresh remeasure할 수 있지만, 재계산한
`candidate_id`, `token_budget`, `context_length`, `target_global_batch_tokens`,
`selected_microbatch_sequences`, `accumulation_steps` 중 하나라도 approved
resource receipt와 다르면 fail-closed다. 수치 projection 자체의 작은 변동은
기록할 수 있으나 다른 candidate, budget 또는 batch layout을 silent selection하지
않는다. Canonical `--device`도 selected candidate를 측정한 worker의 physical
device와 GPU UUID에 결합된다.

이 receipt는 resource feasibility만 고정한다. LM 효과, E26a scientific result,
E26b/E26c 실행 권한을 열지 않는다.
