# E26 Stage-3D Fixed-Physical-Layout BF16 Admissibility Protocol

## 연구 질문

한 가지 사전 고정 physical microbatch/accumulation/backend recipe에서
Projected-Tied와 Dual이 matched experiment를 수행할 만큼 재현 가능하고
numerically admissible한가?

이 단계는 Dual의 우월성이나 transaction 성능을 평가하지 않는다. Main-test
outcome을 읽지 않으며 claim ceiling은 fixed-layout numerical admissibility다.

## Predecessor 보존

Stage-3C의 `BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE`와 다음
결과는 그대로 유지한다.

- FP32 arbitrary partition: 12/12 report, 132/132 nontrivial rows PASS
- BF16 arbitrary partition: 0/12 report PASS
- FP32/BF16 alternative accumulation layout: 각각 0/12 PASS
- Diagnostic: `KNOWN_BF16_AND_OPTIMIZER_LAYOUT_SENSITIVITY`
- 별도 diagnostic: fixed Stage-3C probe의 BF16
  `compiled_scan`-`reference_python` mismatch

Stage-3D 결과는 위 실패를 PASS로 바꾸지 않는다.

등록 predecessor anchor는 다음과 같다.

| Binding | SHA-256 |
|---|---|
| Stage-3C result | `83fab26e7936654b664653776d501c3fdee6cb7f0ffd78c3d9682ed41d319b56` |
| Stage-3C status | `15b896a33e0fe286c80f2c204b7be2be0fbe6aaf8cdc512fafbd31040f8aabda` |
| Registered raw-run aggregate | `296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc` |
| Raw failure status | `dc7ed1837ccf022fe5110fdb44907c5e340391f0bcc5c92b7d5e26dcf2a95616` |

기존 raw aggregate와 새 manifest의 row-rehash aggregate는 알고리즘이 다르므로
서로 대체하지 않고 둘 다 별도 필드로 보존한다.

## Gate

### G0 — Invariance

Stage-3C report/status/raw file SHA, Stage-3C protocol/data locks, repaired-data
readiness와 E00-E25 aggregate를 실행 전에 검증한다. 새로운 run directory만
사용한다.

### G1 — Matched physical recipe

후보별 fixed layout은 prospective YAML 그대로 사용한다. Parameter name/shape,
initialization digest, optimizer signature, token IDs, cursor, autocast, clip 및
scheduler order가 variant 간 같아야 한다.

### G2 — FP32 reference inheritance

Stage-3C raw receipts를 exact hash로 읽고 기존 FP32 report 12/12와 nontrivial
row 132/132가 통과했는지 재검증한다. 다른 source에서 재계산한 값을 대체하지
않는다.

### G3 — Fixed-layout BF16 admissibility

각 candidate × variant × `{zero_state,prefilled_state}`에서 다음을 검사한다.

1. BF16 compiled backend 대 BF16 `reference_python`
2. BF16 `reference_python` 대 FP32 `reference_python`
3. Complete floating gradient-tree relative L2와 finite 여부
4. Recurrent/attention state 및 metadata
5. Clone/no-alias와 graph-break/fallback

Stage-3C의 FP32 `1e-5`/max-abs `1e-5`, BF16 `0.007` 기준을 그대로 사용한다.
Near-zero leaf를 제외하지 않으며 worst-leaf 값은 diagnostic으로만 기록한다.

### G4-G6 — Replay, optimizer, backend

G3가 통과한 경우에만 candidate × variant 6개 same-layout replay를 별도 fresh
process에서 수행한다. 후보당 하나의 common serialized initial checkpoint를
만들고 Projected-Tied/Dual의 A/B process가 이를 독립적으로 load한다. 동일하게
capture한 Python/NumPy/Torch/CUDA RNG, checkpoint cursor, data IDs/order,
optimizer, scheduler, backend recipe와 physical device를 고정한다. 같은
variant의 A/B graph SHA는 같아야 하지만 두 variant의 projection graph SHA가
같을 필요는 없다. Token normalization 및 한 번의
clip/AdamW/scheduler boundary, finite gradient, graph-break/fallback 0과
variant-specific precision/layout override 0을 요구한다.

## 실행 분기

한 hard gate라도 실패하면 resource preflight를 실행하지 않는다. 모든 gate가
통과하면 fixed layout manifest를 immutable하게 보존한 뒤 같은 layout만 사용해
non-evidence throughput, memory, checkpoint I/O와 schedule projection을 측정한다.
Scientific E26a는 별도 사용자 승인 전까지 항상 닫혀 있다.

GO 이후 resource preflight는 Stage-3C protocol에 기록된 E26a config,
tokenizer manifest, general-train corpus manifest의 **정확한 path와 SHA-256**을
사용한다. 같은 bytes를 다른 위치에서 제시하는 것도 허용하지 않는다. 결과는
다음 canonical non-evidence namespace에 새 UTC run으로 보존한다.

```text
/data/minjun_dev/CATENA/artifacts/
  e26_stage3d_resource_preflight/<run_id>/
```

Aggregate receipt는 모든 input과 세 worker receipt/report의 절대 path와 byte
SHA를 묶고, schema와 canonical receipt validator로 다시 읽어 검증한다.
`artifact_audit.json`과 atomic `latest.json`도 생성한다. 종료 코드는 `0`이면
resource feasible, `1`이면 resource infeasible, `2`이면 dependency 또는 execution
error다. 어느 분기에서도 Scientific E26a를 시작하지 않는다.

12개 G3 또는 6개 G4 record가 완결된 상태에서 numerical hard gate가 실패한
경우에만 `STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY`를 기록한다.
Subprocess crash, 누락 artifact, hash/source mismatch처럼 estimand를 평가하지
못한 경우에는 `STAGE3D_NOT_EVALUABLE_IMPLEMENTATION_OR_EXECUTION_ERROR`로
분리하며 numerical failure로 재분류하지 않는다.
