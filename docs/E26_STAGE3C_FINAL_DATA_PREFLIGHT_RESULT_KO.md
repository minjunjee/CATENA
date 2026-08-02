# E26 Stage-3C final-data preflight 결과

## 판정

```text
execution_status: COMPLETED_FAIL_CLOSED
stage3c_disposition: BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE
scientific_evidence: false
scientific_e26a_started: false
restart_audit_started: false
resource_preflight_started: false
```

Zero-tolerance repair data와 source/protocol lock은 모두 통과했지만, 세 model
candidate 모두 등록 numerical contract를 통과하지 못했다. 이 결과는 E26의
과학적 가설 판정이 아니라, Scientific E26a admission 이전의 numerical
implementation boundary다.

## 고정 입력과 실행 provenance

| 항목 | 값 |
|---|---|
| 실행 source commit | `56d6027e51f7aad1a4ab16376bf4e912f26fb4da` |
| source inventory SHA-256 | `913277086a14493a6091e0057d86c06f29c9adb3243a760da520dcbbd6ed8e09` |
| final-data readiness file SHA-256 | `c267ec6b85a354b50d9772dd4a43ae9829d58c010ec4d0be08a94bc23e558ef7` |
| final-data readiness internal SHA-256 | `fd22ea54413905122545393ae0832981e2c2f9d6c594a58cdb9f1be337dd427f` |
| composite final-data lock SHA-256 | `03658622eaa36ed2b6756b680dd64e933ea355c843175257c988bc1393f4db42` |
| frozen E00-E25 aggregate | 2,062 files / `46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b` |
| original run | `/tmp/catena_e26_preflight_fd22ea544139` |
| durable immutable mirror | `/data/minjun_dev/CATENA/artifacts/e26_stage3c_numerical_preflight/20260802T060323Z` |
| mirrored raw-run aggregate (11 files) | `296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc` |
| failure status SHA-256 | `dc7ed1837ccf022fe5110fdb44907c5e340391f0bcc5c92b7d5e26dcf2a95616` |

원본 11개 run 파일은 두 경로에서 byte-for-byte 일치하며 non-finite 값은 없다.
Durable mirror에는 이 1-page summary만 추가했으며 위 aggregate에서 제외했다.
실행 전후 세 worktree는
clean했고 E00-E25 artifact는 변경되지 않았다.

## Numerical 결과

| Gate | 결과 | 등록 기준 |
|---|---:|---:|
| FP32 arbitrary-partition reports | 12/12 PASS | rel-L2 및 max-abs `<=1e-5` |
| FP32 nontrivial partition rows | 132/132 PASS | 동일 |
| BF16 arbitrary-partition reports | 0/12 PASS | rel-L2 `<=0.007` |
| BF16 nontrivial partition rows | 0/132 PASS | 동일 |
| FP32 alternative accumulation layouts | 0/12 PASS | rel-L2 및 max-abs `<=1e-5` |
| BF16 alternative accumulation layouts | 0/12 PASS | rel-L2 `<=0.007` |
| Finite gradients / exact state metadata | 전부 PASS | 필수 |
| Compile fallback / graph break | 0 / 0 | 각각 0 |

FP32 external partition의 최대 relative-L2는 logits `2.79e-6`, recurrent state
`3.53e-6`, attention K/V `2.71e-6`, gradient `4.20e-6`이었다. 반면 BF16의
최대 external-partition error는 logits `0.01140`, recurrent `0.01786`,
attention K/V `0.01179`, gradient `0.01800`이었다. Optimized-vs-reference
BF16 gradient error는 최대 `0.03514`였다.

FP32 microbatch별 pre-optimizer gradient는 relative-L2
`2.35e-6`-`3.20e-6`으로 등록 기준을 통과한다. 그러나 거의 0인 한 gradient가
layout에 따라 `+1.30e-8` 대 `-9.31e-9`로 바뀌면서 첫 AdamW step 뒤 parameter
차이가 `4.56e-5`가 됐다. 전체 후보에서 parameter max-abs는 최대 `8.72e-5`로
`1e-5` 기준을 넘었다. BF16 gradient는 optimizer 이전부터 relative-L2
`0.01262`-`0.01492`로 실패했다.

## 원인과 claim 경계

Trainer의 global-token normalization, single clip/optimizer/scheduler step,
token exposure에는 오류가 없었다. BF16 실패는 batch/sequence shape에 따른
mixed-precision GEMM 차이와, optimized recurrence의 FP32 accumulation과 Python
reference의 BF16 recurrence가 섞인 precision-policy 불일치에서 발생했다.

Full-shape padding이나 광범위한 FP32 FFN은 진단상 오차를 줄일 수 있지만 각각
등록 physical microbatch 의미를 없애거나 dense compute의 대다수를 FP32로
전환하여 7-day resource feasibility를 바꾼다. 이는 minor implementation repair가
아니므로 사후 적용하지 않았다. Threshold, optimizer, candidate, data, seed와
metric은 변경하지 않았다.

따라서 restart/resource preflight와 Scientific E26a는 열리지 않는다. 다음
진행은 별도 prospective arithmetic-policy amendment를 사용자 승인으로 고정한
경우에만 가능하며, 현재 결과를 PASS로 재판정할 수 없다.
