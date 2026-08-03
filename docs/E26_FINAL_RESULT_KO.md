# E26 Final 결과

## 판정

```text
execution_status: BLOCKED_ADMISSION
scientific_disposition: BLOCKED_OFFICIAL_RUNTIME_NAMESPACE_PROVENANCE_VALIDATION
scientific_main_started: false
lm_transfer_claim_open: false
```

E26 Final은 과학적 가설을 평가하지 못했다. Official source, community
checkpoint bytes·shape·strict load, TinyLlama tokenizer bytes, CUDA 13 / PyTorch
2.9 / FlashAttention / API-compatible FLA dependency provenance는 통과했다.
그러나 canonical official-runtime admission이 종료 조건을 통과하지 못했다.

정확한 source pin은 NVLabs `GatedDeltaNet-2`
`95709fc250357c2dd109361c353192f2aa5913f9`이며, license는 commercial use가
허용되지 않는 `NVIDIA Source Code License-NC`다. Starting checkpoint는
community repository의 `model-100b.pth` 17,401,727,659 bytes,
SHA-256 `0322ebeefa96badb24d6b4b511c36b02374b704dc1a65b90eab2ee1383a9ce23`다.
이 weight는 NVIDIA official release가 아니고 `model-95b.pth`와 byte-identical해
100B training-token identity를 입증하지 못한다. TinyLlama tokenizer revision은
`ff3c701f2424c7625fdefb9dd470f45ef18b02d6`이지만 community checkpoint와
cryptographically linked되어 있지 않다. Source license와 model-card license의
차이도 별도 manual-review 경계로 남는다.

최초 Dual/Tied audit는 직전 implementation validation이 derived checkout에 만든
`__pycache__` 때문에 GPU 실행 전에 source-cleanliness gate에서 차단됐다. 두
receipt를 보존하고 prospective R1 amendment에 따라 cache만 제거한 뒤 동일
protocol로 한 번 재실행했다. R1에서는 두 variant 모두 다음 오류로 동일하게
차단됐다.

```text
E26FinalCheckpointAuditError:
Refusing preloaded non-official Python module: lit_gpt.gdn2_ops
```

이 오류는 pinned checkout에서 import된 `lit_gpt.gdn2_ops` namespace package가
일반 module과 같은 `__file__` provenance 검사를 만족하지 못한 검사기 경계로
보인다. 하지만 등록된 R1을 이미 사용했으므로 checker를 고치거나 추가 실행하지
않았다.

## 실행되지 않은 단계

- 1.3B canonical chunk/fused runtime audit
- speed/VRAM/power preflight와 token-budget lock
- common tied bridge와 functionally identical fork
- 5 paired seeds × 2 variants main training
- sealed evaluation, mechanism intervention, PPL/RULER 및 systems benchmark

따라서 data, throughput, VRAM, power, bridge quality와 모든 scientific metric은
`NOT RUN` 또는 `NOT MEASURED`다. Dependency receipt도 provenance만 통과했으며,
external decode-cache clone/restore plumbing은 구현되지 않아
`decode_cache_evaluation_eligible=false`였다. 이 독립적인 later-stage limitation은
관측된 namespace blocker보다 뒤에 있어 실행 분기를 바꾸지는 않았다.

따라서 ADD/INVALIDATE 이득, symmetric equivalence, retention/locality, general
quality, mechanism 또는 speed에 관한 수치는 없다. 이 결과는 E26 hypothesis의
null이나 반증이 아니라 **official runtime admission failure**다.

## 보존·claim 경계

- Stage-3C와 Stage-3D disposition/report SHA는 변경하지 않았다.
- Frozen E00–E25 2,062 files의 aggregate SHA-256
  `46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b`를 재검증했다.
- 허용: 지정 official source와 community checkpoint의 transport/structural
  provenance가 통과했지만 E26 scientific runtime admission은 열리지 않았다.
- 금지: pretrained-LM transfer, official GDN2 superiority, transaction gain,
  mechanism, quality/locality, throughput 또는 production claim.

Canonical terminal artifact:
`/data/minjun_dev/CATENA/artifacts/e26_final_gdn2_1p3b_transactional_transfer/20260803T161043.290986Z`
