# E22b paired-seed sharding 구현 검증

검증일: 2026-07-29 UTC
실행 모드: CPU `DRY_RUN` / non-evidence
Scientific MAIN: 실행하지 않음

## 결과

| 검사 | 결과 |
|---|---|
| Frozen E22b protocol/config hash | PASS |
| Registered 8-seed → 4-shard mapping | PASS |
| Unit CPU serial-vs-seed-shard scientific equality | PASS |
| 4-worker canonical dry pipeline | PASS |
| 기존 serial E22b dry와 새 4-worker dry scientific rows | 2,048/2,048 exact |
| 기존 serial E22b dry와 새 4-worker dry state-dict | 64/64 exact |
| 기존 serial E22b dry와 새 4-worker dry seed summary | byte exact |
| Canonical raw/seed/active row 수 | 2,048 / 16 / 896 |
| Canonical checkpoint 수 | 64 |
| Shard manifest 수 | 4 |
| Report/manifest/latest finalization | PASS |
| MAIN authorization/runtime negative test | PASS |
| Homogeneous-idle GPU inventory negative test | PASS |
| Exclusive launcher lock contention test | PASS |
| Canonical equivalence copy/tamper test | PASS |
| Equivalence exact-six/fixed-field adversarial test | PASS |
| Canonical checkpoint inode isolation test | PASS |

비교에서 `examples_per_second`와 `torch.save` container SHA만 제외했다.
전자는 topology-dependent timing이고, 후자는 동일 state-dict라도 container
serialization에 따라 달라질 수 있다. 실제 shard checkpoint의 file SHA와
state-dict content SHA는 모두 manifest에서 독립적으로 검증했다.

## 개발 검증 artifact

Reviewer hardening을 반영한 최종 검증은 아래 fresh root에서 수행한다. 이
문서는 source lock 대상이므로 생성되는 UTC run id와 artifact SHA는 source
문서에 사후 삽입하지 않고 검증 command output 및 전달 보고에 기록한다.

Canonical four-worker dry run root:

```text
/tmp/catena_e22b_sharded_pipeline_reviewer_final_20260729T130000Z/
```

CLI CPU equivalence proof:

```text
/tmp/catena_e22b_shard_equivalence_reviewer_final_20260729T130000Z/
  E22B_CPU_SERIAL_SHARD_EQUIVALENCE.json
```

네 comparison seed에서 1,024 serial row와 1,024 sharded row가 exact
일치했고, scientific equality check 6/6이 통과했다.

추가 hardening은 MAIN 실행에만 적용된다. 정확한 승인 ACK와 catena-v6
interpreter, 네 동종 idle GPU, exclusive launcher lock을 요구하며, 검증한
CPU equivalence proof를 canonical run 내부에 atomic copy한다. Canonical
checkpoint도 hardlink가 아니라 독립 byte-copy라서 shard file의 변경이
canonical evidence를 변경하지 않는다. CPU dry-run의 scientific 계산과
worker thread 제한은 기존과 동일하다.

Equivalence validator는 producer가 생성하는 여섯 check key의 exact set과
all-True를 요구한다. Frozen config/selection/dry runtime에서 도출한 첫 네
seed, method id, runtime hash, 양쪽 raw row count와 non-evidence/comparison
field도 모두 exact match해야 한다. 각 check의 missing/false, extra check,
각 fixed field의 missing/wrong value를 대상으로 한 negative test가
fail-closed함을 확인한다.

## 사용 제한

위 `/tmp` artifact는 구현 검증용이며 claim에 사용할 수 없다. 또한 final
integration/source capture 전 source fingerprint를 사용했으므로 MAIN
unlock proof가 아니다. 최종 source-lock commit 이후 실제 E22a MAIN
selection을 입력으로 equivalence command를 다시 실행하고, 그 새 report만
E22b MAIN launcher에 전달해야 한다.
