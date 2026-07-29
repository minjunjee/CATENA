# E22b paired-seed sharding 실행 amendment

상태: `EXECUTION_TOPOLOGY_AUTHORIZED_DEPENDENCIES_PENDING`
범위: 성능·실행 topology만 변경
과학 프로토콜: `prospective_active_path_locality_confirmatory_v1` 그대로 유지

사용자는 2026-07-29에 보수적인 GPU 최적화를 명시적으로 승인했다. 승인
범위는 seed-level 실행 topology뿐이다. Scientific protocol, data, model,
optimization, precision, metric, threshold와 claim 변경은 승인 범위 밖이다.

## 목적

기존 E22b는 8개 paired training seed를 한 프로세스에서 직렬 실행한다.
각 seed의 RNG가 등록 seed offset으로 다시 초기화되고 다른 seed의
checkpoint를 읽지 않으므로, seed 전체를 독립 worker에 배정할 수 있다.
이 amendment는 다음 실행 변환만 허용한다.

```text
serial:
seed 1301 -> 1319 -> 1327 -> 1361 -> 1381 -> 1409 -> 1423 -> 1451

sharded:
GPU 0: 1301, 1381
GPU 1: 1319, 1409
GPU 2: 1327, 1423
GPU 3: 1361, 1451
```

한 seed의 `mean_retention`과 selected method, 네 controller variant는 항상
같은 worker에서 완전히 실행한다. Paired statistical unit를 분할하지 않는다.

## 변경하지 않는 항목

다음은 기존 config, E21 parent lock, E22a selection lock 및 E22b protocol
lock에서 그대로 읽는다.

- confirmatory seed와 seed offset
- train/evaluation namespace와 data tensor 생성
- model, optimizer, step, batch, precision
- method, loss, threshold, SESOI, equivalence/non-inferiority margin
- update/gap grid와 모든 metric
- B/C/D estimand, exact sign-flip, 최종 4-way disposition
- evidence tier와 허용·금지 claim 문구

TF32, AMP, `torch.compile`, batch 확대, step 축소, metric 근사, checkpoint
재사용 및 resume는 이 amendment에 포함되지 않는다.

CPU non-evidence pipeline dry-run에 한해서 worker당 OMP/MKL thread를 4로
제한해 host oversubscription을 막는다. CUDA MAIN worker에는 이 제한을
적용하지 않는다.

## fail-closed 실행 계약

Canonical launcher는
`scripts/launch_e22b_seed_shards.py` 하나다. Scientific MAIN 전에 다음을
모두 요구한다.

1. 기존 E22b protocol lock과 그 파일 hash map 검증
2. E22a completed MAIN `selection_lock.json` hash-chain 검증
3. 이 execution amendment lock의 전체 file map 검증
4. 현 HEAD를 가리키는 clean-worktree annotated source-lock tag
5. 현재 source/config/protocol/selection hash에 묶인 CPU
   serial-vs-shard equivalence PASS report
6. 정확한 `CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED`
7. `/home/minjun_dev/miniconda3/envs/catena-v6` prefix와 그
   `bin/python` interpreter
8. active compute process가 없고 memory use가 512 MiB 이하인 동종 GPU
   네 개
9. artifact root별 E22b exclusive non-blocking launcher lock

Launcher lock은 GPU idle 검사를 하기 전에 획득하고 coordinator 종료까지
유지한다. 선택 GPU의 index, UUID, model name, total/used memory와 compute
capability를 execution plan, run manifest와 final report에 고정한다. GPU
worker가 시작된 뒤에는 idle 검사를 다시 실행하지 않고, 시작 전에 잠근
inventory record의 무결성만 검증한다.

외부 CPU equivalence report는 검증 직후 canonical run의
`cpu_serial_shard_equivalence.json`으로 atomic byte-copy한다. Execution
plan, run manifest와 final report는 이 canonical copy의 path와 SHA-256에
묶인다. 따라서 `/tmp`의 원본 proof가 이동·변경돼도 완성된 MAIN artifact의
provenance chain은 자체 완결적이다.

Coordinator는 하나의 새 UTC E22b run을 만든다. Worker는 각자
`shards/shard-NNN/`만 쓰고, 다음을 마지막에 atomic
`shard_manifest.json`으로 봉인한다.

- exact seed assignment와 device
- source/config/protocol/selection/data/plan hash
- raw/seed/active-cell JSONL row count와 SHA-256
- checkpoint file SHA-256와 state-dict content SHA-256
- initialization, codebook, parameter-surface metadata

Worker 하나라도 실패하거나 manifest/hash/row/grid가 다르면 aggregate를
중단한다. 실패 run은 `execution_failure.json`과 log를 보존하고
`latest.json`을 갱신하지 않는다. Partial resume나 성공 shard 재사용은
허용하지 않는다.

## deterministic aggregate

Aggregate는 scientific outcome을 읽어 topology를 선택하지 않는다.
네 registered shard를 모두 검증한 뒤:

1. 8-seed complete grid와 중복·결측·non-finite를 검사
2. 기존 serial loop의 seed→method→variant→condition→family→update→gap
   순서로 raw row를 복원
3. seed summary, active-cell row와 confirmatory assessment를 전체 raw
   row에서 새로 계산
4. checkpoint를 canonical `checkpoints/`에 atomic independent byte-copy로
   배치하고 source/target inode가 다르며 SHA-256이 같은지 검증
5. 기존 E22b report/claim contract에 execution-topology provenance만 추가
6. 마지막에만 report, manifest와 latest pointer를 finalize

따라서 shard worker의 완료 순서는 scientific row 순서나 통계 결과에
영향을 주지 않는다.

## CPU serial-vs-shard equivalence

실제 E22a selection을 입력으로 dry-sized runtime의 첫 4 confirmatory seed를
CPU에서 다음 두 방식으로 실행한다.

- 한 번의 serial grid
- seed별 독립 grid 네 개 후 canonical merge

다음을 exact equality로 요구한다.

- throughput과 checkpoint container hash를 제외한 모든 raw scientific row
- checkpoint state-dict content hash
- initialization hash
- identifier codebook hash
- parameter count

MAIN unlock validator는 `checks`가 아래 여섯 key와 정확히 일치하고 모두
`true`인 report만 허용한다. 일부 key 누락, 추가 key 또는 `false`는
fail-closed한다.

```text
raw_row_count_equal
canonical_scientific_rows_exact
checkpoint_state_hashes_exact
identifier_codebook_hash_exact
initialization_hashes_exact
parameter_counts_exact
```

또한 frozen E22b config와 E22a selection contract에서 도출한 첫 네
confirmatory seed, selected/baseline method id, dry runtime SHA-256와
serial/sharded expected raw row count를 exact binding한다. 다음 non-evidence
및 comparison field도 exact match여야 한다.

```text
scientific_evidence = false
claim_eligible = false
comparison_exclusions =
  [examples_per_second, checkpoint container file SHA-256]
checkpoint_state_hash_comparison = exact
scientific_metric_comparison = exact
```

Equivalence producer는 report를 쓴 직후 동일 validator로 자체 검증하며,
그 검증을 통과한 report만 path를 반환한다.

`examples_per_second`는 실행 topology 자체의 측정값이고 비교에서 제외한다.
`torch.save` container hash는 state가 같아도 serialization container가 달라질
수 있으므로 비교에서 제외하되, 각 실제 shard checkpoint file hash는 shard
manifest와 최종 report에서 별도로 검증·기록한다.

## 실행 순서

E22a MAIN이 완료된 뒤 CPU equivalence:

```bash
cd /home/minjun_dev/CATENA
source /home/minjun_dev/miniconda3/bin/activate catena-v6

python scripts/launch_e22b_seed_shards.py verify-equivalence \
  --config configs/e22b_active_path_locality.yaml \
  --selection-run /data/minjun_dev/CATENA/artifacts/e22a_locality_method_selection/<RUN> \
  --output-root /tmp/catena_e22b_shard_equivalence_<UTC>
```

구현 commit을 고정한 annotated tag message에는 아래 두 줄이 정확히
포함되어야 한다.

```text
E22B_BASE_PROTOCOL_LOCK_SHA256=<E22B protocol lock SHA-256>
E22B_SEED_SHARD_AMENDMENT_LOCK_SHA256=<amendment lock SHA-256>
```

그 뒤에만 MAIN을 실행한다.

```bash
CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED \
python scripts/launch_e22b_seed_shards.py run \
  --config configs/e22b_active_path_locality.yaml \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --selection-run /data/minjun_dev/CATENA/artifacts/e22a_locality_method_selection/<RUN> \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --source-lock-tag <ANNOTATED_TAG> \
  --equivalence-report /tmp/catena_e22b_shard_equivalence_<UTC>/E22B_CPU_SERIAL_SHARD_EQUIVALENCE.json
```

실행 topology 자체는 승인됐지만, E22a 완료, CPU equivalence PASS와 새
source-lock tag, 정확한 runtime, 동종 idle GPU 네 개와 exclusive launcher
lock이 없으면 Scientific MAIN은 여전히 fail-closed한다. Canonical
checkpoint는 shard checkpoint와 inode를 공유하지 않으므로 shard artifact의
사후 변경이 canonical evidence를 바꿀 수 없다.
