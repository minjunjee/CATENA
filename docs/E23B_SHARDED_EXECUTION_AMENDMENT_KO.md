# E23b 등록 Seed Sharding 실행 Amendment

## 목적과 범위

이 amendment는 동결된 `e23b_product_poset_confirmatory`의 계산 배치만
변경한다. Scientific protocol은 변경하지 않는다.

| 항목 | 동결 상태 |
|---|---|
| config 및 namespace | 변경 없음 |
| 등록 seed | `2401, 2411, 2423, 2437, 2441, 2459, 2473, 2477` |
| model/controller projection | 변경 없음 |
| optimizer, steps, batch, data order | 변경 없음 |
| dtype/precision | 변경 없음 |
| metric, SESOI, margin, claim wording | 변경 없음 |
| E18b/E23a/E22b dependency decision | 변경 없음 |
| boundary selection | `theory_boundary_only_v1`, 변경 없음 |
| 변경 대상 | 서로 독립적인 training seed의 물리적 GPU 배치만 |

Lock은 seed/seed offset, controller set, model, optimizer, batch, step,
data/namespace, theory boundary, metric, threshold, precision, dependency,
claim wording이 변경되지 않았고 checkpoint resume/reuse가 없음을 개별
Boolean flag로 선언한다. Entry point는 이 no-change flag 전체가 정확히
`false`인지 실행 전에 검사한다.

이 실행 방식은 E23a outcome을 읽어 confirmatory controller를 고르지 않는다.
E22b disposition에 따른 `safe_minimality`/`capacity_only` 결정과 sparse
objective fail-closed 규칙도 기존 E23b dependency resolver를 그대로
호출한다.

## 실행 구조

기본 4-GPU topology는 등록 seed 순서를 유지한 다음 네 shard다.

```text
shard_00: 2401, 2411
shard_01: 2423, 2437
shard_02: 2441, 2459
shard_03: 2473, 2477
```

각 worker는 기존 `generate_product_poset_rows`를 호출하며 config의 seed
부분집합만 실행 substrate에서 선택한다. Scientific config snapshot과 config
hash는 원본 8-seed config를 사용한다. Worker는 다음을 하지 않는다.

- `report.json` 생성
- claim 판정
- canonical experiment `latest.json` 변경
- 다른 shard 파일 수정
- threshold 또는 metric 변경

MAIN `prepare`는 각 shard에 배정할 물리 GPU의 index, UUID, model name과
compute capability를 `prepared_execution.json`에 먼저 고정한다. 네
index와 UUID는 모두 달라야 하고 model name과 capability는 같아야 한다.
각 worker는 `CUDA_VISIBLE_DEVICES=<physical index>`와 logical `cuda:0`의
name/capability를 이 binding과 대조하고, shard manifest에 같은 identity를
기록한다. Aggregator도 네 manifest의 unique/homogeneous binding을 다시
검사한다.

## Fail-closed aggregation

Aggregator는 다음 항목이 모두 통과한 뒤에만 canonical
`e23b_product_poset_confirmatory/<UTC>/` artifact를 생성한다.

1. clean HEAD를 가리키는 annotated source-lock tag
2. tag message가 원 protocol lock과 amendment lock SHA-256을 함께 bind
3. 현 source/config/dependency에 bind된 CPU serial-vs-shard equivalence PASS
4. 준비 시점과 aggregation 시점의 source fingerprint가 동일
5. 원 E23b protocol lock과 이 amendment lock의 SHA-256 일치
6. E18b, E23a, E22b dependency report SHA-256 재검증
7. 8개 등록 seed의 disjoint exact cover
8. raw grid `38,016`개 exact Cartesian cover
9. training row와 checkpoint 각각 `128`개
10. duplicate, missing, non-finite row 없음
11. checkpoint 파일 SHA-256과 모든 row-level checkpoint provenance 일치
12. row-level E18b/E23a/E22b provenance 일치
13. controller/gap 사이 paired base-transaction digest 일치
14. frozen config seed 순서로 deterministic merge

누락되거나 실패한 shard는 삭제하지 않는다. Aggregator는 scientific report와
`latest.json`을 만들지 않고 중단한다. 동일 workspace의 이중 aggregation도
거부한다.

Aggregator는 workspace의 `.aggregate.lock`에 non-blocking exclusive
filesystem lock을 잡은 상태로 validation, canonical run 생성, report,
`latest.json`, aggregate receipt 기록을 모두 수행한다. 따라서 동시에 두
aggregator가 시작되어도 canonical run은 최대 하나만 만들어진다. Lock을
얻지 못한 process는 즉시 fail-closed하며, 첫 process가 끝난 뒤 실행된
중복 요청은 기존 receipt 때문에 차단된다.

## Canonical artifact

모든 검증 후에는 원래 E23b entry point와 동일한 파일 계약을 생성한다.

```text
config.resolved.yaml
environment.json
run_manifest.json
protocol_lock.json
execution_amendment_lock.json
prepared_execution.json
execution_shard_manifests/shard_00.json
execution_shard_manifests/shard_01.json
execution_shard_manifests/shard_02.json
execution_shard_manifests/shard_03.json
E23B_CPU_SERIAL_SHARD_EQUIVALENCE.json
data_manifest.json
theory_predictions.json
product_poset_raw_metrics.jsonl
product_poset_seed_metrics.jsonl
product_poset_training_runs.jsonl
poset_minimal_demands.jsonl
checkpoints/*.pt
RESULTS_SUMMARY_KO.md
report.json
```

`report.json`의 `execution_topology`만 sharding provenance를 추가한다.
`data_manifest.json`, seed-level 통계, claim gate는 frozen serial E23b와 같은
과학적 계약을 사용한다. Prepared manifest, 네 shard manifest 및 MAIN
equivalence proof는 canonical artifact 안에 byte-for-byte 복사하고 source와
copy의 SHA-256을 대조한다. 따라서 staging workspace가 없어도 canonical
artifact 자체에서 실행 provenance를 감사할 수 있다.

## 실행 명령

MAIN 전에 amendment를 commit하여 clean source를 고정해야 한다.

```bash
cd /home/minjun_dev/CATENA
source /home/minjun_dev/miniconda3/bin/activate catena-v6

PARENT_SHA=$(sha256sum docs/E23B_PRODUCT_POSET_CONFIRMATORY_LOCK.json | awk '{print $1}')
AMENDMENT_SHA=$(sha256sum docs/E23B_SHARDED_EXECUTION_AMENDMENT_LOCK.json | awk '{print $1}')
SOURCE_LOCK_TAG=post-e21-e23b-sharded-<UTC>

git tag -a "$SOURCE_LOCK_TAG" -m "E23b sharded execution source lock
E23B_BASE_PROTOCOL_LOCK_SHA256=$PARENT_SHA
E23B_SHARDED_EXECUTION_AMENDMENT_LOCK_SHA256=$AMENDMENT_SHA"

EQUIVALENCE_ROOT=/tmp/catena_e23b_shard_equivalence_<UTC>
python -m experiments.e23b_product_poset_confirmatory_sharded \
  verify-equivalence \
  --config configs/e23b_product_poset_confirmatory.yaml \
  --output-root "$EQUIVALENCE_ROOT" \
  --e18-freeze /data/minjun_dev/CATENA/artifacts/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json \
  --e23a-screen /data/minjun_dev/CATENA/artifacts/e23a_product_poset_screen/<EXACT_RUN> \
  --e22b-run /data/minjun_dev/CATENA/artifacts/e22b_active_path_locality/<EXACT_RUN> \
  --source-lock-tag "$SOURCE_LOCK_TAG"

CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED \
bash scripts/launch_e23b_sharded.sh \
  /home/minjun_dev/CATENA \
  /data/minjun_dev/CATENA/artifacts \
  /data/minjun_dev/CATENA/artifacts/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json \
  /data/minjun_dev/CATENA/artifacts/e23a_product_poset_screen/<EXACT_RUN> \
  /data/minjun_dev/CATENA/artifacts/e22b_active_path_locality/<EXACT_RUN> \
  "$SOURCE_LOCK_TAG" \
  "$EQUIVALENCE_ROOT/E23B_CPU_SERIAL_SHARD_EQUIVALENCE.json"
```

Launcher는 선택한 GPU들이 동일 model/capability이고 유휴 상태인지 확인한다.
worker가 하나라도 실패하면 partial shard를 보존하고 aggregation을 차단한다.

## 동등성 검증

Source-lock tag가 생성된 뒤, 실제 E18b/E23a/E22b dependency를 bind하고
등록 confirmatory seed 중 앞의 4개를 사용한 CPU dry-sized runtime에서
serial과 4개 single-seed shard를 각각 실행한다. 다음 deterministic
scientific payload의 exact equality를 요구한다.

- raw metrics (artifact 절대경로 제외)
- training loss/optimizer/provenance (throughput/peak-memory 제외)
- seed metrics와 poset cell rows
- assessment와 claim gate
- data SHA-256
- checkpoint state-dict content SHA-256 map

Equivalence report의 `checks`는 다음 9개 key의 exact set이어야 하며 모든
값이 Boolean `true`여야 한다. Key 누락·추가와 truthy 대체값을 허용하지
않는다.

```text
registered_four_seed_subset_exact
raw_row_count_exact
canonical_scientific_raw_rows_exact
canonical_scientific_training_rows_exact
checkpoint_state_hashes_exact
seed_statistics_exact
cell_statistics_exact
assessment_exact
runtime_contract_exact
```

Report의 seed는 frozen config의 앞 4개로 고정한다. `boundary_mode`,
`locality_method`, `locality_risk_scale`은 frozen E22 dependency에서,
`runtime_config_sha256`과 serial/sharded expected raw-row count는 frozen
config의 4-seed CPU dry-sized runtime에서 각각 유도해 exact equality를
검사한다. `comparison_exclusions`는 아래 네 항목의 순서까지 exact하며
`checkpoint_state_hash_comparison`과 `scientific_metric_comparison`은
각각 문자열 `exact`여야 한다.

```text
examples_per_second
peak_memory_bytes
checkpoint absolute path
checkpoint container file SHA-256
```

Checkpoint container hash, 절대경로, throughput, peak-memory만 비교에서
제외한다. Report는 source fingerprint, annotated tag object/commit,
config SHA, 두 protocol SHA 및 E18b/E23a/E22b dependency payload/hash에
묶인다. MAIN에서 full serial 결과를 동시에 생성하지 않는다.

Integration dry-run은 scientific non-evidence 상태에서 등록 seed 앞의
4개를 각각 독립 CPU shard에 배정해 네 shard aggregation path까지 검사한다.
추가로 missing shard, tampered shard artifact, 동시 aggregate 두 process를
각각 fail-closed하는지 검증한다.

## Claim boundary

이 amendment는 runtime speedup 자체를 scientific claim으로 열지 않는다.
E23b의 controlled capacity 또는 safe minimality claim만 원 protocol에 따라
판정한다. Natural language, pretrained LM, agent, official backend, production
latency claim은 계속 금지된다.
