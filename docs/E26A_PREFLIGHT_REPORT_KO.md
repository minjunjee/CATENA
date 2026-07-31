# E26a Stage-2 preflight 보고

## 최종 판정

```text
stage2_disposition: BLOCKED_DATA_SOURCE
blocking_gate: PREREGISTERED_NEAR_DUPLICATE_AUDIT
flagged_pair_count: 541
ready_for_user_approval_e26a: false
```

Data/tokenizer/runner 통합은 완료됐지만 near-duplicate gate가 열리지 않았다.
따라서 final scientific-input receipt, canonical numerical/resource receipt와
E26a candidate/budget lock은 생성하지 않았다. 이는 구현 failure가 아니라
사전등록 data gate의 정상적인 fail-closed 동작이다.

## Repository와 불변성

| 항목 | 값 | 판정 |
|---|---|---:|
| Live repository | `/home/minjun_dev/CATENA` | immutable |
| Live HEAD | `adfdeaf9e87a8602a8e334915d87acb9ff25af39` | PASS |
| E26 worktree/branch | `/home/minjun_dev/CATENA_E26`, `exp/e26-autoregressive-lm` | isolated |
| Stage-2 implementation commit | `PENDING_IMPLEMENTATION_COMMIT` | pending |
| Pre-E26 source | 556 files, `554d5ff5792d28472a3b0d01f558d026f748d0126428a45aed5a3f9242566bf9` | PASS |
| Frozen E00–E21 base | 1,329 files, `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` | PASS |
| Frozen E22–E25 extension | 733 files, `3b9524854ee01d17a9a3f99b8b0ebd08a2ebf0c725d3765dac3496442772564e` | PASS |
| Frozen E00–E25 composite lock | 2,062 files, `46a779594c760fc1103833810bcc5227d7ca4526111911d7856cdfe2ff2d202b` | PASS |
| Composite lock path/SHA | `docs/E26_FROZEN_E00_E25_ARTIFACT_LOCK.json`, `3820d5d00b45a36e5bf207f962840bb3692d4b4be3ea9ffeaef0afe453a18423` | PASS |
| Post-commit frozen receipt | `PENDING_FROZEN_RECEIPT` | pending |
| Canonical E26 artifact | 없음 | PASS |

기존 integration `28f6b868…`, numerical repair `bb5fd9ae…`, data commits
`2b29681`, `28d5449`, `d118335`는 보존했다. 최초 failed smoke와 repaired
non-evidence smoke도 수정하지 않았다.

## Data lock

세부 수치와 hash는 `docs/E26_DATA_LOCK_REPORT_KO.md`에 있다.

| Gate | 결과 |
|---|---:|
| FineWeb-Edu immutable revision/shard provenance | PASS |
| Exact content dedup 및 split disjointness | PASS |
| 16,384 BPE 두 fresh-directory replay | PASS |
| General train/validation/test capacity 및 memmap integrity | PASS |
| Transaction replay/leakage 및 bounded 80/20 schedule | PASS |
| Corrected near-duplicate replay | **541 flags / FAIL** |
| Scientific data-readiness receipt | NOT CREATED |

두 corrected LSH audit의 파일 SHA-256은 동일한
`9bfe3c84df9368955b97ec6285db01450dff912cecd9f2f3334a7b8ced18f17a`다.
검수용 541행 CSV/JSONL은 생성했지만 label은 모두 비어 있다.

## 구현된 execution contract

- candidate: `d512_ctx4096`, `d512_ctx2048`, `d448_ctx4096`
- actual worker-visible CUDA UUID 검증
- arbitrary chunk partition FP32/BF16 state/logit/gradient contract
- 모든 candidate×variant×source-transition checkpoint/restart coverage
- saved training graph identity와 resumed training graph identity 결합
- target context/65,536-token batch에서 accumulation-1 대 selected/더 작은
  preregistered layout의 BF16 gradient-accumulation equivalence
- validation-only population lock 및 per-row split/episode access trace
- resource worker spec/result/report/source/device tamper 검증
- 사용자가 승인한 exact resource-preflight file SHA 요구
- canonical 재측정이 preflight candidate, token budget, context 또는
  microbatch/accumulation layout을 바꾸면 block
- canonical CLI/runtime GPU를 selected resource-worker physical index와 UUID에 결합하고
  별도 artifact에 기록
- clean worktree, no-overwrite canonical namespace, optimized backend only

## Non-evidence validation

```text
full pytest: 817 collected; 813 passed; 4 explicit skips
ruff check: PASS
ruff format --check: PASS
strict mypy (changed src/tools, 23 files): PASS
compileall: PASS
schema JSON parse: 17 / 17 PASS
shell syntax: PASS
scientific launcher default: DRY PRINT ONLY
```

GPU target-context preflight는 실행하지 않았다. 현재 memmap을 사용하면 human
adjudication 뒤 발생할 수 있는 data repair와 hash 변경 때문에 receipt가
즉시 무효가 되기 때문이다. 기존 sequence-256 smoke의 1,963.16 tokens/s는
descriptive reference일 뿐 target ETA lock으로 사용하지 않는다.

## ETA와 storage

현재 exact E26a pilot GPU-hour, E26c wall-time, selected token budget과
checkpoint storage는 `NOT_ESTIMABLE_UNDER_CURRENT_DATA_LOCK`이다. Target
resource sweep을 일부러 실행하지 않았기 때문이다. Data gate가 prospective
repair로 통과하면 outcome을 보지 않고 다음 순서로 한 번 측정한다.

```text
final data hash
→ numerical/restart preflight
→ target-context microbatch/resource preflight
→ user-approved resource receipt SHA
→ canonical scientific E26a
```

Dry-print launcher는 `scripts/run_e26a_scientific_gate.sh`다. 현재는 필수
readiness/resource path가 없으므로 실행 가능한 canonical command를 제시하는
것 자체가 부정확하다. Script가 출력하는 template은 exact approved resource
SHA까지 요구하며 E26b/E26c 호출을 포함하지 않는다.

## 변경 파일

Stage-2 변경은 data preparation/validation tools, prospective config,
scientific admission/executor, numerical/restart/backend/resource provenance,
schemas, regression tests와 amendment/report 문서에 한정된다. 전체 authoritative
목록은 implementation commit의 다음 명령으로 확인한다.

```bash
git -C /home/minjun_dev/CATENA_E26 show \
  --name-status --format=fuller PENDING_IMPLEMENTATION_COMMIT
```

Raw text, parquet, tokenizer model, memmap, SQLite, checkpoint는 Git에 추가하지
않았다.

## Stop confirmation

```text
scientific_e26a_started: false
scientific_e26b_started: false
scientific_main_started: false
canonical_e26_artifact_created: false
e26a_status: BLOCKED_DATA_SOURCE
user_approval_required: true
```

현재 허용되는 주장은 Stage-2 implementation과 deterministic data construction
일부가 검증됐다는 것뿐이다. Autoregressive LM transfer, Dual superiority,
official GDN2/KDA correspondence, E26b GO 또는 final candidate/budget은
주장할 수 없다.
