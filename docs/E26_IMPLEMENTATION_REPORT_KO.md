# CATENA v8.1 E26–E30 구현 보고

## 결론

E26–E30 track은 기존 repository에 파일을 덮어쓰지 않고 별도 worktree에
통합됐다. CPU reference validation과 Blackwell GPU의 100-step non-evidence
smoke가 통과했다. 이 결과는 구현 feasibility만 확인하며 E26a, LM transfer,
official architecture 또는 scientific claim을 열지 않는다.

Scientific E26a 실행은 아직 `BLOCKED_DEPENDENCY`다. 외부에서 고정한 16K
tokenizer와 real-corpus token memmap이 없고, complete E26a parity/data/floor/
throughput runner도 아직 scientific non-dry path에 연결되지 않았다.

## Repository와 불변성

| 항목 | 값 | 판정 |
|---|---|---|
| Live repository | `/home/minjun_dev/CATENA` | immutable |
| Live branch / HEAD | `main` / `adfdeaf9e87a8602a8e334915d87acb9ff25af39` | clean |
| E26 worktree | `/home/minjun_dev/CATENA_E26` | isolated |
| E26 branch | `exp/e26-autoregressive-lm` | clean at smoke |
| Initial integration commit | `28f6b868e9444ad52a6a09b97559691563561d02` | frozen |
| Numerical repair commit | `bb5fd9ae3799652457b15fd038405b43dbbf3d9a` | frozen |
| Pre-E26 tracked source | 556 / 556 unchanged | PASS |
| E00–E21 artifact | 1,329 / 1,329 unchanged | PASS |
| Frozen artifact aggregate | `4ef82dd6fe42543bc1e3b8a63f24781c2ba5d476a39bd84448f9dde6896c87f7` | PASS |
| Canonical E26–E30 artifact | 생성 없음 | PASS |

`adfdeaf9…` 대비 source commit에는 신규 파일 100개가 추가됐고 기존 tracked
파일 수정은 0개다. 구성은 config 14, entry point 14, schema 10, script 7,
`src/catena/lm` 18, test 19, tool 8, 문서 6개다.

## Semantic integration

- 동일 parameter surface의 `dual_delta_lm`과
  `projected_tied_delta_lm`
- compiled fixed-chunk recurrence: inner token loop 없음, graph break/fallback 없음
- recurrent state와 local-attention K/V ring을 함께 clone하는 runtime state
- strict 16K tokenizer/corpus manifest와 paired deterministic cursor
- no-overwrite artifact, strict JSON/JSONL, run-start/completion source fingerprint
- E27–E30 dependency fail-closed scaffold
- reference/synthetic path의 `NON_EVIDENCE_VALIDATION` 강제

상세 semantic diff는 `docs/E26_PACKET_SEMANTIC_DIFF_KO.md`, backend 계약은
`docs/E26_MODEL_BACKEND_SPEC_KO.md`, data 계약은
`docs/E26_DATASET_SPEC_KO.md`를 따른다.

## Validation

Fresh CPU dry-run:

```text
/tmp/catena_e26_dry_revalidation_xeWRhGfj
```

- E26a–E30c 14개 entry point 정상 종료
- artifact validator 14/14 PASS
- JSON Schema 70/70 PASS
- required files 154/154, indexed hash/size 142/142 일치
- E30a–E30c는 unmet dependency로 의도대로 `BLOCKED`
- 모든 run은 `DRY_RUN`, `NON_EVIDENCE_VALIDATION`,
  `scientific_evidence=false`

| Static/unit 검사 | 결과 |
|---|---:|
| Full pytest | 745 PASS, 1 SKIP |
| Ruff check + format check | 신규 Python 59 / 59 PASS |
| strict mypy | `src/catena/lm` 18 / 18 PASS |
| compileall | Python target 326 PASS |
| YAML / 신규 JSON / integrated schema | 14 / 14, 11 / 11, 10 / 10 PASS |
| Packet protocol/schema/entrypoint | 14 / 7 / 14 PASS |
| CLI `--help` / shell `bash -n` | 14 / 14, 7 / 7 PASS |

Pytest의 유일한 skip은 official GDN2 checkout 부재이며, 14개 warning은
TorchScript API deprecation이다. `jsonschema`는 environment를 바꾸지 않고
`/tmp/catena_v81_validation_deps`에서만 주입했다. Run-validator test의 고정
`/tmp` residue는 다음 세션에서 충돌할 수 있어, 기존 residue를 보존 이름으로
옮긴 뒤 clean rerun으로 최종 PASS를 확인했다.

## GPU non-evidence smoke

첫 run은 state-carry numerical gate를 fail-closed로 종료했다.

```text
/tmp/catena_e26_dry_gpu_smoke_20260731T0701Z/
  e26a_operator_data_gate/20260731T070224.840049Z
```

- report SHA: `f64cddaed57ef351997c5c0808dcc0abe2f53813300e2b4525ea62d795c99bbe`
- recurrent carry L2 `0.0108647 > 0.007`
- attention carry L2 `0.00704006 > 0.007`

FP32 분해 진단으로 state semantics가 정상임을 확인한 뒤, threshold·seed·data·
metric을 바꾸지 않고 FFN output projection만 FP32 accumulation으로 고정했다.
원본 run은 그대로 보존했으며 amendment는
`docs/E26A_NON_EVIDENCE_AMENDMENT_001_KO.md`에 기록했다.

Fresh repair run:

```text
/tmp/catena_e26_dry_gpu_smoke_repair_20260731T0716Z/
  e26a_operator_data_gate/20260731T071132.954822Z
```

| 항목 | 관찰값 | 판정 |
|---|---:|---|
| Gate | 20 / 20 | PASS |
| Parameters | 42,025,616 | 35–50M PASS |
| Tied/Dual signature, init, optimizer state | identical | PASS |
| Hybrid output/recurrent/attention carry L2 | 0 / 0 / 0 | PASS |
| FP32 optimized/reference output L2 | `2.17e-7` | PASS |
| BF16/FP32 output/state L2 | `0.005779` / `0.005881` | PASS |
| Gradient norm | `14.7302`, finite | PASS |
| Graph break / fallback | 0 / 0 | PASS |
| Throughput | 1,963.16 tokens/s | descriptive |
| Peak allocated/reserved | 0.988 / 1.170 GB | descriptive |
| Checkpoint save/load | 0.951 / 0.116 s | `/tmp` only |

- disposition: `NON_EVIDENCE_100_STEP_SMOKE_PASS`
- report SHA:
  `aa50a0f75da4a981956f94cfbe575e3c6281242bc1963c747567c0dd01dd7d07`
- backend manifest SHA:
  `822221639c79abd232500b8f35de887d0c0256d21aec2f50dc5285f68eb05adc`
- artifact contract validation: PASS

## Data와 storage plan

외부 scientific input은 내려받거나 생성하지 않았다. 승인 후 준비할 shared
uint16 token memmap은 250M/300M/400M/500M token에서 각각 약
0.466/0.559/0.745/0.931 GiB다. 80% general, 20% transaction stream이며 10개
model run 사이에서 복제하지 않는다.

실측 model+AdamW checkpoint는 504,416,153 bytes(0.470 GiB)다. 10개 final
checkpoint는 약 4.70 GiB이고, 25M-token cadence를 모두 보존하면 token
budget에 따라 약 47.0–94.0 GiB가 필요하다.

## Stop confirmation

```text
scientific_main_started: false
canonical_scientific_artifact_created: false
e26a_status: BLOCKED_DEPENDENCY
user_approval_required_before_e26a: true
```
