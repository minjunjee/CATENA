# CATENA Post-E21 Dependency DAG

```text
E21b-R1 immutable
  -> E22a development-only locality selection
  -> E22b fresh 8-seed confirmatory

E18b immutable
  -> explicit SUPPORTED freeze
  -> E23a learned 16-controller screen
  -> explicit completed E23a screen provenance
  -> E23b learned confirmatory
       ├─ E22b safe PASS     -> safe-minimality
       ├─ E22b completed fail -> capacity-only
       └─ E22b absent         -> BLOCKED_DEPENDENCY

H1 + E10b + E11b immutable
  ├─ E24a approximate-rank stress
  └─ E24b behavioral-attainability stress

pinned GDN2 + pinned FLA + separate environment
  -> E25a source/parity gate
  -> explicit gate report + user replication authorization
  -> E02b/E18 official minimal replication
  -> E22 safe report present
       ├─ selected objective/route implemented -> E22 official subset
       └─ not implemented -> BLOCKED_DEPENDENCY

E25b v4 protocol lock
  -> 300-item immutable audit preparation
  -> external review-work copy + two-reviewer audit
  -> leakage/floor dry gate
  -> E25b main
```

## 즉시 가능한 구현 검증

E22a, E23a, E24a/E24b, E25a dry contract, E25b development dry-run은
서로 독립적이다. 이 단계의 모든 dry-run은 `/tmp`에만 쓰고
`claim_eligible=false`다.

## Main 차단 규칙

- E22b는 명시적으로 봉인된 E22a selection artifact 하나가 필요하다.
- E23a MAIN은 명시적 E18b `PASS/SUPPORTED` freeze가 필요하다.
- E23b MAIN은 같은 E18b freeze, 완료된 E23a screen, 명시적 E22b report를
  모두 요구한다. E23a outcome은 boundary 선택에 사용하지 않는다.
- E25a replication은 explicit PASS gate report와
  `--allow-scientific-replication` 없이는 실행되지 않는다.
- E25b main은 v4 population lock에 묶인 외부 review-work CSV의 정확히
  300행 completed two-reviewer audit가 필요하다.
- Official dependency가 없거나 revision/hash가 다르면 reference
  backend로 대체하지 않는다.

## 구현 완료 시점의 terminal 상태

- E25a parity-only gate는 official source가 구성된 상태에서 `FAIL`이다.
  BF16 relative L2와 tied-to-KDA parity가 등록 tolerance를 넘었으므로
  threshold를 바꾸거나 replication으로 진행하지 않는다.
- E25b v4는 train에서 `ADD`/`INVALIDATE`만 사용하고, primary의
  `SUPERSEDE`를 held-out composition으로 둔다. Asymmetric tied-to-dual
  gain과 symmetric tied/dual equivalence를 별도 gate로 판정한다.
  Old-rule query는 `FULL/PARTIAL/NONE`으로 평가하고 oracle도 동일
  evaluator를 통과한다.
- E22–E24와 E25b scientific MAIN은 아직 실행하지 않았다.

## Claim ceiling

| 경로 | 최대 claim |
|---|---|
| E22 | controlled safe localized assimilation |
| E23 | learned repeated sequence의 declared 4-axis empirical epsilon-minimality |
| E24 | controlled diagnostic의 construction robustness |
| E25a gate | pinned official operator numerical parity |
| E25a replication | registered official magnitude subset; locality는 selected route 구현 시에만 |
| E25b | shared frozen text representation의 controlled transaction anchor |

Pretrained recurrent LM, free-form agent, Transformer/KVEraser superiority와
production latency claim은 이 DAG 어디에서도 열리지 않는다.
