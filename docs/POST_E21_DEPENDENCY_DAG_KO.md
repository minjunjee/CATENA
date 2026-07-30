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

## Scientific MAIN 완료 시점의 terminal 상태

- E25a parity-only gate는 official source가 구성된 상태에서 `FAIL`이다.
  BF16 relative L2와 tied-to-KDA parity가 등록 tolerance를 넘었으므로
  threshold를 바꾸거나 replication으로 진행하지 않는다.
- E22a는 `smoothmax_100`을 development-only 방법으로 선택했다.
  Fresh 8-seed E22b는 recovery 방향과 retention을 유지했지만 absolute
  capacity, worst-cell locality와 paired locality-improvement gate가
  실패해 `NOT_SUPPORTED`다.
- E22b가 safe PASS가 아니므로 E23b는 사전 규칙대로 `capacity_only`
  mode로 실행됐다. E23a outcome은 boundary 선택에 사용하지 않았다.
  E23b는 directional predecessor/incomparable contrast를 보였지만
  absolute adequacy와 minimal-set recovery가 실패해 `NOT_SUPPORTED`다.
- E24a는 descriptor-conditioned learned controller의 OOD spectrum-family
  transfer를 지지하지 않았고, E24b는 construction-robust behavioral
  prediction gate를 통과하지 못했다.
- E25b는 immutable 300-item audit population, Reviewer A/B CSV와 검수
  도구까지만 생성된 `AUDIT_PENDING` 상태다. 실제 독립 인간 2인의
  audit 전에는 leakage/floor gate와 scientific MAIN이 차단된다.

실제 완료 경로:

```text
E21b-R1
  -> E22a SELECTED
  -> E22b NOT_SUPPORTED
       -> E23b capacity_only
            -> E23b NOT_SUPPORTED

E18b SUPPORTED
  -> E23a SCREEN_ONLY
  -> E23b completed

H1/E10b/E11b
  -> E24a OOD transfer NOT SUPPORTED
  -> E24b construction-robust prediction FAILURE

Official source
  -> E25a parity FAIL (terminal)

E25b audit population
  -> audit package complete
  -> two independent humans required
```

Canonical run, SHA와 gate별 수치는
`docs/POST_E21_WAVE1_RESULTS_KO.md`와
`docs/POST_E21_WAVE1_ARTIFACT_AUDIT_KO.md`에 고정했다.

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
