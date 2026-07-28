# E14 — Structured table-state continuation proxy

## 질문과 범위

Repaired E13b-R1의 structured entity-value state를 stale plan-field
proxy로 해석했을 때, 동결된 dual controller가 verified update 이후 실제로
영향을 받은 field를 교정하고 learned distractor no-op 경로에서 untouched
field를 보존하는가?

이 실험은 별도의 planner, action decoder, tool execution, natural-language
input을 포함하지 않는다. 정확한 evidence 범위는
`CONTROLLED_REFERENCE / STRUCTURED_SYNTHETIC_PROXY`다.

## Hard dependency

Main E14는 latest `e13c_r1_transactional_sequence_aggregate` run이 다음
조건을 모두 만족할 때만 실행한다.

1. `run_mode=MAIN`, execution `PASS`, `claim_gate.supported=true`
2. 고정 seed `101, 211, 307, 401, 503`
3. tied/dual 두 variant와 완전한 `3 updates × 4 gaps` grid
4. E13c report의 source provenance JSONL과 report 내 provenance가 동일
5. 각 E13b report, metrics, checkpoint hash가 E13c seal과 동일
6. 각 checkpoint의 경로, seed, variant, payload config가 등록 contract와
   동일

E14는 glob이나 lexicographic order로 checkpoint를 고르지 않는다. E13c가
봉인한 source 열 개 중 정확히 다섯 dual checkpoint만 seed 순서대로
사용한다. Dependency와 checkpoint는 평가 전·후에 다시 hash한다.

## 평가 설계

모든 다섯 training seed에 대해 다음 전체 grid를 평가한다.

| Axis | 고정값 |
|---|---|
| Updates | `1, 4, 8` |
| Distractor gaps | `0, 128, 512, 2048` |
| Evaluation batches | cell당 32 |
| Batch size | 256 |

각 batch seed는 checkpoint seed와 gap 길이에 의존하지 않는다. Repaired V2
generator의 base transaction stream과 distractor stream 분리를 이용하므로,
다섯 checkpoint와 네 gap cell은 동일한 initial state, verified update,
target을 본다. Base-transaction digest가 update별로 모든 checkpoint와 gap
사이에서 같아야 한다. Batches와 episodes는 독립적인 statistical unit으로
취급하지 않는다.

## Prospective identifiability repair

첫 E14 평가 전에 기존 gate의 구조적 문제를 발견했다. Whole-table MSE에서
`updates=1`인 cell의 correction gain은 완벽한 모델에서도 최대

\[
\frac{2}{64\times128}=0.000244140625
\]

다. 이는 사전 설정된 SESOI `0.001`보다 작으므로, whole-table gain으로 모든
cell 통과를 요구하면 claim은 원천적으로 식별 불가능하다.

E14 artifact가 하나도 생성되지 않은 상태에서 다음을 prospective하게
고정했다.

- SESOI 수치 `0.001`은 변경하지 않는다.
- Primary estimand를 affected-entity correction gain으로 명시한다.

\[
G_{\mathrm{affected}}
=
\operatorname{MSE}_{\mathrm{stale,affected}}
-
\operatorname{MSE}_{\mathrm{assimilated,affected}}
\]

- Whole-table correction gain은 descriptive metric으로만 보존한다.
- Untouched retention margin `0.0005`는 변경하지 않는다.
- 다섯 seed의 60개 cell이 모두 두 guardrail을 통과해야 한다.

상세 동결 기록은
`E14_PROSPECTIVE_IDENTIFIABILITY_REPAIR_LOCK_KO.md`에 있다.

## Primary gate와 guardrail

모든 seed×update×gap cell에서 다음을 동시에 요구한다.

1. `affected_plan_correction_gain >= 0.001`
2. `unaffected_plan_retention_mse <= 0.0005`
3. affected/unaffected denominator가 모두 양수
4. 모든 metric이 finite
5. 정확히 5 seed × 3 update × 4 gap = 60개 unique cell

한 cell이라도 누락되거나 실패하면 E14 claim은 열리지 않는다. Dry-run은
dependency plumbing만 검증하며 `supported=false`다.

## 해석 제한

Repaired E13b-R1에서는 distractor `verified` bit가 encoder input일 뿐
hard mask로 update를 차단하지 않는다. 따라서 untouched retention은 learned
no-op path의 guardrail이지만, address와 old/new candidate가 여전히 oracle인
만큼 learned localization이나 일반적인 collateral safety 증거로 해석하지
않는다. Long gap 역시 language-model memory persistence 결과가 아니다.

성공 시 허용되는 문장은 다음 범위다.

> Across five sealed training seeds, the dual controller corrected stale
> affected fields while preserving untouched fields in a structured synthetic
> entity-value continuation proxy.

다음 claim은 명시적으로 금지한다.

- independent plan semantics 또는 downstream action quality
- semantic demand inference 또는 learned addressing
- natural-language/recurrent language-model transfer
- general agent planning 또는 tool orchestration
- official GDN2/KDA backend transfer
- production latency 또는 break-even
