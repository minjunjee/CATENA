# E14 prospective identifiability repair lock

## Freeze timing

- Frozen at: `2026-07-27T18:42:44Z`
- First E14 evaluation artifact existed at freeze: `false`
- Original E14 result reused: `false`
- E13c/E13b result inspected to choose this estimand: `false`
- Repair reason: analytic normalization bound

이 lock은 첫 E14 evaluation 전에 작성됐다. E13c가 `SUPPORTED`가 되기
전에는 E14 main을 실행하지 않는다.

## 발견된 구조적 문제

기존 `plan_correction_gain`은 전체 `64 × 128` table의 평균 MSE 차이였다.
`updates=1`에서 한 entity의 old/new bit 두 개가 모두 변하는 가장 큰
경우에도 perfect correction의 whole-table gain 상한은 다음과 같다.

\[
G_{\mathrm{whole,max}}
=
\frac{2}{64\times128}
=
0.000244140625
<
0.001.
\]

따라서 기존 수치 SESOI를 whole-table estimand에 적용한 all-cell gate는
구조적으로 통과할 수 없었다.

## 동결된 prospective repair

```yaml
primary_estimand: affected_plan_correction_gain
minimum_affected_plan_correction_gain: 0.001
maximum_retention_mse: 0.0005
required_seeds: [101, 211, 307, 401, 503]
required_updates: [1, 4, 8]
required_gap_events: [0, 128, 512, 2048]
required_variant: dual
all_seed_cell_guardrails_required: true
whole_table_gain_role: DESCRIPTIVE_ONLY
```

Affected estimand은 generator가 기록한 verified-update
`affected_entities` mask에서 stale MSE와 assimilated MSE를 같은
분모로 계산한다. SESOI와 retention margin의 숫자는 변경하지 않는다.

## Dependency lock

Main은 latest `e13c_r1_transactional_sequence_aggregate`
`MAIN/PASS/SUPPORTED` run의 source provenance에 봉인된 정확한
`e13b_r1_transactional_sequence_memory` dual checkpoint 다섯 개만
사용한다. Calibration dependency는
`e13a_r2_sequence_floor_throughput`만 허용한다. E13c-R1
report/provenance와 각 E13b-R1 report, metrics, checkpoint hash 및
checkpoint seed/variant/config를 검증한다. 임의 checkpoint override와
filesystem glob 선택은 금지한다.

E14 evaluation은 repaired V2 generator/model interface를 사용한다. Batch
seed에서 gap을 제외하고 base-transaction digest 동일성을 hard-fail
guardrail로 둔다.

## Claim boundary

```text
evidence_tier: CONTROLLED_REFERENCE
proxy_scope: STRUCTURED_SYNTHETIC_ENTITY_VALUE
independent_plan_semantics_tested: false
semantic_inference_tested: false
learned_addressing_tested: false
agent_planning_claim_eligible: false
official_backend_claim_eligible: false
production_break_even_claim_eligible: false
```

Source/config SHA-256는 구현과 config 검증이 끝난 뒤 이 문서 하단에
기록하며, 이후 변경 시 새 protocol version이 필요하다.

## Frozen file hashes

```text
experiments/e14_plan_continuation.py:
262d5519ee0305f9c47bc90a591e295bbb53769c37248d6eb9d503ce191115db

configs/e14_plan_continuation.yaml:
06c3433f774a4420b6f27b1db741267799389189ad44f17eb8c5d7b2cf29d201
```
