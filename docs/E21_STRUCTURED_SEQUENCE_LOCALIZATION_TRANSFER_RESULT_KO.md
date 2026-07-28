# E21 Structured Sequence Localization Transfer 결과 요약

- Source: 5 paired MAIN seed, 각 768 rows·4 checkpoint
- Original E21b: `INCONCLUSIVE_GATE_IMPLEMENTATION`
- Prospective E21b-R1: execution `PASS`, claim `NOT_SUPPORTED`
- Evidence tier: `CONTROLLED_REFERENCE`

## Primary recovery

| Contrast | Mean gain | 양수 seed | Exact p | Gate |
|---|---:|---:|---:|---|
| B: separate-address | 0.0583049 | 5/5 | 0.03125 | PASS |
| C: state-read | 0.1419464 | 5/5 | 0.03125 | PASS |
| D: full-only | 0.0722309 | 5/5 | 0.03125 | PASS |

## Full-conjunction guardrail

| 항목 | 관측값 | 기준 | Gate |
|---|---:|---:|---|
| Max capable affected MSE | 0.00177357 | `<=0.001` | FAIL |
| Max active non-target degradation | 0.000867025 | `<=0.0005` | FAIL |
| Max primary retention degradation | `4.95605e-6` | `<=0.0005` | PASS |
| Max candidate MSE | 0.000120040 | `<=0.001` | PASS |
| Min address accuracy | 1.0 | `>=0.95` | PASS |

최대 affected-floor 실패는 seed 331에서, 최대 non-target 실패는 seed 557의
`D × state_conditioning × updates=1 × gap=0`에서 발생했다. Primary
recovery는 크고 seedwise 일관됐지만 capable path의 absolute accuracy와
cellwise non-target 안정성을 동시에 만족하지 못했다. 따라서 full
structured sequence-transfer claim을 열지 않는다.

Fixed identifier schema와 explicit algebraic demand/provenance field의
controlled result다. H5, semantic/natural-language, novel identifier,
pretrained/recurrent LM, agent/planning, official backend와 runtime claim은
모두 닫혀 있다.

- [R1 run summary](../artifacts/e21b_r1_structured_sequence_localization_aggregate/20260728T091547.163300Z/RESULTS_SUMMARY_KO.md)
- [R1 report](../artifacts/e21b_r1_structured_sequence_localization_aggregate/20260728T091547.163300Z/report.json)
- [Immutable freeze](../artifacts/E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json)
