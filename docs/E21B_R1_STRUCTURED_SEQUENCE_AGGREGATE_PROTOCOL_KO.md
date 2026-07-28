# E21b-R1 Active-Guardrail Aggregate Repair Protocol

## Disposition과 목적

원본 E21b 구현은 그대로 보존하며 다음 disposition으로만 해석한다.

```text
E21b original: INCONCLUSIVE_GATE_IMPLEMENTATION
reason: inactive state-read non-target comparison and averaged guardrails
E21b-R1: prospective aggregate-only repair
```

E21b-R1은 E21a의 data, model, training, evaluation row, seed, threshold 또는
primary estimand를 바꾸지 않는다. 첫 canonical E21a source report가
생성되기 전에 aggregate rule만 고정하며, 등록된 다섯 source run을
명시적으로 받아 한 번 평가한다.

## 유지되는 primary estimand

다음 세 seed-level full-grid mean gain과 stress-cell 방향, exact sign-flip
검정은 원본과 동일하다.

1. `B × address_decoupling`: no-separate minus separate-capable
2. `C × state_conditioning`: no-state-read minus state-read-capable
3. `D × address_decoupling`: best incomplete minus full

SESOI `0.001`, 5/5 positive seed, one-sided exact sign-flip `p<=0.05` 및
모든 floor/activity/provenance gate도 변경하지 않는다.

## 수리된 non-target guardrail

각 아래 항목에서 variant-group 평균 affected MSE의
`treatment - comparison`을 **각 seed × condition × family × updates × gap
cell에서 먼저 계산**한 뒤 전체 maximum을 사용한다.

| Freedom | 활성 조건 | 비교 | 명시적으로 제외하는 identifying target |
|---|---|---|---|
| Separate address | `B`, `D` | `{separate_address, full}` − `{base, state_aware}` | `B/D × address_decoupling` |
| State read | `C`, `D` | `{state_aware, full}` − `{base, separate_address}` | `C × state_conditioning`, `D × address_decoupling` |
| Full conjunction | `D` | `full` − best incomplete | `D × address_decoupling` |

여기서 best incomplete는 해당 seed의 primary `D × address_decoupling`
full-grid affected MSE가 가장 낮은 단일 incomplete variant로 고정한다.
따라서 test cell별 comparator switching을 허용하지 않는다.

State-read guardrail은 oracle-candidate가 state-read route를 우회하는
`A/B`를 사용하지 않는다. `C/D`에서만 계산하므로 실제 current-state read
projection이 활성화된다.

## 수리된 retention guardrail

Primary context 세 개에서 동일 treatment/comparison을 사용해
`retention_treatment - retention_comparison`을 각 updates×gap cell에서
계산하고 maximum을 사용한다.

- `B × address_decoupling`: separate-capable vs no-separate
- `C × state_conditioning`: state-read-capable vs no-state-read
- `D × address_decoupling`: full vs 위에서 고정한 best incomplete

평균이 다른 cell의 손상을 상쇄할 수 없다. 두 repaired maximum의 margin은
원래 값 `0.0005`를 그대로 사용한다.

## Source와 claim boundary

- 정확히 seeds `113,223,331,449,557`의 explicit E21a MAIN run을 요구한다.
- Source report, manifest, metrics, checkpoint, summary hash와 complete
  `4×4×4×3×4=768` row grid를 원본 validator로 재검증한다.
- Source 재학습, checkpoint 선택, optional stopping 또는 row 제외는 없다.
- 결과는 fixed identifier schema, explicit demand/provenance field가 있는
  `CONTROLLED_REFERENCE` repeated-sequence evidence에만 해당한다.
- H5, semantic/natural-language inference, novel identifier, recurrent LM,
  agent/planning, official backend 또는 runtime claim은 열지 않는다.
