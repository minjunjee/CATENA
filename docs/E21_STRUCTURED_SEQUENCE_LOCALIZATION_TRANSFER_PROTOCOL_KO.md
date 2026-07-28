# E21 Structured Sequence Localization / State-Read Transfer Protocol

## 목적과 선행 결과의 경계

E21은 E18 또는 E19를 수정하거나 재판정하지 않는 독립 prospective
bridge다. 질문은 다음 하나로 제한한다.

> 고정된 structured identifier schema와 explicit update-demand fields를
> 공유 event encoder가 처리할 때, E19에서 분리한 learned localization과
> current-state erase-candidate read의 선택적 이득이 E18식 repeated
> sequence와 distractor gap에서도 유지되는가?

E21은 operation label이나 자연어 의미에서 demand를 추론하지 않는다.
Demand algebra는 structured field로 명시되며 old value는 event input에서
제외된다. 따라서 H5를 다시 열거나 semantic factorization을 검정하지
않는다.

```text
e21a: one paired seed, four variants, full condition/family/test grid
e21b: exactly five e21a MAIN sources의 paired aggregate
current_status: PROTOCOL_LOCKED_BEFORE_DRY_OR_MAIN
authorized_execution: CPU_DRY_ONLY
```

## 정보 조건과 controller projection

E19의 네 정보 조건을 그대로 유지한다.

| 조건 | Address | Erase candidate |
|---|---|---|
| `A` | Oracle erase/write | Oracle old value |
| `B` | Learned erase/write | Oracle old value |
| `C` | Oracle erase/write | Current-state read |
| `D` | Learned erase/write | Current-state read |

네 variant는 event encoder, 두 address head, candidate head와 activity head를
모두 동일하게 등록한다. 차이는 forward projection뿐이다.

| Variant | Separate erase/write address | Current-state candidate read |
|---|---:|---:|
| `base` | false | false |
| `separate_address` | true | false |
| `state_aware` | false | true |
| `full` | true | true |

각 paired seed에서 네 variant는 동일 initialization tensor, training
stream, condition/family round-robin order, optimizer와 evaluation stream을
사용한다.

## Structured sequence와 leakage boundary

Model-visible event field는 다음으로 한정한다.

- source와 destination identifier의 고정 random code
- incoming normalized value
- E18의 네 demand-family one-hot
- magnitude-operation one-hot 또는 value-channel mask
- same/different-address relation field
- provenance verification bit

Slot integer, old value와 `update_mask`는 model input에 포함하지 않는다.
Learned-address condition은 identifier code에서 고정 slot schema를
복구한다. Train/test가 동일 codebook을 사용하므로 novel identifier 또는
ontology generalization은 검정하지 않는다.

E18의 demand family와 target algebra를 유지한다.

| Demand family | Structured target rule |
|---|---|
| `magnitude_factorization` | preserve/add/invalidate/supersede scalar demand |
| `value_granularity` | contiguous value-coordinate mask |
| `address_decoupling` | erase와 write slot이 다름 |
| `state_conditioning` | current old-value sign에 따른 erase/write branch |

Verified event는 첫 위치와 gap block 뒤에 배치한다. Distractor도 encoder와
activity head를 통과하며 `update_mask`로 hard masking하지 않는다. Target
state는 verified event만 반영한다.

## 고정 train/evaluation grid

| 항목 | 고정값 |
|---|---|
| Main seeds | `113, 223, 331, 449, 557` |
| Excluded development seed | `99121` |
| Train | updates `4`, gap `128`, steps `3000`, batch `128` |
| Test updates | `1, 4, 8` |
| Test gaps | `0, 128, 512, 2048` |
| Variants × conditions × families | `4 × 4 × 4` |
| Rows per e21a source | `768` |
| Main statistical unit | paired training seed |

Identifier, train, evaluation과 distractor namespace는 config에 독립적으로
고정한다. Base verified transaction digest는 gap, condition과 variant에
걸쳐 같아야 한다.

## E21b 사전등록 gate

Primary selective contrasts는 다음과 같다.

1. `B × address_decoupling`: no-separate minus separate-capable affected MSE
2. `C × state_conditioning`: no-state-read minus state-read-capable affected MSE
3. `D × address_decoupling`: best incomplete minus `full` affected MSE

각 contrast는 full test-grid mean gain `>=0.001`, 5/5 positive seed,
one-sided exact sign-flip `p<=0.05`를 모두 만족해야 한다. 동일 contrast를
stress cell `(updates=8, gap=2048)`에서도 5/5 positive로 요구한다.

Guardrail은 다음과 같다.

| Gate | 기준 |
|---|---:|
| Non-target affected-MSE degradation | `<=0.0005` |
| Maximum retention degradation | `<=0.0005` |
| Learned-address capable accuracy | `>=0.95` |
| State-read capable candidate MSE | `<=0.001` |
| Capable affected MSE | `<=0.001` |
| Oracle-information floor | `<=0.001` |
| Verified activity mean | `>=0.95` |
| Distractor activity mean | `<=0.05` |
| Source grid/provenance/checkpoint/hash pairing | exact |

Main 또는 aggregate가 완료되면 해당 run directory에 1페이지 이내
`RESULTS_SUMMARY_KO.md`를 반드시 생성한다. E21a main summary는
`PENDING_AGGREGATE`, E21b만 최종 gate status를 기록한다. CPU dry summary는
명시적으로 비증거로 표시한다.

## Claim boundary

성공 시 허용되는 최대 주장은 다음이다.

> In controlled repeated structured-event sequences with a fixed identifier
> schema and explicit demand fields, learned separate localization and
> current-state erase-candidate reads selectively recover the update demands
> that require them across the registered sequence lengths and distractor
> gaps.

Evidence tier는 `CONTROLLED_REFERENCE`, `scientific_evidence=false`다.
Natural language, semantic demand inference, novel identifier generalization,
pretrained/recurrent language model, agent/planning, official backend 또는
runtime superiority 주장은 금지한다.
