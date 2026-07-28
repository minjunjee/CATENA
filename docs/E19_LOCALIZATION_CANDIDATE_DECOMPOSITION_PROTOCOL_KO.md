# E19 Learned Localization / Candidate Decomposition Protocol

## 질문과 범위

E19는 E12–E14를 수정하거나 재판정하지 않고 다음 질문만 평가한다.

> Structured address cue에서 update 위치를 학습하고, erase candidate를
> 현재 state에서 읽어야 할 때 separate-address와 state-read freedom이
> 예측된 조건에서 선택적으로 필요한가?

주소 cue는 고정 random codebook의 slot code이며 write candidate는 모든
조건에서 제공된다. 따라서 이 실험은 controlled learned localization과
state-read candidate recovery이지 semantic inference, 자연어, learned
ontology 또는 official backend 실험이 아니다.

## 네 정보 조건

| 조건 | Address | Erase candidate | 식별할 자유도 |
|---|---|---|---|
| A | Oracle erase/write | Oracle old value | Positive-control floor |
| B | Learned erase/write | Oracle old value | Separate address |
| C | Oracle erase/write | Current-state read | State awareness |
| D | Learned erase/write | Current-state read | Separate address + state awareness |

Erase와 write slot은 모든 episode에서 다르다. Descriptor에는 old value가
없으며, state는 descriptor와 독립적으로 생성된다.

## Controller와 공정성

| Variant | Separate erase/write address | State-read erase candidate |
|---|---:|---:|
| `base` | false | false |
| `separate_address` | true | false |
| `state_aware` | false | true |
| `full` | true | true |

네 variant는 encoder, 두 address head와 candidate head를 모두 동일하게
등록한다. Freedom은 maximal surface의 output projection만 바꾸며, seed
내 initial tensor와 training data order를 pair한다. Main statistical
unit은 다섯 training seed `[101, 211, 307, 401, 503]`다.

## Metrics

각 seed × variant × condition에서 다음을 기록한다.

| Metric | 정의 |
|---|---|
| Address accuracy | Erase/write effective address argmax 정확도의 평균 |
| Candidate recovery MSE | Effective erase candidate와 true old value의 MSE |
| Affected MSE | True erase/write slot correction MSE의 평균 |
| Retention MSE | 두 affected slot을 제외한 state MSE |
| Old residual | True erase slot의 post-update squared residual |
| Architecture extra error | 같은 seed·condition의 `affected_mse - full affected_mse` |

## 사전등록 aggregate gate

| Gate | Estimand | 기준 |
|---|---|---:|
| B separate-address recovery | mean(no-separate) − mean(separate-capable) affected MSE | gain `>=0.001`, 5/5 positive, exact one-sided `p<=0.05` |
| C state-aware recovery | mean(no-state-read) − mean(state-read-capable) affected MSE | gain `>=0.001`, 5/5 positive, exact one-sided `p<=0.05` |
| D full-only maintenance | best incomplete architecture − full affected MSE | gain `>=0.001`, 5/5 positive, exact one-sided `p<=0.05` |
| Retention | 각 treatment−comparison seed contrast | 최대 `<=0.0005` |
| Learned address assay | B separate-capable와 D full | 최소 accuracy `>=0.95` |
| State-read assay | C state-capable와 D full | 최대 candidate MSE `<=0.001` |
| Capable floor | B/C의 capable group과 D full | 최대 affected MSE `<=0.001` |
| A oracle floor | 모든 variant | 최대 affected MSE `<=1e-6` |

E19b conjunction이 모두 통과해야 claim이 열린다. 개별 E19a run은
`PENDING_AGGREGATE`이며, dry-run은 gate evidence가 아니다.

## Artifact와 실행 계약

- E19a main은 seed 하나당 새 UTC run directory 하나를 생성한다.
- 각 E19a run은 16 metric row와 variant checkpoint 4개를 저장한다.
- E19b는 정확한 5-seed source grid, source config, report, manifest,
  metric과 checkpoint hash를 검증한다.
- CPU dry-run은 개발 검증 전용이며 GPU main을 열거나 claim을 만들지 않는다.
- 기존 E12–E14 source, config, report와 artifact는 dependency도 아니며
  수정하지 않는다.

## Claim 경계

성공 시 허용되는 최대 주장은 다음이다.

> In a controlled fixed-slot address-code setting, learned separate
> localization and current-state erase-candidate reads provide selective and
> complementary correction capacity.

Natural-language semantics, novel entity generalization, learned addressing in
pretrained recurrent models, agents, official GDN2/KDA/KVEraser 또는 runtime
superiority 주장은 금지한다.
