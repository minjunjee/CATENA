# E22 Active-Path Locality 사전 프로토콜

상태: `PROTOCOL_FROZEN_BEFORE_MAIN`  
증거 등급: `CONTROLLED_REFERENCE`

## 질문과 원본 보존

E21b-R1은 primary recovery를 통과했지만 capable affected floor와 active
non-target locality guardrail을 통과하지 못했다. E22는 E21을 재평가하거나
소급 수정하지 않는다. 다음의 새로운 질문만 다룬다.

> 평균 retention 대신 tail-aware 또는 실제 sparse-route locality objective를
> 사용하면, E21의 B/C/D recovery를 유지하면서 active non-target cell의
> worst degradation을 줄일 수 있는가?

E22도 fixed identifier codebook, explicit algebraic demand code와
model-visible provenance bit를 사용한다. H5, 자연어, novel identifier,
pretrained/recurrent LM, agent/planning, official backend와 runtime claim은
열지 않는다.

## Threshold 상속

새 YAML에는 SESOI나 margin 숫자를 기록하지 않는다. 두 entry point는 실행
시점에
`docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json`의
`registered_thresholds` 전체를 읽는다. 정적 E22 lock에는 다음을 기록한다.

1. E21 parent lock의 exact SHA-256
2. 런타임에 읽어야 할 threshold key 전체
3. 당시 상속된 값의 snapshot
4. E22 source/config/protocol의 byte SHA-256

Parent hash나 값이 하나라도 달라지면 실행 전 중단한다.

## E22a — Development-only 방법 선택

독립 development seed는 `1201, 1213, 1229` 세 개다. 모든 방법은 동일
initialization, data order, maximal controller surface와 evaluation
transaction을 공유한다.

각 method는 E21의 `base`, `separate_address`, `state_aware`, `full` 네
variant를 모두 독립 학습·평가한다. B/C/D recovery와 active non-target 및
retention guard는 `E21b-R1`의 contrast와 cellwise maximum 코드를 그대로
재사용해 method별로 다시 계산한다. Full controller 하나의
method-vs-mean 차이는 recovery estimand로 사용하지 않는다.

| Family | 등록 grid |
|---|---|
| Mean | `mean_retention` baseline |
| CVaR | tail `0.05, 0.10, 0.20` |
| Smooth maximum | normalized temperature `0.25, 0.50, 1.00` |
| Sparse route | active fraction `0.125, 0.25, 0.50`, CVaR tail `0.10` |
| Protected | active non-target readout-subspace projection oracle diagnostic |

CVaR은 unaffected entity error의 등록 upper-tail mean이다. Smooth maximum의
실제 temperature는 `normalized_temperature × inherited
maximum_nontarget_degradation`다. 즉 dimensionless normalization scale은
상속 locality margin `5e-4`이며 capable-floor `1e-3`를 사용하지 않는다.
Sparse route는 CVaR `0.10`에 distractor activity penalty(weight `0.25`)를
더할 뿐 아니라, 각 event의 erase/write address route에
`k=max(1, ceil(active_fraction × slots))` hard top-k mask를 실제 forward
update 전에 적용한다. 학습 estimator는
`hard_mask + soft_address - stop_gradient(soft_address)`인
hard-forward/soft-backward straight-through 방식으로 고정하고, masked
address weight를 합 1로 다시 정규화한다. 평가에는 raw applied route mask,
activity-thresholded active route mask와 support, post-mask event-update RMS,
최종 update RMS 및 compute proxy를 모두 기록한다. 선택 tie-break의
`active-path support`는 active architectural cell에서 실제 적용된 raw
route mask의 평균 support이고, compute proxy는 activity threshold와
무관하게 실제 masked forward가 수행한 route/update 연산량이다.
Protected diagnostic은 associative state-slot basis에서 각 verified event의
oracle erase/write target subspace만 남기고, active non-target readout
subspace의 update 성분을 recurrent state carry 전에 제거한다. Distractor
event에는 등록 target이 없으므로 update를 0으로 projection한다. Controller
freedom 자체를 condition별로 바꾸지 않는다.

Protected projection은 구현 가능성 진단일 뿐 선택 대상이 아니다. Mean도
baseline라 선택 대상이 아니다. 나머지 후보는 다음을 만족해야 한다.

- mean B/C/D recovery gain 각각 `>= selective_gain`
- primary-context retention non-inferiority

Hard gate는 위 B/C/D mean recovery SESOI와 primary-context retention뿐이다.
Seed-direction, stress-direction, address/candidate/capable-path absolute 값은
E22a에서 diagnostic으로 기록하고 E22b의 confirmatory gate에서 판정한다.
통과 후보는 validation maximum active
non-target degradation, smaller active route support, lower update compute,
lexicographic method id 순으로 deterministic하게 선택한다. Active
non-target margin 자체는 선택 hard gate가 아니라 E22b의 독립 판정
대상이다. 통과 후보가 없으면
`NO_SELECTION`이고 E22b MAIN은 차단된다. E22a는 어떤 경우에도
confirmatory claim을 열지 않는다.

E22a MAIN은 명시적인
`--parent-e21-freeze E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json`
dependency를 요구한다. E21 freeze는 immutable, E21b-R1
`NOT_SUPPORTED`, exact parent lock binding을 모두 만족해야 한다.

## E22b — Prospective confirmatory

E22b MAIN은 완료된 E22a MAIN run을 `--selection-run`으로 명시해야 한다.
`selection_lock.json`, raw/seed metrics, selection scores, report와 manifest의
hash chain을 재검증한다. E22a dry-run은 E22b MAIN을 열 수 없다.

Confirmatory paired seeds는 development와 겹치지 않는 다음 8개다.

```text
1301, 1319, 1327, 1361, 1381, 1409, 1423, 1451
```

비교는 선택된 방법 대 `mean_retention`이며, 두 방법 각각의 네 E21
variant를 다시 학습한다. Train은 update 4회·gap 128,
test grid는 update `(1,4,8)` × gap `(0,128,512,2048)`다. 두 방법은 seed
내 initialization, train/eval stream과 parameter surface를 공유한다.

### Recovery/capacity gate

선택 method 자체에서 E21의 세 estimand를 그대로 요구한다.

- `B`: separate-address recovery
- `C`: state-read recovery
- `D`: full-only conjunction recovery
- 각 mean gain `>= inherited selective_gain`
- 각 양수 방향 `8/8`, one-sided exact sign-flip `p <= inherited alpha`
- 세 stress gain 모두 `8/8` 양수
- capable affected/address/candidate/oracle/activity absolute guard 통과

### Locality/retention gate

- E21b-R1 정의의 모든 active non-target cell에서 selected method의
  cellwise maximum degradation `<= inherited maximum_nontarget_degradation`
- identifying target의 cellwise maximum retention degradation
  `<= inherited retention_noninferiority`
- selected-vs-mean 비교는 seed별
  `mean.max_nontarget - selected.max_nontarget`로 계산하며 양수 `8/8`,
  one-sided exact sign-flip `p <= inherited alpha`
- distractor activity가 상속 기준 통과
- exact 8-seed paired grid와 data/protocol/checkpoint provenance 완결

최종 status는 다음 4-way enum을 그대로 사용한다.

```text
SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED
CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED
OVERREGULARIZED_LOCALITY_TRADEOFF
NOT_SUPPORTED
```

첫 status만 claim eligible이다. Dry-run도 관찰 gate에 따른 같은 enum을
출력하되 `dry_run_non_evidence=true`, `supported=false`로 고정한다.
Dependency validation 실패는 `BLOCKED_DEPENDENCY` pre-run 상태이며 새
scientific artifact를 만들지 않는다.

## Artifact 계약

모든 run은 새 UTC directory를 사용하고 최소 다음을 포함한다.

```text
protocol_lock.json
config.resolved.yaml
environment.json
run_manifest.json
data_manifest.json
raw_metrics.jsonl
seed_metrics.jsonl
active_cell_metrics.jsonl
RESULTS_SUMMARY_KO.md
report.json
checkpoints/
```

E22a는 `selection_scores.jsonl`, `selection_lock.json`을 추가하고 E22b는
`selection_provenance.json`을 추가한다. Summary는 55줄 이하다.
Dry-run artifact root는 `/tmp` 아래 fresh path만 허용한다. 본 프로토콜
구현·검증 단계에서는 canonical MAIN을 실행하지 않는다.
