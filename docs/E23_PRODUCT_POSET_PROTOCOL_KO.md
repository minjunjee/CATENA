# E23 Product-Poset Sequence Protocol

## 연구 질문

> Magnitude, value granularity, address separation, state conditioning의
> 네 자유도가 독립적인 Boolean product poset을 이룰 때, demand가 요구하는
> 최소 controller가 sequence stress에서도 이론적 absolute-adequacy
> poset-minimal element와
> 일치하는가?

E23은 E18의 실제 repeated-sequence tensor, maximal head, AdamW
training/evaluation contract를 확장하는 별도 `CONTROLLED_REFERENCE`
실험이다. E18/E21의 원 판정을 재판정하지 않는다. 사용자 제공 generic 파일은
`mocks/post_e21_packet/{experiments,configs}/` 아래에 원문 그대로 보존한
interface mock이며 scientific entry point로 사용하지 않는다.

## 고정 controller poset

Bit 순서는 다음으로 고정한다.

```text
(magnitude, value, address, conditioning)
```

따라서 controller는 `c0000`부터 `c1111`까지 정확히 16개다.
부분순서는 componentwise Boolean order다.

```text
c_abcd <= c_efgh
iff
a<=e, b<=f, c<=g, d<=h
```

Demand는 다음 11개다.

- single axis 4개: `magnitude`, `value`, `address`, `conditioning`
- pairwise 6개: 네 축의 모든 2-combination
- guardrail 1개: `preserve`

각 demand의 theory-minimal controller는 해당 required bit만 1인 유일한
controller다. `preserve`의 최소 원소는 `c0000`이다.

## 고정 grid

| 축 | 등록값 |
|---|---|
| intensity | `0.25, 0.5, 1.0` |
| updates | `1, 4, 8` |
| gap events | `0, 512, 2048` |
| E23a screen seed | `2301, 2311, 2333` |
| E23b confirmatory seed | `2401, 2411, 2423, 2437, 2441, 2459, 2473, 2477` |

E23a와 E23b seed는 겹치지 않는다.

## Learned application contract

Application metric은 theory floor에 noise를 더해 만들지 않는다. 각 seed에서
동일한 maximal parameter surface를 한 번 초기화한 뒤, 그 state dict를
`c0000`부터 `c1111`까지 16개 controller에 정확히 복제한다. 네 bit는
forward projection만 제한한다.

| bit | 0 | 1 |
|---|---|---|
| magnitude | erase/write tied | erase/write independent |
| value | scalar value gate | value-diagonal gate |
| address | erase address에 write | separate write address |
| conditioning | state read 0-mask | current old-state read |

모든 controller는 동일 entity/event encoder, 동일 `2+2D` maximal head,
동일 parameter count를 가진다. 같은 seed 안에서는 initialization,
training/evaluation data order, optimizer와 steps가 동일하다.

```text
state: 32 x 32
embedding: 128
hidden: 512
optimizer: AdamW
train: 3,000 steps, batch 128, updates 4, gap 128
eval: 2 batches x 32, registered intensity/update/gap grid
```

Pairwise demand는 E18의 네 axis field를 두 개 동시에 활성화한다. Oracle
address/candidate, explicit algebraic descriptor, model-visible verified bit
경계는 E18과 동일하게 유지된다. Conditioning demand에서는 operation
one-hot을 제거하고 current state branch로 demand를 결정한다.

Raw application row는 checkpoint와 initialization hash와 함께 다음 observed
metric을 기록한다.

- affected-state MSE와 no-update 대비 correction gain
- unaffected retention mean MSE
- 실제 관측된 모든 unaffected entity 중 worst-cell MSE
- paired base-transaction digest

`theoretical_affected_mse`는 outcome-independent diagnostic으로만 남으며
`affected_mse`를 생성하거나 대체하지 않는다.

## Absolute-adequacy minimal estimand

각 seed×demand에서 등록 intensity/update/gap 전체의 worst-cell metric을
집계한다. Capacity adequacy는 다음 절대 조건으로 정의한다.

```text
A_capacity = {
  c:
  max affected_mse(c) <= 0.0001
  and max retention_mse(c) <= 0.0005
}
```

Safe mode에서는 여기에 다음을 추가한다.

```text
max active_nontarget_degradation(c) <= 0.0005
```

Primary set은 이 absolute adequate set의 poset-minimal element다. 관측된
best error에 `epsilon`을 더하는 relative rule은 application 판정에 사용하지
않는다. Exact minimal-set match, Jaccard, theory-adequate set 대비
false-adequate/false-inadequate 수를 모두 기록한다.

Confirmatory boundary는 결과 독립적으로 다음 역할의 합집합을 사용한다.

1. theory-minimal controller
2. required bit 하나를 제거한 immediate lower cover
3. 아직 필요하지 않은 bit 하나를 추가한 immediate upper cover
4. 같은 cardinality의 incomparable alternative
5. maximal controller `c1111`

11개 demand boundary의 union은 16 controller 전체다. 따라서 16개 모두
학습하지만, demand별 판정은 미리 고정된 해당 boundary에서만 수행한다.
E23a 결과, observed MSE, seed 방향, E22b 수치는 이 선택에 사용하지 않는다.

Seed-level primary gate는 다음과 같다.

- single-axis exact match `4/4`
- pairwise exact match 최소 `5/6`
- 모든 immediate predecessor가 absolute inadequate이고 target보다
  affected MSE가 최소 `0.0001` 큼
- same-rank incomparable alternative의 affected-error 방향이 target보다 큼
- `c1111`이 single-axis와 preserve에서 adequate이며 target 대비 degradation
  `<=0.0005`

## E23a — learned screen

Entry point:

```text
experiments/e23a_product_poset_screen.py
```

MAIN은 `--e18-freeze`로 immutable E18b `PASS/SUPPORTED` freeze를 명시해야
한다. 세 등록 seed와 16개 learned controller의 full grid를 학습·검사한다.
E23a는 MAIN으로 실행하더라도 screen diagnostic일 뿐 confirmatory claim을
열 수 없고 E23b boundary를 바꿀 수 없다.

Dry-run은 canonical E18 artifact를 읽지 않는 synthetic dependency fixture와
첫 seed, 2 training step, intensity/update/gap 첫 cell만 사용한다. 모든 16
projection의 training/checkpoint/evaluation path는 실제로 통과하지만
`claim_eligible=false`다.

## E23b — confirmatory dependency

Entry point:

```text
experiments/e23b_product_poset_confirmatory.py
```

MAIN은 다음 세 explicit dependency를 모두 요구한다.

```text
--e18-freeze  immutable E18b PASS/SUPPORTED freeze
--e23a-screen completed E23a MAIN report
--e22b-run    completed E22b report
```

`latest.json`이나 glob으로 선택하지 않는다. E23a report는 pipeline
provenance와 E18 freeze 일치만 확인하며, screen outcome은 confirmatory
boundary·threshold·claim wording을 바꾸지 않는다.
E22b report의 `protocol_lock.sha256`은 E23b lock에 포함된 exact frozen
E22b protocol SHA-256과 일치해야 하며, 이 값은 dependency payload와
report에 보존한다.

| E22b 상태 | E23b 고정 boundary |
|---|---|
| 완료된 safe PASS | `safe_minimality` |
| 완료된 non-safe disposition | `capacity_only` |
| report 없음, 미완료, contract 불일치 | `BLOCKED_DEPENDENCY` |

Safe PASS는 다음 두 조건을 동시에 요구한다.

```text
claim_gate.status =
SUPPORTED_SAFE_LOCALIZED_ASSIMILATION_CONTROLLED

claim_gate.supported = true
```

완료된 non-safe status는
`CAPACITY_SUPPORTED_LOCALITY_NOT_SUPPORTED`,
`OVERREGULARIZED_LOCALITY_TRADEOFF`, `NOT_SUPPORTED`로 고정한다.
`capacity_only`에서는 locality claim을 계산하거나 열지 않는다.

`safe_minimality`에서는 E22b `phase_dependency.selected_method`를 frozen
E22a method grid와 exact-match한 뒤 동일 retention objective를 학습에
사용한다. 현재 faithful wiring은 `mean`, `cvar`, `smoothmax`다. E22에서
`sparse`가 선택되면 hard top-k route semantics가 E23 controller에 없으므로
surrogate penalty로 대체하지 않고 `safe_objective_implemented=false`,
`BLOCKED_DEPENDENCY`로 닫는다. `protected_projection`은 애초 선택 비대상이다.
Non-safe E22 disposition은 항상 frozen `mean_retention` objective로
capacity-only 학습을 수행한다. Method payload와 risk scale은 manifest,
training/raw row, report에 기록한다.

Dry-run은 내부 synthetic non-evidence dependency를 사용할 수 있지만
반드시 `claim_eligible=false`와 `DRY_RUN_ONLY`를 기록한다.

## Prospective lock 전략

두 lock은 별도로 유지한다.

```text
docs/E23A_PRODUCT_POSET_SCREEN_LOCK.json
docs/E23B_PRODUCT_POSET_CONFIRMATORY_LOCK.json
```

각 lock은 다음을 포함한다.

- `schema_version=1`
- 정확한 experiment ID
- `protocol_frozen_before_main=true`
- `main_execution_started=false`
- config와 E23 전용 source의 SHA-256 map
- E18-compatible learned data/model/training source의 SHA-256 map
- 16 controller, 11 demand, absolute-adequacy minimal set과 boundary set의 canonical
  payload 및 SHA-256

Lock 생성 후 source/config/theory prediction을 변경하면 entry point가
artifact 생성 전에 중단된다. E23a screen 결과를 본 뒤 E23b lock이나
boundary를 수정하지 않는다.

## 필수 artifact

각 run은 새 UTC directory에 다음을 생성한다.

```text
config.resolved.yaml
environment.json
run_manifest.json
protocol_lock.json
data_manifest.json
theory_predictions.json
product_poset_raw_metrics.jsonl
product_poset_seed_metrics.jsonl
product_poset_training_runs.jsonl
poset_minimal_demands.jsonl
checkpoints/c....pt
RESULTS_SUMMARY_KO.md
report.json
```

각 controller×seed checkpoint와 combined checkpoint SHA-256을 report에
기록한다. E23a의 raw/training row에는 exact E18 freeze SHA-256을,
E23b의 row에는 여기에 E23a screen report, E22b report, E22b protocol-lock
SHA-256을 추가해 dependency provenance를 row 수준에서도 보존한다.
MAIN dependency가 차단되면
row/checkpoint는 생성하지 않고
`BLOCKED_DEPENDENCY`를 기록한다.

## Claim boundary

E23이 열 수 있는 최대 범위는 controlled four-axis product poset에서의
capacity 또는 safe absolute-adequacy minimality다. 다음 claim은 항상 닫혀 있다.

- semantic 또는 natural-language demand inference
- pretrained/recurrent language model
- agent/planning
- official backend
- runtime superiority

이 구현 단계에서는 scientific MAIN을 실행하지 않는다.
