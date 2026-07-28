# E25b Shared-Text-Encoder Transaction Anchor Protocol

이 문서는 prospective v4 protocol이다. v3 dry/audit-preparation 검토에서
`VALUE`의 부분 old-rule 보존을 이진 authoritative label로 평가하면 완전한
target도 오답이 되는 결함을 발견했다. Scientific MAIN은 시작되지 않았으며,
v4는 main namespace와 audit filename을 새로 분리하고 old-rule query를
`FULL/PARTIAL/NONE` categorical status로 고정한다.

## 질문과 범위

E25b는 encoder를 controller별로 factorize하지 않는다. 동일한 frozen
text encoder가 모든 transaction representation을 만들고 controller의
reachable projection만 바꾼다.

```text
text transaction
  -> one frozen/shared encoder
  -> tied / dual / diagonal / separate-address / state-aware
```

허용되는 최대 해석은 controlled text-form transaction과 held-out
identifier/template/domain에서 architecture-demand interaction이 같은
방향으로 나타나는지다. Pretrained LM, free-form generation, agent,
official backend와 production claim은 열지 않는다.

## Model-visible 정보

입력 문장에는 operation label, erase/write bit, exact coordinate mask,
integer address ID가 없다. 다음 lexical cue도 token-level blacklist로
차단한다.

```text
add delete revoke invalidate replace supersede erase write
```

Private operation/demand, active-state branch, address, target state와
query gold는 scoring 및 human audit에만 사용한다. Audit CSV의
private/gold column은 reviewer에게만 제공되고 training/evaluation tensor를
만드는 경로에서는 읽지 않는다. `MatchedTextTransactionController.forward`가
받는 것은 transaction text, current value state와 각 slot에 저장된 opaque
entity string뿐이다. Slot 번호나 gold address는 받지 않으며, 동일 frozen
encoder로 entity string을 key로 만든 뒤 content-addressing한다.

Model-visible transaction은 old-value token을 직접 포함하지 않는다.
Old-rule probe의 private 평가 label만 해당 token을 보유하며, controller는
stored state에서 erase candidate를 읽어야 한다.

Encoder는 deterministic frozen hash-ngram encoder다. Pretrained language
model이 아니며 controller 사이에서 table, seed와 fingerprint가 완전히
동일하다. 모든 controller는 동일 maximal parameter surface를 등록하고
erase candidate 자체는 모든 controller가 자신의 erase address에서 current
state를 읽어 사용한다. `state_aware` projection만 그 preliminary state
read를 gate/address hidden state에도 결합해 같은 text가 state에 따라 다른
update demand를 만들 수 있다. 나머지 projection도 동일 candidate/state-context
parameter를 보유시키되 state-context conditioning은 reachable computation에서
제외하여 parameter matching을 유지한다.

Visible policy token `policy-N`은 별도의 learned incoming head를 거치지
않는다. 모든 controller가 동일한 seed `25000000990000`과 동일한
deterministic hash-to-unit-vector decoder로 token을 하나의 candidate
`B`로 바꾼다. Transaction text에 policy token이 둘 이상이면 실행을
차단하고, transaction이 제거된 state-only control에서는 `B=0`이다.
`MatchedTextTransactionController`에는 controller-specific
`incoming_head`가 없다.

Entity identifier에는 숫자 slot suffix를 넣지 않는다. Identifier OOD에서는
train과 겹치지 않는 opaque name namespace를 쓰되, transaction과 memory key
양쪽에서 동일 문자열을 제공한다.

## Split

| Split | 목적 |
|---|---|
| primary | seen domain/template의 held-out magnitude composition |
| paraphrase | seen demand의 held-out surface |
| identifier | held-out compositional entity identifier |
| domain | held-out domain vocabulary |
| combined | composition + paraphrase + identifier + domain stress |

Magnitude protocol은 H2 geometry를 그대로 사용한다. Stored association을
`A`, visible policy candidate를 `B`라고 하면:

| Semantic condition | Train | Target | Oracle control |
|---|---|---|---|
| ADD anchor | yes | `A+B` | `e=0,w=1` |
| INVALIDATE anchor | yes | `0` (`B`는 보이지만 설치되지 않음) | `e=1,w=0` |
| SUPERSEDE composition | no | `B` | `e=1,w=1` |

Operation name과 `(e,w)` bit는 text에 노출하지 않는다. Primary namespace는
train과 독립적으로 생성된 ADD/INVALIDATE anchor와, training에서 한 번도
나오지 않은 SUPERSEDE semantic composition을 모두 포함한다.

State-conditioning example은 동일한 entity, policy, day, template 문장을
active state와 inactive state에 각각 한 번씩 배치한다. 따라서 private
active branch와 template/item parity 사이에 surface shortcut이 없고 두
조건은 current state로만 구분된다. 이 두 행은 demand 4-way
`minimal_pair_id`와 별개인 `state_counterpair_id`로 묶는다. 전자는
동일 state/old/new/day에서 demand relation만 바꾸고, 후자는 동일
text/template/evidence/entity/policy/day에서 private current-state branch만
바꾼다.

State/episode 생성 seed는 split마다 분리하지만, 같은 visible semantic value
token은 모든 split에서 동일한 target vector를 뜻하도록 별도의 고정
`semantic_value_seed`를 공유한다. 따라서 OOD split이 label inconsistency로
변하지 않는다.

모든 example은 다음 네 query를 함께 가진다.

1. direct fact
2. derived action
3. old-rule probe
4. unaffected retention

Direct-fact MSE는 changed-slot mask와 별도로 query target row에서 항상
계산한다. 따라서 inactive state-conditioned no-op도 자동 정답이 아니다.
Old-rule probe는 이진 authoritative label을 사용하지 않는다. Private
demand로 다음 categorical gold를 고정한다.

| Demand/branch | Gold old-rule status |
|---|---|
| magnitude ADD (`A+B`) | `FULL` |
| magnitude INVALIDATE (`0`) | `NONE` |
| magnitude SUPERSEDE (`B`) | `NONE` |
| value coordinate update | `PARTIAL` |
| address source row | `NONE` |
| state-conditioning inactive/active | `FULL` / `NONE` |

Evaluator는 erase-source row에 대해 demand-aware `FULL/PARTIAL/NONE`
prototype을 구성하고 predicted row와 MSE가 가장 가까운 status를 선택한다.
세 prototype은 candidate 부분을 동일하게 유지하고 old contribution만
full, registered partial mask, zero로 바꾼다. 따라서 candidate error가
old-rule class를 직접 결정하지 않는다. `old_rule_accuracy`는 predicted
status와 private categorical gold의 일치 여부다. 완전한 target과 exact
one-hot address를 사용하는 oracle도 이 evaluator를 그대로 통과하며 query
metric을 하드코딩하지 않는다. 기존 `old_rule_residual`과
`old_rule_component_coefficient`는 연속 진단값으로 함께 보존한다.

Derived-action audit gold는 opaque integer만 제시하지 않는다. Reviewer-only
CSV에 canonical direct-fact vector 값을 기록하고, 최대 coordinate index
modulo 4를 `HOLD/AUTHORIZE/ESCALATE/MONITOR`에 매핑하는 taxonomy와 rule을
명시한다. 이 private column은 모델 입력으로 사용되지 않는다.

## Controls

- oracle-demand upper bound
- shuffled policy/day text: entity/address string은 그대로 유지
- wrong entity/address: policy/day/demand semantics는 그대로 유지
- transaction-only zero-state
- state-only

두 text control은 서로 다른 factor만 교란한다. Shuffled control은 같은
demand 안에서 policy/day content만 순열화하고, wrong-entity control은
원문을 다시 render하면서 entity/address string만 바꾼다.

## Human audit

Main 전 `--prepare-audit` stage가 정확히 300행의 population을 만들고
다음 세 immutable preparation artifact와 별도의 review-work filename을
protocol에 고정한다.

```text
E25B_V4_HUMAN_AUDIT_ITEMS_LOCKED.csv
E25B_V4_HUMAN_AUDIT_REVIEW_TEMPLATE.csv
E25B_V4_HUMAN_AUDIT_POPULATION_LOCK.json
E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv
```

처음 세 파일은 finalized `AUDIT_PREPARATION` run 안에서 immutable이다.
Reviewer는 template을 preparation directory 밖의 별도 workspace로 복사해
정확히 `E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv`라는 이름으로 검수한다.
Finalized template 자체를 편집한 audit는 거부한다.

Population lock은 immutable audit column의 canonical SHA-256, items CSV
SHA-256, config SHA-256, 행 수와 exact filename을 포함한다. Main은
`--audit-csv`와 `--audit-population-lock`을 함께 요구하며, reviewed
CSV의 audit/example ID, text, split, minimal-pair/state-counterpair ID와
모든 private/gold
field가 준비 population과 동일한지 다시 생성해 검증한다. 중복
audit/example ID, 중복 pair member, 불완전 4-way pair, prepared items
hash drift는 모두 main directory 생성 전에 차단된다.

300은 `4 demands × 2 state branches`로 나누어떨어지지 않으므로 audit
sampling count는 split별 demand당 `16/16/16/14/13`으로 고정한다. 이로써
296행은 완전한 state counterpair를 이루고 combined split의 마지막 4행만
deterministic singleton tail이다. Singleton도 counterpair ID와 전체
population hash에는 묶이지만 active/inactive pair audit 통계에는 넣지
않는다.

두
reviewer가 semantic preservation, operation leakage, entity ambiguity,
old-value lexical leakage와 gold consistency를 독립적으로 PASS/FAIL
판정한다. 동일 split 안에서 entity/old/new/day를 공유하고 demand
relation만 바뀌는 4-way `minimal_pair_id`를 audit 전용으로 고정한다.
Reviewer가 gold consistency를 실제로 판단할 수 있도록 private demand,
erase/write entity, active branch, old/new label과 hash, target hash,
direct-fact gold, derived-action gold, categorical old-rule gold와
`FULL/PARTIAL/NONE` 정의를 CSV에 제공한다.
Agreement는 `0.80` 이상이어야 하며 disagreement는 adjudication이
필요하다. 다섯 판단 항목은 각각 독립된 adjudication column을 갖는다.
미해결 또는 최종 critical FAIL이 하나라도 있으면 main을 차단한다.

## Primary gate

Paired training seed가 상위 독립 단위다. Magnitude는 asymmetric
reachability와 symmetric composition을 하나의 dual-gain gate로 합치지
않는다.

| Demand | Gate |
|---|---|
| ADD anchor | tied-minus-dual affected-MSE gain |
| INVALIDATE anchor | tied-minus-dual affected-MSE gain |
| SUPERSEDE composition | tied/dual absolute equivalence + 각 absolute floor |
| value | dual -> diagonal |
| address | diagonal -> separate-address |
| state conditioning | separate-address -> state-aware |

ADD와 INVALIDATE는 각각, 그리고 equal-weight 평균에서 mean gain이
SESOI `0.001` 이상이고 전 seed 방향이 양수이며
one-sided exact sign-flip `p <= 0.05`여야 한다. 또한 oracle affected MSE
`<=0.001`, primary `state_aware` affected MSE `<=0.001`, primary
erase/write address accuracy `>=0.95`, unaffected retention `<=0.0005`,
각 negative control의 affected degradation `>=0.001`을 모두 요구한다.
따라서 낮은 oracle floor만으로 shared encoder/controller floor가 확보된
것으로 간주하지 않는다.

SUPERSEDE에서는 dual gain을 요구하지 않는다. Seed-paired
`dual-minus-tied` bootstrap CI 전체가 absolute margin `±0.0005` 안에
있고 tied 및 dual affected MSE가 각각 `0.001` 이하여야 한다. 이는
`e=w=1`인 symmetric target이 두 controller 모두에게 reachable하다는
composition guard다.

Oracle headroom이 식별 가능한 각 seed×split×demand에서 다음 recovery도
기록한다.

```text
(tied affected MSE - controller affected MSE)
------------------------------------------------
(tied affected MSE - oracle-demand affected MSE)
```

분모가 등록값 `1e-12` 이하이면 이 값으로 gate를 열지 않고
`oracle_headroom_identifiable=false`로 기록한다. 이 metric은 controller
interaction의 해석을 보조하며, 위의 절대 oracle floor와 primary contrast
gate를 대체하지 않는다.

## Artifact와 실행 경계

각 run은 새 UTC directory에 다음을 쓴다.

```text
protocol_lock.json
run_manifest.json
data_manifest.json
text_transaction_metrics.jsonl
text_transaction_seed_metrics.jsonl
checkpoint_seed<seed>_<controller>.pt
report.json
RESULTS_SUMMARY_KO.md
```

Dry-run은 development namespace만 사용하고 `claim_eligible=false`다.
Main은 reviewed audit CSV와 그 audit-preparation population lock이
모두 검증되기 전에는 run directory, run manifest 또는 `latest.json`을
생성하지 않는다. 이 구현 작업에서는 main data 생성이나 training을
실행하지 않는다.
