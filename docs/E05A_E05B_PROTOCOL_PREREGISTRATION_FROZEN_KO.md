# E05a/E05b prospective semantic-anchor protocol

## 상태와 목적

상태: **FROZEN BEFORE ANY NEW E05 DRY RUN, PILOT, MAIN NAMESPACE, OR MODEL
OUTCOME**

- Legacy `e05_semantic_demand_inference.py`와
  `configs/e05_semantic_demand_inference.yaml`은 v6.1 evidence에 재사용하지
  않는다.
- 이 동결 시점에 `/data/minjun_dev/CATENA/artifacts` 아래 E05 run directory는
  `0`개다.
- E04 `20260727T054917.678326Z`와 additive
  `E04_ARTIFACT_FREEZE_V1.json`은 immutable dependency다.
- 원본 E02는 confirmatory `INCONCLUSIVE`, E02b만 prospective
  `SUPPORTED`로 유지한다.
- E05a는 design-validity 실험이며 H5 claim을 열지 않는다.
- E05b는 E05a `GO`, completed human audit, sealed-validation pass가 모두 있을
  때만 primary를 한 번 평가한다.

연구 질문은 다음 하나다.

> Operation label이나 oracle erase/write 값을 받지 않은 controller가
> structured transaction의 version·time·validity 관계에서 update demand를
> 추론하고, 학습에서 보지 않은 `SUPERSEDE=(erase,write)=(1,1)` 조합에
> 일반화하는가?

## Claim ceiling

모든 gate가 통과해도 최대 허용 표현은 다음과 같다.

> Frozen `CONTROLLED_REFERENCE` oracle-address structured-record setting에서
> old lexical value를 숨기고 old state content는 제공된 address에서만 읽었을
> 때, operation/oracle-demand label 없이 seen-domain/template held-out
> `SUPERSEDE`에서 등록된 factorized-control advantage 방향이 유지됐다.

자연어 이해, hidden-address recovery, semantic gate의 인과적 식별,
pretrained LM/agent behavior, official GDN2/KDA transfer, domain/paraphrase
일반화, latency 또는 systems 우위는 주장하지 않는다.

## E05a와 E05b 분리

| Stage | 역할 | 결과 사용 |
|---|---|---|
| E05a dry | 별도 development namespace의 schema/runner smoke test | claim 불가 |
| E05a main | P/A/I-only 4-seed learnability·leakage-control validity | GO/NO-GO |
| E05a registry seal | GO 뒤 E05b split/control/audit registry 생성 | outcome 미사용 |
| Human audit | 두 reviewer와 adjudication | E05b training dependency |
| E05b validation | 8 fixed final-step checkpoint의 seen P/A/I parity | main unseal gate |
| E05b primary | held-out SUP seen domain/template, 1회 | H5-lite gate |
| E05b secondary | paraphrase/domain/combined stress | descriptive only |

E05a는 어떤 stage에서도 `SUPERSEDE`를 학습·평가하지 않는다. Dry run은 E05b
main/paraphrase/domain/combined namespace를 생성하지 않는다.

## Structured semantic schema

Model-visible immutable record에는 다음만 허용한다.

```text
entity description
domain
current relation
incoming evidence
prior version
evidence version
observation day
evidence timestamp day
prior validity interval
evidence validity interval
scope
source
provenance
incoming value token
template surface
```

Old lexical value는 record에서 제거한다. 다만 제공된 oracle address의 visible
state read는 허용한다. 정확한 표현은 다음과 같다.

> The old lexical value is hidden from the transaction record, while old state
> content is read at the oracle address.

Model-visible dataclass와 feature encoder에는 다음을 넣을 수 없다.

```text
operation / operation_features
oracle (erase, write) / demand
target / target_state
exact mask / affected_index
erase_candidate / write_candidate
old_value / old_value_token
split / namespace
```

Surface에는 `add`, `delete`, `revoke`, `invalidate`, `replace`, `supersede`
토큰을 금지한다. Demand는 derived boolean이 아니라 raw interval 관계로만
정의한다.

```text
erase* = 1  iff prior_valid_to < observation_day
write* = 1  iff evidence_valid_from <= observation_day <= evidence_valid_to
                 and evidence_version > prior_version
                 and evidence scope matches current scope
```

P/A/I는 위 두 predicate의 `(0,0)/(0,1)/(1,0)`이고, E05b primary
`SUPERSEDE`는 학습에서 보지 않은 `(1,1)`이다. 모든 operation에 incoming
evidence와 candidate value가 존재하므로 “값이 있음/없음” 하나로 demand를
알 수 없다.

## 정보 접근과 update 경로

모든 model/control은 다음 공개 경로만 사용한다.

\[
\widehat S =
S_{\rm visible}
-\widehat e\,A(S_{\rm visible},a_{\rm visible})
+\widehat w\,B(x_{\rm visible},a_{\rm visible}).
\]

`A`는 visible address에서 읽은 visible state content로 만들고, `B`는
model-visible transaction의 incoming candidate와 visible address로 만든다.
Private evaluation bundle의 operation, old value, target, precomputed oracle
candidate는 update 적용 함수에서 읽지 않는다.

Memory key는 32차원 orthonormal key 16개다. 따라서 correct address의 state
read가 old content를 정확히 복구하며 oracle-demand positive control의 expected
error는 numerical zero다. 이 설정은 oracle address를 숨기지 않으며 addressing
claim을 열지 않는다.

## Controller comparison과 학습 budget

Factorized와 shared controller는 동일 parameter tensor, initialization,
optimizer, batch order, step 수와 dense compute graph를 사용한다. 두 64차원
path를 모두 계산한 뒤 fixed `2x2` routing만 다르다.

- Factorized: identity routing, erase/write가 별도 path를 사용
- Shared: rank-one average routing, 두 output이 하나의 shared latent를 사용

두 조건 모두 두 gate를 출력하며, shared condition에 tied scalar demand를
강제하지 않는다. Parameter count와 registered dense multiply-add count는
완전히 같아야 한다.

학습은 fixed final step 하나만 사용한다.

```text
steps                 2400
batch size             128
learning rate         .002
weight decay             0
affected weight           1
retention weight          1
state-MSE weight          0
```

Gate label supervision, validation checkpoint selection, main 결과를 이용한
재학습은 금지한다.

## Dataset와 namespace

### E05a

| Split | Seed | Cell | Count/seed |
|---|---:|---|---:|
| Train | 101,202,303,404 | P/A/I × 2 domain × 2 template | 3,600 |
| Control validation | same 4 | same 12 cells | 1,536 |

Dry는 seed `9101`, train cell당 24, validation cell당 16이며 별도 namespace다.

### E05b

| Split | Seed당 균형 | Count/seed |
|---|---|---:|
| Train | P/A/I × 2 seen domain × 2 seen template, cell 300 | 3,600 |
| Sealed validation | same 12 cells, cell 128 | 1,536 |
| Primary | SUP × 2 seen domain × 2 seen template, cell 512 | 2,048 |
| Paraphrase | P/A/I × 2 seen domain, cell 256 | 1,536 |
| Held-out domain | P/A/I × 2 seen template, cell 256 | 1,536 |
| Combined stress | SUP × new domain × paraphrase, cell 1,024 | 1,024 |

E05a, E05b train, validation, primary, paraphrase, domain, combined의 entity,
old-value token, new-value token vocabulary 교집합은 모두 공집합이다. Token
문자열에는 split/namespace 이름을 넣지 않는다. Transaction/episode ID 중복은
허용하지 않는다. Numeric episode namespace는 `5e12` 이상이며 prior experiment
registered range와 겹치지 않는다.

같은 seed/episode에서는 factorized, shared, oracle, 모든 control이 완전히 paired
된다.

## Leakage와 negative controls

Mapping은 outcome 전에 registry와 hash로 고정한다.

| Condition | 고정 개입 |
|---|---|
| Full | correct semantics, address, visible state |
| Oracle demand | gate만 `(e*,w*)`; candidate 경로는 Full과 동일 |
| Transaction-only | semantics 유지, visible state를 0으로 하고 `A`도 그 zero state에서 생성 |
| State-only | state/address 유지, semantic feature와 transaction-derived `B`를 0으로 mask |
| Shuffled fields | domain×template 안에서 operation-changing fieldwise derangement, self-map 0 |
| Wrong address | semantics 유지, 같은 episode의 deterministic `j!=a`; `A_j,B_j`를 correct candidate norm에 deterministic rescale |
| Wrong semantics | entity/address/new value를 유지하고 coherent validity/version tuple만 다른 P/A/I demand로 교체 |

Wrong-address candidate norm mismatch는 `<=1e-6`이어야 한다. Shuffled와
wrong-semantics mapping은 operation-equal이며 output/error를 사용하지 않는다.

E05a control gate는 ADD와 INVALIDATE를 equal-weight로 계산하고 PRESERVE는
descriptive로만 남긴다. E05b primary의 shuffled/wrong-semantics donor는 sealed
validation의 P/A/I에서 미리 정한다.

## E05a GO/NO-GO

Primary model/control outcome은 affected-read MSE다. Retention은 별도 MSE로
판정한다. Bootstrap은 4 checkpoint를 모두 고정한 채 seed의
domain×template×operation cell 내부 episode를 5,000회 paired resample한다.
Pilot seed p-value는 claim gate에 쓰지 않는다.

| Gate | 사전 기준 |
|---|---:|
| Forbidden access | hidden candidate/operation/e,w/mask access 0 |
| Namespace integrity | vocab intersection 0, duplicate ID 0 |
| Budget match | parameter/MAC/initialization/optimizer/steps identical |
| Oracle affected and retention | pooled 95% CI upper `<=1e-8` |
| Factorized oracle excess affected/retention | CI upper `<=0.0005` |
| Shared oracle excess affected/retention | CI upper `<=0.0005` |
| Seen model parity, shared−factorized affected | CI inside `[-0.0005,0.0005]` |
| Each five-control degradation | lower CI `>0.001` |
| Seed direction | each of 4 raw control degradations positive |

Five controls are shuffled fields, wrong address, transaction-only, state-only,
wrong semantics. 하나라도 실패하면 `NO_GO`이며 E05b registry와 audit sample을
생성하지 않는다. 수정 재시도는 새 protocol version과 새 namespace가 필요하다.

## Human naturalization audit

E05a `GO` 뒤 model outcome을 보기 전에 E05b primary/paraphrase/domain/combined
각 75개, 총 300개를 고정한다. Audit item registry는 수정하지 않는다.

Reviewer A와 B는 서로의 판정을 보지 않고 별도 파일에 meaning preservation과
direct answer leakage를 0/1로 기록한다. Adjudicator는 300행 전부에 최종값을
기록한다. Reviewer는 E05b model outcome을 볼 수 없다.

E05b training dependency:

```text
adjudicated meaning preservation >= .95  (>=285/300)
adjudicated answer leakage       <= .02  (<=6/300)
raw A/B agreement on meaning     >= .80
raw A/B agreement on leakage     >= .80
adjudicated rows                 = 300/300
```

Reviewer가 사람이라는 조건을 AI agent review로 대체하거나 자동으로 채우지
않는다. Review 결과는 E05a artifact를 수정하지 않고 별도 additive adjudication
artifact로 저장한다.

## E05b sealed validation과 primary gate

E05b는 seeds `11,22,33,44,55,66,77,88`의 fixed final-step checkpoint를
먼저 sealed validation에서만 평가한다. Validation parity가 실패하면 primary
registry를 열지 않고 `NO_GO_MAIN_SEALED`로 종료한다.

Seed \(s\)의 primary affected-read MSE를 \(Y^m_{si}\)라 두고

\[
D_s=\overline{Y^{shared}-Y^{factorized}},\quad
H_s=\overline{Y^{shared}-Y^{oracle}},\quad
Q_s=D_s/H_s,
\]
\[
R_s=\overline{Y^{factorized}_{retain}-Y^{shared}_{retain}},\quad
C_{c,s}=\overline{Y^{factorized,c}-Y^{factorized,full}}
\]
로 고정한다. \(Q\)는 seed별 ratio를 동일 가중 평균한다.

| Confirmatory gate | 사전 기준 |
|---|---:|
| Validation parity | affected CI inside `±0.0005`, exact TOST both `p<=.05` |
| Oracle low | affected/retention CI upper `<=1e-8` |
| Shared headroom | `H` lower CI `>0.001`, shifted sign-flip `p<=.05` |
| Raw improvement | `D` lower CI `>0.001`, shifted sign-flip `p<=.05`, `D_s>0` 8/8 |
| Headroom closure | `Q` lower CI `>0.10`, shifted sign-flip `p<=.05` |
| Retention NI | `R` upper CI `<=0.0005`, shifted less sign-flip `p<=.05` |
| Absolute factorized retention | factorized−oracle CI upper `<=0.0005` |
| Five controls | each `C_c` lower CI `>0.001`, shifted sign-flip `p<=.05`, 8/8 |

어떤 point estimate 또는 bootstrap replicate에서라도 seed-level
`H_s<=0.001`이면 ratio를 계산하지 않고
`INCONCLUSIVE_ORACLE_HEADROOM`으로 판정한다. Episode를 사후 제외하지 않는다.

Bootstrap은 모든 8 checkpoint를 고정하고 seed별 domain×template stratum에서
paired 5,000회 수행한다. Seed uncertainty는 exact sign-flip/TOST로 분리한다.
Conjunction은 intersection-union test이며 하나라도 실패하면 H5-lite는 열리지
않는다.

Primary는 한 번만 평가한다. Paraphrase/domain/combined는 같은 descriptive
estimand와 CI를 보고하되 primary gate를 보완하거나 대체하지 않는다.

## 상태와 evidence

```text
execution_status
e05a_design_status
human_audit_status
sealed_validation_status
primary_semantic_anchor_status
full_h5_lite_claim_open
```

Evidence tier는 항상 `CONTROLLED_REFERENCE`, `scientific_evidence=false`다.
Official backend나 pretrained model이 없을 때 reference result를 그 claim으로
자동 승격하지 않는다.
