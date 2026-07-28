# CATENA v6.1 experiment runbook

## 0. 현재 실행 중인 E00/E01

현재 E00/E01은 중단하거나 live source를 수정하지 않는다. 완료된 artifact는 v6.0 pilot로 보존한다. Process가 종료된 뒤 `CATENA_v6.1_post_E01_joint_patch`를 적용하고 E01b부터 confirmatory evidence를 생성한다.

```bash
pgrep -af 'e00_protocol_lock.py|e01_local_controllability.py'
```

아무 출력이 없을 때 patch를 적용한다.

---

## E01b - H1 constrained behavioral reachability

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.e01b_constrained_behavioral_reachability \
  --config configs/e01b_constrained_behavioral_reachability.yaml \
  --device cuda:0
```

### 데이터

- Candidate mode: `OracleCandidate`, `RecurrentRead`
- Operation: PRESERVE, ADD, INVALIDATE, SUPERSEDE
- Train/test geometry 분리
- 변화 요인: association load, key correlation, old/new norm, old-new cosine

### 계산

- state `R_span`, state `R_feas`
- equal-weight correction/retention `R_beh`
- actual learned behavioral MSE
- operation fixed-effect regression
- operation-only 대비 conditional unseen-geometry R2
- unseen test operation-adjusted slope와 calibration slope
- learned excess over the behavioral bound

### Claim gate

OracleCandidate/tied에서 conditional minimum OOS R2와 unseen-test positive-slope
seed sign-flip을 모두 통과해야 한다. Calibration slope는 사전 equivalence
margin이 없어 descriptive primary로만 보고한다. RecurrentRead gap은 addressing이
아니라 candidate-recovery/content interference로만 해석한다.

---

## E02 - H2 magnitude factorization

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.e02_magnitude_factorization \
  --config configs/e02_magnitude_factorization.yaml \
  --device cuda:0
```

현재 구현의 정식 main은 GPU 한 장에서 8 paired seeds를 한 run으로 실행한다.
seed-lane merge/aggregation이 구현되기 전에는 여러 process가 `latest.json`과
완결성 gate를 공유할 수 없으므로 4-GPU 분할 실행을 사용하지 않는다. 동일 seed
안에서 tied와 dual은 같은 initialization tensor를 사용한다.

### Confirmatory comparison

- Tied: identical two-output head를 diagonal subspace로 projection
- Dual: erase/write를 독립 사용
- OracleCandidate와 fixed address
- strict same-recipe가 primary
- equal-budget LR tuning은 robustness

### Primary metrics

- raw affected-read MSE
- tied-to-oracle headroom normalized gain
- PRESERVE absolute equivalence
- SUPERSEDE relative equivalence
- asymmetric-minus-symmetric raw DID
- unaffected retention non-inferiority
- practical no-op-to-oracle gap closure는 secondary

### Claim gate

E01b의 main/full H1 artifact가 hash와 lineage 검사를 통과하고 H1이 supported일
때만 E02 main을 시작한다.

다음 여섯 조건이 모두 필요하다.

1. asymmetric gain이 SESOI 초과
2. PRESERVE equivalence
3. SUPERSEDE equivalence
4. DID > 0
5. retention non-inferiority
6. tuned comparison이 strict 방향을 뒤집지 않음

DID만 양수이면 H2 claim을 열지 않는다.

---

## E03 - H3 joint diagonalizability

```bash
python -m experiments.e03_granularity_orientation \
  --config configs/e03_granularity_orientation.yaml \
  --device cpu
```

### Demand families

- axis-aligned commuting
- common-rotated commuting
- noncommuting
- seed별 24개 operator로 shared basis를 fit하고, 별도 8개 operator에서 평가
- dry-run seed와 8개 main seed는 완전히 분리

### Control classes

- fixed diagonal
- learned shared-basis diagonal, multi-restart optimization
- transaction-conditioned low-rank oracle upper bound
- full matrix oracle

### Evidence

- fixed-basis rotation penalty
- common-rotation shared-basis recovery
- noncommuting joint-diagonalization gap
- richer-control recovery
- commutator norm vs JD regret relation
- error-control-dimension frontier
- held-out isotropic probe에서 normalized JD regret가 실제 operator-application
  MSE를 예측하는지 calibration

네 contrast는 normalized operator-entry MSE에서 사전 고정한 practical gap
`0.001`을 넘어야 하며, 8 seed exact sign-flip을 함께 통과해야 한다. 상대
contrast만으로 H3를 열지 않는다. Axis fixed residual, common-rotation learned
residual, commuting commutator, low-rank/full oracle residual은 각각 사전 고정한
absolute tolerance도 통과해야 한다.

E03는 CPU-only이며 작은 QR workload의 oversubscription을 막기 위해 config의
thread 수를 고정한다. E00 current-source infrastructure lock만 요구하며
E01b/E02의 claim 결과에는 의존하지 않는다. Low-rank condition은
parameter-matched learned model이 아니라 upper bound로만 해석한다.

---

## E04 - H4 functional mediation

E02의 모든 8 paired-seed checkpoint가 고정된 뒤 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.e04_functional_mediation \
  --config configs/e04_functional_mediation.yaml \
  --device cuda:0
```

### Counterfactual design

각 quartet은 동일한 state, address, old candidate, new candidate를 공유하고 operation만 바뀐다.

### Intervention package

- relevant/irrelevant dose: 0, .25, .5, .75, 1
- independent same-operation norm-matched donor
- same-base cross-operation norm-matched donor
- relevant-component same-operation/oracle rescue
- post-hoc scalarization mediation fraction
- trained tied checkpoint comparison

### Primary

ADD/INVALIDATE에서 relevant damage - irrelevant damage의 seed-level interaction.

### Support

- dose monotonicity
- same-operation transplant superiority
- nontrivial rescue superiority
- retention damage secondary

Zeroing 또는 heatmap만으로 mediation claim을 열지 않는다.

---

## E05 - H5-lite semantic anchor

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.e05_semantic_demand_inference \
  --config configs/e05_semantic_demand_inference.yaml \
  --device cuda:0
```

### Input restrictions

- no operation one-hot
- no oracle erase/write magnitude
- no exact mask
- oracle address + hidden old value만 primary

### Splits

1. held-out SUPERSEDE, seen domain/template - primary
2. seen operation, held-out paraphrase
3. seen operation, held-out domain
4. SUPERSEDE + new domain + paraphrase - combined stress

### Comparisons

- factorized semantic controller
- parameter-matched shared controller
- shuffled-text negative control
- oracle-demand upper bound

### Audit

`naturalization_audit.csv`의 300개를 두 연구자가 독립 검수한다.

- meaning preserved >= .95
- answer leakage <= .02
- meaning/leakage reviewer agreement >= .80
- 모든 row adjudication 완료

Audit가 통과되기 전에는 semantic external-validity claim을 열지 않는다.

---

## E06/E07 - post-workshop only

E06은 multi-update, plan continuation, retrieve-every-query, retrieve-once cached snapshot, one-time assimilation의 quality-constrained break-even을 다룬다. E07은 official KVEraser와 localization parity가 확보된 경우에만 scientific evidence가 된다. REALM critical path에 넣지 않는다.

---

## E08 - evidence freeze

```bash
python -m experiments.e08_claim_freeze \
  --config configs/e08_claim_freeze.yaml \
  --device cpu
```

E08은 raw report와 audit를 읽어 hierarchical claim gate를 고정한다.

```text
H1 -> H2 -> H4
H3: independent theory branch
H5: exploratory anchor + completed audit
H6/RQ-T: post-workshop
```

논문에는 `claim_freeze.json`에서 허용된 문구만 사용한다.
