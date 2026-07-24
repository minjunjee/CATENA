# CATENA 실험 실행 명세

이 문서는 실험을 **시간 순서**로 설명한다. 각 단계는 앞 단계의 gate를 통과한 뒤에만 다음 단계로 넘어간다. 메인 연구 질문은 하나다.

> 검증된 외부 메모리 transaction이 주어졌을 때, 이미 형성된 recurrent execution state를 짧은 native forward로 갱신하여 최신 상태를 전체 재실행한 exact-refresh behavior와 일치시킬 수 있는가?

## 공통 표기와 평가 단위

- `M_t`: transaction 이전의 canonical external state
- `f_t`: 검증된 transaction, `M_{t+1}=f_t(M_t)`
- `S_t`: transaction 이전까지의 모델 execution state
- `S_exact`: 최신 canonical view와 유효한 작업 문맥을 처음부터 다시 처리한 operational reference
- `S_catena`: `S_t`에 transaction slots만 native forward로 처리해 얻은 transported state
- affected query: 변경된 사실이나 그 파생 행동을 묻는 질의
- retention query: 변경과 무관한 기존 작업 문맥을 묻는 질의

핵심 coherence 지표는 다음 셋이다.

- `C_update`: affected query에서 exact oracle과 같은 결정을 한 비율
- `C_retain`: retention query에서 exact oracle과 같은 결정을 한 비율
- `C_joint`: 두 값의 조화평균

Raw gold accuracy와 own-oracle agreement를 함께 보고한다. 이는 backbone 자체의 과제 능력과 state repair 능력을 분리하기 위해서다.

---

## E00. 서버 audit와 환경 고정

### 목적

CUDA 13.0/driver 580.126.16 환경에서 실제 GPU 4장, VRAM, compute capability, P2P topology, MIG 여부, storage와 compiler를 기록한다. 제품명이나 96 GB VRAM은 audit 이전에는 가정하지 않는다.

### 입력과 실행

데이터나 모델은 사용하지 않는다.

```bash
bash scripts/00_bootstrap_and_audit.sh
```

권장 기준 환경은 Python 3.11, PyTorch 2.12.1 `cu130`, BF16이다. 시스템 CUDA toolkit은 custom extension을 컴파일할 때 사용하고, PyTorch wheel은 자체 CUDA runtime을 사용한다.

### 기록 항목

- GPU 이름, UUID, VRAM, compute capability
- `nvidia-smi topo -m`, MIG mode, P2P 가능 여부
- PyTorch/CUDA/cuDNN 버전
- BF16 matmul smoke
- local scratch와 model cache의 읽기/쓰기 속도
- `pip freeze`, git SHA, kernel build log

### 통과 기준

- `torch.cuda.device_count()==4`
- 네 GPU에서 BF16 연산 성공
- 각 GPU 단독 프로세스가 정상 실행
- custom extension용 `nvcc`와 compiler 확인

이 단계가 실패하면 어떤 과학적 실험도 시작하지 않는다.

---

## E01. 모델 runtime hard gate

### 목적

CATENA 효과를 측정하기 전에 state/cache adapter가 수치적으로 올바른지 확인한다. 이 단계의 실패를 method failure로 해석하면 안 된다.

### 모델

- RWKV-7 0.4B FLA/HF: 빠른 debug
- RWKV-7 2.9B FLA/HF: H1-H4 메인 backend (`fla-hub/rwkv7-2.9B-g1`)
- RWKV-7 2.9B PTH/pip: 동일 원본 weight `rwkv7-g1-2.9b-20250519-ctx4096.pth` 교차 검증
- Qwen2.5-3B-Instruct: Transformer 경계 비교

### 확인 항목

1. RWKV full-sequence prefill과 chunked prefill의 candidate ranking 일치
2. token ID 입력과 동일 embedding 직접 입력의 logits/ranking 일치
3. state clone 이후 alias가 없고 원본 상태가 변하지 않음
4. continuous slots에서 encoder까지 gradient가 유한하게 전달됨
5. Transformer KV clone/crop 이후 prefix reuse 결과가 full prefix 결과와 일치
6. affected-suffix re-prefill이 실제 stale prefix KV를 보존한 채 수정 suffix만 계산함

### 메트릭과 gate

- max/mean absolute logit error
- candidate ranking agreement
- state/cache bytes
- finite gradient 여부
- BF16 허용 오차 내 parity

RWKV 2.9B differentiable backend가 통과하지 않으면 H1/H2는 PTH text path로 진단할 수 있지만, H3/H4는 시작하지 않는다. 자세한 제약은 `BACKEND_STATUS_KO.md`를 따른다.

---

## E02. CATENA-UpdateBench 생성과 누출 검사

### 목적

외부 상태 변경과 내부 execution state 불일치를 인과적으로 분리할 수 있는 통제 데이터셋을 만든다.

### 데이터 구조

각 episode에는 다음이 포함된다.

1. transaction 이전 canonical state
2. old rule을 실제로 사용한 history/plan/tool trace
3. verified transaction과 dependency closure
4. transaction 이후 canonical state
5. affected-direct, affected-derived, old-rule probe, unaffected-retention query
6. exact refresh input과 Transformer affected-suffix boundary

### 도메인과 변수

- 도메인: API configuration, access-control policy, user/workflow state
- operation: `SUPERSEDE`, `AMEND`, `INVALIDATE`, `ADD_EXCEPTION`
- history: 1K, 4K, 16K; 32K stress
- dependency depth: 0, 1, 2; stress 3
- query gap: 0, 128, 512; stress 2,048 token
- schema family holdout: train/validation/test 분리

### 생성 크기

- pilot: train 1,200 / val 240 / test 800
- main: train 6,000 / val 800 / test 2,400
- stress: test 1,200
- chain: train 3,000 / val 400 / test 1,000

### 검증

- transaction을 적용한 current state가 query gold와 일치
- affected query와 retention query가 동시에 존재
- closure는 query 생성 전에 고정되고 최종 tool action을 직접 포함하지 않음
- Tx-only, shuffled closure가 데이터 누출 통제로 작동할 수 있음
- exact refresh는 과거 사건을 재작성하지 않고 최신 view를 추가함

```bash
bash scripts/02_generate_and_validate_data.sh
```

---

## E03 / H1. Stale execution-state failure 진단

### 가설

외부 state가 `M_t -> M_{t+1}`로 바뀐 뒤 기존 `S_t`를 그대로 사용하면, exact refresh보다 폐기된 행동을 선택하는 비율이 높다.

### 모델과 데이터

- 메인: RWKV-7 2.9B
- pilot test 800 episode로 gate 후 main test 2,400 episode
- 학습 없음

### 비교 정책

- `stale`: 기존 execution state 그대로 재사용
- `exact_refresh`: 최신 canonical view와 유효 문맥을 전체 재처리

### 메트릭

- gold accuracy
- own-oracle agreement
- teacher-correct agreement
- `C_update`, `C_retain`, `C_joint`
- stale-action rate
- exact-to-stale behavioral KL
- history/dependency/query-gap strata별 paired bootstrap CI

### 가설 지지 결과

Exact teacher의 gold accuracy가 해석 가능한 상한을 제공하고, affected query에서 stale가 exact보다 유의하게 낮아야 한다. Retention query에서는 큰 차이가 없어야 하며, history/gap 증가에 따라 stale failure가 커지는 추세가 있으면 더 강한 증거다.

### 결과별 허용 주장

- exact teacher가 약함: backbone/task ceiling만 보고
- stale-exact 차이 없음: 이 설정에서 coherence failure가 유도되지 않았다는 부정 결과
- H1 성립: persistent recurrent state의 stale-state failure 진단까지만 주장

```bash
bash scripts/03_h1_h2_pilot_4gpu.sh
```

---

## E04 / H2. Update signal의 충분성과 비누출성

### 가설

Operation, target, version, validity, dependency closure를 포함한 typed transaction은 plain correction보다 affected behavior를 더 잘 고치면서 retention을 유지한다. 이 개선은 transaction만으로 정답을 직접 제공한 결과가 아니다.

### 모델과 데이터

- RWKV-7 2.9B
- pilot/main test
- 학습 없음

### 비교 정책

1. stale
2. plain correction
3. typed transaction
4. typed + dependency closure
5. shuffled closure
6. transaction-only + zero state
7. reset + compact current snapshot
8. query-time latest-memory retrieval
9. exact refresh

### 주요 독립 변수

- dependency depth
- query kind
- query gap
- history length
- operation type

### 메트릭

H1 메트릭에 update token 수, update latency, correction-retention-cost Pareto를 추가한다.

### 가설 지지 결과

- closure가 특히 affected-derived/tool query를 개선
- retention은 plain/typed 대비 유지
- Tx-only가 CATENA-style state reuse보다 낮음
- shuffled closure가 올바른 closure보다 낮음

### 결과별 허용 주장

- Tx-only와 동일: 기존 state를 이동했다는 주장은 불가
- shuffled와 동일: dependency semantics 기여 없음
- reset/retrieval이 같은 정확도와 비용: learned transport의 실용적 필요성 약화
- H1+H2 성립: typed transaction/closure가 execution-state update signal이라는 주장 가능

---

## E05. Exact teacher materialization

### 목적

H3/H4 학습마다 긴 exact refresh를 반복하지 않도록 exact candidate distribution을 미리 계산하고 data/model manifest를 고정한다.

### 입력

- main train/val episode
- RWKV-7 2.9B exact refresh
- 모든 affected/retention query와 후보

### 산출물

- episode/query별 exact candidate log-likelihood
- exact prediction과 gold index
- teacher-correct flag
- dataset/model/config hash

State object 자체는 backend serialization이 안정적으로 검증되기 전까지 필수 cache로 사용하지 않는다. 기본 H3 trainer는 base history를 재-prefill하며, fixed-size RWKV state serialization이 E01에서 통과하면 별도 최적화로 켠다.

```bash
bash scripts/04_build_rwkv_teacher_4gpu.sh
```

---

## E06 / H3a. Transport pilot과 slot sweep

### 목적

큰 grid를 돌리기 전에 learned native transport가 학습 가능한지 확인하고, slot bottleneck `K`를 validation에서 선택한다.

### 모델과 학습 대상

- Frozen RWKV-7 2.9B
- TransactionSlotEncoder 약 5-20M parameters
- backbone parameter는 optimizer에 넣지 않음

### 네 GPU 동시 run

- GPU0: typed, K=4, seed 11
- GPU1: typed, K=8, seed 11
- GPU2: typed, K=16, seed 11
- GPU3: generic soft-slot, K=8, seed 11

먼저 300-step pilot로 finite loss/gradient와 validation oracle agreement 상승을 확인하고, 통과하면 2K-4K step sweep으로 확장한다.

### Loss

`L = λ_aff KL(p_exact || p_student) + λ_ret KL_retention + λ_gold CE_gold`

기본 weight는 1.0 / 0.6 / 0.5다. Query는 encoder 입력에 주지 않으며 한 번 transport한 state에서 여러 미래 query를 평가한다.

### 선택 기준

Validation `C_joint`를 1순위, update latency와 KL을 2순위로 사용한다. Test는 이 단계에서 열지 않는다.

```bash
PILOT_ONLY=1 bash scripts/05_train_h3_4gpu.sh
bash scripts/05_train_h3_4gpu.sh
```

---

## E07 / H3b. Main transport와 ablation

### 가설

Query-independent compact transaction slots를 RWKV native forward로 처리하면 text patch, reset, retrieval, generic soft slots보다 더 나은 exact-coherence/비용 Pareto를 만든다.

### 학습 run

Validation에서 선택한 K를 고정한다. 기본 config는 K=8이다.

- typed CATENA: seeds 11, 22, 33
- typed no-closure: seed 11
- untyped structured: seed 11
- generic soft-slot: seed 11; 시간이 남으면 seed 22/33

### 평가 데이터

- main test: 1K/4K/16K, held-out schema
- stress test: 16K/32K, dependency 1-3, gap 512/2,048
- 하나의 transported state에서 direct/derived/retention/old-rule queries 모두 평가

### Baseline

- stale
- plain correction
- typed closure text
- reset + snapshot
- query-time retrieval
- generic soft slots
- typed no-closure/untyped
- exact refresh
- direct hidden delta는 작은 negative-control subset만

### 메트릭

- gold/own-oracle/teacher-correct agreement
- `C_update`, `C_retain`, `C_joint`
- stale-action rate, behavioral KL
- update latency p50/p95
- resident state bytes, processed update tokens
- accuracy-cost Pareto, coherence per millisecond
- 3-seed mean, standard deviation, paired bootstrap CI

### 가설 지지 결과

CATENA가 text/generic/reset/retrieval보다 Pareto frontier를 확장해야 한다. 즉, 같은 `C_joint`에서 더 빠르거나 같은 비용에서 더 높은 `C_joint`를 보여야 한다.

### 결과별 허용 주장

- generic ≈ typed: typed structure 기여 주장 불가
- no-closure ≈ closure: closure의 latent transport 기여 없음
- reset/retrieval ≈ CATENA: transport 실용 우위 없음
- Tx-only ≈ CATENA: state transport 주장 불가
- H1-H3 성립: compact native state transport 주장 가능

```bash
bash scripts/06_train_h3_ablations_4gpu.sh
bash scripts/06_eval_h3_4gpu.sh
```

---

## E08. Transformer 경계와 공정성 실험

### 목적

Transformer를 메인 주제로 바꾸는 것이 아니라, RWKV 결과가 단순한 update prompting인지, fixed recurrent state가 어떤 workload에서 실제 이점을 갖는지 경계를 찾는다.

### 모델과 데이터

- Qwen2.5-3B-Instruct
- RWKV와 동일 episode/query/candidate
- 각 모델을 자기 full-refresh oracle에 대해 정규화

### Inference 정책

1. stale KV reuse
2. plain append
3. typed append
4. closure append
5. reset + compact memory capsule
6. oracle affected-suffix re-prefill
7. full re-prefill

Affected-suffix re-prefill은 stale KV의 영향받지 않은 prefix를 실제로 crop/reuse하고, 최초 affected position 이후 current-view suffix만 재계산한다.

### Learned fairness check

RWKV H3와 parameter/slot 수를 맞춘 Qwen soft patch를 최소 seed 11로 학습한다. 시간이 허용되면 3 seeds로 확장한다. 이것이 없으면 architecture superiority는 주장하지 않고 inference-only boundary로 제한한다.

### 메트릭

- gold/own-oracle/joint coherence
- update/re-prefill latency
- TTFA, 32-token decode latency
- resident KV bytes
- recomputed token 수
- history length와 update frequency scaling

### 허용 주장

- RWKV가 긴 history/high-update에서만 Pareto 우세: 해당 regime에 한정한 fixed-state advantage
- Transformer가 전 조건 우세: recurrent advantage 철회, architecture boundary 분석만 유지

```bash
bash scripts/07_transformer_boundary_4gpu.sh
```

---

## E09 / H4. Composition과 long-chain drift

### 가설

길이 1-2의 transaction chain에서 sequential transport와 composed path를 정렬하면, 같은 chain data를 본 distillation-only control보다 test length 4/8/16에서 drift가 작다.

### 데이터

- chain train: length 1-2
- validation: 1/2/4
- test: 4/8/16
- same-key version chain 중심
- unseen operator pair/schema 포함

### 학습

H3 best checkpoint로 encoder를 초기화한다.

- control: exact distillation + retention, composition weight 0
- CATENA-composition: exact + retention + symmetric KL between sequential and composed paths

`L_H4 = λ_exact KL_exact + λ_ret KL_ret + λ_path KL_sym(seq, composed) + λ_gold CE`

### 평가

- final `C_joint`
- chain length별 mean exact KL
- drift slope
- retention decay
- stale recurrence
- sequential/composed path disagreement
- query gap 0/128/512/2,048 및 old-rule resurfacing prompt

### 가설 지지 결과

Composition model이 matched distillation-only control보다 길이 증가에 따른 KL/drift slope를 낮춰야 한다. 단일 길이 성능이 아니라 길이 외삽 추세가 핵심이다.

### 허용 주장

- 차이 없음: H4/compositional novelty 철회, H3 single-update paper로 제한
- 긴 chain에서 일관된 개선: compositional regularization의 OOD chain generalization 주장 가능

```bash
H3_INIT_CHECKPOINT=artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt \
  bash scripts/08_h4_composition_4gpu.sh
bash scripts/08_eval_h4_4gpu.sh
```

---

## E10. 시스템 profiling

### 목적

모델 FLOPs 추정 대신 실제 동일 서버에서 state/KV 구조의 비용을 측정한다.

### 조건

- RWKV-7 2.9B, Qwen2.5-3B
- history 1K/4K/16K/32K
- warm-up 5회, 측정 30회
- BF16, single GPU, 동일 프로세스 pinning

### 메트릭

- full prefill p50/p95
- transaction update p50/p95
- TTFA
- 32-token decode latency
- resident state/KV bytes
- peak allocated bytes

CATENA learned encoder latency는 encoder + K-slot native forward 전체를 포함해야 한다.

```bash
bash scripts/09_profile_4gpu.sh
```

---

## E11. Naturalized prompt와 실제 tool-call 생성

### 목적

Candidate scoring 결과가 실제 agent action 생성으로 이어지는지 확인한다.

### 데이터

Stress set 중 최대 300 episode를 자연어 표현으로 변형하고, JSON tool schema를 사용한다.

### 정책

- RWKV typed closure / exact refresh / CATENA best checkpoint
- Qwen closure append / full re-prefill

### 메트릭

- tool name exact match
- argument exact match
- JSON/schema validity
- executable simulator success
- stale field rate

Candidate agreement는 높지만 schema generation이 낮으면 internal coherence와 agent action reliability를 분리해 보고한다.

```bash
bash scripts/10_naturalized_toolcalls_4gpu.sh
CATENA_H3_CHECKPOINT=artifacts/checkpoints/e07_h3_main/seed_11/encoder_final.pt \
  bash scripts/10b_toolcalls_catena.sh
```

---

## E12. Claim gate와 clean rerun

Test 결과를 본 뒤 method나 threshold를 다시 고르지 않는다. Validation에서 선택한 config와 seed를 고정하여 clean directory에서 main table을 재생성한다.

필수 산출물은 다음과 같다.

- raw predictions JSONL
- summary JSON과 bootstrap interval
- all config snapshots / model hashes / git SHA / host audit
- H1 stale gap figure
- H2 representation ablation
- H3 coherence-cost Pareto
- H4 chain drift
- RWKV-Transformer regime map

결과별 허용 주장은 `CLAIM_GATES_KO.md`를 따른다. 최종 bundle은 `bash scripts/12_bundle_results.sh`로 생성한다.
