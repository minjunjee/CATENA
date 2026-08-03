# E26 Final Scientific Protocol

Experiment ID: `E26_FINAL_GDN2_1P3B_TRANSACTIONAL_TRANSFER`  
상태: `PROSPECTIVE_LOCK`  
Evidence tier: `PRETRAINED_AUTOREGRESSIVE_LM / PROSPECTIVE_MATCHED_ARCHITECTURE`

수치 threshold, seed, data split, model/checkpoint 및 source revision의 유일한
machine-readable authority는
`configs/e26_final_gdn2_1p3b_transactional_transfer.yaml`이다.

## 연구 질문

Official NVLabs GDN2 recurrent operator와 community 100B-token 1.3B checkpoint를
사용한 matched autoregressive LM에서 independent erase/write control이 자연어
`ADD`/`INVALIDATE` transaction에 선택적인 이득을 주는지 평가한다. Generic
capacity 차이를 제거하기 위해 `PRESERVE`/balanced `SUPERSEDE`를 대칭 대조군으로
사용하고, retention, worst-cell locality, general PPL, RULER 및 speed를 hard
guardrail로 둔다.

이 실험은 기존 Stage-3C/3D 판정을 수정하지 않는다. 두 numerical failure는
그대로 보존되며, E26 Final은 pinned official runtime과 pretrained checkpoint를
사용하는 별도의 prospective experiment다.

## 비교 정책

각 official GDN2 layer의 raw projection logits를 `z_b`, `z_w`라 한다.

```text
Dual:
  b = sigmoid(z_b)
  w = sigmoid(z_w)

Projected-Tied:
  z = 0.5 * (z_b + z_w)
  b = sigmoid(z)
  w = sigmoid(z)
```

두 조건 모두 `b_proj`와 `w_proj`를 등록·학습하며 parameter name/shape/count와
optimizer state surface가 같다. `allow_neg_eigval=false`만 허용한다. Gate 계산
외 official `chunk_gdn2`, `fused_recurrent_gdn2`, q/k/v, decay, normalization 및
state dtype 경로는 수정하지 않는다.

## Common-function bridge

Community checkpoint의 모든 layer에서 두 gate projection weight를 평균화하고
동일 값으로 복사한다. 하나의 Projected-Tied common model을 90% general / 10%
transaction calibration mix, next-token CE로 100M tokens 학습한다. Runtime
integrity가 정상이나 PPL/headroom gate만 실패할 때 등록된 단 한 번의 200M total
extension을 허용한다. Main test는 열지 않는다.

Bridge final model/optimizer/RNG/data cursor를 byte-identical clone한 뒤 Tied와
Dual로 분기한다. Fork boundary의 logits, recurrent state와 loss가 동일해야 한다.

## Main training

- seeds: `26011, 26022, 26033, 26044, 26055`
- paired runs: 5 × 2 variants
- sequence length: 4096
- global batch: 32 sequences / 131,072 tokens per run
- objective: autoregressive next-token CE only
- optimizer: AdamW, betas 0.9/0.95, LR 3e-5→3e-6 cosine, WD 0.1,
  clip 1.0, warmup 2%
- primary checkpoint: fixed 100% token exposure
- no early stopping or best-checkpoint selection

Token budget은 scientific outcome을 보기 전에 official-kernel speed measurement로
`350M, 500M, 750M, 1B` 중 36-hour resource contract를 만족하는 가장 큰 값을
선택한다.

## Data와 evaluation

Training token mix는 general language 75%, transaction 25%다. Operation name,
erase/write bit, gate target, address mask를 token input이나 auxiliary loss로 주지
않는다. 같은 paired seed의 variants는 token IDs와 sequence boundaries가 exact
match한다. Main-test manifest/template는 training 완료 전 접근할 수 없다.

Frozen evaluation은 4 operations × 3 update counts × 4 gaps × 4 OOD splits × 32
episodes = 6,144 episodes다. 동일 update-prefix state를 no-alias clone해 current
fact, derived action/schema JSON, stale suppression, unaffected retention query를
각각 평가한다. Primary는 length-normalized candidate likelihood 기반 composite다.

## Primary estimand와 claim gate

```text
I[s,o] = score(Dual,s,o) - score(Tied,s,o)
DID[s] = mean(I[s,ADD], I[s,INVALIDATE])
         - mean(I[s,PRESERVE], I[s,SUPERSEDE])
```

`LM_TRANSFER_SUPPORTED`는 5/5 `DID>0`, exact sign-flip p=0.03125,
operation별 seed direction, 2pp absolute SESOI와 10% headroom recovery,
symmetric equivalence, retention/locality/PPL/RULER/long-gap/system matching 및
gate-mechanism 조건이 모두 통과할 때만 열린다.

## Fail-closed boundary

Official source/checkpoint/tokenizer/strict model load, official kernel dispatch,
speed, common bridge 중 hard gate가 실패하면 checkpoint/backend/model/precision,
threshold나 data를 바꾸지 않는다. Fresh immutable artifact에 registered
`BLOCKED_*` disposition과 허용·금지 claim을 기록하고 중단한다. Admission이 모두
통과하면 사용자 재승인 없이 bridge, main, evaluation과 report까지 진행한다.

Community checkpoint는 NVIDIA official weight가 아니며 `model-95b.pth`와
`model-100b.pth`가 동일 blob인 provenance 경고가 있다. 따라서 training-token
label은 uploader claim으로만 보고하고 final report에 제한을 명시한다.
