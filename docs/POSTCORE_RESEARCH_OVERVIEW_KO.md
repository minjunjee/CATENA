# CATENA post-core 연구 개요

## 현재까지 확보된 핵심

1. Bounded behavioral reachable regret가 operation identity를 통제한 unseen learned error를 예측했다.
2. 독립 erase/write control은 asymmetric `ADD`와 `INVALIDATE`의 tied-control error를 제거하고 symmetric operation과 unrelated retention은 보존했다.
3. Shared diagonal control의 충분성은 demand family의 joint diagonalizability로 결정됐고, graded JD regret는 held-out application error를 정량 calibration했다.
4. Dose, transplant, rescue, scalarization이 operation-specific하게 architecture gap을 매개했다.
5. Semantic front-end factorization은 tested protocol에서 practically identifiable한 이득을 보이지 않았다. 따라서 shared semantic encoder와 geometry-matched memory controller를 분리한다.

## 다음 중심 질문

> Controlled geometry에서 발견한 설계 원리가, control operator와 representation을 실제로 학습하고 여러 update를 sequence로 처리하는 모델에서도 필요한 architecture를 예측하는가?

## 새 가설

### P1. Learned rank sufficiency

Best-rank oracle이 아니라 learned rank-r controller도 held-out transaction에서 해당 reachable floor에 접근하며, 필요한 최소 learned rank는 target operator family의 intrinsic rank와 함께 증가한다.

### P2. Representation-control co-adaptation

하나의 shared representation은 common-rotated commuting family를 diagonal하게 만들 수 있지만 noncommuting family 전체를 동시에 diagonalize할 수 없다. 후자에서는 richer control이 필요하다.

### P3. Architecture-demand lattice

Magnitude factorization, coordinate granularity, erase/write address separation, state conditioning은 각각 이를 요구하는 demand family에서만 선택적 이득을 제공한다.

### P4. Transactional event-sequence transfer

Shared semantic encoder 뒤에서 독립 erase/write control은 repeated updates와 long distractor gap에서도 tied control보다 correction을 높이고 retention을 유지한다.

### P5. Plan-state continuation

Update 이전에 형성된 structured plan state가 있는 경우 one-time assimilation은 stale plan을 교정하면서 unaffected fields를 유지한다.

## Claim boundary

새 실험은 H1-H4를 변경하지 않는다. Reference operator와 event-sequence 결과를 pretrained recurrent LM, general agent, world model 또는 Transformer superiority로 확대 해석하지 않는다.
