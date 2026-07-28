# E18 Scientific Claim-Boundary Audit

## 감사 범위와 결론

이 문서는 E18의 prospective lock, YAML, data/model/training source,
source-run entry point와 aggregate만 독립적으로 읽은 감사 기록이다. 실행 중인
artifact와 완료 결과는 읽거나 수정하지 않았고 locked file도 변경하지 않았다.

감사 시점의 lock SHA-256은
`7c465ceb60b6979e717d85599533bd7c0dd884f10b191fa29c42771ccc9c9989`이며,
lock에 열거된 10개 파일은 모두 기록된 hash와 일치했다.

Controller–demand 대응, paired evaluation, simpler-demand/retention
non-inferiority, 2,048-event stress와 aggregate conjunction의 구현은 등록
protocol과 일치한다. 다만 현재 allowed-claim 문구보다 좁게 해석해야 하는
식별 경계가 있다. E18은 **명시적인 oracle demand descriptor가 주어진
controlled execution-capacity 실험**이지 semantic demand inference
실험이 아니다. 또한 primary gain은 grid 평균이므로 모든 cell에서의 일관된
개선을 뜻하지 않는다.

## Controller–demand 구현 대조

| Adjacent contrast | 추가된 실제 정보/행동 자유도 | Target 생성 | 판정 가능한 내용 |
|---|---|---|---|
| tied → dual scalar | erase/write scalar 분리 | operation one-hot으로 preserve/write/erase/both 지정 | 명시된 magnitude demand에서 grid-mean adjacent gain |
| dual scalar → diagonal value | coordinate별 erase/write gate | exact contiguous channel mask 제공 | 명시된 partial-channel demand에서 grid-mean adjacent gain |
| diagonal → separate address | oracle write address를 erase address와 다르게 실행 | erase/write address가 항상 다름 | oracle-address execution에서 grid-mean adjacent gain |
| separate address → state-aware | current erase-row value를 encoder에 제공 | old value 첫 coordinate 부호로 erase-only/write-only 결정 | local current-state read가 필요한 demand에서 grid-mean adjacent gain |

모든 variant는 동일한 registered maximal head와 encoder shape를 갖지만,
forward에서 허용하는 active degree of freedom은 다르다. 따라서
`same registered parameter surface`는 맞지만 `same active control dimension`
또는 `same effective pathway`라고 표현해서는 안 된다.

## Confirmatory gate 구현 감사

### 대응 demand gain

각 seed에서 대응 demand의 `3 updates × 4 gaps = 12` cell에 대한
`baseline affected MSE − treatment affected MSE`를 평균하고, 그 다섯
seed 평균이 `0.001` 이상인지 검사한다. Statistical unit을 seed로 유지하고
episode 수를 독립 표본처럼 사용하지 않는 점은 protocol과 일치한다.

그러나 seed별 방향성이나 cell별 방향성은 이 primary gate에 포함되지 않는다.
비음수 MSE만 사용하는 구성에서도 10/12 cell이 악화되고 stress gain이
`0.0001`인 동시에 overall mean이 `0.0010083`이 되어 해당 contrast가
통과할 수 있음을 source-level counterexample로 확인했다.

### Simpler-demand와 retention guard

두 guard는 평균 상쇄를 사용하지 않는다.

- Simpler demand: 모든 seed×simpler-demand×update×gap의 **cell-mean**
  treatment degradation 중 최댓값
- Retention: 모든 seed×demand×update×gap의 **cell-mean** retention
  degradation 중 최댓값

따라서 등록 cell-mean margin에서는 strict하다. 하지만 둘 다 adjacent
baseline 대비 상대 non-inferiority일 뿐이다. Treatment의 absolute error가
낮다는 gate가 없고, 한 cell 내부 example/subgroup 간 상쇄도 배제하지 않는다.
`accurate preservation`, `near-zero retention`, `solves the demand`는 별도의
absolute 결과 없이는 허용되지 않는다.

특히 magnitude family는 네 operation을 하나의 affected MSE로 평균한다.
E02b의 PRESERVE/SUPERSEDE equivalence가 sequence setting에서 다시
검증됐다고 말할 수 없다. State-conditioned erase-only/write-only subgroup와
channel-mask width subgroup도 별도 gate가 없다.

### 2,048-event stress

등록 stress는 `updates=8, gap=2048` 한 cell이다. 각 contrast가 다섯 seed
모두에서 strictly positive이고 one-sided exact sign-flip
`p <= 0.05`여야 한다. 다섯 값이 모두 양수이면 최소 p는 `1/32=0.03125`로
구현이 맞다.

Stress에는 별도 `0.001` SESOI가 없다. 따라서 허용되는 표현은
`positive in 5/5 paired seeds at the registered stress cell`이며
`stress SESOI maintained` 또는 `uniform long-gap persistence`가 아니다.

## Pairing과 provenance

다음은 구현상 확인된다.

- 5 seeds × 5 variants의 exact source grid와 1,200 metric rows
- seed 안에서 동일 initialization hash와 registered parameter count
- AdamW, 동일 locked config와 family/data seed schedule
- variant 간 동일 evaluation seed와 verified base-transaction digest
- gap 간 동일 verified transaction/target digest
- report/manifest identity와 report hash
- checkpoint 존재와 metric row에 기록된 checkpoint hash
- protocol-lock hash와 source/config hash

Train seed stream 3,000개와 evaluation seed stream 96개는 각 registered
seed에서 겹치지 않음을 정적으로 확인했다.

한 가지 provenance 한계가 있다. E18a completion manifest는 report hash는
commit하지만 source metric/checkpoint hash를 report나 manifest에
commit하지 않는다. E18b가 aggregate 시점의 metric/checkpoint를 검증하고
hash로 고정하므로 **aggregate 이후**의 provenance는 강하지만, source-run
완료와 aggregate 사이의 byte immutability를 completion manifest 하나로
증명하지는 못한다. Paper에서는 `hash-verified at aggregation and frozen
thereafter`라고 쓰고 더 강한 end-to-end timestamp claim은 피한다.

최종 freeze는 이 구분을 다음처럼 보존한다.

- `registered_report_claim_gate`: 원 E18b report의 gate와 문구를 그대로 보존
- `audited_claim_gate`: 이 문서의 좁은 estimand·입력·stress·guardrail 해석
- `claim_boundary`: grid-mean/5-of-5 direction claim만 조건부로 열고
  every-cell, stress SESOI, absolute preservation과 외부 transfer는 닫음

따라서 원 report/history를 사후 수정하지 않으면서 paper-facing claim
boundary를 별도 provenance field로 감사할 수 있다.

## Leakage와 active-path 식별 경계

모델은 address/candidate 외에 다음 정답 구조를 직접 입력받는다.

- demand-family one-hot
- magnitude operation one-hot
- exact value-channel mask
- verified/relevance bit

`update_mask` tensor 자체는 model input에 없지만 verified bit는 정상
sequence에서 true-update와 distractor 위치를 정확히 구분한다. 따라서 이
실험은 relevance나 demand semantics를 추론하는 실험이 아니다.

정상 distractor는 hard mask로 건너뛰지 않고 encoder와 update loop를 지난다.
Random-init full-gap/no-gap delta와 verified-bit activation assay는 code-level
active path를 검증한다. 다만 activation assay는 distractor를 update-like
event로 바꾼 뒤 target은 그대로 두는 intervention이다. 이것은 hard-mask
부재를 보이지만, relevance cue 없는 distractor robustness나 natural event
selection을 입증하지 않는다.

Sequence layout도 첫 verified update 뒤의 **하나의 contiguous distractor
block**으로 제한된다. Arbitrary interleaving이나 여러 gap block으로
일반화해서는 안 된다.

## 최종 허용 claim

E18b가 등록 conjunction을 통과한 경우에만 다음 수준이 허용된다.

> In a controlled associative-memory sequence probe with oracle erase/write
> addresses, oracle candidates, explicit oracle demand descriptors, and a
> model-visible verified-event bit, each adjacent controller expansion
> achieved the preregistered mean affected-MSE gain on its matched demand over
> the registered update-by-gap grid, while cell-mean simpler-demand and
> retention degradation remained within the adjacent non-inferiority margins.
> At the registered 8-update/2,048-distractor stress cell, the matched gain was
> positive in all five paired training seeds.

더 짧은 paper 문구:

> With demand identity, operation/channel descriptors, addresses, candidates,
> and event verification supplied explicitly, the registered controller
> lattice showed matched **grid-mean** adjacent gains. The long-gap stress
> contrast was positive in 5/5 paired seeds; this does not imply a positive
> gain in every grid cell.

## 금지 표현

- every cell improved / uniform persistence across the grid
- stress SESOI was maintained
- accurate or near-zero preservation, unless separately reported descriptively
- each freedom helps only its matched demand
- the minimal sequence controller is sufficient or solves the task
- semantic demand inference or learned relevance filtering
- learned address/candidate recovery
- arbitrary event interleaving, natural language, recurrent LM, agent/planning
  또는 official-backend transfer

`selective`가 필요하면 반드시
`matched adjacent gain with simpler-demand and retention non-inferiority`로
정의하고, 다른 더 복잡한 demand에서 이득이 전혀 없다는 exclusivity 의미로
사용하지 않는다.
