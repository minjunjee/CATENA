# E26 Stage-2 gradient-accumulation numerical metric note

이 문서는 scientific evidence가 아닌 실행 의미론 검증 중 발견한 BF16 수치
진단을 기록한다. Registered BF16 relative-L2 threshold `0.007` 및 seed,
global-token batch, optimizer, scheduler와 model은 변경하지 않았다.

## 최초 관찰

`(4,)`, `(2,2)`, `(1,1,1,1)` microbatch partition을 비교했을 때 `(2,2)`의
다음 primary update 값은 모두 threshold 안이었다.

| 값 | relative error |
|---|---:|
| gradient | 0.00350996 |
| final parameter | 0.00096527 |
| gradient norm | 0.00000839 |

하지만 optimizer state를 tensor leaf별 relative-L2의 최댓값으로 집계하면,
norm이 거의 0인 한 Adam `exp_avg_sq` leaf에서 `0.00742971`이 관찰됐다.
해당 leaf의 max absolute discrepancy는 `1.05e-9`였다.

## 고정한 집계

기존 contract는 tensor-tree 전체의 relative-L2와 global max-absolute error다.
따라서 optimizer primary gate도 모든 floating optimizer tensor를 합친 aggregate
relative-L2와 global max-absolute error로 계산한다. Scalar step counter 및
non-floating bookkeeping tensor는 exact match를 요구한다.

같은 해석을 gradient tree에도 적용한다. Tiny actual-candidate smoke의 BF16
optimized-vs-monolithic reference gradient에서 전체 external partition gate와
logit은 통과했지만, 가장 작은 norm의 단일 gradient leaf를 primary로 사용한
초기 구현은 다음 worst-leaf relative-L2를 기록했다.

| variant/state | initial worst-leaf relative-L2 |
|---|---:|
| dual / zero | 0.0124831 |
| dual / prefilled | 0.0107780 |
| projected-tied / zero | 0.0121856 |
| projected-tied / prefilled | 0.00784634 |

이 초기 over-conservative failure는 삭제하거나 재판정하지 않는다. Registered
`relative L2`는 complete floating gradient tree를 flatten한 global L2로
구현하고, 기존 `0.007` threshold를 그대로 적용한다. External
partition-vs-full gradient 비교도 같은 registered global-tree contract로 계속
gate한다.

Worst-leaf relative-L2는 각각 `gradient_worst_leaf_error`,
`gradients_worst_leaf`, `reference_gradients_worst_leaf` diagnostic으로
receipt에 보존한다. Optimizer의 기존 `0.00742971`도
`optimizer_worst_leaf_error`로 남는다. Max-absolute error는 config에 이미
등록된 경우에만 primary gate에 사용하며 새 threshold를 추가하지 않는다.

Clarified implementation의 별도 tiny compiled-model non-evidence validation에서는
BF16 optimized-vs-monolithic global gradient-tree relative-L2가
`0.00208–0.00225`, gradient-accumulation global gradient-tree relative-L2가
`0.00154–0.00161` 범위였고 모두 기존 `0.007` 안이었다. 같은 receipt의
worst-leaf diagnostic은 최대 `0.01144`였으며 primary 판정에는 사용하지 않았다.

## Target-layout admission 보강

Tiny sequence-64 audit는 산술 구현을 검증하지만 resource sweep이 선택할 실제
context/microbatch layout을 대신하지 않는다. 따라서 resource worker는 각 candidate의
locked context와 `65,536` global input-token batch에서, 두 variant 모두에 대해
all-sequences-at-once BF16 accumulation-1을 mandatory reference로 사용한다. 여기에
resource-selected layout과 그보다 작은 모든 preregistered divisor layout을 비교한다.
Mandatory reference가 OOM이면 threshold나 layout을 완화하지 않고 preflight를
fail-closed한다.

Target-layout audit의 context, selected microbatch, accumulation step, 모든 비교
layout과 audit SHA는 `resource_preflight.json`에 포함된다. Canonical 재측정은
candidate와 token budget뿐 아니라 이 네 execution-layout 값까지 동일해야 한다.
이는 throughput 결과를 보고 batch semantics를 사후 변경하지 못하게 하는
admission contract이며, 새로운 scientific metric이나 threshold가 아니다.
