# E10b — Numerical-floor-aware learned rank scaling

## 원본 보존과 목적

원본 E10 main run `20260727T184326.484361Z`의 판정은 다음과 같이
변경하지 않는다.

```text
execution_status: PASS
original_e10_claim_status: NOT_OPENED
reason: STRICT_ALL_PAIR_MONOTONICITY_FAILED_AT_NUMERICAL_SATURATION
mean_strict_monotonic_fraction: 0.59
```

원본 E10은 40/40 cell에서 exact-target recovery `0.95`를 처음 만족하는
learned rank가 intrinsic rank와 일치했고, seedwise minimum rank도
8/8에서 nondecreasing이었으며, low-vs-high contrast는 8-seed exact
sign-flip `p=0.00390625`였다. 그러나 recovery threshold를 이미 넘은
controller들 사이의 `1e-9`–`1e-7` numerical-floor fluctuation도 strict
monotonicity 위반으로 세어 full gate가 열리지 않았다.

E10b는 원본 결과를 재판정하지 않는다. 240개 checkpoint를 동결한 채
새 descriptor test namespace에서, saturation 이전에 식별 가능한
monotonicity만 prospective하게 평가한다.

## 동결 사항

- 원본 checkpoint 240개를 retrain하거나 선택하지 않는다.
- 원본 report, manifest, metrics와 checkpoint-set digest를 고정한다.
- 원본 test descriptor와 metric row는 평가 outcome으로 재사용하지 않는다.
- demand family의 `U`, `V`, coefficient map과 bias는 동일하게 재구성한다.
- training seed 8개가 그대로 독립 통계 단위다.
- intrinsic-rank grid `{1,2,4,8,16}`과 learned-rank grid
  `{1,2,4,8,16,32}`를 유지한다.
- exact-target recovery threshold `0.95`, `max_rank_factor=2`,
  minimum-rank match fraction `0.8`, one-sided exact sign-flip
  `alpha=0.05`를 유지한다.

## Fresh test namespace

각 source training seed \(s\)와 intrinsic rank \(k\)에서 test descriptor
seed는 평가 전에 다음처럼 고정한다.

\[
\operatorname{seed}_{E10b}(s,k)
=62{,}000{,}000+10{,}000s+k.
\]

이는 원본 E10의 \(30{,}000s+k\)와 겹치지 않는다. 각 cell에서 2,048개의
descriptor를 새로 생성하며 원본 test prediction이나 error는 불러오지
않는다.

## Prospective monotonicity repair

learned rank가 오름차순인 adjacent pair \((r_i,r_{i+1})\)에 대해 lower
rank의 fresh-set exact-target recovery가 `0.95` 미만일 때만
`ELIGIBLE_PRE_SATURATION`으로 등록한다.

\[
\operatorname{recovery}(r_i)<0.95
\quad\Longrightarrow\quad
E(r_{i+1})\le E(r_i).
\]

lower rank가 이미 `0.95`를 만족하면 해당 pair는
`SATURATED_EXCLUDED`로 기록한다. 결과 row는 버리지 않지만 monotonicity
분모에는 포함하지 않는다. 새로운 tolerance나 error floor는 도입하지
않으며, eligible pair는 raw error가 엄밀히 non-increasing이어야 한다.

## Gate

다음을 모두 요구한다.

1. 모든 frozen checkpoint 240개의 SHA-256이 source metric row와 일치
2. eligible pre-saturation adjacent pair의 100%가 non-increasing
3. exact-target recovery로 정한 minimum sufficient rank가 기존
   `[intrinsic rank, 2 × intrinsic rank]` tracking gate를 만족
4. 각 seed에서 minimum sufficient rank가 intrinsic rank에 따라
   nondecreasing
5. low-rank 대 high-rank gain이 8-seed one-sided exact sign-flip 통과

Dry-run은 source checkpoint 240개의 hash를 모두 확인하되 축소된 fresh
evaluation만 수행하며 claim을 열 수 없다.

## Claim boundary

성공 시 허용되는 주장은 smooth synthetic operator family의 fresh
descriptor set에서 learned control rank가 demand intrinsic rank를
추적하고, 식별 가능한 pre-saturation 구간의 error가 rank와 함께
non-increasing이라는 것이다.

Evidence tier는 `CONTROLLED_REFERENCE`다. Official backend, pretrained
language model, natural-language transaction 또는 보편적 scaling law
주장은 열지 않는다.
