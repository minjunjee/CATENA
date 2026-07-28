# E10 Learned Control-Rank Scaling 결과

## 판정

```text
execution_status: PASS
rank_tracking_status: SUPPORTED_DESCRIPTIVELY
strict_monotonicity_status: FAILED
full_e10_claim_open: false
failure_diagnosis: POST_SATURATION_NUMERICAL_FLOOR_SENSITIVITY
```

Main run은 `20260727T184326.484361Z`다. 원본 report와 gate는 수정하지
않는다.

## 주요 결과

| 지표 | 결과 | Gate |
|---|---:|---:|
| Minimum-rank tracking | 40/40 cell, 1.000 | `>=0.8` PASS |
| Seedwise minimum-rank nondecreasing | 8/8 seed, 1.000 | PASS |
| Low-vs-high rank sign-flip | `p=0.00390625` | PASS |
| Mean strict monotonic fraction | 0.590 | `>=0.9` FAIL |

모든 seed에서 최소 qualifying learned rank는 정확히 다음과 같았다.

| Intrinsic rank | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| Minimum qualifying rank | 1 | 2 | 4 | 8 | 16 |

즉 rank-tracking 신호 자체는 완전하게 일치했다. Full gate가 닫힌 이유는
충분 rank에 도달한 뒤 test error가 numerical floor
(`약 1e-9`–`2e-7`)에서 미세하게 오르내린 pair도 strict violation으로
계산했기 때문이다. 관찰된 최대 post-floor 증가는 약 `4.36e-8`이다.
반면 exact-target recovery가 등록 threshold `0.95`에 도달하기 전 rank
구간에서는 error가 큰 폭으로 감소했다.

## 개발 중 확인한 특기사항

- 최초 extension gate는 rank 1이 모든 intrinsic family를 풀어도 통과할
  수 있는 upper-bound-only 구조였다. Main 전에 lower bound와 seedwise
  nondecreasing 조건을 prospective하게 추가했다.
- 세 incomplete run은 report가 생성되지 않았으며 claim에 사용하지
  않는다: `20260727T180703.792069Z`,
  `20260727T180747.984294Z`, `20260727T183537.502519Z`.
- 최종 main은 240개 model checkpoint를 저장했고 metric row의 SHA-256과
  모두 일치한다.
- Strict monotonic gate 실패를 사후 수정하지 않는다. 별도 E10b에서
  checkpoint를 동결하고 fresh descriptor test를 사용해, 기존 0.95
  threshold 이전의 pre-saturation monotonicity만 prospective하게
  평가한다.

## Immutable provenance

| 항목 | SHA-256 |
|---|---|
| Report | `1c04ee78b184934ff233c2ef1786005557020c8b48f47169449088dbd51dc562` |
| Run manifest | `f358ae8d8cbe4c55cbf1e2f230dcf00c54f064aeeeec8b8114dcbab4e960136b` |
| Main metrics | `2d396f7edcfe4d2d00d569536edccf394022a07422a9b40fe96fad25f2e20333` |
| Seed effects | `0c4a71dc286abb1ca7e4ce5c83c66acfde7b4572569861e799a4118f18e679d8` |
| Cell tracking | `c319843574528df24e271e83da3aecfafa8f56a0c03476948cebfe343a7068f9` |
| Seed tracking | `1c0cb948c98d84176b0fbd04e6a67320cf9817d240ddc9b48202869f7628c23f` |
| Protocol lock V2 | `a2baae578fe3f932284ef253209c3eefbc764a6a7e57abe58e19d802279833e3` |

Evidence tier는 `CONTROLLED_REFERENCE`다.
