# E26 scientific data-mix implementation amendment

## 발견된 구현 결함

기존 `PairedTrainingCursor`는 일반 corpus sequence 네 개와 transaction
sequence 한 개를 번갈아 내보냈다. 일반 sequence는 4,096 token을 거의 모두
사용하지만 transaction sequence의 평균 non-padding 길이는 약 273 token이었다.
따라서 nominal 4:1 sequence 비율은 실제 loss-bearing token 기준으로 약
98.36% general / 1.64% transaction이었고, 등록한 80/20 token mix를 구현하지
못했다.

이 결함은 model outcome 또는 E26 metric을 보기 전에 발견됐다. Canonical E26a,
E26b, E26c는 시작되지 않았으며 기존 legacy cursor와 non-evidence 결과는
삭제하거나 재판정하지 않는다.

## Prospective repair

Scientific path만 다음 versioned cursor를 사용한다.

1. `PackedTransactionCursor`는 transaction example을 자르지 않고 고정 context에
   deterministic하게 packing한다.
2. `TokenBalancedPairedTrainingCursor`는 다음 complete transaction row와 general
   row 사이에서 누적 `|4T-G|`를 최소화한다.
3. 최초 1M actual-token probe와 resume 후 1M probe가 byte-identical paired
   replay를 보이고 다음 bound를 만족해야 한다.

```text
abs(4 * transaction_nonpadding_tokens - general_nonpadding_tokens)
<= 4 * context_length
```

Complete example은 indivisible하므로 정확히 0.800000인 비율을 요구하거나
주장하지 않는다. 첫 synthetic contract probe의 realized mix는
80.084% / 19.916%였으며, canonical run은 exact token count와 fraction을 다시
기록한다.

## Runner contract

- throughput과 pilot budget은 nominal padded allocation이 아니라 actual
  non-padding token으로 보고한다.
- loss-mask denominator와 source별 valid-prediction token accounting이
  일치하지 않으면 즉시 차단한다.
- paired variants의 byte/cursor/token exposure가 다르면 차단한다.
- bounded-discrepancy gate가 실패하면 candidate selection과 E26b dependency를
  열지 않는다.
- operation, metric, seed, scientific threshold는 변경하지 않았다.
