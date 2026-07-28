# E13a Sequence Floor/Throughput Calibration 결과

## 판정

| Protocol | Run | 판정 | 용도 |
|---|---|---|---|
| Original E13a | `20260727T180703.836996Z` | `CALIBRATION_PILOT_ONLY` | E13b dependency로 사용하지 않음 |
| Prospective E13a-R1 | `20260727T183609.755945Z` | `PASS / GO_FOR_E13B` | E13b 실행 dependency |

원본 E13a는 tied/dual의 seed pairing, affected-entity floor와 throughput
측정 계약이 충분하지 않아 pilot으로만 보존했다. E13a-R1은 동일
initialization, 동일 training/evaluation seed, affected-entity metric과
paired alternating timing을 evaluation 전에 고정한 독립 repair다.

## E13a-R1 주요 결과

| 지표 | Tied | Dual | 판정 |
|---|---:|---:|---|
| Affected-entity MSE | 0.0040357850 | 0.0000000586 | Dual floor PASS |
| Affected exact match | 0.494450 | 1.000000 | Exact floor PASS |
| Retention MSE | 0.0 | 0.0 | PASS |
| Forward examples/s | 23,630 | 24,270 | PASS |
| Parameter count | 123,394 | 123,394 | Matched |

Tied-minus-dual affected-MSE gain은 `0.0040357264`다. E13b-scale short
training의 실제 처리량은 tied `1,535.1`, dual `1,523.7` examples/s였고,
등록된 30,000-step run의 예상 최장 시간은 약 42분, 세 wave 합계는 약
126분이다. 등록된 각 runtime cap을 모두 통과했다.

## 개발 중 확인한 특기사항

- 기존 E13a의 whole-state exact match는 affected update floor를 직접
  검증하지 못했다. R1에서는 affected entity만 별도로 집계했다.
- Variant별 다른 seed는 architecture contrast와 training noise를
  혼합한다. R1은 initialization hash와 data seed를 tied/dual 사이에
  일치시켰다.
- Throughput은 data generation과 training을 제외한 paired alternating
  forward timing으로 측정하고, 별도로 E13b-scale short-training
  처리량을 측정했다.
- 이 결과는 calibration gate이며, 그 자체로 repeated-sequence claim을
  열지 않는다. 해당 claim은 E13b 전체 paired seed와 E13c aggregate가
  통과해야 한다.

## Immutable provenance

| 항목 | SHA-256 |
|---|---|
| R1 report | `ba5eb2df15d409b01cfde1194d21e52ef3bc16d4d2ceee15af5f7a0573b72fe3` |
| R1 manifest | `376378a8393ab2a129b4eead1e940f39439040cb8f8b28c00e171ac26aabb058` |
| Calibration metrics | `1bdeaf07e6e0c28968fe6d3dc98a318d1713c9f5c243be3f0b407a8d7e15b1a2` |
| Scale-feasibility metrics | `3dd59658cce1678c27a18579fb3b5713a06e36f4177237ab7ab7af805080acd4` |
| Protocol lock | `5a589d6a51d0b1150429c22633355b5a4b99121d06f325e58ab47b7f7d542935` |
| Tied checkpoint | `28cc04c5ac8626c51de93810d58d023f850780e9f9578c2fa5cb67e7429bc185` |
| Dual checkpoint | `68e2d2609eea902075f611d8348aa50428552ad26841dca5c98ad5bc6f201f14` |

Evidence tier는 `CONTROLLED_REFERENCE`이며 natural-language,
pretrained-LM 또는 official-backend claim에는 사용할 수 없다.
