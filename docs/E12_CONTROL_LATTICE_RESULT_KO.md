# E12 Architecture-Demand Control Lattice 결과

## 판정

```text
execution_status: PASS
architecture_demand_lattice_status: SUPPORTED
evidence_tier: CONTROLLED_REFERENCE
```

Artifact-complete main run은 `20260727T184511.437394Z`다. 앞선 supported
run `20260727T182449.721061Z`의 metric/gate를 바꾸지 않고 checkpoint와
schema-v2 provenance만 추가했으며, 네 contrast 값이 정확히 재현됐다.

## 주요 결과

| 추가 control freedom | 대응 demand | Mean selective MSE gain | 더 단순 task 최대 악화 | Exact sign-flip |
|---|---|---:|---:|---:|
| Tied → dual scalar | Magnitude factorization | 0.0065376581 | 0.0 | 0.00390625 |
| Dual scalar → diagonal value | Value granularity | 0.0086808868 | -0.0000002104 | 0.00390625 |
| Diagonal → separate address | Address decoupling | 0.0156250368 | 0.0000001155 | 0.00390625 |
| Separate address → state-aware | State conditioning | 0.0131247569 | 0.0000574804 | 0.00390625 |

네 selective gain은 모두 등록 SESOI `0.001`을 넘었고, simpler-demand
non-inferiority margin `0.0005` 안에 들었다. 각 contrast는 8/8 seed에서
같은 방향이었다.

## 개발 중 확인한 특기사항

- 초기 extension은 value-granularity mask 시작점을 model input에
  노출하지 않았고 state-conditioned target/readout index도 불일치했다.
  Main 전에 demand generator와 controller를 수정했다.
- Variant마다 다른 initialization RNG를 쓰던 부분을 고쳐, 같은 seed의
  모든 controller가 동일 maximal parameter surface와 초기 tensor를
  공유하도록 했다.
- Retention denominator의 hard-coded slot 수를 실제 unaffected count로
  교체했다.
- 첫 corrected run은 결과는 supported였으나 checkpoint가 없어
  artifact contract가 불완전했다. Config, metric, threshold를 바꾸지
  않고 새 run을 만들었고, 40개 checkpoint hash를 160개 metric row에서
  검증했다.

## Immutable provenance

| 항목 | SHA-256 |
|---|---|
| Report | `5d300ea84fbe004370a2a44854637b2d60a0a0ccf5c63fc2ee2a24bce8fa2562` |
| Run manifest | `6af5002065a2d53af6bc445ee89c3457aaf4b0828e36fbb30d0080209eb1207d` |
| Metrics | `594d22c7d2a076931a9187c171da0231cfbff289709266814c437582406d57e8` |
| Artifact-completion lock | `aaa6d423f6860ea89c2d34d8425517f67129296d3fc46ffc4a97c5857d4b9198` |

이 결과는 architecture-demand selectivity를 지지하지만 official
GDN2/KVEraser, language model 또는 runtime superiority claim은 열지
않는다.
