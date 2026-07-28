# E20 Quality-Constrained Break-Even Prospective Protocol

## 상태와 연구 질문

이 문서는 GPU main timing을 보기 전에 고정한 E20 protocol이다. E14,
E19의 source·config·artifact·claim을 수정하거나 재판정하지 않는다.

E20의 질문은 다음으로 제한한다.

> 동일한 structured state, update, query workload에서 update를 한 번
> internal state에 assimilate하는 정책이 세 external-state baseline보다
> 언제 quality guardrail을 만족하면서 total latency 기준으로 break-even
> 하는가?

이 실험은 실제 storage service, network, official recurrent backend 또는
pretrained language model을 실행하지 않는 controlled microbenchmark다.
따라서 main이 실행되더라도 evidence tier는
`CONTROLLED_SYSTEMS_PROXY`, `scientific_evidence=false`다.

## 고정 workload

- State: batch × 128 slot × 64 value channel의 structured tensor
- Update: episode마다 한 slot의 현재 값을 새 value로 교체
- Query grid:
  \(m\in\{1,2,4,8,16,32,64\}\)
- 한 logical query는 다음 두 read를 함께 포함한다.
  1. updated slot의 affected correction read
  2. update와 겹치지 않는 matched retention read
- Retention address는 같은 episode 안에서 중복되지 않는다.
- 모든 \(m\)은 같은 initial state와 update를 사용하고 query prefix만
  확장한다.

각 \(m\)에 대해 initial state, canonical post-update state, update,
query address, compact-cache mapping 전체의 canonical SHA-256을
계산한다. 네 policy row의 `paired_workload_sha256`이 다르면 hard fail한다.
State/update만의 `base_workload_sha256`도 모든 \(m\)에서 같아야 한다.

## 네 정책과 timing boundary

| Policy | Timed operation | Timed region 밖에서 허용되는 준비 |
|---|---|---|
| `one_time_internal_assimilation` | device-resident state의 affected slot을 한 번 갱신하고 \(m\) query pair를 read | persistent initial state와 query descriptor의 device 배치 |
| `external_canonical_state_per_query` | 각 query마다 CPU canonical post-update state에서 두 row를 선택하고 device로 전달 | query descriptor |
| `retrieve_once_cached_compact_snapshot` | 현재 query set에 필요한 affected/retention row만 한 번 모아 device로 전달한 뒤 cache에서 answer | compact inverse mapping |
| `full_refresh` | CPU canonical post-update state 전체를 device로 복사한 뒤 query | query descriptor |

External policy의 CPU canonical state는 external-state proxy이고 실제
database/network latency를 나타내지 않는다. Internal initial-state load는
persistent memory라는 estimand에 따라 timing 밖이며, assimilation update는
timing 안이다. Cached snapshot이 internal assimilation보다 빠르면 이를
제거하거나 불리하게 만들지 않는다.

각 policy × \(m\)에서:

1. 등록 warmup을 수행한다.
2. 각 measured repeat 전에 policy context를 새로 준비한다.
3. CUDA main에서는 시작 직전과 종료 직후
   `torch.cuda.synchronize(device)`를 호출한다.
4. total seconds의 raw repeat 전체, median, minimum, maximum과
   per-logical-query median을 저장한다.

Primary latency statistic은 median total seconds다.

## Quality guardrail

기존 E14/E13 계열의 절대 기준을 변경하지 않고 다음처럼 고정한다.

| Metric | 등록 epsilon |
|---|---:|
| Affected correction MSE \(\epsilon_c\) | `0.001` |
| Unaffected retention MSE \(\epsilon_r\) | `0.0005` |

두 비교 policy가 모두 자기 absolute guardrail을 만족한 cell만 break-even
후보가 된다. Production error budget이나 statistical equivalence
margin으로 재해석하지 않는다.

## Primary estimand와 판정

Internal policy를 \(I\), baseline을 \(b\)라 할 때:

\[
m_b^\star =
\min\{m:
E_{c,I},E_{c,b}\le\epsilon_c,\;
E_{r,I},E_{r,b}\le\epsilon_r,\;
L_I(m)\le L_b(m)\}.
\]

세 baseline 각각의 \(m_b^\star\)를 별도로 보고한다. 등록 grid에서
후보가 없으면 `null`이며 grid 밖으로 extrapolation하지 않는다.

`retrieve_once_cached_compact_snapshot`이 모든 등록 \(m\)에서 quality
guardrail을 만족하고 internal latency 이하이면 최종 판정은 그대로:

```text
claim_gate.status: NOT_SUPPORTED_BOUNDARY
reason: CACHED_COMPACT_SNAPSHOT_DOMINATES_REGISTERED_GRID
```

그 결과를 숨기거나 cached baseline을 사후 제거하지 않는다. Cached
dominance가 없고 세 baseline 모두에서 \(m^\star\)가 관측된 경우에만
`SUPPORTED_CONTROLLED_SYSTEMS_PROXY`다. 그 밖에는
`NOT_SUPPORTED_BOUNDARY`다.

## 실행 분리

```text
CPU + --dry-run
  → implementation/artifact contract만 확인
  → claim_gate = NOT_EVALUATED_DRY_RUN

CUDA + MAIN
  → 등록 timing grid
  → controlled systems-proxy 판정만 가능
```

CPU main과 CUDA dry-run은 모두 hard fail한다. Dry-run 결과를 main
evidence에 합치지 않는다.

## Artifact 계약

모든 실행은 새 UTC run directory에 다음을 쓴다.

- `config.resolved.yaml`
- `environment.json`
- `run_manifest.json`
- `quality_break_even_metrics.jsonl`
- `report.json`

Metrics에는 raw timing repeats, 두 quality metric, workload digest,
device/run mode와 evidence boundary가 포함된다. Protocol lock은 artifact
생성 전에 검증하고 실행 종료 직전에 다시 검증한다.

## Claim 경계

성공 시 최대 허용 문장:

> On the registered controlled structured-state workload, one-time internal
> assimilation reached the reported quality-constrained latency break-even
> points relative to the three in-process external-state proxies.

금지:

- production storage/network break-even
- official GDN2/KDA/KVEraser runtime
- pretrained language-model 또는 agent efficiency
- general semantic transaction assimilation
- E14 proxy를 general planning evidence로 재표기
- E19 fixed-code localization을 natural-language localization으로 재표기

