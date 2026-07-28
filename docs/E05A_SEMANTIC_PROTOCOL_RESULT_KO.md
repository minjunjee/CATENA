# E05a Semantic Protocol and Leakage Lock — 공식 결과

## 판정

| 항목 | 결과 |
|---|---|
| Exact run | `20260727T081532.073522Z` |
| Execution | `PASS` |
| Static design gates | 4/4 PASS |
| Registered statistical checks | 9/12 PASS |
| E05a design | `NO_GO` |
| E05b registry | 생성하지 않음 |
| Human audit | 생성·실행하지 않음 |
| H5-lite | `NOT TESTED / NOT OPENED` |
| Evidence tier | `CONTROLLED_REFERENCE` |

E05a는 H5 confirmatory evidence가 아니라 E05b 진입 전 design-validity
preflight다. 따라서 이 결과는 H5의 반증이 아니다. 누수·control assay는
작동했지만 learned controller가 seen `PRESERVE/ADD/INVALIDATE`의 새
namespace에서 사전 성능 조건을 만족하지 못해 E05b를 열지 않은 결과다.

## 실패한 사전 gate

| Gate | Estimate | 95% paired-bootstrap CI | 사전 기준 | 판정 |
|---|---:|---:|---:|---|
| Factorized−oracle affected excess | 0.0059955 | [0.0057477, 0.0062518] | upper ≤ 0.0005 | FAIL |
| Shared−oracle affected excess | 0.0072113 | [0.0069503, 0.0074664] | upper ≤ 0.0005 | FAIL |
| Shared−factorized affected parity | 0.0012158 | [0.0009920, 0.0014460] | CI inside ±0.0005 | FAIL |

Shared−factorized seed 효과는 `0.001583`, `0.002161`, `0.000464`,
`0.000655`로 네 seed 모두 shared가 더 나빴다. 이 차이를 held-out
`SUPERSEDE`의 factorization 효과로 해석해서는 안 된다. E05a에는
`SUPERSEDE`가 없고, 본 차이는 seen-operation preflight 실패다.

## 통과한 leakage·control 조건

Oracle affected/retention, factorized/shared retention excess와 retention
parity는 모두 정확히 0이었다. Forbidden access, namespace integrity,
parameter/MAC/initialization/schedule match, E05a 내 `SUPERSEDE` 부재도
통과했다.

Formal control estimand는 `ADD/INVALIDATE` equal-weight다.
`pilot_seed_effects.jsonl`의 P/A/I 전체 평균이 아니라 아래 등록 수치를
사용한다.

| Negative control degradation | Estimate | 95% paired-bootstrap CI | 판정 |
|---|---:|---:|---|
| Shuffled fields | 0.0354732 | [0.0347199, 0.0362037] | PASS |
| Wrong address | 0.0249852 | [0.0246681, 0.0252901] | PASS |
| Transaction-only | 0.0101612 | [0.0098534, 0.0104642] | PASS |
| State-only | 0.0215773 | [0.0211626, 0.0219790] | PASS |
| Wrong semantics | 0.0327849 | [0.0321750, 0.0333768] | PASS |

다섯 control은 모두 lower CI가 SESOI `0.001`보다 컸고 네 seed의 raw
방향도 모두 양수였다. 따라서 `wrong-address`와 `transaction-only` assay가
쉬워서 생긴 NO_GO는 아니다.

## 개발 단계 진단

동결 checkpoint를 재학습하지 않고 train/validation을 replay한 사후
descriptive 진단은 다음과 같다.

| Model | Train affected MSE | Validation affected MSE | 배율 |
|---|---:|---:|---:|
| Factorized | 0.000353 | 0.005996 | 17.0× |
| Shared | 0.000293 | 0.007211 | 24.6× |

Operation별 validation gate 평균도 목표 demand에 충분히 수렴하지 않았다.

| Model | PRESERVE `(e,w)` | ADD `(e,w)` | INVALIDATE `(e,w)` |
|---|---:|---:|---:|
| Factorized | (0.194, 0.076) | (0.025, 0.874) | (0.622, 0.054) |
| Shared | (0.317, 0.081) | (0.038, 0.830) | (0.609, 0.029) |
| Target | (0, 0) | (0, 1) | (1, 0) |

2,400×128 update는 약 85 dataset pass이며 train behavior는 대부분
near-oracle이었다. 따라서 단순 step 부족보다는 새 namespace에서 안정적인
version/time/scope 관계를 학습하지 못한 일반화 문제가 더 잘 맞는다.

사후 feature ablation은 gate 입력의 32-D state read를 0으로 만들었을 때
train MSE는 악화됐지만 validation MSE는 factorized에서 20.9%, shared에서
15.1% 개선됐다. 이는 random state content와 opaque nuisance field에 대한
shortcut 의존을 시사한다. 이 ablation은 exploratory diagnosis이며 E05a
판정을 바꾸거나 새 claim을 열지 않는다.

## 구현·provenance 특기사항

- Runner는 `python -m experiments...`로 실행했다. 파일 경로 직접 실행 시
  repository root가 import path에서 빠져 `ModuleNotFoundError`가 날 수 있다.
- E05 결과 전 protocol/config/hash를 고정했고 legacy E05는 사용하지 않았다.
- Learned forward/update context에는 visible semantic feature, state, address,
  incoming value와 candidate scale만 들어간다. Private operation, oracle
  demand, target state는 loss/scoring 또는 명시적 oracle branch에만 존재한다.
- Shuffled control은 multi-donor fieldwise derangement이며 wrong-semantics와
  같은 대체 demand를 유도한다. 모든 mapping은 outcome-independent다.
- Main run이 `NO_GO`였으므로 E05b registry, paraphrase set, audit item과
  reviewer template은 생성되지 않았다.
- 동일 protocol/config/namespace 재실행이나 threshold 변경은 금지한다.

## 연구 흐름에서의 짧은 해석

H1–H4 controlled-mechanism core의 판정은 변하지 않는다. E05a는 semantic
anchor를 추가하기 전에 shortcut-sensitive controller generalization 문제를
검출해 confirmatory E05b 진입을 차단했다. 현재 논문은 H1–H4로 마감할 수
있으며 semantic transfer는 미해결 한계로 남는다.

이 진단을 바탕으로 별도 hash-lock과 새 namespace를 쓴 단 한 번의
`E05a-R1`을 수행했다. State read를 gate에서 제거하고
version/time/scope 관계만 사용하자 두 controller 모두 oracle 근처로
일반화했지만, shared controller에도 충분한 headroom이 없어 R1은 다시
`NO_GO`였다. 자세한 판정은
`docs/E05A_R1_SEMANTIC_DESIGN_REPAIR_RESULT_KO.md`에 별도로 보존한다.
사전등록대로 이번 제출에서는 추가 semantic repair를 하지 않는다.

## Immutable sources

| Artifact | SHA-256 |
|---|---|
| `report.json` | `34bab0288d5bbe82e1debcfa81e51493f4fa280475b86f0d097cb2a4aff8057c` |
| `run_manifest.json` | `c2571fa8c4ec184068dff3bb002dc08be1c503c147348bb19ada1fd1199b5e2b` |
| `pilot_semantic_metrics.jsonl` | `3b74eef5032628ede316799e57e030cd46df0bdf73bba1437ecc01712f016463` |
| `pilot_control_pairings.jsonl` | `be1bb055888d73e39af3293c50c0eb5f1ee79291071c5bd9f0716efe82044ec9` |
| `E05A_ARTIFACT_FREEZE_V1.json` | `f6e6edebd303fb1b6d48cff9630516a8864dc317386778202da58a2a6c189122` |

Canonical artifact root는 `/data/minjun_dev/CATENA/artifacts`이며 repository의
`artifacts` symlink가 같은 위치를 가리킨다.
