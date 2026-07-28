# E05a-R1 Semantic Design Repair — 공식 결과

## 판정

| 항목 | 결과 |
| --- | --- |
| Exact main run | `20260727T142609.591935Z` |
| Execution | `PASS` |
| Static design gates | 9/9 PASS |
| R1 design | `NO_GO` |
| Original E05a | `NO_GO` 유지 |
| Human audit pool | 생성하지 않음 |
| E05b-R1 registry/training | 생성·실행하지 않음 |
| H5 | `TERMINATED_NOT_REFUTED` |
| Evidence tier | `CONTROLLED_REFERENCE` |

E05a-R1은 원본 E05a의 네 seed나 episode를 합치지 않고, 8개 fresh
training seed와 root `6_000_000_000_000`의 새 train/validation namespace에서
수행한 단 한 번의 prospective representation repair다. 실행은
성공했지만 사전등록 conjunction을 통과하지 못했다. 따라서 H5가
반증된 것은 아니며, 사전등록대로 이번 제출에서 추가 semantic repair와
E05b-R1을 종료한다.

## Primary 결과

Primary estimand는 seed마다 `ADD`와 `INVALIDATE`에 같은 가중치를 준
shared−factorized affected-read MSE다.

| 지표 | 값 | 사전 기준 | 판정 |
| --- | ---: | ---: | --- |
| Mean gain | 0.0000118048 | ≥ 0.001 | FAIL |
| Seed-cluster 95% CI | [0.0000031714, 0.0000191686] | lower > 0 | PASS |
| Exact sign-flip | \(p=0.01953125\) | ≤ 0.05 | PASS |
| Seed direction | 7/8 positive | 8/8 positive | FAIL |

절대 gain은 SESOI의 약 1.18%다. Shared 대비 상대 차이는 약 33.3%지만,
두 모델의 error 자체가 수치적으로 매우 작기 때문에 이 상대값을
factorization evidence로 사용하지 않는다.

| Seed | Factorized A/I MSE | Shared A/I MSE | Shared−Factorized |
| ---: | ---: | ---: | ---: |
| 1103 | 0.0000200010 | 0.0000342061 | 0.0000142052 |
| 2207 | 0.0000428691 | 0.0000573190 | 0.0000144498 |
| 3301 | 0.0000158623 | 0.0000366169 | 0.0000207546 |
| 4409 | 0.0000152767 | 0.0000188851 | 0.0000036084 |
| 5501 | 0.0000191244 | 0.0000255431 | 0.0000064187 |
| 6607 | 0.0000307802 | 0.0000185240 | -0.0000122562 |
| 7703 | 0.0000166691 | 0.0000379141 | 0.0000212450 |
| 8807 | 0.0000287069 | 0.0000547198 | 0.0000260130 |

## 실패 원인

```text
INSUFFICIENT_ORACLE_HEADROOM
PRIMARY_MEAN_BELOW_SESOI
PRIMARY_SEED_DIRECTION_INCONSISTENT
```

| Gate | Estimate | 95% seed-cluster CI | 사전 기준 | 판정 |
| --- | ---: | ---: | ---: | --- |
| Factorized A/I oracle excess | 0.0000236612 | [0.0000181046, 0.0000303383] | upper ≤ 0.0005 | PASS |
| Shared A/I oracle headroom | 0.0000354660 | [0.0000261373, 0.0000453058] | lower > 0.001, 8/8 > 0.001 | FAIL |

핵심은 factorized가 학습에 실패한 것이 아니라 shared도 oracle 근처까지
학습했다는 점이다. 현재 6-D relational representation에서는 P/A/I seen
demand가 parameter-matched shared controller에도 충분히 쉬워, factorized
우위를 검정할 등록 headroom이 생기지 않았다.

## 통과한 guardrail

| Gate group | 결과 |
| --- | --- |
| Oracle affected / retention | 정확히 0, PASS |
| Factorized asymmetric oracle excess | PASS |
| Factorized/shared `PRESERVE` | 각각 0.000141816 / 0.000182600, PASS |
| Factorized/shared retention | 정확히 0, PASS |
| Retention non-inferiority | \(p=0.00390625\), PASS |
| Forbidden access / namespace / 11-stratum balance | PASS |
| Wrong-address norm match / matched parameter-compute | PASS |

다섯 control도 모두 CI 하한이 SESOI 0.001보다 컸고 shifted exact
sign-flip \(p=0.00390625\), 8/8 raw positive였다.

| Control degradation | Estimate | 95% seed-cluster CI |
| --- | ---: | ---: |
| Shuffled fields | 0.0462011 | [0.0459771, 0.0464274] |
| Wrong address | 0.0312263 | [0.0312196, 0.0312320] |
| Transaction-only | 0.0156264 | [0.0156221, 0.0156297] |
| State-only | 0.0312158 | [0.0311998, 0.0312283] |
| Wrong semantics | 0.0462011 | [0.0459708, 0.0464295] |

## 개발 단계 특기사항

| Run | 상태 | 내용 |
| --- | --- | --- |
| `20260727T142227.284941Z` | incomplete dry-run | 기존 E05a fieldwise donor search가 R1 factorial에서 valid derangement를 찾지 못해 학습 전 중단 |
| `20260727T142548.301652Z` | completed dry-run | R1 전용 outcome-blind multi-donor pairing 추가 후 `PASS / NOT_EVALUATED_DRY_RUN`; static 9/9 |
| `20260727T142609.591935Z` | main | `PASS / NO_GO`; 아래 immutable hash로 동결 |

원본 E05 control 코드는 수정하지 않았다. R1 pairing은 먼저 coherent
operation-changing donor를 고정하고, demand에 관여하지 않는 두 predicate
field와 nuisance field를 다른 donor에서 가져오며, wrong address는 같은
base state에서 candidate norm을 맞춘다. 모든 mapping은 outcome-blind다.

이 구현에서 shuffled-fields와 wrong-semantics의 gate-visible relational
content는 같고, 두 조건의 차이는 encoder가 사용하지 않는 field에만 있다.
따라서 두 control estimate가 같은 것은 예상되는 결과이며, 이 둘을 독립된
두 종류의 semantic robustness evidence로 세지 않는다. Main이 다른 이유로
`NO_GO`였으므로 이 중복성은 판정을 유리하게 바꾸지 않는다.

원본 E05a 대비 validation affected MSE는 factorized에서 약 99.61%,
shared에서 약 99.51% 감소했다. 이는 raw relational representation이
fresh-namespace shortcut 문제를 실질적으로 수리했다는 개발 진단이다.
그러나 shared도 함께 수리되어 factorized-specific semantic advantage는
확보되지 않았다.

## 연구 흐름에서의 짧은 해석

H1–H4 controlled core는 변하지 않는다. R1은 좋은 update-control geometry와
semantic demand inference가 별도 병목이라는 경계를 더 명확히 했다.
정확한 결론은 “semantic representation은 안정화됐지만, seen P/A/I에서
factorized interface의 등록된 추가 이점은 없었다”이다.

## Immutable sources

| Artifact | SHA-256 |
| --- | --- |
| `report.json` | `fdb1a397ccc526f9546b473d63e2ab3351529f184879baaefb3f686b505f6bb3` |
| `run_manifest.json` | `e8c75ef56a78c64169aa5c03e46b0267d1805b20b3dc4fc8dc48d588f2f0e2fd` |
| `r1_semantic_metrics.jsonl` | `3aad00926db3bed028e35ee8bfcde8ce0be02ce9c582858422c71fe0b17e3c6a` |
| `r1_seed_effects.jsonl` | `f5a59a5be634f96646cbf11807464574165ca36168ef46d7e71bb4958ef7ae4f` |
| `r1_control_pairings.jsonl` | `c9465d0f55d6180e0c89fff41ece7bed995ca2e12414b414e2d2e0529670525b` |
| `E05A_R1_ARTIFACT_FREEZE_V1.json` | `b5cdf6036d25060d2bb05d77cd712769c141bf8a8ced0e6944ad78c95ed12aad` |
| `E05A_R1_CLAIM_STATUS.json` | `6847805450e30d08c5b6216865a0ea9c0459b8f7a7c485f3973604b84829b238` |

Canonical artifact root는 `/data/minjun_dev/CATENA/artifacts`다.
