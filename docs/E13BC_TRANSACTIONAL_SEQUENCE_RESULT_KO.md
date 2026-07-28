# E13b-R1 / E13c-R1 Transactional Sequence 결과

## 판정

```text
e13b_r1_execution_status: PASS (10 / 10 runs)
e13c_r1_execution_status: PASS
e13c_r1_claim_status: SUPPORTED
evidence_tier: CONTROLLED_REFERENCE
scientific_evidence: false
```

E13c-R1 aggregate run은 `20260727T214126.954177Z`다. 등록된 5개 paired
seed, 2개 controller, 3개 update 수, 4개 distractor gap으로 구성된
60개 paired cell을 모두 집계했고, 12개 claim gate를 모두 통과했다.

## Dependency 경계

| 단계 | Run | 최종 용도 |
|---|---|---|
| Original E13a | `20260727T180703.836996Z` | 초기 calibration pilot; main dependency 아님 |
| E13a-R1 | `20260727T183609.755945Z` | hard-masked distractor pipeline calibration으로만 보존 |
| E13a-R2 | `20260727T190642.222102Z` | learned-distractor R1 pipeline의 유일한 GO dependency |
| E13b-R1 | 아래 10개 run | 5-seed tied/dual main |
| E13c-R1 | `20260727T214126.954177Z` | paired aggregate와 confirmatory gate |

Original/R1 pipeline에서는 distractor가 model update에서 구조적으로
hard-masked되어 long-gap 효과를 식별할 수 없었다. R2는 distractor를
첫 verified update 뒤의 실제 model-visible path에 넣고, `update_mask`를
audit metadata로만 제한했다. E13b-R1은 R2 report
`9071aee334da3170d79e23b5b2cbf57cffe041bb640a157c400687ddd2565218`
와 manifest
`36a4a6a03d6dc14287ee7e6ef58b68dc7ba7e07817803dc50e89f81c76c12d43`
를 dependency로 사용했다.

## Aggregate 주요 결과

Affected gain은 `tied MSE - dual MSE`로 정의한다. 양수일수록 dual
controller의 affected correction error가 낮다.

| 지표 | 관측값 | 등록 기준 | 판정 |
|---|---:|---:|---|
| 전체-grid mean affected gain | 0.001999458350 | `>= 0.001` | PASS |
| 전체-grid positive seed fraction | 1.000000 | `>= 1.0` | PASS |
| 전체-grid one-sided exact sign-flip | 0.03125 | `<= 0.05` | PASS |
| 최대 seed-mean dual−tied retention | 1.517934e-10 | `<= 0.0005` | PASS |
| Stress mean affected gain (`updates=8`, `gap=2048`) | 0.002035470479 | `>= 0.001` | PASS |
| Stress positive seed fraction | 1.000000 | `>= 1.0` | PASS |
| Stress one-sided exact sign-flip | 0.03125 | `<= 0.05` | PASS |
| 최대 dual stress retention MSE | 6.267470e-10 | `<= 0.001` | PASS |
| 최대 dual affected-MSE gap degradation | 3.974866e-10 | `<= 0.0005` | PASS |
| 최소 active-path retention harm | 0.124313574 | `>= 0.001` | PASS |
| Same-base transaction digest | 전 cell 일치 | required | PASS |
| R2 calibration hash chain | 일치 | required | PASS |

## Seed별 결과

`gap degradation`은 stress update 수 8에서 dual의
`affected MSE(gap=2048) - affected MSE(gap=0)`이다. Active harm은
max-gap distractor의 verified bit만 `0 -> 1`로 바꾼 assay에서
`intervened retention MSE - baseline retention MSE`다.

| Seed | 전체-grid mean gain | Stress gain | Dual stress retention | Gap degradation | Active harm |
|---:|---:|---:|---:|---:|---:|
| 101 | 0.002002561725 | 0.002026104287 | 5.728873e-10 | 3.664639e-10 | 0.124313574 |
| 211 | 0.001993000074 | 0.002039526948 | 5.817311e-10 | 3.708579e-10 | 0.124380203 |
| 307 | 0.001994093730 | 0.002039928709 | 5.480983e-10 | 3.539199e-10 | 0.124383696 |
| 401 | 0.001998611315 | 0.002033722845 | 6.267470e-10 | 3.974866e-10 | 0.124353179 |
| 503 | 0.002009024906 | 0.002038069605 | 6.011820e-10 | 3.888471e-10 | 0.124341179 |

모든 seed에서 전체-grid와 stress gain의 방향이 양수였다. Gap
degradation과 stress retention은 등록 margin보다 각각 약 6자릿수
작았다. Active-path assay의 큰 harm은 정상 long-gap 보존이 고정 hard
mask가 아니라 model-visible verified semantics에 의존함을 확인한다.

## E13b-R1 source run provenance

각 run은 12개 evaluation row를 포함하며 report status는 `PASS`,
per-run claim status는 `PENDING_AGGREGATE`였다. 아래 SHA-256은 E13c-R1
source provenance와 독립 재계산 결과가 일치한다.

| Seed | Variant | Run ID | Checkpoint SHA-256 |
|---:|---|---|---|
| 101 | tied | `20260727T191308.358170Z` | `974fac062c86c7e5ad5ce8f28b11d479ee57584d4efbf743e62244149b235b62` |
| 101 | dual | `20260727T191308.445971Z` | `2709bee75a071cea5ffe56751b407bcb659fb3e21ae68fffe125f22f3a5ae6d0` |
| 211 | tied | `20260727T191308.433045Z` | `6e76769508ffdd188ffde0ace8ab4e4347b59f2a0f9b7f2d2226304d96ca6879` |
| 211 | dual | `20260727T191308.460552Z` | `7a54a27d35fcfea3e4fc3b8bcd62a788090fd3f766c586468a8e96c716029731` |
| 307 | tied | `20260727T200441.551652Z` | `6aa0850147d74fbd586fa57583d5e26886f8f2835dbfd5d3e6344ddc0ee978d2` |
| 307 | dual | `20260727T200441.692316Z` | `443ce04f6ba046af10073abd2203b9a8a1e8996e8ddf25e8a037f9c17fed4fed` |
| 401 | tied | `20260727T200441.610008Z` | `21580bb56bcfffa87e4f7706a253b04d16aba50c518da32893ba9f9f492132e1` |
| 401 | dual | `20260727T200441.613674Z` | `db60b5fd07cafbbab257eca2231e63d72fe3373981be29489b9c0956faa9c8f2` |
| 503 | tied | `20260727T205713.761721Z` | `ca95e681a137243636f86444839f5601ffec5a3e5ac689b9f93d10f70e845c0e` |
| 503 | dual | `20260727T205713.767858Z` | `ed3752534847fe4cf63c1eb4f9b5693da8af300d8f9ac185d2bdbee9bf5290ed` |

| Seed | Variant | Report SHA-256 | Manifest SHA-256 | Metrics SHA-256 |
|---:|---|---|---|---|
| 101 | tied | `3547445e04c275243cbe2f873ffea86600b8aec7150f709db5eb2af0a7fba583` | `95781ee3b897240c170f20ea939d3d98d34fc57cafd7e218b3f36ff56aececc3` | `75477f8392d50d08cd98b7cbf1dd4406c4408135367b906dca9620e39948e067` |
| 101 | dual | `f1d818b5ab8a2f49d9f2cb10eacd6bfff5d67d29b9cbdd3434c1ba646d2451d8` | `4393a0c3f370f00c1581dd6a5d207baff9c41a5f3c29dd142a7c1b6c54d4b752` | `bc66bb476648c1e26e0cb1af8eef0f94bf0b197269bca85525c2cc257d3d8788` |
| 211 | tied | `819785647264abfbfb8bb20013e22ff0d5d243d7992b07754eaf444691398266` | `b65de665fca573b2cd827a6f628bb37fdef85c30f5e593ed6bf03dd53a89e01d` | `1222317007e31ab40425be512b04be3a0f0970b9073a1e01362b7f1a14d0333a` |
| 211 | dual | `2a06b76d072c2e652aba59eaacbccfd3b2c8e5182979104cdbd815e402dda679` | `a59f8e407a67bedbc98468b7120aec0b82a95811637b7bd008ed8b8b081d6c56` | `9579d655e409ca442fa88836b0bb9090924adfdffd17a0960d3e2f915b677e07` |
| 307 | tied | `2df7cbf18cc6dadd1b73201299fd18ea91436927498e901e7bf764a80e479aac` | `4a8bcffb5d7e321241d6bb70a900fb71bf40c9a984e0ed18b3c9f265560217f5` | `92100570d7455ea4f0a7c8125db97cbf7bfa8f50cffd046d5a80ac21792e8d7c` |
| 307 | dual | `9fcec5af6938a5cad6e4cb62480885c7f4addf90d79cde37f36108d5c56cb769` | `c055cc55206eb47d68d77973683e0d9acdb466c91e281c94de3ea0685fd6b731` | `8c6b402894b2e832bc891a6c6f525a21bb573706dea5092c4798cf846f1506ef` |
| 401 | tied | `b3348fa1b2fb10165d25d1d7a0e33b080d622964a7cfbddee35945e9e0230790` | `dc44807f544116f5010c98eb981598a79faaca68fb0077412d7cce22691cf904` | `c0e24d2916f70e38d983a5e065781c9af08869ffdc8b1baab337abc4136045db` |
| 401 | dual | `56703c8f774cbc33280f79e1904257bf8f8a50bbec21452b6b1a41c8b740e9f9` | `386c5797868189fcaa09f86540db5fcece1770fe2d741175c31831287df504b0` | `337ece7e85962f662d95a3256ee60c375cb775545e00085bdba4a54f6791570e` |
| 503 | tied | `d71f9126be54f3790535eabfc3553a72abcc1141d3e609738cd986f181e421cd` | `975830d9a2ce2ee5924640e63e8dea72d5c207035623795c091dbcd76779ba06` | `fb5e9bf7603069e8c4f0153079760adbce29b9b301018e1c23a04703dcd1b678` |
| 503 | dual | `be4e6ed24fd0f2a05fd10fc7dccf2156e36e3d66c66ba972f8cf4b49ef388b2f` | `21e957748f47e61caec1a90723392effa598eae5e948d61599e52ebac4f6f344` | `7a2d3ca25b01ef574b96f067b48cad64f84f492e7b9d819bcee087426987602e` |

## Operational incomplete records

첫 launcher 시도는 shell 종료와 함께 child process가 종료되어 과학적
output을 만들지 못했다. 아래 4개 directory는 정확히
`config.resolved.yaml`, `environment.json`, `run_manifest.json`만 가진
`RUN_START` provenance다. E13c-R1은 사전 고정된 strict filter에 따라
이들만 제외했으며, partially completed 또는 unknown-file run은
제외할 수 없도록 hard error로 유지했다.

| Run ID | Run manifest SHA-256 | Disposition |
|---|---|---|
| `20260727T191226.039404Z` | `86040ddb071df0860c183545a6dffdc73c5e8c878daf9d2b4ffce55d3b71249b` | `EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY` |
| `20260727T191226.069595Z` | `74036acff82110e17a57575e484d8e810e15b9c53fd58fa7a300b24062e05d8c` | `EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY` |
| `20260727T191226.084356Z` | `adce8ba201689ac758c626a6727aa023fd7ec3dba5989363ae942ca1f6f4645b` | `EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY` |
| `20260727T191226.089631Z` | `92379d213fa7fbda47a826e3689dd77ae81072f129c0152f47f6b37bc709de2e` | `EXCLUDED_OPERATIONAL_INCOMPLETE_RUN_START_ONLY` |

## Source-fingerprint 차이

| Wave | Seeds | Source files | Full-tree fingerprint |
|---|---|---:|---|
| Wave 1 | 101, 211 | 279 | `77860b5215c933c9c3d5c003ea4d7fdeffdb362e12b1a387225651bcbffc93b5` |
| Wave 2/3 | 307, 401, 503 | 285 | `e8b2e7d6d09829c78750f226a2cf2deac50c9a3061d8bf7ea3bf95a8410ece29` |

Repository full-tree fingerprint는 `.md`, test와 tooling source도 포함한다.
Wave 사이에는 failed-launch provenance를 처리하는 E13c operational
filter, 그 lock/test, status/audit 문서·도구가 추가됐다. E13b-R1
scientific entry point, config, V2 data/model/training source는 변경되지
않았고 다음 고정 hash가 모든 source run의 canonical config/hash chain과
일치한다.

| Scientific file | SHA-256 |
|---|---|
| `configs/e13b_r1_transactional_sequence_memory.yaml` | `514278b573ef838b748be0642b9851376e23d66e7bc24c74988b7e384a198ae1` |
| `experiments/e13b_r1_transactional_sequence_memory.py` | `f59ad81899fd2679ec6b3492c4518c61d04231c572069263606f10868fefaf37` |
| `src/catena/data/transactional_sequence_v2.py` | `8c635d6ec9451258016e83762a90a626576d72ef49b6d49b119712537a588e92` |
| `src/catena/models/sequence_memory_v2.py` | `6e059b7f43475be07668c828cfd48ebc2c14342b24cff35b56433a3f47831e92` |
| `src/catena/training/sequence_training_v2.py` | `9a0a5717ac73f6053673e8fcc13aac185151e8c27b2f1a367bdd1d124126365e` |

## Aggregate provenance

| 파일 | SHA-256 |
|---|---|
| `report.json` | `78121aa5dc7bd423065d8aef0fab908fc9fcd9e2412a49d9f0601a4f722e6a30` |
| `run_manifest.json` | `cc94ec353d216d434f131fbdfe4366aa221f0e93ac5c95f5ee5b1db8fca6eeac` |
| `sequence_paired_metrics.jsonl` | `f8a483819a2a6f08d00537c5d88d0553094e876b7f0353a0e02b9d50e4361cd7` |
| `sequence_stress_seed_metrics.jsonl` | `d75ecf1bec9065502540e264dd0a37da49c8665e142f66a7f97290fd44ccb97f` |
| `source_run_provenance.jsonl` | `6d5d5b11b4242d6fb374ad6181ff80c6de9de136c4f7159cc4c993d6b7fcd394` |
| `excluded_operational_incomplete_runs.jsonl` | `42f91cb3461f9e2a9cea285d73ffb4718a49b51c2618a514bc7e46b3cc0d19f5` |

## Claim boundary

허용되는 주장은 aggregate report에 고정된 다음 범위다.

> In the structured learned-distractor sequence bridge, independent erase/write
> control improves repeated-update correction and remains effective through the
> registered 2,048-event distractor stress.

이 결과는 shared structured event encoder와 fixed-address controlled
finite-memory setting의 evidence다. Natural language, learned addressing,
recurrent language model, agent/planning 또는 official-backend transfer
주장은 허용하지 않는다.
