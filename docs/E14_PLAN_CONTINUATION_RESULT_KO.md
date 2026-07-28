# E14 — Structured entity-value continuation 결과

## 판정

| 항목 | 결과 |
|---|---:|
| 실행 상태 | `PASS` |
| 등록 claim gate | `SUPPORTED` |
| 평가 grid | 5 seed × 3 update 수 × 4 gap = 60 cell |
| 통과 cell | 60 / 60 |
| Seed-level effect 방향 | 5 / 5 양수 |
| One-sided exact sign-flip | \(p=0.03125\) |
| Evidence tier | `CONTROLLED_REFERENCE` |
| Scientific/official evidence | `false` |

Run은
[`20260727T214143.455051Z`](/data/minjun_dev/CATENA/artifacts/e14_plan_continuation/20260727T214143.455051Z)
이며, 원본
[`report.json`](/data/minjun_dev/CATENA/artifacts/e14_plan_continuation/20260727T214143.455051Z/report.json)은
수정하지 않았다.

이 실험 이름의 `plan`은 독립적인 planner나 plan semantics를 뜻하지 않는다.
평가 대상은 E13b-R1의 동결된 dual checkpoint가 처리하는
**structured synthetic entity-value table의 stale affected field correction
proxy**다.

## Primary 결과

Primary estimand는 affected entity에 한정한 stale-to-assimilated MSE gain이다.
등록 SESOI는 `0.001`, untouched retention margin은 `0.0005`다.

| 요약 지표 | 값 |
|---|---:|
| 전체 60-cell 평균 affected gain | 0.007920057317525046 |
| 전체 60-cell 최소 affected gain | 0.007840155891628973 |
| 최대 affected assimilated MSE | \(6.635468077775819\times10^{-10}\) |
| 최대 untouched retention MSE | \(6.282057396804778\times10^{-10}\) |
| Whole-table gain 범위 — descriptive only | 0.000122501818653292 – 0.000949800008319698 |

최악 cell에서도 affected gain은 SESOI의 약 7.84배였고, 최대 retention
MSE는 margin보다 약 \(7.96\times10^5\)배 작았다. Whole-table gain은
prospective lock에 따라 claim 판정에 사용하지 않았다.

### Seed별 재집계

| Training seed | Cell | 평균 affected gain | 최소 cell gain | 최대 affected MSE | 최대 retention MSE |
|---:|---:|---:|---:|---:|---:|
| 101 | 12 | 0.007920057319617582 | 0.007840155949559278 | \(6.056165031270710\times10^{-10}\) | \(5.733063706309898\times10^{-10}\) |
| 211 | 12 | 0.007920057321721985 | 0.007840155942689851 | \(6.124859300293828\times10^{-10}\) | \(5.822000218785432\times10^{-10}\) |
| 307 | 12 | 0.007920057328537585 | 0.007840155963433790 | \(5.917419904771792\times10^{-10}\) | \(5.482768736374596\times10^{-10}\) |
| 401 | 12 | 0.007920057307732880 | 0.007840155891628973 | \(6.635468077775819\times10^{-10}\) | \(6.282057396804778\times10^{-10}\) |
| 503 | 12 | 0.007920057310015209 | 0.007840155895421414 | \(6.597543674574657\times10^{-10}\) | \(6.009845682824071\times10^{-10}\) |

다섯 seed의 평균 gain이 모두 양수여서, seed를 inference unit으로 한
one-sided exact sign-flip 값은 등록 가능한 최소값 \(1/2^5=0.03125\)다.
Batch나 episode는 통계적 독립 단위로 사용하지 않았다.

## Gap별 forward latency

아래 값은 RTX PRO 6000 Blackwell의 `cuda:0`, batch size 256에서 관측한
forward-only 시간이다. 각 행은 5 checkpoint × 3 update 조건, 즉 15개
cell의 기술통계다.

| Gap events | 평균 ms/batch | 최소–최대 ms/batch |
|---:|---:|---:|
| 0 | 1.129852 | 0.394626 – 3.136531 |
| 128 | 20.910628 | 20.357699 – 21.498223 |
| 512 | 80.260075 | 79.468388 – 81.396843 |
| 2048 | 318.195714 | 316.864008 – 319.678219 |

이 수치는 device-local descriptive latency일 뿐이며 production throughput,
break-even 또는 공식 backend 속도 주장을 열지 않는다.

## Base-transaction pairing

Evaluation seed는 `base + 100000 * updates + batch_index`이며 gap과 training
seed를 포함하지 않는다. 각 update 조건에서 모든 5 checkpoint × 4 gap이
동일한 base transaction digest를 가졌다.

| Updates | Unique digest 수 | Base transaction digest |
|---:|---:|---|
| 1 | 1 | `bfddd0977cd71ec5f1c16c6799d9db63dfb871e4da781253922a6b2a97e00192` |
| 4 | 1 | `2acf251d1f14bc62d22bfea9c9a9a3edf90d667c8e4f277a8593f82a961ca4e4` |
| 8 | 1 | `b246a1846a0841542745c4323da71b251e51ccb9cd0a2acd499584d2d6f148c9` |

따라서 gap 비교에서 initial state, verified update와 target이 바뀌는
confound는 관찰되지 않았다.

## 선택된 checkpoint provenance

E14는 E13c-R1이 봉인한 열 개 source 중 아래 다섯 `dual` checkpoint만
seed 순서대로 사용했다. 모든 checkpoint와 source report, metrics,
manifest hash를 E14 report의 봉인값에 대해 재검증했다.

| Seed | E13b-R1 source run | Checkpoint SHA-256 | Source report / metrics / manifest SHA-256 |
|---:|---|---|---|
| 101 | `20260727T191308.445971Z` | `2709bee75a071cea5ffe56751b407bcb659fb3e21ae68fffe125f22f3a5ae6d0` | `f1d818b5ab8a2f49d9f2cb10eacd6bfff5d67d29b9cbdd3434c1ba646d2451d8` / `bc66bb476648c1e26e0cb1af8eef0f94bf0b197269bca85525c2cc257d3d8788` / `4393a0c3f370f00c1581dd6a5d207baff9c41a5f3c29dd142a7c1b6c54d4b752` |
| 211 | `20260727T191308.460552Z` | `7a54a27d35fcfea3e4fc3b8bcd62a788090fd3f766c586468a8e96c716029731` | `2a06b76d072c2e652aba59eaacbccfd3b2c8e5182979104cdbd815e402dda679` / `9579d655e409ca442fa88836b0bb9090924adfdffd17a0960d3e2f915b677e07` / `a59f8e407a67bedbc98468b7120aec0b82a95811637b7bd008ed8b8b081d6c56` |
| 307 | `20260727T200441.692316Z` | `443ce04f6ba046af10073abd2203b9a8a1e8996e8ddf25e8a037f9c17fed4fed` | `9fcec5af6938a5cad6e4cb62480885c7f4addf90d79cde37f36108d5c56cb769` / `8c6b402894b2e832bc891a6c6f525a21bb573706dea5092c4798cf846f1506ef` / `c055cc55206eb47d68d77973683e0d9acdb466c91e281c94de3ea0685fd6b731` |
| 401 | `20260727T200441.613674Z` | `db60b5fd07cafbbab257eca2231e63d72fe3373981be29489b9c0956faa9c8f2` | `56703c8f774cbc33280f79e1904257bf8f8a50bbec21452b6b1a41c8b740e9f9` / `337ece7e85962f662d95a3256ee60c375cb775545e00085bdba4a54f6791570e` / `386c5797868189fcaa09f86540db5fcece1770fe2d741175c31831287df504b0` |
| 503 | `20260727T205713.767858Z` | `ed3752534847fe4cf63c1eb4f9b5693da8af300d8f9ac185d2bdbee9bf5290ed` | `be4e6ed24fd0f2a05fd10fc7dccf2156e36e3d66c66ba972f8cf4b49ef388b2f` / `7a2d3ca25b01ef574b96f067b48cad64f84f492e7b9d819bcee087426987602e` / `21e957748f47e61caec1a90723392effa598eae5e948d61599e52ebac4f6f344` |

전체 provenance는
[`sealed_checkpoint_provenance.jsonl`](/data/minjun_dev/CATENA/artifacts/e14_plan_continuation/20260727T214143.455051Z/sealed_checkpoint_provenance.jsonl)에
보존되어 있다.

## Dependency와 protocol seal

| 항목 | 경로 또는 SHA-256 |
|---|---|
| E13c-R1 dependency run | [`20260727T214126.954177Z`](/data/minjun_dev/CATENA/artifacts/e13c_r1_transactional_sequence_aggregate/20260727T214126.954177Z) |
| E13c-R1 report | `78121aa5dc7bd423065d8aef0fab908fc9fcd9e2412a49d9f0601a4f722e6a30` |
| E13c-R1 manifest | `cc94ec353d216d434f131fbdfe4366aa221f0e93ac5c95f5ee5b1db8fca6eeac` |
| E13c-R1 source provenance | `6d5d5b11b4242d6fb374ad6181ff80c6de9de136c4f7159cc4c993d6b7fcd394` |
| Prospective identifiability lock | [`E14_PROSPECTIVE_IDENTIFIABILITY_REPAIR_LOCK_KO.md`](E14_PROSPECTIVE_IDENTIFIABILITY_REPAIR_LOCK_KO.md), `8984159ab392d27672cc8426be0aa6126c419fe066e55ca3d3af3bdabf6417b0` |
| Frozen E14 source | `262d5519ee0305f9c47bc90a591e295bbb53769c37248d6eb9d503ce191115db` |
| Frozen E14 config | `06c3433f774a4420b6f27b1db741267799389189ad44f17eb8c5d7b2cf29d201` |

E13c-R1 dependency는 `MAIN/PASS/SUPPORTED`이며, E14는 그 report가 지정한
동결 checkpoint만 평가했다.

## 허용되는 해석과 evidence boundary

허용되는 해석은 다음 범위로 제한한다.

> 다섯 개의 봉인된 E13b-R1 dual checkpoint에서, controller는 시험한
> structured synthetic entity-value continuation proxy의 stale affected
> field를 교정하면서 untouched field를 보존했다.

| 경계 | 상태 |
|---|---|
| Proxy type | `STRUCTURED_SYNTHETIC_ENTITY_VALUE` |
| Oracle entity address | `true` |
| Oracle old/new candidate | `true` |
| Learned distractor no-op path guardrail | 평가함 |
| Independent plan semantics | 평가하지 않음 |
| Semantic demand inference | 평가하지 않음 |
| Learned addressing | 평가하지 않음 |
| Long-gap persistence 일반화 | claim 불가 |
| Natural-language / recurrent LM transfer | claim 불가 |
| General agent planning / tool orchestration | claim 불가 |
| Official GDN2/KDA backend | claim 불가 |
| Production latency / break-even | claim 불가 |

따라서 E14의 `SUPPORTED`는 structured proxy 내부의 correction/retention
gate에만 적용된다. Agent planning, 자연어 planning, 공식 backend 또는
실서비스 성능의 증거로 확장해서는 안 된다.
