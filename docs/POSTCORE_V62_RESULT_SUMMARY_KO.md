# CATENA v6.2 Post-Core 결과 종합

완료된 E10–E21 run의 1페이지 요약 67개와 SHA-256은
[run별 결과 요약 index](../artifacts/POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md)에
모아 두었다. E15a/E15b 초기 split dry-run은 원본 artifact를 사후
수정하지 않고 별도 sidecar 요약으로 보존했다. 전체 69개 completed run의
누락 여부는
[coverage 감사](POSTCORE_E10_E21_RESULT_SUMMARY_COVERAGE_AUDIT_KO.md)에서
확인할 수 있다.

## 증거 경계와 현재 상태

| 항목 | 현재 기록 |
|---|---|
| Post-core evidence tier | `CONTROLLED_REFERENCE` |
| Scientific evidence | `false` |
| Official-backend claim | 닫힘 |
| Language-model / natural-language claim | 닫힘 |
| H5 semantic anchor | `CLOSED` — E05a와 E05a-R1의 `NO_GO` 유지, E05b 미실행 |
| E15 official backend | E15a-R1 `FAIL` (4/6); E15b KVEraser `NOT_CONFIGURED` |
| Artifact root | `/data/minjun_dev/CATENA/artifacts` |

`SUPPORTED`는 각 report에 등록된 controlled-reference claim 범위에서만
사용한다. Reference 결과를 official GDN2/KDA/KVEraser, pretrained
language model, natural-language transaction 또는 agent evidence로
확장하지 않는다.

## 판정 요약

| 실험 | Execution | 현재 claim 판정 | 핵심 artifact |
|---|---|---|---|
| E10 | `PASS` | `NOT_OPENED` — numerical-floor 이후 strict monotonicity 실패 | `20260727T184326.484361Z` |
| E10b | `PASS` | `SUPPORTED` — prospective floor-aware repair | `20260727T190906.272784Z` |
| E11 | `PASS` | `NOT_OPENED_SCALE_RESTRICTION` | `20260727T180703.763554Z` |
| E11b | `PASS` | `SUPPORTED` — prospective scale-normalized repair | `20260727T183004.928280Z` |
| E12 | `PASS` | `SUPPORTED` | `20260727T184511.437394Z` |
| E13a-R2 | `PASS` | `GO_FOR_E13B_R1`; calibration only | `20260727T190642.222102Z` |
| E13b-R1 | 10/10 `PASS` | Per-run `PENDING_AGGREGATE` | 5 paired seed × tied/dual |
| E13c-R1 | `PASS` | `SUPPORTED` | `20260727T214126.954177Z` |
| E14 | `PASS` | `SUPPORTED` — structured proxy only | `20260727T214143.455051Z` |
| E15 | `DRY_RUN` | Original combined gate `NOT_CONFIGURED` | `20260727T184517.578907Z` |
| E15a-R1 | `FAIL` | 4/6 gate; official operator claim 닫힘 | `20260728T062259.987278Z` |
| E16 | `PASS` | H1–H5 evidence registry 동결 완료 | `20260727T181946.576248Z` |
| E17 | `PASS` | E10–E15 post-core evidence registry 동결 완료 | `20260728T055329.507772Z` |
| E18b | `PASS` | `SUPPORTED` — sequence architecture-demand lattice | `20260728T074753.618843Z` |
| E19b | `PASS` | `SUPPORTED` — learned localization/candidate decomposition | `20260728T060318.439621Z` |
| E20 | `PASS` | `SUPPORTED_CONTROLLED_SYSTEMS_PROXY` | `20260728T062658.423644Z` |
| E21b | `PASS` | Original `INCONCLUSIVE_GATE_IMPLEMENTATION` | `20260728T091532.426698Z` |
| E21b-R1 | `PASS` | `NOT_SUPPORTED` — primary 3/3 PASS, guardrail 2개 FAIL | `20260728T091547.163300Z` |

## 전체 연구 해석

E10b, E11b와 E12는 demand의 rank, basis algebra와 필요한 control freedom이
최소 memory-control architecture를 예측한다는 controlled-reference
결과를 제공한다. E13c-R1과 E18b는 이 선택적 이득이 반복 update와
2,048-event distractor stress에서도 유지됨을 보였다. E19b는 fixed random
codebook에서 learned address와 current-state candidate bottleneck을
분해했고, E14는 structured stale-field continuation proxy를 교정했다.
E20은 별도 in-process systems proxy일 뿐 production benchmark가 아니다.
H5는 닫혔고 E15a-R1은 official parity 4/6에서 실패했으므로 semantic
inference, natural language, official backend 또는 agent planning 전이는
주장하지 않는다. E21b-R1에서는 learned address/state-read recovery의
방향은 5/5 seed로 재현됐지만 absolute affected floor와 cellwise
non-target guardrail이 실패해 full structured sequence-transfer claim도
열리지 않았다.

## E10 — Learned Control-Rank Scaling 원본

| 항목 | 결과 |
|---|---|
| 현재 판정 | Execution `PASS`; full claim `NOT_OPENED` |
| 핵심 metric | Minimum-rank tracking 40/40; seedwise nondecreasing 8/8; sign-flip \(p=0.00390625\) |
| 실패 gate | Strict adjacent-rank monotonic fraction `0.590` (`>=0.9` 필요) |
| Main 전 protocol audit | Upper-bound-only gate가 rank 1이 모든 family를 풀어도 통과할 수 있어, 첫 evaluable main 전에 intrinsic-rank lower bound와 seedwise nondecreasing 조건을 고정 |
| 개발 중 이슈 | Exact-target recovery 뒤 `약 1e-9`–`2e-7` numerical floor의 미세 증감도 strict violation으로 계산됨 |
| 처리 | 원본 report와 gate를 변경하지 않고 E10b를 별도 prospective experiment로 생성 |
| Artifact | [`20260727T184326.484361Z/report.json`](../artifacts/e10_learned_rank_scaling/20260727T184326.484361Z/report.json) |
| 상세 결과 | [E10 결과](E10_LEARNED_RANK_SCALING_RESULT_KO.md) |

모든 seed에서 intrinsic rank `1, 2, 4, 8, 16`의 최소 qualifying learned
rank가 각각 `1, 2, 4, 8, 16`이었지만, 이 사실로 원본 full gate를
사후 개방하지 않는다.

## E10b — Floor-Aware Learned Control-Rank Scaling

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / SUPPORTED` |
| 핵심 metric | Fresh-set pre-saturation pair 80/80 non-increasing; minimum-rank tracking 40/40; sign-flip \(p=0.00390625\) |
| Repair | E10의 240개 checkpoint를 재학습·선택하지 않고 hash 검증 후 fresh descriptor namespace에서 평가 |
| Gate 식별성 | 기존 exact-target recovery `0.95` 이전 pair만 monotonic estimand; 포화된 120 pair는 기록하되 분모에서 제외 |
| Artifact | [`20260727T190906.272784Z/report.json`](../artifacts/e10b_floor_aware_rank_scaling/20260727T190906.272784Z/report.json) |
| 상세 결과 | [E10b 결과](E10B_FLOOR_AWARE_RANK_SCALING_RESULT_KO.md) |

허용 claim은 등록된 smooth synthetic operator family에서 exact recovery에
필요한 learned rank가 demand intrinsic rank를 추적한다는 범위다.

## E11 — Representation-Control Co-adaptation 원본

| 항목 | 결과 |
|---|---|
| 현재 판정 | Execution `PASS`; claim `NOT_OPENED_SCALE_RESTRICTION` |
| 방향성 | Common recovery, noncommuting gap, low-rank recovery가 모두 8/8 seed 동일 방향; 각 \(p=0.00390625\) |
| 개발 중 이슈 | 등록 raw-MSE SESOI `0.001`이 held-out target 평균 에너지 `약 0.000595`보다 커 scale상 식별 불가능 |
| 처리 | 원본 row, report와 SESOI를 바꾸지 않고 fresh-family E11b를 사전 고정 |
| Artifact | [`20260727T180703.763554Z/report.json`](../artifacts/e11_representation_control_coadaptation/20260727T180703.763554Z/report.json) |
| 상세 결과 | [E11/E11b 결과](E11_REPRESENTATION_COADAPTATION_RESULT_KO.md) |

## E11b — Scale-Normalized Representation-Control Co-adaptation

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / SUPPORTED`; 10/10 gate 통과 |
| Common rotation | Shared basis가 fixed-basis headroom의 `0.997103` 회복; residual/target energy `0.002655` |
| Noncommuting | Shared-basis residual/target energy `0.182039`; common 대비 gap `0.179384` |
| Richer learned control | Rank-8이 shared residual의 `0.988539` 회복; residual/target energy `0.002087` |
| Repair | Fresh 8 seed와 target-energy/headroom ratio를 사용한 prospective scale-normalized gate |
| 특기사항 | Low-rank controller는 diagonal controller와 parameter matched가 아니므로 efficiency claim은 열리지 않음 |
| Artifact | [`20260727T183004.928280Z/report.json`](../artifacts/e11b_scale_normalized_coadaptation/20260727T183004.928280Z/report.json) |
| 상세 결과 | [E11/E11b 결과](E11_REPRESENTATION_COADAPTATION_RESULT_KO.md) |

## E12 — Architecture-Demand Control Lattice

| 추가 control freedom | 대응 demand | Mean selective MSE gain | Simpler-task 최대 악화 | Sign-flip |
|---|---|---:|---:|---:|
| Tied → dual scalar | Magnitude factorization | 0.0065376581 | 0.0 | 0.00390625 |
| Dual scalar → diagonal value | Value granularity | 0.0086808868 | -0.0000002104 | 0.00390625 |
| Diagonal → separate address | Address decoupling | 0.0156250368 | 0.0000001155 | 0.00390625 |
| Separate address → state-aware | State conditioning | 0.0131247569 | 0.0000574804 | 0.00390625 |

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / SUPPORTED`; 네 selective gate 모두 통과 |
| 개발 중 이슈 | Value-mask 위치 미노출, state target/readout index 불일치, variant별 initialization RNG, retention denominator의 hard-coded slot 수를 발견 |
| Repair | Main evaluation 전에 generator/controller/pairing/denominator를 수정; 동일 maximal parameter surface와 paired initialization 유지 |
| Artifact 이슈 | 첫 corrected run은 checkpoint contract가 불완전해 claim source로 사용하지 않음 |
| 최종 처리 | Metric·SESOI·gate를 바꾸지 않은 새 artifact-complete run에서 contrast를 재현하고 checkpoint 40개 hash 검증 |
| Artifact | [`20260727T184511.437394Z/report.json`](../artifacts/e12_control_algebra_lattice/20260727T184511.437394Z/report.json) |
| 상세 결과 | [E12 결과](E12_CONTROL_LATTICE_RESULT_KO.md) |

## E13a-R2 — Learned-Distractor Sequence Calibration

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / GO_FOR_E13B_R1`; sequence claim 자체는 열지 않음 |
| Affected MSE | Tied `약 0.00407916`; dual `1.359995e-7` |
| Tied−dual gain | `0.0040790231` |
| Dual affected exact match | `1.000000` |
| Parameter contract | Tied/dual 각 123,394, initialization hash 동일 |
| Throughput | Tied `1544.4`, dual `1520.6` examples/s |
| Scale estimate | 30,000-step 최장 run `약 42.1분`; 3-wave 합계 `약 126.3분` |
| Artifact | [`20260727T190642.222102Z/report.json`](../artifacts/e13a_r2_sequence_floor_throughput/20260727T190642.222102Z/report.json) |
| 상세 결과 | [E13a-R2 결과](E13A_R2_LEARNED_DISTRACTOR_RESULT_KO.md) |

Original E13a/R1에서는 distractor가 update 뒤에 놓이고 oracle mask 두 개로
제거되어 long-gap path가 식별 불가능했다. R2는 첫 verified update 뒤의
model-visible path에 distractor block을 넣고, `verified`를 semantic
input으로만 사용하며 `update_mask`는 audit metadata로 제한했다.

## E13b-R1 — Paired Transactional Sequence Main

모든 run은 12개 evaluation cell을 포함하고 `PASS`였으며, per-run claim은
사전 계약대로 E13c-R1 집계 전까지 `PENDING_AGGREGATE`였다.

| Seed | Tied run | Dual run |
|---:|---|---|
| 101 | [`20260727T191308.358170Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T191308.358170Z/report.json) | [`20260727T191308.445971Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T191308.445971Z/report.json) |
| 211 | [`20260727T191308.433045Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T191308.433045Z/report.json) | [`20260727T191308.460552Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T191308.460552Z/report.json) |
| 307 | [`20260727T200441.551652Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T200441.551652Z/report.json) | [`20260727T200441.692316Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T200441.692316Z/report.json) |
| 401 | [`20260727T200441.610008Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T200441.610008Z/report.json) | [`20260727T200441.613674Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T200441.613674Z/report.json) |
| 503 | [`20260727T205713.761721Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T205713.761721Z/report.json) | [`20260727T205713.767858Z`](../artifacts/e13b_r1_transactional_sequence_memory/20260727T205713.767858Z/report.json) |

| 개발·운영 특기사항 | 처리 |
|---|---|
| Shell launcher 종료 시 `nohup` child가 정리되어 최초 네 시도가 `RUN_START` provenance만 남김 | Artifact를 삭제·이동하지 않고 보존; config/metric/gate를 바꾸지 않은 동일 command를 장기 PTY에서 실행 |
| 불완전 directory가 aggregate source 탐색에 존재 | E13c 평가 전에 exact 3-file `RUN_START`만 hash와 reason을 기록해 제외하는 operational amendment를 고정; partial/tampered run은 hard error |
| Wave 사이 repository full-tree fingerprint 변화 | Scientific entry point, config와 V2 data/model/training hash가 동일함을 별도로 확인; 추가된 문서·test·운영 filter만 provenance에 기록 |

상세 provenance와 10개 checkpoint hash는
[E13b-R1/E13c-R1 결과](E13BC_TRANSACTIONAL_SEQUENCE_RESULT_KO.md)에
정리돼 있다.

## E13c-R1 — Transactional Sequence Aggregate

| 지표 | 결과 | Gate |
|---|---:|---:|
| Paired grid | 5 seed × 3 updates × 4 gaps = 60 cell | Complete |
| Overall mean tied−dual affected gain | 0.001999458350 | `>=0.001` PASS |
| Overall direction / sign-flip | 5/5; \(p=0.03125\) | PASS |
| Stress gain (`updates=8`, `gap=2048`) | 0.002035470479 | `>=0.001` PASS |
| Stress direction / sign-flip | 5/5; \(p=0.03125\) | PASS |
| Max seed-mean retention degradation | `1.517934e-10` | `<=0.0005` PASS |
| Max dual stress retention MSE | `6.267470e-10` | `<=0.001` PASS |
| Max dual gap degradation | `3.974866e-10` | `<=0.0005` PASS |
| Minimum active-path retention harm | 0.124313574 | `>=0.001` PASS |

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / SUPPORTED`; 12/12 registered condition 통과 |
| Same-base contract | 동일 seed×variant×updates의 4개 gap에서 transaction digest 일치 |
| Active-path assay | Max-gap distractor의 `verified`만 0→1로 바꾸면 큰 retention harm 발생; hard-mask 우회가 아님을 확인 |
| Artifact | [`20260727T214126.954177Z/report.json`](../artifacts/e13c_r1_transactional_sequence_aggregate/20260727T214126.954177Z/report.json) |
| 상세 결과 | [E13b-R1/E13c-R1 결과](E13BC_TRANSACTIONAL_SEQUENCE_RESULT_KO.md) |

허용 claim은 shared structured event encoder와 fixed-address
controlled-memory setting에서 independent erase/write control의 이득이
repeated update와 등록된 2,048-event distractor stress에서도 유지됐다는
범위다.

## E14 — Structured Entity-Value Continuation Proxy

| 지표 | 결과 | Gate |
|---|---:|---:|
| Evaluation grid | 5 seed × 3 updates × 4 gaps = 60 cell | Complete |
| Mean affected stale-to-assimilated gain | 0.007920057318 | `>=0.001` PASS |
| Minimum cell affected gain | 0.007840155892 | `>=0.001` PASS |
| Max affected assimilated MSE | `6.635468e-10` | Diagnostic |
| Max untouched retention MSE | `6.282057e-10` | `<=0.0005` PASS |
| Seed direction / sign-flip | 5/5; \(p=0.03125\) | PASS |
| Forward latency, gap 0 → 2048 | 평균 1.130 → 318.196 ms/batch | Descriptive only |

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS / SUPPORTED` — structured synthetic proxy 범위 |
| 개발 중 이슈 | 기존 whole-table MSE gain은 `updates=1`의 analytic 최대값 `0.000244140625`가 등록 SESOI `0.001`보다 작아 gate가 구조적으로 식별 불가능 |
| Prospective repair | 첫 E14 evaluation 전에 수치 threshold를 유지하고 primary estimand를 affected entity gain으로 고정; whole-table gain은 descriptive only |
| Dependency | E13c-R1이 봉인한 정확한 dual checkpoint 5개와 모든 source hash를 검증 |
| Artifact | [`20260727T214143.455051Z/report.json`](../artifacts/e14_plan_continuation/20260727T214143.455051Z/report.json) |
| 상세 결과 | [E14 결과](E14_PLAN_CONTINUATION_RESULT_KO.md) |

`SUPPORTED`는 oracle entity address와 oracle old/new candidate가 주어진
structured entity-value stale-field correction/retention proxy에만
적용된다. Independent plan semantics, learned addressing, long-gap
persistence 일반화, agent planning과 production break-even은 평가하지
않았다.

## E15 — Official Backend Gate

| 항목 | 결과 |
|---|---|
| 실행 상태 | `DRY_RUN` |
| 구성 상태 | `NOT_CONFIGURED`; `official_backend_ready=false` |
| GDN2 / KVEraser | 둘 다 `DRY_RUN`; scientific evidence `false` |
| 누락 dependency | Official repository path, exact commit, `catena_official_plugins` |
| Fallback | Reference/mock 자동 대체 없음 |
| 현재 판정 | 실제 official E15 미실행; official architecture claim 닫힘 |
| Artifact | [`20260727T184517.578907Z/report.json`](../artifacts/e15_official_backend_gate/20260727T184517.578907Z/report.json) |
| 상세 결과 | [E15 dry gate 결과](E15_OFFICIAL_BACKEND_GATE_RESULT_KO.md) |

실제 official-backend claim은 별도 환경에서 exact repository commit,
plugin parity와 official gate가 모두 `PASS`한 뒤에만 열 수 있다.
이후 분리한 gate의 초기 assay는
[E15a GDN2/KDA dry-run](E15A_OFFICIAL_GDN2_KDA_DRY_RUN_RESULT_KO.md)과
[E15b KVEraser dry-run](E15B_OFFICIAL_KVERASER_DRY_RUN_RESULT_KO.md)에
각각 기록했다.

## E16 — Completed-Core Evidence Freeze

| 항목 | 결과 |
|---|---|
| 현재 판정 | `PASS`; `core_registry_complete=true`; invalid claim 0개 |
| 동결 범위 | H1, H2 original/E02b, H3 original/E03b, H4, H5 original/R1의 8개 evidence record |
| H5 최종 상태 | E05a `NO_GO`; E05a-R1 `NO_GO_H5_CLOSED`; H5 claim 닫힘 |
| Evidence policy | Registry에 고정된 report와 SHA-256만 core paper 수치·판정의 출처로 사용 |
| Artifact | [`20260727T181946.576248Z/report.json`](../artifacts/e16_core_evidence_freeze/20260727T181946.576248Z/report.json) |
| Registry | [`evidence_registry.json`](../artifacts/e16_core_evidence_freeze/20260727T181946.576248Z/evidence_registry.json) |
| 계약 문서 | [E16 evidence freeze](E16_EVIDENCE_FREEZE_KO.md) |

E16은 기존 H1–H5 report를 수정하거나 재판정하지 않고 경로와 hash,
claim disposition만 새 artifact에 고정했다. 모든 core record의 evidence
tier는 `CONTROLLED_REFERENCE`, `scientific_evidence`는 `false`다.

## E15a-R1 — Official GDN2/KDA Operator Gate

| Gate | 관측값 | 기준 | 판정 |
|---|---:|---:|---|
| Full/chunk FP32 parity | 0 | `<=1e-5` | PASS |
| Tied GDN2/KDA FP32 parity | `1.3783e-4` | `<=1e-5` | FAIL |
| BF16/FP32 parity | `5.9033e-3` | `<=5e-3` | FAIL |
| Backward finite | 7/7 | all finite | PASS |
| State carry/clone/restore | true | true | PASS |
| Intervention confinement | true | true | PASS |

Pinned official GDN2와 FLA source는 구성되어 실행됐지만 등록 gate는
4/6만 통과했다. Threshold 변경이나 reference fallback 없이
`official_operator_claim_eligible=false`를 유지하며, E02b/E12/E13 official
replication은 실행하지 않는다.

- [Run summary](../artifacts/e15a_r1_official_gdn2_kda_gate/20260728T062259.987278Z/RESULTS_SUMMARY_KO.md)
- [No-patch audit](E15A_R1_OFFICIAL_PARITY_FAILURE_AUDIT_KO.md)
- [Disposition freeze](../artifacts/E15A_OFFICIAL_GATE_DISPOSITION_FREEZE_V1.json)

## E17 — Post-Core Evidence Freeze

E10–E15의 원본과 prospective repair를 분리해 11/11 exact record의
run/report/manifest/freeze-anchor hash와 claim scope를 동결했다. Invalid
record는 0개이며 이 registry는 새 과학 결과가 아니라 논문 수치의
canonical provenance다.

- [Run summary](../artifacts/e17_postcore_evidence_freeze/20260728T055329.507772Z/RESULTS_SUMMARY_KO.md)
- [Canonical freeze](../artifacts/E17_POSTCORE_EVIDENCE_FREEZE_V1.json)

## E18b — Sequence Architecture–Demand Lattice

| Added freedom | Registered-grid mean affected gain | Stress 방향 |
|---|---:|---:|
| Magnitude factorization | 0.0067189 | 5/5 |
| Value granularity | 0.0091577 | 5/5 |
| Address decoupling | 0.0164596 | 5/5 |
| State conditioning | 0.0111915 | 5/5 |

25개 source run의 1,200 row를 집계했고 네 adjacent conjunction이 모두
통과했다. 이 값은 update×gap grid 평균이다. Stress에는 별도 SESOI가
없고, simpler-demand/retention은 relative cell-mean guardrail이다.
Oracle address/candidate, explicit demand descriptor와 model-visible
verified bit가 주어지므로 every-cell, minimal sufficiency, semantic,
learned relevance 또는 official-backend claim은 열리지 않는다.

- [Aggregate summary](../artifacts/e18b_sequence_control_lattice_aggregate/20260728T074753.618843Z/RESULTS_SUMMARY_KO.md)
- [Claim-boundary audit](E18_CLAIM_BOUNDARY_AUDIT_KO.md)
- [Freeze](../artifacts/E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json)

## E19b — Learned Localization/Candidate Decomposition

| Condition | Mean affected gain | 방향 / exact p |
|---|---:|---:|
| B: separate-address recovery | 0.0528356 | 5/5 / 0.03125 |
| C: state-read recovery | 0.1252249 | 5/5 / 0.03125 |
| D: full-only recovery | 0.0692067 | 5/5 / 0.03125 |

Fixed-slot random codebook에서 address accuracy 1.0, candidate MSE
`4.74e-12`, capable affected MSE `8.98e-9`와 retention gate가 모두
통과했다. Learned localization과 current-state candidate read의 병목은
분리됐지만 semantic localization, novel entity와 natural-language
generalization은 평가하지 않았다.

- [Aggregate summary](../artifacts/e19b_localization_candidate_aggregate/20260728T060318.439621Z/RESULTS_SUMMARY_KO.md)
- [Freeze](../artifacts/E19_LOCALIZATION_CANDIDATE_FREEZE_V1.json)

## E20 — Quality-Constrained Break-Even Proxy

세 external-state baseline 모두 등록 query grid의 첫 지점 `m=1`에서
quality-constrained break-even을 보였다. `m=64` median은 internal
`36.40 µs`, external-per-query `3,391.59 µs`, compact-cache `332.42 µs`,
full-refresh `225.69 µs`였고 28개 policy×query-count cell의 correction과
retention MSE는 0이었다.

이는 persistent device state와 CPU-to-device transfer를 포함한 등록
in-process workload에 한정된다. Database/network, production serving,
official recurrent backend, pretrained LM 또는 agent latency claim은
열리지 않는다.

- [Run summary](../artifacts/e20_quality_constrained_break_even/20260728T062658.423644Z/RESULTS_SUMMARY_KO.md)
- [Freeze](../artifacts/E20_QUALITY_BREAK_EVEN_FREEZE_V1.json)

## E21b / E21b-R1 — Structured Sequence Localization Transfer

원본 E21b의 aggregate implementation은 inactive route를 non-target에
포함하고 cell을 먼저 평균하는 결함이 있어 결과와 무관하게
`INCONCLUSIVE_GATE_IMPLEMENTATION`으로 보존한다. 첫 source report 전에
동결한 E21b-R1만 최종 gate 판정에 사용했다.

| Registered gate | 관측값 | 기준 | 판정 |
|---|---:|---:|---|
| B separate-address mean gain | 0.0583049 | `>=0.001`, 5/5, `p<=0.05` | PASS |
| C state-read mean gain | 0.1419464 | `>=0.001`, 5/5, `p<=0.05` | PASS |
| D full-only mean gain | 0.0722309 | `>=0.001`, 5/5, `p<=0.05` | PASS |
| Stress direction | 1.0 | 5/5 | PASS |
| Max capable affected MSE | 0.00177357 | `<=0.001` | FAIL |
| Max active non-target degradation | 0.000867025 | `<=0.0005` | FAIL |
| Max primary retention degradation | `4.95605e-6` | `<=0.0005` | PASS |

따라서 localization과 current-state candidate read의 primary recovery는
repeated structured sequence에서도 강하고 seedwise 일관됐지만, capable
path가 등록 absolute floor에 도달하지 못했고 특정 non-target cell의
손상이 margin을 넘었다. 정확한 판정은 execution `PASS`, R1 claim
`NOT_SUPPORTED`이며 threshold를 완화하거나 primary 3/3 통과만으로 full
claim을 열지 않는다.

Setting은 fixed identifier schema, explicit algebraic demand/provenance
field를 사용한다. H5, semantic/natural-language, novel identifier,
recurrent/pretrained LM, agent/planning, official backend와 runtime claim은
모두 닫혀 있다.

- [Original aggregate summary](../artifacts/e21b_structured_sequence_localization_aggregate/20260728T091532.426698Z/RESULTS_SUMMARY_KO.md)
- [R1 aggregate summary](../artifacts/e21b_r1_structured_sequence_localization_aggregate/20260728T091547.163300Z/RESULTS_SUMMARY_KO.md)
- [1페이지 E21 결과 요약](E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_RESULT_KO.md)
- [Immutable freeze](../artifacts/E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json)
- [Operational run log](E21_OPERATIONAL_RUN_LOG_KO.md)
