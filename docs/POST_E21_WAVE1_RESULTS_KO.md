# CATENA Post-E21 Scientific MAIN 결과

Artifact root는 `/data/minjun_dev/CATENA/artifacts`다. Source provenance는
실행 시점에 따라 세 그룹으로 분리된다.

| Run group | Repository source binding |
|---|---|
| E22a, E23a, E24a, E24b, E25b | commit `51156242dfc429cb66d577c144b8d38a5ae38551`; tag `post-e21-main-source-lock-20260728T174131Z`; fingerprint `9f97b8cbdae1ed1abec7784b79893dea860bda535a1942188161d6ab5875e4ac` (472 files) |
| E22b, E23b | commit `1ef51cdd488b411d51a3dac85eaa4fc8d5b04d1b`; tag `post-e21-sharded-source-lock-20260729T122800Z`; fingerprint `86e30646577a5bd18a608d060a03cd98d3de6b15dee83a702d6b8d7a37571683` (482 files) |
| E25a | Internal source fingerprint `37ef731d5a1bbb288756e71f8d011227d15746007e1ba6fc24b9f35dcb5a6c97` (472 files); internal commit was not recorded in the immutable run. Official GDN2/FLA commits are listed below. |

따라서 `1ef51cdd...`는 전체 Wave 1의 단일 source가 아니라 이후 sharded
confirmatory source lock이다. 이 문서는 frozen report를 재판정하지 않고
각 report의 disposition과 claim boundary를 그대로 요약한다.

## 종합 판정

| 실험 | 실행 | 과학적 disposition | 다음 dependency |
|---|---|---|---|
| E22a | `PASS` | `SELECTED` — development only | E22b를 `smoothmax_100`으로 열었음 |
| E22b | `PASS` | `NOT_SUPPORTED` | E23b를 `capacity_only`로만 열었음 |
| E23a | `PASS` | `SCREEN_ONLY_NO_CONFIRMATORY_CLAIM` | Outcome은 E23b boundary에 사용하지 않음 |
| E23b | `PASS` | `NOT_SUPPORTED` | Product-poset minimality claim 닫힘 |
| E24a | `PASS` | `OOD_SPECTRUM_FAMILY_TRANSFER_NOT_SUPPORTED` | 없음 |
| E24b | `PASS` | `CONSTRUCTION_ROBUST_PREDICTION_FAILURE` | 없음 |
| E25a | `FAIL` | Terminal preregistered parity-gate failure | Official replication 닫힘 |
| E25b | `PASS` | `AUDIT_PENDING` — preparation only | 실제 인간 2인 audit 필요 |

실행 성공과 가설 지지는 분리한다. E22b, E23b, E24a, E24b는 artifact
계약을 만족한 정상 MAIN이지만 등록 claim은 열리지 않았다. E25a는
official source가 구성됐으나 parity gate가 실패한 terminal boundary다.
E25b는 scientific MAIN이 아니라 immutable human-audit package까지만
완료됐다.

## E22a — Active-Path Locality Method Selection

- Run:
  `/data/minjun_dev/CATENA/artifacts/e22a_locality_method_selection/20260728T174341.363711Z`
- Report SHA-256:
  `00ca1962a6829daceb74cf7fd4a54de544d782df57ae6cc463b0ee6f410b3ba9`
- 개발 seed: `1201, 1213, 1229`
- 선택: `smoothmax_100`
- Selection lock:
  `3d80098f929bf1e1e00a4979cb7384f8cfb4afbf3b4c14d3720afbb00a9c1e28`

| Metric | Mean retention | `smoothmax_100` |
|---|---:|---:|
| B recovery gain | 0.058940 | 0.059762 |
| C recovery gain | 0.141036 | 0.141705 |
| D recovery gain | 0.073344 | 0.073633 |
| Max active non-target degradation | 0.001004 | **0.000197** |
| Max retention degradation | 3.93e-6 | 6.70e-6 |
| Max capable affected MSE | 0.001796 | 0.001796 |

사전 선택 규칙에 따라 validation worst-cell degradation이 가장 낮은
`smoothmax_100`이 선택됐다. 다만 3-seed selection은 confirmatory
evidence가 아니며 capable affected MSE도 `0.001` guardrail보다 컸다.

허용 claim은 development-only method selection이다. Safe locality,
E21 재판정, semantic/LM/agent/official/runtime claim은 금지된다.

## E22b — Active-Path Locality Confirmatory

- Run:
  `/data/minjun_dev/CATENA/artifacts/e22b_active_path_locality/20260730T005512.389198Z`
- Report SHA-256:
  `56f65f8519dc69a981c477c17bced73319a741108282444d756a9572209702f7`
- 비교: `smoothmax_100` vs `mean_retention`
- 독립 단위: fresh paired seed 8개
- Disposition: `NOT_SUPPORTED`

| Gate | 관측값 | 판정 |
|---|---:|---|
| B separate-address recovery | 0.059011; 8/8; p=0.003906 | PASS |
| C state-read recovery | 0.142568; 8/8; p=0.003906 | PASS |
| D full-only recovery | 0.073427; 8/8; p=0.003906 | PASS |
| Absolute capacity | max affected MSE 0.001767 > 0.001 | FAIL |
| Absolute locality | max non-target 0.001144 > 0.0005 | FAIL |
| Retention | max degradation 9.02e-6 ≤ 0.0005 | PASS |
| Selected-vs-mean locality | gain 1.049e-4; 4/8; p=0.316406 | FAIL |
| Address/candidate | accuracy 1.0; candidate MSE 0.000147 | PASS |

Recovery의 상대 방향은 강했지만 absolute capacity와 worst-cell locality,
그리고 paired locality-improvement gate가 실패했다. 이는 구현 오류가
아니라 clean negative result다. Worst-cell-aware smooth-max만으로 safe
localized assimilation을 열 수 없으며 세 번째 locality repair는 수행하지
않는다.

Reportable observation은 recovery 방향과 retention/address/candidate
guardrail, 그리고 preregistered safe-locality 실패다. `claim_eligible=false`
이므로 positive safe-locality claim은 열리지 않는다. Safe assimilation,
E21의 사후 수리 및 외부 architecture transfer claim은 금지된다.

Frozen E22b protocol은 최신 실행 지시문의 3-way 목록과 달리
`NOT_SUPPORTED`를 포함한 4-way disposition을 main 전에 등록했다. 실제
결과는 이 immutable 4-way lock을 따라 `NOT_SUPPORTED`이며 사후 relabel하지
않는다.

## E23a — Full Product-Poset Screening

- Run:
  `/data/minjun_dev/CATENA/artifacts/e23a_product_poset_screen/20260728T174341.359934Z`
- Report SHA-256:
  `d0de1b91b3ec04a5b806357f72576e1de7399176a62e06326d64078e34325b7d`
- Seeds: 3
- Disposition: `SCREEN_ONLY_NO_CONFIRMATORY_CLAIM`

| Metric | 결과 |
|---|---:|
| Single-axis exact match | 0/4 |
| Pairwise exact match | 0/6 |
| Minimum minimal-set Jaccard | 0 |
| False adequate / false inadequate | 0 / 139 |
| Minimum predecessor gain | 0.000725 |
| Minimum incomparable gap | 0.000104 |

상대 predecessor/incomparable 방향은 보였지만 absolute epsilon-adequate
set이 비어 confirmatory minimality를 지지하지 않았다. Screen outcome은
E23b controller boundary나 threshold 선택에 사용되지 않았다.

## E23b — Product-Poset Confirmatory

- Run:
  `/data/minjun_dev/CATENA/artifacts/e23b_product_poset_confirmatory/20260730T113127.332498Z`
- Report SHA-256:
  `5dea3af8d434a8963198f6b1a912c2bb8cc55357845342723a74a85c58d1d11d`
- Boundary mode: `capacity_only`
- Seeds: 8
- Disposition: `NOT_SUPPORTED`

| Metric | 결과 |
|---|---:|
| Single-axis exact match | 0/4 |
| Pairwise exact match | 0/6 |
| Minimum minimal-set Jaccard | 0 |
| False adequate / false inadequate | 0 / 375 |
| Minimum predecessor gain | 0.000359 |
| Minimum incomparable gap | 0.000124 |
| Maximal-controller simpler degradation | 0.002358 |
| Preserve retention MSE max | 0.000902 |

E22b가 safe PASS가 아니므로 locality를 제외한 capacity+retention만
검정했다. Frozen theory boundary의 즉시 predecessor와 incomparable
contrast는 방향성 신호를 남겼지만, 등록 absolute tolerance에서 adequate
controller가 없어 empirical minimal set을 복구하지 못했다.
추가 guardrail도 maximal-controller simpler degradation
`0.002358 > 0.0005`, preserve retention
`0.000902 > 0.0005`로 실패했다.

허용 claim은 frozen 4-axis product-poset의 capacity-only confirmatory
검정이 실패했다는 것이다. Absolute minimality, safe locality, semantic,
official backend와 runtime transfer claim은 금지된다.

## E24a — Approximate-Rank Stress

- Run:
  `/data/minjun_dev/CATENA/artifacts/e24a_approximate_rank_stress/20260728T174341.503129Z`
- Report SHA-256:
  `3acf8be7877c27ec9aa7e23fe083467508bdaba2567c8ab7a24f4290f685a0ba`
- Disposition: `OOD_SPECTRUM_FAMILY_TRANSFER_NOT_SUPPORTED`

Primary descriptor-conditioned learned controller는 세 leave-one-spectrum
fold를 모두 실패했다. OOD epsilon-minimal rank match는 `0`, oracle floor
대비 mean excess는 `778.1204`였다. 반면 target-wise direct SVD diagnostic은
136/160, 85% match와 mean excess `2.386e-5`를 보였지만
diagnostic-only이며 primary failure를 구제하지 않는다.

등록 gate는 OOD rank-match fraction `>=0.50`과 mean normalized
excess-over-oracle `<=0.25`였다. 둘 다 실패했다. Canonical MAIN report의
`claim_boundary.allowed_claim`에는 잘못 `"None; dry-run is non-evidence."`
가 남아 있다. 이는 `run_mode=MAIN`과 모순되는 stale report-field defect이며
frozen report는 수정하지 않았다. 따라서 위 수치는 descriptive diagnostic과
negative disposition으로만 보고하고 positive E24a claim은 열지 않는다.
Universal learned-rank scaling, unseen geometry/basis, parameter efficiency와
official/LM claim은 금지된다.

## E24b — Behavioral-Attainability Stress

- Run:
  `/data/minjun_dev/CATENA/artifacts/e24b_behavioral_attainability_stress/20260728T174341.918677Z`
- Report SHA-256:
  `d4bd3f8bc21a4171f9c0ef9f4fa6e69f415236b74c3e6ac2995d02ef528f8abb`
- Disposition: `CONSTRUCTION_ROBUST_PREDICTION_FAILURE`

| Subset | Holdout | R² | Pearson | NRMSE |
|---|---|---:|---:|---:|
| Linear H=1 | Controller | -0.323 | 0.616 | 1.150 |
| Linear H=1 | Demand | 0.568 | 0.754 | 0.657 |
| Linear H=1 | Geometry | 0.591 | 0.769 | 0.640 |
| Noisy nonlinear multistep | Controller | -9.960 | 0.380 | 3.311 |
| Noisy nonlinear multistep | Demand | -1.133 | 0.528 | 1.460 |
| Noisy nonlinear multistep | Geometry | -1.110 | 0.561 | 1.452 |

어떤 holdout axis도 등록 gate `R²≥0.8`, `Pearson≥0.9`,
`NRMSE≤0.25`를 통과하지 못했다. 기존 H1의 제한된 construction을
소급 반증하지 않지만 construction-robust prescriptive predictor claim은
닫힌다.

## E25a — Official GDN2/KDA Gate

- Run:
  `/data/minjun_dev/CATENA/artifacts/e25a_official_gdn2_gate/20260728T130831.653416Z`
- Report SHA-256:
  `3e071908cc849844312cb1223c64a6b7f5fbb3ae1eb831313d27a3f6d456bc41`
- GDN2 commit: `95709fc250357c2dd109361c353192f2aa5913f9`
- FLA commit: `4b02d15d6a68700181b180235be62a9fb95d2a38`

FP32 full/chunk parity, backward finite gradient, state carry/clone/restore와
intervention confinement은 통과했다. BF16 relative L2
`0.00590334 > 0.005`와 tied-GDN2/KDA relative L2
`1.37832e-4 > 1e-5`가 실패했다. Threshold 완화, 다른 commit 탐색,
official replication과 reference fallback은 수행하지 않았다.

## E25b — Shared-Text-Encoder Audit Preparation

- Preparation run:
  `/data/minjun_dev/CATENA/artifacts/e25b_text_transaction_anchor/20260728T174341.332797Z`
- Report SHA-256:
  `4fc0dcbe5c5d8b5693adffd96b203bd245abff8376e74989208b1a33ae1159d2`
- Audit package:
  `/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z`
- Package manifest SHA-256:
  `2976a2073a0f3813a273b52811ff5ed0961e25bb71aaea12a1288d851341565d`
- Population: immutable 300 items

Reviewer A/B CSV와 instructions, merge/validation tool까지 준비됐다.
Codex는 human label을 작성하지 않았고 scientific row/checkpoint도
생성하지 않았다. 실제 독립 인간 2인의 audit가 끝나기 전에는 leakage
gate와 E25b MAIN을 실행하지 않는다.

Reviewer 파일:

```text
/data/minjun_dev/CATENA/artifacts/e25b_human_audit_packages/20260728T174341Z/
  reviewer_a/E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv
  reviewer_b/E25B_V4_HUMAN_AUDIT_REVIEW_WORK.csv
  REVIEWER_INSTRUCTIONS_KO.md
  merge_validate_audit.py
```

Reviewer A는 `reviewer_a_{semantic_preservation,operation_leakage,
entity_ambiguity,old_value_leakage,gold_consistency}`, Reviewer B는 동일한
`reviewer_b_` prefix 열만 작성한다. 허용 label은 `PASS`/`FAIL`이며 결측은
허용되지 않는다. Merge, raw agreement, Cohen's kappa, disagreement
adjudication template 생성과 최종 validate 명령은 package의
`REVIEWER_INSTRUCTIONS_KO.md`에 고정돼 있다. Adjudication CSV는 A/B 완료
후 merge 도구가 disagreement만 포함해 생성하므로 현재는 존재하지 않는다.

## 실제 dependency DAG

```text
E21b-R1
  -> E22a SELECTED
  -> E22b NOT_SUPPORTED
       -> E23b capacity_only
            -> E23b NOT_SUPPORTED

E18b SUPPORTED
  -> E23a SCREEN_ONLY
  -> E23b completed (screen outcome not used for boundary)

H1/E10b/E11b
  -> E24a OOD transfer NOT SUPPORTED
  -> E24b construction-robust prediction FAILURE

Official source
  -> E25a parity FAIL (terminal; replication not run)

E25b immutable population
  -> audit package complete
  -> two independent humans required
  -> leakage/floor gate and MAIN remain blocked
```

## 논문 반영 계획

Paper 수치는 이 문서 작성 단계에서 수정하지 않았다. 다음 변경은 별도
paper-update commit으로 수행한다.

1. E22b:
   `papers/transactional_control_algebra_long/tex/sections/07_boundaries.tex`
   의 locality subsection과 gate table에 relative recovery와 absolute
   capacity/locality 실패를 함께 둔다. Paired-seed plot은 appendix로 보낸다.
2. E23b:
   `papers/transactional_control_algebra_long/tex/sections/05_lattice.tex`
   의 E18 relative covering result와 분리된 capacity-only adequacy table로
   넣는다. Positive product-poset minimality figure로 표현하지 않는다.
3. E24a/E24b:
   `papers/transactional_control_algebra_long/tex/sections/appendix.tex`에
   construction-stress table을 추가하고 theorem-aligned diagnostic과
   learned/OOS generalization을 분리한다.
4. E25a:
   `papers/transactional_control_algebra_long/tex/sections/07_boundaries.tex`
   에 official evidence가 열리지 않았음을 한 문장으로 쓰고
   `appendix.tex`에 parity table을 둔다.
5. E25b는 human audit 전에는 figure/table에 넣지 않고
   `07_boundaries.tex`에서 pending protocol로만 기록한다.

현재 논문의 strongest supported result는 여전히 frozen controlled core와
E18의 registered relative sequence contrasts다. Post-E21 MAIN은 이를
확장하는 새 positive claim을 열지 않았고, locality·absolute minimality·
construction robustness·official transfer의 경계를 더 정확히 닫았다.

## Immutable artifact errata

Frozen artifact는 변경하지 않고 다음 label defect를 이 aggregate 문서에만
기록한다.

- E24a report의 `allowed_claim`은 MAIN인데도 dry-run 문구를 포함한다.
  Scientific disposition과 `claim_eligible=false`는 일관되므로 positive
  claim을 열지 않는다.
- E23a `RESULTS_SUMMARY_KO.md`의 `E22b dependency` label은 값이 가리키는
  E18b freeze validation의 오기다. Dependency DAG와 report는 E18b를
  올바르게 사용한다.
