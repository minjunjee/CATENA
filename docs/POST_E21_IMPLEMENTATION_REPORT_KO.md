# CATENA Post-E21 구현 보고서

기준 repository: `/home/minjun_dev/CATENA`  
구현 branch: `exp/post-e21`  
기준 HEAD: `c23986c12a199024a30fecdf94ae1bb55f67c071`

## 구현 범위

기존 E00–E21 scientific source와 canonical artifact를 수정하지 않고 다음
실험을 additive하게 구현했다.

| Experiment | Entry point | Config | Protocol |
|---|---|---|---|
| E22a | `experiments/e22a_locality_method_selection.py` | `configs/e22a_locality_method_selection.yaml` | E21 threshold를 상속한 3-seed development selection |
| E22b | `experiments/e22b_active_path_locality.py` | `configs/e22b_active_path_locality.yaml` | selected-vs-mean, fresh paired 8-seed confirmatory |
| E23a | `experiments/e23a_product_poset_screen.py` | `configs/e23a_product_poset_screen.yaml` | learned 16-controller screen |
| E23b | `experiments/e23b_product_poset_confirmatory.py` | `configs/e23b_product_poset_confirmatory.yaml` | result-independent absolute-adequacy poset boundary |
| E24a | `experiments/e24a_approximate_rank_stress.py` | `configs/e24a_approximate_rank_stress.yaml` | descriptor-conditioned approximate-rank LOFO stress |
| E24b | `experiments/e24b_behavioral_attainability_stress.py` | `configs/e24b_behavioral_attainability_stress.yaml` | noisy-teacher, geometry, multi-step/nonlinear attainability stress |
| E25a | `experiments/e25a_official_gdn2_gate.py` | `configs/e25a_official_gdn2_gate.yaml` | pinned official operator parity, replication fail-closed |
| E25b | `experiments/e25b_text_transaction_anchor.py` | `configs/e25b_text_transaction_anchor.yaml` | shared frozen text encoder, content addressing, human audit gate |

각 entry point는 새 UTC run directory를 만들고 `latest.json`,
`run_manifest.json`, `protocol_lock.json`, `data_manifest.json`,
`report.json`, raw/seed JSONL과 1-page 이내 `RESULTS_SUMMARY_KO.md`를
생성한다. MAIN report에는 config/data/checkpoint/protocol/source hash와
claim boundary가 기록된다.

## 핵심 구현 결정

- E22는 E21b-R1 checkpoint를 재사용하지 않는다. E21 lock의 B/C/D SESOI,
  locality, retention, address와 candidate guardrail만 읽어 새 lock에
  복사한다.
- E22 sparse route는 hard-forward/soft-backward top-k mask를 실제 recurrent
  update에 적용한다. Protected projection은 target-slot oracle diagnostic이며
  선택 대상이 아니다.
- E23 controller ID는 `(m,v,a,s)`의 canonical 4-bit tuple이다. 이론 boundary는
  predicted minimal, immediate predecessor, same-rank incomparable,
  immediate upper와 maximal controller를 outcome 이전에 고정한다.
- E23의 empirical minimal set은 affected correction과 retention, 그리고
  E22 safe mode에서 locality의 absolute tolerance를 먼저 적용한 뒤 poset
  minimal element를 계산한다. E22 selected objective를 정확히 구현할 수
  없는 경우 safe mode는 fail-closed한다.
- E24a primary learner는 test target별 SVD가 아니라 descriptor-conditioned
  low-rank controller다. Direct SVD는 oracle floor diagnostic에만 사용한다.
- E24b는 clean application target과 noisy teacher를 분리하고, affected
  update row와 structurally retained row를 분리한다. Predictor와
  controller-specific attainable floor는 test application outcome을 보기
  전에 생성한다.
- E25a는 별도 prefix, exact GDN2/FLA commit, license와 module origin을
  검증한다. Official dependency가 없거나 parity가 실패하면
  `NOT_CONFIGURED`/`FAIL`이며 reference fallback은 없다.
- E25b v4는 숫자 address ID 대신 opaque identifier의 shared-encoder
  content-addressing을 사용한다. Operation label, erase/write bit, exact
  mask와 old-value lexical cue는 model input에서 제외한다. 모든 controller는
  visible `policy-N` token의 동일 deterministic candidate decoder를 사용한다.
  Train에는 `ADD=A+B`와 `INVALIDATE=0`만 두고, primary의
  `SUPERSEDE=B`는 held-out composition과 symmetric-equivalence guard로
  분리한다. Old-rule query는 demand-aware `FULL/PARTIAL/NONE` categorical
  gold를 사용하며 oracle도 동일 evaluator를 통과한다.
- 사용자 제공 mock 11개는 `mocks/post_e21_packet/`에 원문 보존했으며
  scientific source에서 import하지 않는다.

## 새 파일

주요 신규 경로:

```text
experiments/e22a_*.py ... experiments/e25b_*.py
configs/e22a_*.yaml ... configs/e25b_*.yaml
src/catena/post_e21/
src/catena/data/controller_poset.py
src/catena/data/product_poset_sequence.py
src/catena_official_plugins/e25a_replication.py
environments/e25a_official_gdn2_{environment.yaml,observed_lock.json}
tests/test_e22_*.py ... tests/test_e25b_*.py
scripts/{run,launch,check}_post_e21*.sh
tools/post_e21_status.py
docs/E22* ... docs/E25*
```

기존 tracked scientific 파일의 수정은 없다. 신규 E23 data module 두 개가
root-level `data/` ignore 규칙에 가려지지 않도록 `.gitignore`에 두 파일만
예외로 추가했다. 구현 전 사용자 제공 untracked mock은 충돌 방지를 위해
`mocks/post_e21_packet/`으로 이동했다.

## 검증과 실행 상태

- E22/E23/E24/E25b scientific MAIN: `NOT_RUN`
- E25a scientific replication: `NOT_RUN`
- E25a parity-only gate: `FAIL` (configured official source, 4/6 checks PASS)
- E25b 300-item audit preparation: `/tmp` validation만 `PASS`
- E25b v4 독립 adversarial review: `ACCEPT` (blocker 없음)
- E25b human review/main: `NOT_RUN`
- Canonical E22–E24/E25b artifact: main 실행 전에는 생성하지 않음
- Canonical E25a artifact: parity-only gate report 하나 생성
- CPU dry-run: `/tmp` fresh root에서만 실행
- E18–E21 targeted regression: PASS
- E00–E21 immutable artifact hash: 1,329/1,329 PASS, changed 0

E25a gate에서 source/import, FP32 full-vs-chunk, backward finite gradient,
state carry/clone/restore와 intervention confinement은 통과했다. 다음 두
preregistered tolerance가 실패했으므로 threshold를 변경하거나 replication을
실행하지 않았다.

| E25a check | Observed | Gate |
|---|---:|---:|
| BF16 vs FP32 relative L2 | `0.0059033381` | `<= 0.005` |
| tied GDN2 vs KDA relative L2 | `0.0001378316` | `<= 0.00001` |

Gate artifact:
`/data/minjun_dev/CATENA/artifacts/e25a_official_gdn2_gate/20260728T130831.653416Z/`

최종 validation:

| Check | Result |
|---|---|
| Repository pytest | 654/654 PASS |
| Ruff check/format | 43 files PASS |
| strict mypy | 30 source files PASS |
| compileall | PASS |
| YAML/JSON parse | 9/12 files PASS |
| entry-point `--help` | 8/8 PASS |
| shell syntax | 3/3 PASS |
| protocol source binding | 90 bindings across 9 locks PASS |
| integrated non-evidence dry-run | 9/9 PASS |
| pre-E22 artifact immutability | 1,329/1,329 PASS, changed 0 |

통합 dry-run root:
`/tmp/catena_post_e21_dry_final_20260728T143000Z/`

주요 protocol lock SHA-256:

```text
E22a  1012bea7d097d8b0079273883384f28ad0f31a4a1f41a2164e481f5b9ebf0244
E22b  e19dfd26018e53d7ab601d1bd1b0e94c3bd922e1849c35cdcffec7ae38474598
E23a  93f7a4e7c7df2505d984e887b8b0308f8d1e889faa45bce4c377530e83f8aa6e
E23b  ffcff888c026a4cbf03e6443965f49f42d3fbd9034d6f7374e1f5203fb02b1ca
E24a  981d8c0383db519775508dc5585461691108f7e51cafe06c4dbf4dfa8627d52e
E24b  233caf6a55748766c800dab2c8c59f8bec80b2a54f243f3b97dd9e2a0f1e0fcb
E25a  f6c8c966a97266b680458589992f25bce4940a851375b7084f8922254bb36262
E25b  398882e4a8c3f18449c7d04698d432328eb758b0c6a10a2ec13c1679ab8f0610
```

## 검토 후 실행 명령

전체 non-evidence dry-run:

```bash
cd /home/minjun_dev/CATENA
source /home/minjun_dev/miniconda3/bin/activate catena-v6

bash scripts/run_post_e21_dry.sh \
  /tmp/catena_post_e21_dry_manual_20260728
```

Canonical status:

```bash
bash scripts/check_post_e21_status.sh \
  /home/minjun_dev/CATENA \
  /data/minjun_dev/CATENA/artifacts
```

Wave-1 launcher는 기본적으로 preflight만 한다.

```bash
bash scripts/launch_post_e21_wave1.sh \
  /home/minjun_dev/CATENA \
  /data/minjun_dev/CATENA/artifacts
```

사용자가 implementation/dry artifact를 검토한 뒤에만 다음 acknowledgement로
E22a, E23a와 E24a/E24b MAIN, 그리고 claim 비대상 E25b audit preparation을
시작한다. E25a에는 이미 terminal parity-gate report가 있으므로 launcher가
이를 감지해 자동 재실행하지 않는다.

```bash
CATENA_POST_E21_MAIN_ACK=POST_E21_MAIN_AUTHORIZED \
bash scripts/launch_post_e21_wave1.sh \
  /home/minjun_dev/CATENA \
  /data/minjun_dev/CATENA/artifacts
```

E22b는 completed E22a run, E23b는 explicit E18 freeze, completed E23a와
completed E22b report가 모두 필요하다. E25b MAIN은 v4 population lock과
분리된 review-work copy에서 완료된 300-item two-reviewer audit CSV 없이는
차단된다. E25a replication은 PASS gate report와
`--allow-scientific-replication`을 동시에 요구하며 본 구현 검증에서
실행하지 않는다.
