# CATENA Post-E21 로컬 저장소 감사

감사 시각: 2026-07-28 UTC  
작업 기준: `/home/minjun_dev/CATENA`

## 판정

E18–E21의 source, config, protocol lock, test와 실제 artifact가 모두 로컬에
존재한다. 따라서 Post-E21 구현을 진행할 수 있다. 로컬 작업 트리를 유일한
기준으로 사용하며 원격 상태로 reset, pull 또는 checkout하지 않는다.

## Git와 작업 트리

| 항목 | 감사 결과 |
|---|---|
| Local HEAD | `c23986c12a199024a30fecdf94ae1bb55f67c071` |
| Audit-time local branch | `master` |
| Post-audit implementation branch | `exp/post-e21` |
| Upstream | `origin/master` |
| `origin/master` | local HEAD와 동일 |
| `origin/main` | 존재하지 않음 |
| Remote | `https://github.com/minjunjee/CATENA.git` |
| Tracked dirty file | 없음 |
| Untracked file | 아래 Post-E21 mock 11개 |

감사 당시 untracked 파일:

```text
configs/e22_active_path_locality.yaml
configs/e23_product_poset_sequence.yaml
configs/e24_theory_stress.yaml
configs/e25a_official_gdn2_gate.yaml
configs/e25b_text_transaction_anchor.yaml
docs/EXPECTED_FILE_MAP.md
experiments/e22_active_path_locality.py
experiments/e23_product_poset_sequence.py
experiments/e24_construction_independent_stress.py
experiments/e25a_official_gdn2_gate.py
experiments/e25b_text_transaction_anchor.py
```

이 파일들은 `status: mock_contract_only` 또는
`Interface mock ... Do not copy blindly`로 표시된 사용자 제공 계약
참고물이다. 실제 scientific entry point로 사용하지 않는다. 혼동을 막기
위해 내용은 보존한 채 `mocks/post_e21_packet/` 아래로 이동하고, 실제
구현은 E18–E21 배치 규칙에 맞는 새 파일로 추가한다.

감사를 완료한 뒤 dirty/untracked 파일을 잃지 않은 상태에서
`exp/post-e21` branch를 생성했다. HEAD는 계속
`c23986c12a199024a30fecdf94ae1bb55f67c071`이며 remote reset, pull,
checkout은 수행하지 않았다. 이후 변경은 모두 Post-E21용 신규 파일 또는
위 reference mock의 보존 이동이다.

## E18–E21 존재성 및 배치 규칙

| 실험 | Entry point | Config | 핵심 module | 실제 artifact |
|---|---|---|---|---|
| E18a | `experiments/e18a_sequence_control_lattice.py` | `configs/e18a_sequence_control_lattice.yaml` | `src/catena/{data,models,training}/sequence_control_lattice.py` | `/data/minjun_dev/CATENA/artifacts/e18a_sequence_control_lattice/` |
| E18b | `experiments/e18b_sequence_control_lattice_aggregate.py` | `configs/e18b_sequence_control_lattice_aggregate.yaml` | E18 aggregate/statistics | `/data/minjun_dev/CATENA/artifacts/e18b_sequence_control_lattice_aggregate/20260728T074753.618843Z/` |
| E19a | `experiments/e19a_localization_candidate_decomposition.py` | `configs/e19a_localization_candidate_decomposition.yaml` | `src/catena/{data,models,training}/localization_candidate.py` | `/data/minjun_dev/CATENA/artifacts/e19a_localization_candidate_decomposition/` |
| E19b | `experiments/e19b_localization_candidate_aggregate.py` | `configs/e19b_localization_candidate_aggregate.yaml` | E19 aggregate/statistics | `/data/minjun_dev/CATENA/artifacts/e19b_localization_candidate_aggregate/20260728T060318.439621Z/` |
| E20 | `experiments/e20_quality_constrained_break_even.py` | `configs/e20_quality_constrained_break_even.yaml` | `src/catena/eval/quality_break_even.py` | `/data/minjun_dev/CATENA/artifacts/e20_quality_constrained_break_even/20260728T062658.423644Z/` |
| E21a | `experiments/e21_structured_sequence_localization_transfer.py` | `configs/e21_structured_sequence_localization_transfer.yaml` | `src/catena/{data,models,training,eval}/structured_sequence_localization.py` | `/data/minjun_dev/CATENA/artifacts/e21a_structured_sequence_localization_transfer/` |
| E21b-R1 | `experiments/e21b_r1_structured_sequence_localization_aggregate.py` | `configs/e21b_r1_structured_sequence_localization_aggregate.yaml` | `src/catena/eval/structured_sequence_localization_r1.py` | `/data/minjun_dev/CATENA/artifacts/e21b_r1_structured_sequence_localization_aggregate/20260728T091547.163300Z/` |

확인된 frozen dependency:

- E18b: `SUPPORTED`
- E21b-R1: execution `PASS`, claim `NOT_SUPPORTED`
- E21b-R1 boundary: target recovery는 통과했으나 capable floor와 active
  non-target locality guardrail은 실패

E22는 E21b-R1을 재판정하지 않고, 해당 lock의 기준만 읽기 전용으로
상속하는 독립 repair다.

## Artifact 경로

```text
/home/minjun_dev/CATENA/artifacts
  -> /data/minjun_dev/CATENA/artifacts
```

`artifacts`는 symlink이며 realpath는
`/data/minjun_dev/CATENA/artifacts`다. 새 main run의 유일한 root도 이
경로로 유지한다. 구현 검증용 dry-run은 canonical root가 아닌 `/tmp`의
fresh directory만 사용한다.

감사 시점 immutable anchor:

| Anchor | SHA-256 |
|---|---|
| `docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json` | `7c465ceb60b6979e717d85599533bd7c0dd884f10b191fa29c42771ccc9c9989` |
| `docs/E19_LOCALIZATION_CANDIDATE_LOCK.json` | `8550fef23f938d84e35f584f16fd625cdb36c8422a6eefacf86f198f614dd3ec` |
| `docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json` | `e07139064b6f2cf1ca990f4f595d38c64f295cd7b25ef2fd3a935cbefe498579` |
| `docs/E21B_R1_STRUCTURED_SEQUENCE_AGGREGATE_LOCK.json` | `57fc7615f39f07c1d1e8377bc7877ef1fbc274d820b22bd87a6bc8386e99422f` |
| `E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json` | `39416476994963900305e04682dabd458719a0a94c92820405feeb639c33e67c` |
| `E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json` | `c27eff42a9ee0fdb1d88f8c78882f3a08ee06857012db78715ab8781b295efce` |

## 재사용할 구현

| 역할 | 재사용 대상 |
|---|---|
| Sequence data/runner | `sequence_control_lattice.py`, `structured_sequence_localization.py`, E18/E21 entry-point CLI 패턴 |
| Controller lattice | `src/catena/models/sequence_control_lattice.py`, `src/catena/models/lattice_controllers.py` |
| Localization/state read | `src/catena/{data,models,training}/localization_candidate.py`, `structured_sequence_localization.py` |
| Statistics | `src/catena/eval/seed_inference.py`, `src/catena/eval/postcore_metrics.py`, E18/E21 paired aggregate 패턴 |
| Artifact writer | `experiments/common.py`, `src/catena/core/io.py` |
| Provenance/integrity | `src/catena/core/provenance_v61.py`, E18/E21 lock validators와 freeze tooling |
| Official adapter boundary | `src/catena/models/official_adapters.py`, `src/catena/eval/official_operator_gate.py`, E15a-R1 gate |

새 코드는 frozen E18–E21 module을 수정하지 않고 위 API를 import하거나
Post-E21 전용 module에서 composition한다.

## E21에서 상속할 고정값

값은 새 코드에 중복 하드코딩하지 않고
`docs/E21_STRUCTURED_SEQUENCE_LOCALIZATION_TRANSFER_LOCK.json`의
`registered_thresholds`에서 읽어 E22 protocol lock에 복사한다.

| 항목 | 값 |
|---|---:|
| Recovery SESOI (`selective_gain`) | `0.001` |
| Active non-target locality margin | `0.0005` |
| Primary-context retention margin | `0.0005` |
| Minimum address accuracy | `0.95` |
| Maximum candidate recovery MSE | `0.001` |
| Maximum capable affected MSE | `0.001` |
| Exact sign-flip alpha | `0.05` |

## 구현 파일 계획

새 파일:

```text
experiments/e22a_locality_method_selection.py
experiments/e22b_active_path_locality.py
experiments/e23a_product_poset_screen.py
experiments/e23b_product_poset_confirmatory.py
experiments/e24a_approximate_rank_stress.py
experiments/e24b_behavioral_attainability_stress.py
experiments/e25a_official_gdn2_gate.py
experiments/e25b_text_transaction_anchor.py
configs/e22a_locality_method_selection.yaml
configs/e22b_active_path_locality.yaml
configs/e23a_product_poset_screen.yaml
configs/e23b_product_poset_confirmatory.yaml
configs/e24a_approximate_rank_stress.yaml
configs/e24b_behavioral_attainability_stress.yaml
configs/e25a_official_gdn2_gate.yaml
configs/e25b_text_transaction_anchor.yaml
src/catena/post_e21/*
src/catena/data/controller_poset.py
tests/test_e22_*.py
tests/test_e23_*.py
tests/test_e24_*.py
tests/test_e25a_*.py
tests/test_e25b_*.py
scripts/launch_post_e21_wave1.sh
scripts/check_post_e21_status.sh
scripts/verify_pre_e22_artifacts.py
docs/E22_ACTIVE_PATH_LOCALITY_PROTOCOL_KO.md
docs/E23_PRODUCT_POSET_PROTOCOL_KO.md
docs/E24_THEORY_STRESS_PROTOCOL_KO.md
docs/E25A_OFFICIAL_GDN2_PROTOCOL_KO.md
docs/E25B_TEXT_TRANSACTION_PROTOCOL_KO.md
docs/POST_E21_DEPENDENCY_DAG_KO.md
docs/POST_E21_IMPLEMENTATION_REPORT_KO.md
docs/POST_E21_ARTIFACT_HASH_VERIFICATION_KO.md
```

Reference mock는 새 구현과 이름 충돌이 없도록
`mocks/post_e21_packet/{experiments,configs,docs}/`로 이동한다.

## 기존 파일 수정 계획

기존 scientific source, config, lock, artifact는 수정하지 않는다.
패키지 export가 꼭 필요한 경우에도 `__init__.py` 변경 없이 직접 module
import를 우선한다. 구현 후 Git 가시성 검증에서 root-level `data/` ignore
pattern이 신규 `src/catena/data/controller_poset.py`와
`product_poset_sequence.py`까지 가리는 것을 확인했다. 따라서 기존 scientific
source가 아닌 `.gitignore`만 수정해 이 두 신규 파일을 명시적으로 unignore한다.
이 변경의 regression은 `git check-ignore`와 전체 repository test로 확인한다.

새 기능 때문에 기존 파일 수정이 불가피해지면 먼저 이유와 영향 범위를 이
문서에 amendment로 기록한 뒤, 전체 기존 test와 frozen-file hash 검증을
통과시킨다.

## Regression 검증

1. Post-E21 unit test
2. 기존 E18–E21 targeted regression
3. 전체 `pytest`
4. `compileall`
5. 신규 Python에 대한 Ruff와 mypy
6. 모든 entry point의 `--help`
7. canonical artifact root 밖에서 CPU dry-run
8. shell syntax 검사
9. tracked E00–E21 source가 기준 commit 대비 변경되지 않았는지 검사
10. immutable freeze와 그 내부 sealed file SHA-256 재검증

Scientific main, 4-GPU run, model download와 official minimal replication은
이 구현 단계에서 시작하지 않는다.
