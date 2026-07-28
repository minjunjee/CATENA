# E05a-R1 사전등록 Hash Lock

동결 시각: `2026-07-27T14:07:33Z`  
동결 시점의 R1 data/model execution 수: `0`

이 문서는 E05a-R1의 결과를 생성하기 전에 config와 protocol 문서의
정확한 byte/canonical hash를 고정한다.

| 파일 | SHA-256 |
| --- | --- |
| `configs/e05a_r1_semantic_design_repair.yaml` | `0f56777e4f8283154e317afa21c60c01f80d138457098155a3bfba49fafe190c` |
| 위 YAML의 canonical JSON | `159880727a3a3e0e10edc876466238022828252118b1ebff0fecb0f4c9984b91` |
| `docs/E05A_R1_SEMANTIC_DESIGN_REPAIR_PREREGISTRATION_FROZEN_KO.md` | `5b06d68aa0fe5c3ec9ef0dceb1963b9ed114aa6570d16970df5fa1c1c9a100ee` |

원본 E05a dependency도 다음 hash로 고정한다.

| 대상 | SHA-256 |
| --- | --- |
| `artifacts/E05A_ARTIFACT_FREEZE_V1.json` | `f6e6edebd303fb1b6d48cff9630516a8864dc317386778202da58a2a6c189122` |
| `artifacts/E05A_CLAIM_STATUS.json` | `f1c1e1585829048c47ac725ce03ffaa3293bfe67c10a23a1b205aaa4af432ec3` |
| 원본 E05a `run_manifest.json` | `c2571fa8c4ec184068dff3bb002dc08be1c503c147348bb19ada1fd1199b5e2b` |
| 원본 E05a `report.json` | `34bab0288d5bbe82e1debcfa81e51493f4fa280475b86f0d097cb2a4aff8057c` |

실행기는 위 hash와 이 lock 문서 자신의 hash를 data generation 전에
검증해야 한다. 불일치가 있으면 새 run directory를 만들기 전에
중단한다.

동결 후에는 다음을 변경하지 않는다.

- R1 seed, namespace, train/validation row
- relational encoder 정의
- controller와 training budget
- primary estimand, SESOI, CI 단위, sign-flip 정의
- oracle, retention, control guardrail
- R1 `NO_GO` 시 H5 종료 규칙
- human audit와 E05b-R1의 별도 namespace 경계

원본 E05a artifact/report/config는 어떤 경우에도 수정하거나 재판정하지
않는다.
