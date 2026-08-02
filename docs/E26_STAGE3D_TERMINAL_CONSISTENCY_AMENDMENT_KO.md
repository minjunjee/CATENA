# E26 Stage-3D Terminal Consistency Amendment

## 판정

다음 기존 run은 수정하거나 재판정하지 않는다.

```text
/data/minjun_dev/CATENA/artifacts/
  e26_stage3d_fixed_layout_bf16_admissibility/
  20260802T144040.692630Z
```

이 run의 validated `report.json`은 다음을 기록한다.

```text
execution_status: FAILED_IMPLEMENTATION_OR_EXECUTION
disposition: STAGE3D_NOT_EVALUABLE_IMPLEMENTATION_OR_EXECUTION_ERROR
```

반면 같은 run의 `status.json`, `RESULTS_SUMMARY_KO.md`, 당시 `latest.json`은
`STAGE3D_BLOCKED_FIXED_LAYOUT_NUMERICAL_INSTABILITY`를 기록했다.

감사 결과 두 구현 결함을 분리했다.

1. **Primary rich-G3 schema defect:** 실제 G3 worker receipt에는 compiled/reference
   비교와 observed backend-integrity를 포함한 valid 8-key comparison이 기록됐다.
   당시 generic `_comparison_observed` validator와 JSON Schema의
   `$defs/comparison`은 이 richer G3 comparison을 허용하지 않았다. 그 결과 report
   builder는 실제 비교 결과를 numerical failure로 판정하지 않고
   `FAILED_IMPLEMENTATION_OR_EXECUTION / NOT_EVALUABLE`로 닫았다.
2. **Terminal propagation defect:** G3 조기 종료 runner가 위 validated report
   disposition을 사용하지 않고 status를 `BLOCKED`로 직접 기록했다. Summary와
   latest가 status를 우선 읽으면서 모순이 전파됐다.

Prospective successor는 exact `g3Comparison` schema와 G3-specific observed-field
validator를 추가한다. Corrected validator로 rich G3 row를 재검증하는 regression은
이 schema gap을 해소하지만, 이는 기존 run의 사후 재판정이 아니다. 이 run에서
disposition의 authoritative source는 원본 `report.json`이며, 나머지 상충 terminal
metadata는 claim 또는 dependency를 여는 데 사용할 수 없다.

## 원본 hash

| 파일 | SHA-256 |
| --- | --- |
| `report.json` | `510700aefa61d56985267ab6f2d31f43e48e411ecddf6c4161b644952039d221` |
| `status.json` | `24721e865774b5930b02cc398d2bc8665ba666c37617727e22cfc494ea3bdaba` |
| `RESULTS_SUMMARY_KO.md` | `81311fd4bad34478b0b749e11191d0d8bc015e59383b2732461fab837810c766` |
| `artifact_audit.json` | `f39febfedfc41ab803c69b05ebf0a847406cae6b982aad333c296666ab115afb` |
| 당시 namespace `latest.json` | `9ae273139c8ea743f9af3ca1c12d413057de7095cf4ecd5cf56fa4c04d7de705` |

## Prospective repair

후속 fresh run부터 다음 규칙을 적용한다.

1. `status.json`의 `execution_status`, `disposition`, resource eligibility와 evidence
   flag는 이미 작성·검증된 `report.json`에서만 생성한다.
2. `RESULTS_SUMMARY_KO.md`와 `latest.json`도 동일 report에서 생성한다.
3. status의 report SHA와 terminal semantics가 report와 다르면 summary,
   artifact audit, latest pointer를 발행하지 않는다.
4. 기존 summary가 report-derived 내용과 다르면 덮어쓰지 않고 fail-closed한다.
5. `NOT_EVALUABLE`은 numerical `BLOCKED`와 구분하며 process exit code 2를 사용한다.
6. 기존 run과 artifact는 byte 단위로 보존한다.
7. Rich G3 row의 evaluability는 raw observed fields로 재계산하고, cached
   `passed` boolean만으로 승격하지 않는다.

## Claim boundary

이 amendment는 numerical gate 결과를 개선하지 않는다. 해당 기존 run은
`NOT_EVALUABLE`이며 Scientific E26a와 resource preflight를 열지 않는다. 후속
fresh run도 validated report가 `STAGE3D_GO_FIXED_LAYOUT_BF16_ADMISSIBLE`일 때만
resource preflight 대상이 된다.
