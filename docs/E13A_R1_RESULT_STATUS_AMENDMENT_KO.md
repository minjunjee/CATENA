# E13a-R1 결과 상태 Additive Amendment

## 목적

이 문서는 기존 E13a-R1 결과 문서나 artifact를 수정·대체하지 않는다.
원본 결과 문서
`docs/E13A_SEQUENCE_CALIBRATION_RESULT_KO.md`의 SHA-256은 다음과 같이
고정되어 있다.

```text
ff1f13a6955719ada91120891404cbdb43e57d24c4e522f48e007f822e56dd4e
```

원본 문서의 `PASS / GO_FOR_E13B` 및 E13b dependency 표현은 E13a-R1
calibration 직후, distractor path의 후속 구조 진단 전에 작성된 당시
상태를 기록한다.

## 후속 진단과 최종 상태

후속 감사에서 E13a-R1의 distractor event path가 model update에서
구조적으로 hard-masked되어 있음을 확인했다. 따라서 E13a-R1은 원래의
hard-masked pipeline에 대한 calibration으로만 보존한다.

```text
execution_status: PASS
calibration_status: GO_FOR_ORIGINAL_HARD_MASKED_PIPELINE_ONLY
repaired_e13b_dependency_eligible: false
diagnosis: DISTRACTOR_PATH_STRUCTURALLY_HARD_MASKED
```

이 판정은 E13a-R1의 실행 결과나 metric을 변경하지 않는다. Repaired
learned-distractor sequence pipeline의 dependency는 별도 prospective
E13a-R2만 제공한다.

```text
repaired_pipeline_dependency:
  experiment: E13a-R2
  run_id: 20260727T190642.222102Z
  calibration_status: GO_FOR_E13B_R1
```

## 불변성 경계

- E13a-R1 원본 문서, report, manifest, metric, checkpoint 및 freeze를
  수정하지 않는다.
- 원본의 `go_for_e13b=true`는 당시 hard-masked pipeline calibration
  report의 값이며 repaired learned-distractor claim을 열지 않는다.
- Repaired E13b-R1은 E13a-R2 report와 manifest만 dependency로 사용한다.
- 이 amendment는 상태 해석과 dependency boundary만 추가하며 metric,
  SESOI, gate 또는 scientific source를 변경하지 않는다.
