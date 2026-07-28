# E04 implementation amendment 01

## 배경

최초 prospective E04 protocol과 config를 고정한 뒤 dry run
`20260727T054548.378221Z`를 별도 dry namespace에서 실행했다. 계산 및 raw
artifact 생성은 끝났으나, strict JSON report writer가
`seed_values` object의 integer key를 거부해 final report와 completed manifest를
쓰지 못했다.

해당 dry run은 삭제하거나 덮어쓰지 않는다.

```text
run_mode: dry_run
manifest_status: RUNNING
main_eligible: false
full_eligible: false
quartet_registry_rows: 8
intervention_design_lock_rows: 16
intervention_metric_rows: 480
seed_level_effect_rows: 1
main_namespace_inspected: false
```

## 허용되는 수정

Report에 들어가는 checkpoint-seed keyed object의 key를 JSON string으로
직렬화한다.

```text
11 -> "11"
22 -> "22"
...
88 -> "88"
```

동일한 오류가 다시 발생하지 않도록 strict report serialization regression
test를 추가한다.

## 변경하지 않는 항목

- E04 config와 모든 threshold
- operation, geometry, seed, dry/main namespace
- checkpoint와 donor pairing
- intervention, outcome, estimand, bootstrap, sign-flip 정의
- gate conjunction과 claim wording
- original E02 및 E02b disposition
- evidence tier

실패 dry run의 수치 결과는 threshold, gate, donor, claim 또는 구현 수식 선택에
사용하지 않는다. Main namespace는 이 amendment 전에 생성하거나 열어보지
않았다.

## 재실행 조건

수정 source로 repository test와 새 E00 source lock을 통과한 뒤 새 UTC dry run을
만든다. 새 dry가 완결된 `NOT_EVALUATED_DRY_RUN` artifact를 만들고 schema,
row count, provenance를 통과한 경우에만 main을 실행한다.
