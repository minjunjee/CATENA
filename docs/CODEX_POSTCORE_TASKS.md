# Codex post-core implementation contract

## 목표

기존 H1-H4 artifact를 변경하지 않고 E10-E16을 실행·검증한다.

## 절대 변경 금지

- 기존 `e01b`, `e02`, `e02b`, `e03`, `e03b`, `e04`, `e05a`, `e05a_r1` artifact
- 기존 claim status
- H5 NO-GO disposition
- 기존 threshold와 source fingerprint

## 작업 순서

1. 새 Python 파일이 import되는지 compile/test
2. E10-E13a CPU dry-run
3. E16 registry 생성 및 실제 artifact ID 보완
4. 4-GPU Wave 1 실행
5. prospective E13a-R1 GO 확인
6. E13b 5-seed tied/dual 실행
7. E13c aggregate
8. E13c supported일 때만 E14
9. Official backend는 별도 environment와 pinned commit에서만 E15

## Acceptance tests

- 모든 새 experiment는 YAML config, manifest, report, latest pointer를 생성
- `--dry-run`은 CPU에서 2분 이내 종료
- NaN/Inf 발견 시 즉시 FAIL
- checkpoint path와 parameter count 기록
- reference/official evidence tier 혼합 금지
- main 결과를 본 뒤 threshold 변경 금지

## 결과 해석

Codex는 결과를 `SUPPORTED`로 만들기 위해 task나 metric을 수정하지 않는다. Gate 실패 시 원인과 허용 claim ceiling을 기록한다.
