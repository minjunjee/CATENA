# E16 - Core Evidence Freeze

## 목적

기존 H1-H5 artifact를 수정하지 않고, report path와 SHA-256을 하나의 registry에 동결한다.

## 출력

- `evidence_registry.json`
- `results_macros.tex`
- `report.json`

Config의 각 claim은 여러 candidate experiment ID를 가질 수 있다. 실제 존재하는 첫 번째 latest pointer를 선택한다. Missing result는 숨기지 않고 `MISSING`으로 기록한다.

## 사용

```bash
python experiments/e16_core_evidence_freeze.py \
  --config configs/e16_core_evidence_freeze.yaml \
  --device cpu \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

논문 수치는 registry에 포함된 report만 사용한다.
