# Post-E21 통합 dry-run

`scripts/run_post_e21_dry.sh`는 E22–E25의 non-evidence 경로만 순차
검증한다. Scientific MAIN과 E25a official replication은 실행하지 않는다.
E25b는 train/eval dry 뒤에 같은 `/tmp` root에서 300-item
`AUDIT_PREPARATION` 계약까지 검사한다.

## 안전 계약

- 인자는 정확히 하나의 **존재하지 않는** absolute path여야 한다.
- 경로는 `/tmp/catena_post_e21_dry_*` direct child만 허용한다.
- 기존 directory를 삭제하거나 재사용하지 않는다.
- 모든 entrypoint에 `--device cpu`, `--dry-run`, 동일한 explicit
  `--artifact-root`를 전달한다.
- `/data/minjun_dev/CATENA/artifacts`에는 쓰지 않는다.
- `catena-v6` prefix가 아닌 interpreter는 거부한다.

실행 예:

```bash
source /home/minjun_dev/miniconda3/bin/activate catena-v6

bash scripts/run_post_e21_dry.sh \
  /tmp/catena_post_e21_dry_manual_20260728
```

경로 안전성만 확인하고 아무것도 생성하지 않으려면:

```bash
CATENA_POST_E21_DRY_VALIDATE_ONLY=1 \
bash scripts/run_post_e21_dry.sh \
  /tmp/catena_post_e21_dry_preflight_20260728
```

## 고정 실행 순서

1. E22a locality method selection dry
2. E22b selected-vs-mean dry — E22a `selection_lock.json`을 명시적으로 전달
3. E23a product-poset screen dry
4. E23b confirmatory-pipeline dry
5. E24a approximate-rank stress dry
6. E24b behavioral-attainability stress dry
7. E25a official operator **gate contract** dry
8. E25b text transaction train/eval dry
9. E25b 300-item audit preparation — review는 수행하지 않음

E23a/E23b dry는 canonical E18 freeze, E23a MAIN report, E22b report를
읽지 않는 protocol-fixed synthetic non-evidence dependency fixture를
사용한다. 16 learned controller projection은 첫 seed, 2 training step,
첫 intensity/update/gap cell로 축소해 실제 checkpoint/evaluation 경로를
검사한다. 실제 dependency를 소비하는 경로는 MAIN에서만 열린다.

## 산출물

각 experiment의 기존 artifact 계약 외에 root에 다음이 생성된다.

```text
_logs/
POST_E21_DRY_RUN_MANIFEST.json
```

Manifest는 9개 run directory와 report SHA-256을 기록하고 다음을 명시한다.

```text
claim_eligible: false
scientific_main_executed: false
official_replication_executed: false
audit_preparation_executed: true
```

각 train/eval phase가 끝날 때 runner는 `run_manifest.json`과
`report.json`의 `run_mode=DRY_RUN`, protocol snapshot, 결과 summary,
non-claim 상태를 재검증한다. E25b audit phase는
`run_mode=stage=AUDIT_PREPARATION`, locked CSV/JSONL filenames와
`claim_eligible=false`를 별도로 검증한다. 실패 artifact는 진단을 위해
`/tmp`에 그대로 보존한다.
