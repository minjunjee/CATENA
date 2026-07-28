# E21 Evidence Freeze Tooling

`scripts/freeze_e21_structured_sequence_transfer.py`는 완료된 다음 artifact를
명시적 경로로 검증하고 하나의 새 top-level freeze JSON에 묶는다.

- E21a MAIN source 정확히 5개
- 원본 E21b aggregate 1개
- prospective E21b-R1 aggregate 1개

원본 E21b report의 observed boolean과 무관하게 freeze disposition은 항상
`INCONCLUSIVE_GATE_IMPLEMENTATION`, `claim_eligible=false`다. 유효 claim
gate는 E21b-R1 하나뿐이다.

## 생성

```bash
python scripts/freeze_e21_structured_sequence_transfer.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --source-run /absolute/e21a/seed113/run \
  --source-run /absolute/e21a/seed223/run \
  --source-run /absolute/e21a/seed331/run \
  --source-run /absolute/e21a/seed449/run \
  --source-run /absolute/e21a/seed557/run \
  --original-aggregate-run /absolute/e21b/run \
  --r1-aggregate-run /absolute/e21b_r1/run
```

출력은 다음 하나이며 `O_EXCL`로 생성되어 기존 파일을 덮어쓰지 않는다.

```text
/data/minjun_dev/CATENA/artifacts/
└── E21_STRUCTURED_SEQUENCE_TRANSFER_FREEZE_V1.json
```

먼저 `--dry-run`으로 전체 검증과 payload를 확인할 수 있다. 이 도구는
source나 aggregate를 실행하지 않는다.

## 기존 freeze 재검증

```bash
python scripts/freeze_e21_structured_sequence_transfer.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --validate-existing
```

재검증은 freeze에 기록된 exact run path만 사용하며 `latest`를 탐색하지
않는다. 다음을 원본 파일에서 다시 계산한다.

- source/aggregate report, manifest, metrics, contrast, provenance, summary hash
- source checkpoint hash
- source E21 lock/config와 E21b-R1 lock/config hash
- 5 source × 768 rows와 두 aggregate의 3,840-row multiset 일치
- source provenance report/metrics/summary/checkpoint hash 일치
- `CONTROLLED_REFERENCE`, `scientific_evidence=false`
- 원본 E21b 무효 disposition과 E21b-R1 단독 claim eligibility
