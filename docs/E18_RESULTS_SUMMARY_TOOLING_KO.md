# E18 run별 1페이지 결과 요약 도구

`scripts/write_e18_result_summaries.py`는 완료된 E18 artifact에 누락된
`RESULTS_SUMMARY_KO.md`만 추가하는 비과학적 문서화 도구다. E18 config,
protocol lock, metric, report, manifest, checkpoint를 수정하지 않으며
aggregate를 실행하거나 scientific input을 선택하지 않는다.

## 안전 계약

- 최종 pre-main lock SHA-256
  `7c465ceb60b6979e717d85599533bd7c0dd884f10b191fa29c42771ccc9c9989`를
  요구한다.
- `latest.json`을 source 선택에 사용하지 않는다.
- E18a의 incomplete·duplicate·live MAIN이 하나라도 있으면 아무것도 쓰지
  않는다.
- E18b live/incomplete/duplicate MAIN도 동일하게 차단한다.
- E18a는 aggregate 코드가 사용하는 원래 provenance validator로 report,
  manifest, config, 48-row grid, checkpoint hash와 lock hash를 재검증한다.
- E18b는 25 source run과 1,200 metric row에서 contrast·gate를 다시
  계산하고 report 및 세 derived JSONL과 exact-match일 때만 요약한다.
- 기존 summary는 건너뛰며 `O_EXCL`로 새 파일만 생성한다. 덮어쓰지 않는다.
- 각 summary는 60줄·UTF-8 8,000 byte 이하여야 한다.

## 사용법

먼저 쓰기 없는 검사를 수행한다.

```bash
python scripts/write_e18_result_summaries.py \
  --repo-root /home/minjun_dev/CATENA \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --dry-run
```

E18a wave가 모두 정지한 시점에는 완료된 source run의 누락 요약만 추가할
수 있다.

```bash
python scripts/write_e18_result_summaries.py \
  --scope source
```

E18b가 별도 CPU command로 완료된 뒤 source와 aggregate의 누락 요약을
함께 생성한다.

```bash
python scripts/write_e18_result_summaries.py \
  --require-aggregate
```

`--scope aggregate`는 provenance-valid E18b MAIN이 실제로 존재할 때만
aggregate summary action을 만든다. `--require-aggregate`를 함께 쓰면
aggregate 부재도 명시적 오류로 처리한다.

요약 생성과 재현 검증이 끝난 뒤에만 E18 top-level freeze를 쓴다.

```bash
python scripts/freeze_e18_sequence_lattice.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts

python scripts/freeze_e18_sequence_lattice.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --validate-existing
```

Freeze 도구는 source 25/25, aggregate 1개, 1,200 metric rows, derived
JSONL 재현, protocol lock 및 `RESULTS_SUMMARY_KO.md` 내용 일치를 다시
검사한다. 기존 `E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json`은 절대
덮어쓰지 않는다.

개별 E18a summary는 `PENDING_AGGREGATE`를 유지한다. Architecture 간
paired claim은 E18b summary에서만 기록하며, 두 문서 모두 oracle
address/candidate뿐 아니라 explicit demand descriptor와 model-visible
verified bit가 제공되는 `CONTROLLED_REFERENCE` 경계를 명시한다. E18b의
primary 값은 registered-grid mean이며 every-cell 또는 uniform persistence를
뜻하지 않는다. Stress는 5/5 positive 방향성 gate이고 별도 SESOI가 없으며,
simpler-demand/retention은 absolute accuracy가 아닌 adjacent cell-mean
non-inferiority다.
