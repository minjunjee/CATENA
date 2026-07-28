# E10–E21 결과 요약 감사 index tooling

`scripts/write_postcore_results_summary_index.py`는 E18b aggregate와 각 run의
`RESULTS_SUMMARY_KO.md` 생성이 끝난 뒤 한 번 실행한다.

## 안전 계약

- 출력은 artifact root 최상단의
  `POSTCORE_E10_E21_RESULTS_SUMMARY_INDEX_KO.md`로 고정한다.
- `O_EXCL`로만 생성하므로 기존 파일을 덮어쓰지 않는다.
- 기존 `POSTCORE_E10_E16_RESULTS_SUMMARY_INDEX_KO.md`를 읽거나 수정하지
  않는다.
- run directory 내부에는 아무 파일도 쓰지 않는다.
- E10–E20 번호별 summary coverage와 E18b aggregate summary가 없으면
  차단한다.
- E21은 completed summary가 실제로 존재할 때만 검증하고 연결한다.
- 각 summary에 대해 canonical UTC run path, report/manifest pair,
  manifest identity, 가능한 경우 report SHA-256, UTF-8, 최대 60 lines,
  최대 8,000 bytes를 검증한다.
- index에는 summary의 상대 링크, line/byte 수, SHA-256을 기록한다.

## E18 완료 후 실행

먼저 E18 run별/aggregate summary와 freeze를 완료한다. 그다음:

```bash
python scripts/write_postcore_results_summary_index.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --dry-run
```

Dry-run 결과를 확인한 뒤 새 index를 생성한다.

```bash
python scripts/write_postcore_results_summary_index.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts
```

생성 후 재현 검증:

```bash
python scripts/write_postcore_results_summary_index.py \
  --artifact-root /data/minjun_dev/CATENA/artifacts \
  --validate-existing
```

E21이 아직 없을 때 생성하면 E21 section은 부재 상태로 고정된다. E21을 같은
index에 포함하려면 E21 summary까지 완료된 후 최초 생성을 수행해야 한다.
