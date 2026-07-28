# Post-core artifact 계약

모든 신규 실험은 `/data/minjun_dev/CATENA/artifacts/<experiment_id>/<run_id>/` 아래에 새 run을 만든다. 기존 run, `latest.json`, 완료된 H1-H5 report는 수정하지 않는다.

각 run은 최소한 다음을 포함한다.

- `config.resolved.yaml`: 실행 시점의 해석된 설정
- `environment.json`: Python, PyTorch, CUDA, GPU 정보
- `run_manifest.json`: source/config hash와 dry/main mode
- `report.json`: execution status와 claim gate
- 실험별 raw JSONL
- 학습 실험은 checkpoint와 optimizer-free inference metadata

## 금지 사항

1. dry-run과 main 결과를 같은 run directory에 쓰지 않는다.
2. `latest.json`이 가리키는 report를 사후 편집하지 않는다.
3. E13c는 등록된 5개 seed 각각에서 tied와 dual의 완전한 update×gap
   paired cell, 유일한 eligible source run, 일치하는 source config가 모두
   존재할 때만 실행한다.
4. E13b는 원본 E13a가 아니라 prospective E13a-R1 report hash와 GO를
   dependency로 기록한다.
5. E14는 E13b checkpoint를 직접 hash하고, 해당 경로를 결과 row에 저장한다.
6. E15가 `PASS`하지 않은 backend를 scientific evidence로 승격하지 않는다.
7. H5 NO-GO를 다시 열기 위한 새 semantic threshold를 만들지 않는다.

## 상태 점검

```bash
bash scripts/check_postcore_status.sh   /home/minjun_dev/CATENA   /data/minjun_dev/CATENA/artifacts
```
